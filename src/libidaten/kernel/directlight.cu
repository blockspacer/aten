#include "kernel/directlight.h"
#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/bvh.cuh"
#include "kernel/compaction.h"
#include "kernel/pt_common.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

template <bool needAOV>
__global__ void shade(
	cudaSurfaceObject_t outSurface,
	cudaSurfaceObject_t* aovs,
	float3 posRange,
	int width, int height,
	idaten::PathTracing::Path* paths,
	int* hitindices,
	int hitnum,
	const aten::Intersection* __restrict__ isects,
	aten::ray* rays,
	int bounce, int rrBounce,
	const aten::ShapeParameter* __restrict__ shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	const aten::LightParameter* __restrict__ lights, int lightnum,
	cudaTextureObject_t* nodes,
	const aten::PrimitiveParamter* __restrict__ prims,
	cudaTextureObject_t vtxPos,
	cudaTextureObject_t vtxNml,
	const aten::mat4* __restrict__ matrices,
	cudaTextureObject_t* textures)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= hitnum) {
		return;
	}

	Context ctxt;
	{
		ctxt.geomnum = geomnum;
		ctxt.shapes = shapes;
		ctxt.mtrls = mtrls;
		ctxt.lightnum = lightnum;
		ctxt.lights = lights;
		ctxt.nodes = nodes;
		ctxt.prims = prims;
		ctxt.vtxPos = vtxPos;
		ctxt.vtxNml = vtxNml;
		ctxt.matrices = matrices;
		ctxt.textures = textures;
	}

	idx = hitindices[idx];

	auto& path = paths[idx];
	const auto& ray = rays[idx];

	aten::hitrecord rec;

	const auto& isect = isects[idx];

	auto obj = &ctxt.shapes[isect.objid];
	evalHitResult(&ctxt, obj, ray, &rec, &isect);

	aten::MaterialParameter mtrl = ctxt.mtrls[rec.mtrlid];

	// 交差位置の法線.
	// 物体からのレイの入出を考慮.
	aten::vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

	// Apply normal map.
	if (mtrl.type == aten::MaterialType::Layer) {
		// 最表層の NormalMap を適用.
		auto* topmtrl = &ctxt.mtrls[mtrl.layer[0]];
		auto normalMap = (int)(topmtrl->normalMap >= 0 ? ctxt.textures[topmtrl->normalMap] : -1);
		AT_NAME::material::applyNormalMap(normalMap, orienting_normal, orienting_normal, rec.u, rec.v);
	}
	else {
		mtrl.albedoMap = (int)(mtrl.albedoMap >= 0 ? ctxt.textures[mtrl.albedoMap] : -1);
		mtrl.normalMap = (int)(mtrl.normalMap >= 0 ? ctxt.textures[mtrl.normalMap] : -1);
		mtrl.roughnessMap = (int)(mtrl.roughnessMap >= 0 ? ctxt.textures[mtrl.roughnessMap] : -1);

		AT_NAME::material::applyNormalMap(mtrl.normalMap, orienting_normal, orienting_normal, rec.u, rec.v);
	}

	// Implicit conection to light.
	if (mtrl.attrib.isEmissive) {
		float weight = 1.0f;

		if (bounce > 0 && !path.isSingular) {
			auto cosLight = dot(orienting_normal, -ray.dir);
			auto dist2 = aten::squared_length(rec.p - ray.org);

			if (cosLight >= 0) {
				auto pdfLight = 1 / rec.area;

				// Convert pdf area to sradian.
				// http://www.slideshare.net/h013/edubpt-v100
				// p31 - p35
				pdfLight = pdfLight * dist2 / cosLight;

				weight = path.pdfb / (pdfLight + path.pdfb);
			}
		}

		path.contrib += path.throughput * weight * mtrl.baseColor;

		// When ray hit the light, tracing will finish.
		path.isTerminate = true;
		return;
	}

	// Explicit conection to light.
	if (!mtrl.attrib.isSingular)
	{
		real lightSelectPdf = 1;
		aten::LightSampleResult sampleres;

		// TODO
		// Importance sampling.
		int lightidx = aten::cmpMin<int>(path.sampler.nextSample() * lightnum, lightnum - 1);
		lightSelectPdf = 1.0f / lightnum;

		auto light = ctxt.lights[lightidx];

		sampleLight(&sampleres, &ctxt, &light, rec.p, orienting_normal, &path.sampler);

		const auto& posLight = sampleres.pos;
		const auto& nmlLight = sampleres.nml;
		real pdfLight = sampleres.pdf;

		auto lightobj = sampleres.obj;

		auto dirToLight = normalize(sampleres.dir);
		auto distToLight = length(posLight - rec.p);

		real distHitObjToRayOrg = AT_MATH_INF;

		// Ray aim to the area light.
		// So, if ray doesn't hit anything in intersectCloserBVH, ray hit the area light.
		auto hitobj = lightobj;

		aten::Intersection isectTmp;
		aten::ray shadowRay(rec.p, dirToLight);

		bool isHit = intersectCloserBVH(&ctxt, shadowRay, &isectTmp, distToLight - AT_MATH_EPSILON);

		if (isHit) {
			hitobj = (void*)&ctxt.shapes[isectTmp.objid];
		}

		isHit = AT_NAME::scene::hitLight(
			isHit,
			light.attrib,
			lightobj,
			distToLight,
			distHitObjToRayOrg,
			isectTmp.t,
			hitobj);

		if (isHit) {
			auto cosShadow = dot(orienting_normal, dirToLight);

			real pdfb = samplePDF(&ctxt, &mtrl, orienting_normal, ray.dir, dirToLight, rec.u, rec.v);
			auto bsdf = sampleBSDF(&ctxt, &mtrl, orienting_normal, ray.dir, dirToLight, rec.u, rec.v);

			bsdf *= path.throughput;

			// Get light color.
			auto emit = sampleres.finalColor;

			if (light.attrib.isSingular || light.attrib.isInfinite) {
				if (pdfLight > real(0) && cosShadow >= 0) {
					// TODO
					// ジオメトリタームの扱いについて.
					// singular light の場合は、finalColor に距離の除算が含まれている.
					// inifinite light の場合は、無限遠方になり、pdfLightに含まれる距離成分と打ち消しあう？.
					// （打ち消しあうので、pdfLightには距離成分は含んでいない）.
					auto misW = pdfLight / (pdfb + pdfLight);
					path.contrib += (misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf;
				}
			}
			else {
				auto cosLight = dot(nmlLight, -dirToLight);

				if (cosShadow >= 0 && cosLight >= 0) {
					auto dist2 = aten::squared_length(sampleres.dir);
					auto G = cosShadow * cosLight / dist2;

					if (pdfb > real(0) && pdfLight > real(0)) {
						// Convert pdf from steradian to area.
						// http://www.slideshare.net/h013/edubpt-v100
						// p31 - p35
						pdfb = pdfb * cosLight / dist2;

						auto misW = pdfLight / (pdfb + pdfLight);

						path.contrib += (misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf;
					}
				}
			}
		}
	}

	real russianProb = real(1);

	if (bounce > rrBounce) {
		auto t = normalize(path.throughput);
		auto p = aten::cmpMax(t.r, aten::cmpMax(t.g, t.b));

		russianProb = path.sampler.nextSample();

		if (russianProb >= p) {
			//path.contrib = aten::vec3(0);
			path.isTerminate = true;
		}
		else {
			russianProb = p;
		}
	}
			
	AT_NAME::MaterialSampling sampling;

	sampleMaterial(
		&sampling,
		&ctxt,
		&mtrl,
		orienting_normal,
		ray.dir,
		rec.normal,
		&path.sampler,
		rec.u, rec.v);

	auto nextDir = normalize(sampling.dir);
	auto pdfb = sampling.pdf;
	auto bsdf = sampling.bsdf;

	real c = 1;
	if (!mtrl.attrib.isSingular) {
		// TODO
		// AMDのはabsしているが....
		//c = aten::abs(dot(orienting_normal, nextDir));
		c = dot(orienting_normal, nextDir);
	}

	if (pdfb > 0 && c > 0) {
		path.throughput *= bsdf * c / pdfb;
		path.throughput /= russianProb;
	}
	else {
		path.isTerminate = true;
	}

	// Make next ray.
	rays[idx] = aten::ray(rec.p, nextDir);

	path.pdfb = pdfb;
	path.isSingular = mtrl.attrib.isSingular;
}

namespace idaten {
	void DirectLightRenderer::onShade(
		cudaSurfaceObject_t outputSurf,
		int hitcount,
		int width, int height,
		int bounce, int rrBounce,
		cudaTextureObject_t texVtxPos,
		cudaTextureObject_t texVtxNml)
	{
		dim3 blockPerGrid((hitcount + 64 - 1) / 64);
		dim3 threadPerBlock(64);

		bool enableAOV = (bounce == 0 && m_enableAOV);
		float3 posRange = make_float3(m_posRange.x, m_posRange.y, m_posRange.z);

		if (enableAOV) {
			shade<true> << <blockPerGrid, threadPerBlock >> > (
			//shade<true> << <1, 1 >> > (
				outputSurf,
				m_aovCudaRsc.ptr(), posRange,
				width, height,
				paths.ptr(),
				m_hitidx.ptr(), hitcount,
				isects.ptr(),
				rays.ptr(),
				bounce, rrBounce,
				shapeparam.ptr(), shapeparam.num(),
				mtrlparam.ptr(),
				lightparam.ptr(), lightparam.num(),
				nodetex.ptr(),
				primparams.ptr(),
				texVtxPos, texVtxNml,
				mtxparams.ptr(),
				tex.ptr());
		}
		else {
			shade<false> << <blockPerGrid, threadPerBlock >> > (
			//shade<false> << <1, 1 >> > (
				outputSurf,
				m_aovCudaRsc.ptr(), posRange,
				width, height,
				paths.ptr(),
				m_hitidx.ptr(), hitcount,
				isects.ptr(),
				rays.ptr(),
				bounce, rrBounce,
				shapeparam.ptr(), shapeparam.num(),
				mtrlparam.ptr(),
				lightparam.ptr(), lightparam.num(),
				nodetex.ptr(),
				primparams.ptr(),
				texVtxPos, texVtxNml,
				mtxparams.ptr(),
				tex.ptr());
		}

		checkCudaKernel(shade);
	}
}
