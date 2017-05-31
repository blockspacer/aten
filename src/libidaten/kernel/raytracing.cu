#include "kernel/raytracing.h"
#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/bvh.cuh"
#include "kernel/common.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

struct ShadowRay : public aten::ray {
	real distToLight;
	int targetLightId;

	struct {
		uint32_t isActive : 1;
	};
};

struct Path {
	aten::ray ray;
	aten::vec3 throughput;
	aten::Intersection isect;
	bool isHit;
	bool isTerminate;
};

__global__ void genPathRayTracing(
	Path* paths,
	int width, int height,
	aten::CameraParameter* camera)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = iy * camera->width + ix;

	float s = (ix + 0.5f) / (float)(camera->width - 1);
	float t = (iy + 0.5f) / (float)(camera->height - 1);

	AT_NAME::CameraSampleResult camsample;
	AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

	auto& path = paths[idx];

	path.ray = camsample.r;
	path.throughput = aten::make_float3(1);
	path.isHit = false;
	path.isTerminate = false;
}

__global__ void hitTestRayTracing(
	Path* paths,
	int width, int height,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum,
	cudaTextureObject_t* nodes,
	aten::PrimitiveParamter* prims,
	cudaTextureObject_t vertices,
	aten::mat4* matrices)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = iy * width + ix;

	auto& path = paths[idx];
	path.isHit = false;

	if (path.isTerminate) {
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
		ctxt.vertices = vertices;
		ctxt.matrices = matrices;
	}
	
	bool isHit = intersectBVH(&ctxt, path.ray, AT_MATH_EPSILON, AT_MATH_INF, &path.isect);

	path.isHit = isHit;
}

__global__ void raytracing(
	cudaSurfaceObject_t outSurface,
	Path* paths,
	ShadowRay* shadowRays,
	int width, int height,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum,
	cudaTextureObject_t* nodes,
	aten::PrimitiveParamter* prims,
	cudaTextureObject_t vertices,
	aten::mat4* matrices)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
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
		ctxt.vertices = vertices;
		ctxt.matrices = matrices;
	}

	const auto idx = iy * width + ix;

	auto& path = paths[idx];

	shadowRays[idx].isActive = false;

	if (!path.isHit) {
		return;
	}
	if (path.isTerminate) {
		return;
	}

	aten::vec3 contrib = aten::make_float3(0);

	const aten::MaterialParameter* mtrl = &ctxt.mtrls[path.isect.mtrlid];

	if (mtrl->attrib.isEmissive) {
		contrib = path.throughput * mtrl->baseColor;

		path.isTerminate = true;
		//p[idx] = make_float4(contrib.x, contrib.y, contrib.z, 1);
		surf2Dwrite(make_float4(contrib.x, contrib.y, contrib.z, 1), outSurface, ix * sizeof(float4), iy, cudaBoundaryModeTrap);
		
		return;
	}

	aten::hitrecord rec;

	auto obj = &ctxt.shapes[path.isect.objid];
	evalHitResult(&ctxt, obj, path.ray, &rec, &path.isect);

	// 交差位置の法線.
	// 物体からのレイの入出を考慮.
	const aten::vec3 orienting_normal = dot(rec.normal, path.ray.dir) < 0.0 ? rec.normal : -rec.normal;

	if (mtrl->attrib.isSingular || mtrl->attrib.isTranslucent) {
		AT_NAME::MaterialSampling sampling;
			
		sampleMaterial(
			&sampling,
			mtrl,
			orienting_normal, 
			path.ray.dir,
			rec.normal,
			nullptr,
			rec.u, rec.v);

		auto nextDir = normalize(sampling.dir);

		path.throughput *= sampling.bsdf;

		// Make next ray.
		path.ray = aten::ray(rec.p, nextDir);
	}
	else {
		// TODO
		int lightidx = 0;

		auto light = lights[lightidx];
		if (light.object.idx >= 0) {
			light.object.ptr = &ctxt.shapes[light.object.idx];
		}

		aten::LightSampleResult sampleres;
		sampleLight(&sampleres, &ctxt, &light, rec.p, nullptr);

		aten::vec3 dirToLight = sampleres.dir;
		auto len = dirToLight.length();

		dirToLight.normalize();

		shadowRays[idx].isActive = true;
		shadowRays[idx].org = rec.p;
		shadowRays[idx].dir = dirToLight;
		shadowRays[idx].distToLight = len;
		shadowRays[idx].targetLightId = lightidx;

		aten::hitrecord tmpRec;

		if (light.attrib.isInfinite) {
			len = 1.0f;
		}

		const auto c0 = max(0.0f, dot(orienting_normal, dirToLight));
		float c1 = 1.0f;

		if (!light.attrib.isSingular) {
			c1 = max(0.0f, dot(sampleres.nml, -dirToLight));
		}

		auto G = c0 * c1 / (len * len);

		path.throughput = path.throughput * (mtrl->baseColor * sampleres.finalColor) * G;
	}
}

__global__ void hitShadowRay(
	cudaSurfaceObject_t outSurface,
	Path* paths,
	ShadowRay* shadowRays,
	int width, int height,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum,
	cudaTextureObject_t* nodes,
	aten::PrimitiveParamter* prims,
	cudaTextureObject_t vertices,
	aten::mat4* matrices)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = iy * width + ix;

	auto& path = paths[idx];

	if (path.isTerminate) {
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
		ctxt.vertices = vertices;
		ctxt.matrices = matrices;
	}

	auto& shadowRay = shadowRays[idx];

	if (shadowRay.isActive) {
		auto& path = paths[idx];

		aten::Intersection isect;
		bool isHit = intersectBVH(&ctxt, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, &isect);

		real distHitObjToRayOrg = AT_MATH_INF;
		aten::ShapeParameter* obj = nullptr;

		aten::hitrecord rec;

		if (isHit) {
			obj = &ctxt.shapes[isect.objid];

			evalHitResult(&ctxt, obj, shadowRay, &rec, &isect);

			distHitObjToRayOrg = (rec.p - shadowRay.org).length();
		}

		auto light = ctxt.lights[shadowRay.targetLightId];
		if (light.object.idx >= 0) {
			light.object.ptr = &ctxt.shapes[light.object.idx];
		}

		shadowRay.isActive = AT_NAME::scene::hitLight(
			isHit,
			light.attrib,
			light.object.ptr,
			shadowRay.distToLight,
			distHitObjToRayOrg,
			isect.t,
			obj);

		if (shadowRay.isActive) {
			path.isTerminate = true;

			auto contrib = path.throughput;

			//p[idx] = make_float4(contrib.x, contrib.y, contrib.z, 1);
			surf2Dwrite(
				make_float4(contrib.x, contrib.y, contrib.z, 1), outSurface, ix * sizeof(float4), iy, cudaBoundaryModeTrap);
		}
	}
}


namespace idaten {
	void RayTracing::prepare()
	{
		addFuncs();
	}

	void RayTracing::render(
		aten::vec4* image,
		int width, int height)
	{
		dim3 block(16, 16);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		int depth = 0;

		idaten::TypedCudaMemory<Path> paths;
		paths.init(width * height);

		CudaGLResourceMap rscmap(&glimg);
		auto outputSurf = glimg.bind();

		auto vtxTex = vtxparams.bind();

		std::vector<cudaTextureObject_t> tmp;
		for (int i = 0; i < nodeparam.size(); i++) {
			auto nodeTex = nodeparam[i].bind();
			tmp.push_back(nodeTex);
		}
		nodetex.writeByNum(&tmp[0], tmp.size());

		idaten::TypedCudaMemory<ShadowRay> shadowRays;
		shadowRays.init(width * height);

		genPathRayTracing << <grid, block >> > (
			paths.ptr(),
			width, height,
			cam.ptr());

		//checkCudaErrors(cudaDeviceSynchronize());

		while (depth < 5) {
			hitTestRayTracing << <grid, block >> > (
			//hitTestRayTracing << <1, 1 >> > (
				paths.ptr(),
				width, height,
				shapeparam.ptr(), shapeparam.num(),
				mtrlparam.ptr(),
				lightparam.ptr(), lightparam.num(),
				nodetex.ptr(),
				primparams.ptr(),
				vtxTex,
				mtxparams.ptr());

			auto err = cudaGetLastError();
			if (err != cudaSuccess) {
				AT_PRINTF("Cuda Kernel Err(hitTest) [%s]\n", cudaGetErrorString(err));
			}

			//checkCudaErrors(cudaDeviceSynchronize());

			raytracing << <grid, block >> > (
				outputSurf,
				paths.ptr(),
				shadowRays.ptr(),
				width, height,
				shapeparam.ptr(), shapeparam.num(),
				mtrlparam.ptr(),
				lightparam.ptr(), lightparam.num(),
				nodetex.ptr(),
				primparams.ptr(),
				vtxTex,
				mtxparams.ptr());

			err = cudaGetLastError();
			if (err != cudaSuccess) {
				AT_PRINTF("Cuda Kernel Err(raytracing) [%s]\n", cudaGetErrorString(err));
			}

			hitShadowRay << <grid, block >> > (
				//hitShadowRay << <1, 1 >> > (
				outputSurf,
				paths.ptr(),
				shadowRays.ptr(),
				width, height,
				shapeparam.ptr(), shapeparam.num(),
				mtrlparam.ptr(),
				lightparam.ptr(), lightparam.num(),
				nodetex.ptr(),
				primparams.ptr(),
				vtxTex,
				mtxparams.ptr());

			err = cudaGetLastError();
			if (err != cudaSuccess) {
				AT_PRINTF("Cuda Kernel Err(hitShadowRay) [%s]\n", cudaGetErrorString(err));
			}

			//checkCudaErrors(cudaDeviceSynchronize());

			depth++;
		}

		checkCudaErrors(cudaDeviceSynchronize());

		vtxparams.unbind();
		for (int i = 0; i < nodeparam.size(); i++) {
			nodeparam[i].unbind();
		}
		nodetex.reset();

		//dst.read(image, sizeof(aten::vec4) * width * height);
	}
}