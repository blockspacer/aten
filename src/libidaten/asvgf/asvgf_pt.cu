#include "asvgf/asvgf.h"

#include "kernel/StreamCompaction.h"

#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/accelerator.cuh"
#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__global__ void genPathASVGF(
    idaten::TileDomain tileDomain,
    bool isFillAOV,
    idaten::SVGFPathTracing::Path* paths,
    aten::ray* rays,
    int width, int height,
    int maxBounces,
    unsigned int frame,
    const aten::CameraParameter* __restrict__ camera,
    cudaTextureObject_t blueNoise,
    int blueNoiseResW, int blueNoiseResH, int blueNoiseLayerNum,
    const unsigned int* __restrict__ random)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    const auto idx = getIdx(ix, iy, width);

    paths->attrib[idx].isHit = false;

    if (paths->attrib[idx].isKill) {
        paths->attrib[idx].isTerminate = true;
        return;
    }

#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
    auto scramble = random[idx] * 0x1fe3434f;
    paths->sampler[idx].init(frame, 0, scramble, samplerValues);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
    auto rnd = random[idx];
    auto scramble = rnd * 0x1fe3434f * ((frame + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
    paths->sampler[idx].init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 0, scramble);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_BLUENOISE
    paths->sampler[idx].init(
        ix, iy, frame,
        maxBounces,
        idaten::SVGFPathTracing::ShadowRayNum,
        blueNoiseResW, blueNoiseResH, blueNoiseLayerNum,
        blueNoise);
#endif

    float r1 = paths->sampler[idx].nextSample();
    float r2 = paths->sampler[idx].nextSample();

    if (isFillAOV) {
        r1 = r2 = 0.5f;
    }

    ix += tileDomain.x;
    iy += tileDomain.y;

    float s = (ix + r1) / (float)(camera->width);
    float t = (iy + r2) / (float)(camera->height);

    AT_NAME::CameraSampleResult camsample;
    AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

    rays[idx] = camsample.r;

    paths->throughput[idx].throughput = aten::vec3(1);
    paths->throughput[idx].pdfb = 0.0f;
    paths->attrib[idx].isTerminate = false;
    paths->attrib[idx].isSingular = false;

    paths->contrib[idx].samples += 1;

    // Accumulate value, so do not reset.
    //path.contrib = aten::vec3(0);
}

__global__ void shadeASVGF(
    idaten::TileDomain tileDomain,
    float4* aovNormalDepth,
    float4* aovTexclrMeshid,
    aten::mat4 mtxW2C,
    int width, int height,
    idaten::SVGFPathTracing::Path* paths,
    const int* __restrict__ hitindices,
    int* hitnum,
    const aten::Intersection* __restrict__ isects,
    aten::ray* rays,
    int frame,
    int bounce, int rrBounce,
    const aten::GeomParameter* __restrict__ shapes, int geomnum,
    const aten::MaterialParameter* __restrict__ mtrls,
    const aten::LightParameter* __restrict__ lights, int lightnum,
    const aten::PrimitiveParamter* __restrict__ prims,
    cudaTextureObject_t vtxPos,
    cudaTextureObject_t vtxNml,
    const aten::mat4* __restrict__ matrices,
    cudaTextureObject_t* textures,
    unsigned int* random,
    cudaTextureObject_t blueNoise,
    idaten::SVGFPathTracing::ShadowRay* shadowRays)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= *hitnum) {
        return;
    }

    Context ctxt;
    {
        ctxt.geomnum = geomnum;
        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lightnum = lightnum;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.vtxPos = vtxPos;
        ctxt.vtxNml = vtxNml;
        ctxt.matrices = matrices;
        ctxt.textures = textures;
    }

    idx = hitindices[idx];

    __shared__ idaten::SVGFPathTracing::ShadowRay shShadowRays[64 * idaten::SVGFPathTracing::ShadowRayNum];
    __shared__ aten::MaterialParameter shMtrls[64];

    const auto ray = rays[idx];

#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
    auto scramble = random[idx] * 0x1fe3434f;
    paths->sampler[idx].init(frame, 4 + bounce * 300, scramble);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
    auto rnd = random[idx];
    auto scramble = rnd * 0x1fe3434f * ((frame + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
    paths->sampler[idx].init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 4 + bounce * 300, scramble);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_BLUENOISE
    // Not need to do.
#endif

    aten::hitrecord rec;

    const auto& isect = isects[idx];

    auto obj = &ctxt.shapes[isect.objid];
    evalHitResult(&ctxt, obj, ray, &rec, &isect);

    bool isBackfacing = dot(rec.normal, -ray.dir) < 0.0f;

    // �����ʒu�̖@��.
    // ���̂���̃��C�̓��o���l��.
    aten::vec3 orienting_normal = rec.normal;

    if (rec.mtrlid >= 0) {
        shMtrls[threadIdx.x] = ctxt.mtrls[rec.mtrlid];

#if 1
        if (rec.isVoxel)
        {
            // Replace to lambert.
            const auto& albedo = ctxt.mtrls[rec.mtrlid].baseColor;
            shMtrls[threadIdx.x] = aten::MaterialParameter(aten::MaterialType::Lambert, MaterialAttributeLambert);
            shMtrls[threadIdx.x].baseColor = albedo;
        }
#endif

        if (shMtrls[threadIdx.x].type != aten::MaterialType::Layer) {
            shMtrls[threadIdx.x].albedoMap = (int)(shMtrls[threadIdx.x].albedoMap >= 0 ? ctxt.textures[shMtrls[threadIdx.x].albedoMap] : -1);
            shMtrls[threadIdx.x].normalMap = (int)(shMtrls[threadIdx.x].normalMap >= 0 ? ctxt.textures[shMtrls[threadIdx.x].normalMap] : -1);
            shMtrls[threadIdx.x].roughnessMap = (int)(shMtrls[threadIdx.x].roughnessMap >= 0 ? ctxt.textures[shMtrls[threadIdx.x].roughnessMap] : -1);
        }
    }
    else {
        // TODO
        shMtrls[threadIdx.x] = aten::MaterialParameter(aten::MaterialType::Lambert, MaterialAttributeLambert);
        shMtrls[threadIdx.x].baseColor = aten::vec3(1.0f);
    }


    // Render AOVs.
    // NOTE
    // �����ɖ@����AOV�ɕێ�����Ȃ�A�@���}�b�v�K�p�シ��ׂ�.
    // �������Atemporal reprojection�Aatrous�Ȃǂ̃t�B���^�K�p���ɖ@�����Q�Ƃ���ۂɁA�@���}�b�v���ׂ������Ă͂�����Ă��܂����Ƃ�����.
    // ����ɂ��A�t�B���^�����������悤�ɂ����炸�t�B���^�̕i�����������Ă��܂���肪��������.
    if (bounce == 0) {
        int ix = idx % tileDomain.w;
        int iy = idx / tileDomain.w;

        ix += tileDomain.x;
        iy += tileDomain.y;

        const auto _idx = getIdx(ix, iy, width);

        // World coordinate to Clip coordinate.
        aten::vec4 pos = aten::vec4(rec.p, 1);
        pos = mtxW2C.apply(pos);

        // normal, depth
        aovNormalDepth[_idx] = make_float4(orienting_normal.x, orienting_normal.y, orienting_normal.z, pos.w);

        // texture color, meshid.
        auto texcolor = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec3(1.0f));
#if 0
        aovTexclrMeshid[_idx] = make_float4(texcolor.x, texcolor.y, texcolor.z, isect.meshid);
#else
        aovTexclrMeshid[_idx] = make_float4(texcolor.x, texcolor.y, texcolor.z, isect.mtrlid);
#endif

        // For exporting separated albedo.
        shMtrls[threadIdx.x].albedoMap = -1;
    }
    // TODO
    // How to deal Refraction?
    else if (bounce == 1 && paths->attrib[idx].mtrlType == aten::MaterialType::Specular) {
        int ix = idx % tileDomain.w;
        int iy = idx / tileDomain.w;

        ix += tileDomain.x;
        iy += tileDomain.y;

        const auto _idx = getIdx(ix, iy, width);

        // World coordinate to Clip coordinate.
        aten::vec4 pos = aten::vec4(rec.p, 1);
        pos = mtxW2C.apply(pos);

        // normal, depth
        aovNormalDepth[_idx] = make_float4(orienting_normal.x, orienting_normal.y, orienting_normal.z, pos.w);

        // texture color.
        auto texcolor = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec3(1.0f));
#if 0
        aovTexclrMeshid[_idx] = make_float4(texcolor.x, texcolor.y, texcolor.z, isect.meshid);
#else
        aovTexclrMeshid[_idx] = make_float4(texcolor.x, texcolor.y, texcolor.z, isect.mtrlid);
#endif

        // For exporting separated albedo.
        shMtrls[threadIdx.x].albedoMap = -1;
    }

    // Implicit conection to light.
    if (shMtrls[threadIdx.x].attrib.isEmissive) {
        if (!isBackfacing) {
            float weight = 1.0f;

            if (bounce > 0 && !paths->attrib[idx].isSingular) {
                auto cosLight = dot(orienting_normal, -ray.dir);
                auto dist2 = aten::squared_length(rec.p - ray.org);

                if (cosLight >= 0) {
                    auto pdfLight = 1 / rec.area;

                    // Convert pdf area to sradian.
                    // http://www.slideshare.net/h013/edubpt-v100
                    // p31 - p35
                    pdfLight = pdfLight * dist2 / cosLight;

                    weight = paths->throughput[idx].pdfb / (pdfLight + paths->throughput[idx].pdfb);
                }
            }

            auto contrib = paths->throughput[idx].throughput * weight * shMtrls[threadIdx.x].baseColor;
            paths->contrib[idx].contrib += make_float3(contrib.x, contrib.y, contrib.z);
        }

        // When ray hit the light, tracing will finish.
        paths->attrib[idx].isTerminate = true;
        return;
    }

    if (!shMtrls[threadIdx.x].attrib.isTranslucent && isBackfacing) {
        orienting_normal = -orienting_normal;
    }

    // Apply normal map.
    int normalMap = shMtrls[threadIdx.x].normalMap;
    if (shMtrls[threadIdx.x].type == aten::MaterialType::Layer) {
        // �ŕ\�w�� NormalMap ��K�p.
        auto* topmtrl = &ctxt.mtrls[shMtrls[threadIdx.x].layer[0]];
        normalMap = (int)(topmtrl->normalMap >= 0 ? ctxt.textures[topmtrl->normalMap] : -1);
    }
    AT_NAME::applyNormalMap(normalMap, orienting_normal, orienting_normal, rec.u, rec.v);

    auto albedo = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec3(1), bounce);

#if 1
#pragma unroll
    for (int i = 0; i < idaten::SVGFPathTracing::ShadowRayNum; i++) {
        shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].isActive = false;
    }

    // Explicit conection to light.
    if (!(shMtrls[threadIdx.x].attrib.isSingular || shMtrls[threadIdx.x].attrib.isTranslucent))
    {
        auto shadowRayOrg = rec.p + AT_MATH_EPSILON * orienting_normal;

        for (int i = 0; i < idaten::SVGFPathTracing::ShadowRayNum; i++) {
            real lightSelectPdf = 1;
            aten::LightSampleResult sampleres;

            // TODO
            // Importance sampling.
            int lightidx = aten::cmpMin<int>(paths->sampler[idx].nextSample() * lightnum, lightnum - 1);
            lightSelectPdf = 1.0f / lightnum;

            aten::LightParameter light;
            light.pos = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 0];
            light.dir = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 1];
            light.le = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 2];
            light.v0 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 3];
            light.v1 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 4];
            light.v2 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 5];
            //auto light = ctxt.lights[lightidx];

            sampleLight(&sampleres, &ctxt, &light, rec.p, orienting_normal, &paths->sampler[idx], bounce);

            const auto& posLight = sampleres.pos;
            const auto& nmlLight = sampleres.nml;
            real pdfLight = sampleres.pdf;

            auto dirToLight = normalize(sampleres.dir);
            auto distToLight = length(posLight - rec.p);

            auto tmp = rec.p + dirToLight - shadowRayOrg;
            auto shadowRayDir = normalize(tmp);

            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].isActive = true;
            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].rayorg = shadowRayOrg;
            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].raydir = shadowRayDir;
            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].targetLightId = lightidx;
            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].distToLight = distToLight;
            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].lightcontrib = aten::vec3(0);
            {
                auto cosShadow = dot(orienting_normal, dirToLight);

                real pdfb = samplePDF(&ctxt, &shMtrls[threadIdx.x], orienting_normal, ray.dir, dirToLight, rec.u, rec.v);
                auto bsdf = sampleBSDF(&ctxt, &shMtrls[threadIdx.x], orienting_normal, ray.dir, dirToLight, rec.u, rec.v, albedo);

                bsdf *= paths->throughput[idx].throughput;

                // Get light color.
                auto emit = sampleres.finalColor;

                if (light.attrib.isSingular || light.attrib.isInfinite) {
                    if (pdfLight > real(0) && cosShadow >= 0) {
                        // TODO
                        // �W�I���g���^�[���̈����ɂ���.
                        // singular light �̏ꍇ�́AfinalColor �ɋ����̏��Z���܂܂�Ă���.
                        // inifinite light �̏ꍇ�́A���������ɂȂ�ApdfLight�Ɋ܂܂�鋗�������Ƒł����������H.
                        // �i�ł����������̂ŁApdfLight�ɂ͋��������͊܂�ł��Ȃ��j.
                        auto misW = pdfLight / (pdfb + pdfLight);

                        shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].lightcontrib =
                            (misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf / (float)idaten::SVGFPathTracing::ShadowRayNum;
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

                            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].lightcontrib =
                                (misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf / (float)idaten::SVGFPathTracing::ShadowRayNum;;
                        }
                    }
                }
            }
        }
    }
#endif

    real russianProb = real(1);

    if (bounce > rrBounce) {
        auto t = normalize(paths->throughput[idx].throughput);
        auto p = aten::cmpMax(t.r, aten::cmpMax(t.g, t.b));

        russianProb = paths->sampler[idx].nextSample();

        if (russianProb >= p) {
            //shPaths[threadIdx.x].contrib = aten::vec3(0);
            paths->attrib[idx].isTerminate = true;
        }
        else {
            russianProb = max(p, 0.01f);
        }
    }
            
    AT_NAME::MaterialSampling sampling;

    sampleMaterial(
        &sampling,
        &ctxt,
        &shMtrls[threadIdx.x],
        orienting_normal,
        ray.dir,
        rec.normal,
        &paths->sampler[idx],
        rec.u, rec.v,
        albedo);

    auto nextDir = normalize(sampling.dir);
    auto pdfb = sampling.pdf;
    auto bsdf = sampling.bsdf;

    real c = 1;
    if (!shMtrls[threadIdx.x].attrib.isSingular) {
        // TODO
        // AMD�̂�abs���Ă��邪....
        c = aten::abs(dot(orienting_normal, nextDir));
        //c = dot(orienting_normal, nextDir);
    }

    if (pdfb > 0 && c > 0) {
        paths->throughput[idx].throughput *= bsdf * c / pdfb;
        paths->throughput[idx].throughput /= russianProb;
    }
    else {
        paths->attrib[idx].isTerminate = true;
    }

    // Make next ray.
    rays[idx] = aten::ray(rec.p, nextDir);

    paths->throughput[idx].pdfb = pdfb;
    paths->attrib[idx].isSingular = shMtrls[threadIdx.x].attrib.isSingular;
    paths->attrib[idx].mtrlType = shMtrls[threadIdx.x].type;

#pragma unroll
    for (int i = 0; i < idaten::SVGFPathTracing::ShadowRayNum; i++) {
        shadowRays[idx * idaten::SVGFPathTracing::ShadowRayNum + i] = shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i];
    }
}

__global__ void gatherASVGF(
    idaten::TileDomain tileDomain,
    cudaSurfaceObject_t dst,
    float4* aovColorVariance,
    float4* aovMomentTemporalWeight,
    const idaten::SVGFPathTracing::Path* __restrict__ paths,
    float4* contribs,
    int width, int height)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    auto idx = getIdx(ix, iy, tileDomain.w);

    auto r = paths->sampler[idx].nextSample();

    if (dst) {
        surf2Dwrite(
            make_float4(r, r, r, 1),
            dst,
            ix * sizeof(float4), iy,
            cudaBoundaryModeTrap);
    }
}

namespace idaten
{
    void AdvancedSVGFPathTracing::onGenPath(
        int maxBounce,
        int seed,
        cudaTextureObject_t texVtxPos,
        cudaTextureObject_t texVtxNml)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        bool isFillAOV = m_mode == Mode::AOVar;

        auto blueNoise = m_bluenoise.bind();
        auto blueNoiseResW = m_bluenoise.getWidth();
        auto blueNoiseResH = m_bluenoise.getHeight();
        auto blueNoiseLayerNum = m_bluenoise.getLayerNum();

        genPathASVGF << <grid, block, 0, m_stream >> > (
            m_tileDomain,
            isFillAOV,
            m_paths.ptr(),
            m_rays.ptr(),
            m_tileDomain.w, m_tileDomain.h,
            maxBounce,
            m_frame,
            m_cam.ptr(),
            blueNoise,
            blueNoiseResW, blueNoiseResH, blueNoiseLayerNum,
            m_random.ptr());

        checkCudaKernel(genPath);
    }

    void AdvancedSVGFPathTracing::onShade(
        cudaSurfaceObject_t outputSurf,
        int width, int height,
        int bounce, int rrBounce,
        cudaTextureObject_t texVtxPos,
        cudaTextureObject_t texVtxNml)
    {
        m_mtxW2V.lookat(
            m_camParam.origin,
            m_camParam.center,
            m_camParam.up);

        m_mtxV2C.perspective(
            m_camParam.znear,
            m_camParam.zfar,
            m_camParam.vfov,
            m_camParam.aspect);

        m_mtxC2V = m_mtxV2C;
        m_mtxC2V.invert();

        m_mtxV2W = m_mtxW2V;
        m_mtxV2W.invert();

        aten::mat4 mtxW2C = m_mtxV2C * m_mtxW2V;

        dim3 blockPerGrid(((m_tileDomain.w * m_tileDomain.h) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        int curaov = getCurAovs();

        auto blueNoise = m_bluenoise.bind();

        shadeASVGF << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            m_tileDomain,
            m_aovNormalDepth[curaov].ptr(),
            m_aovTexclrMeshid[curaov].ptr(),
            mtxW2C,
            width, height,
            m_paths.ptr(),
            m_hitidx.ptr(), hitcount.ptr(),
            m_isects.ptr(),
            m_rays.ptr(),
            m_frame,
            bounce, rrBounce,
            m_shapeparam.ptr(), m_shapeparam.num(),
            m_mtrlparam.ptr(),
            m_lightparam.ptr(), m_lightparam.num(),
            m_primparams.ptr(),
            texVtxPos, texVtxNml,
            m_mtxparams.ptr(),
            m_tex.ptr(),
            m_random.ptr(),
            blueNoise,
            m_shadowRays.ptr());

        checkCudaKernel(shade);

        onShadeByShadowRay(bounce, texVtxPos);

        m_bluenoise.unbind();
    }

    void AdvancedSVGFPathTracing::onGather(
        cudaSurfaceObject_t outputSurf,
        int width, int height,
        int maxSamples)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        int curaov = getCurAovs();

        gatherASVGF << <grid, block, 0, m_stream >> > (
            m_tileDomain,
            outputSurf,
            m_aovColorVariance[curaov].ptr(),
            m_aovMomentTemporalWeight[curaov].ptr(),
            m_paths.ptr(),
            m_tmpBuf.ptr(),
            width, height);

        checkCudaKernel(gather);
    }
}
