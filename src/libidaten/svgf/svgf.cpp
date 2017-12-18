#include "svgf/svgf_pt.h"
#include "kernel/compaction.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

namespace idaten
{
	void SVGFPathTracing::update(
		GLuint gltex,
		int width, int height,
		const aten::CameraParameter& camera,
		const std::vector<aten::GeomParameter>& shapes,
		const std::vector<aten::MaterialParameter>& mtrls,
		const std::vector<aten::LightParameter>& lights,
		const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
		const std::vector<aten::PrimitiveParamter>& prims,
		const std::vector<aten::vertex>& vtxs,
		const std::vector<aten::mat4>& mtxs,
		const std::vector<TextureResource>& texs,
		const EnvmapResource& envmapRsc)
	{
		idaten::Renderer::update(
			gltex,
			width, height,
			camera,
			shapes,
			mtrls,
			lights,
			nodes,
			prims,
			vtxs,
			mtxs,
			texs, envmapRsc);

		m_hitbools.init(width * height);
		m_hitidx.init(width * height);

		m_sobolMatrices.init(AT_COUNTOF(sobol::Matrices::matrices));
		m_sobolMatrices.writeByNum(sobol::Matrices::matrices, m_sobolMatrices.maxNum());

		auto& r = aten::getRandom();
		m_random.init(width * height);
		m_random.writeByNum(&r[0], width * height);

		for (int r = 0; r < (int)Resolution::Num; r++) {
			int w = width >> r;
			int h = height >> r;

			for (int i = 0; i < 2; i++) {
				m_aovNormalDepth[r][i].init(w * h);
				m_aovTexclrTemporalWeight[r][i].init(w * h);
				m_aovColorVariance[r][i].init(w * h);
				m_aovMomentMeshid[r][i].init(w * h);
			}
		}

		for (int i = 0; i < AT_COUNTOF(m_atrousClrVar); i++) {
			m_atrousClrVar[i].init(width * height);
		}

		m_tmpBuf.init(width * height);
	}

	void SVGFPathTracing::setAovExportBuffer(GLuint gltexId)
	{
		m_aovGLBuffer.init(gltexId, CudaGLRscRegisterType::WriteOnly);
	}

	void SVGFPathTracing::setGBuffer(GLuint gltexGbuffer)
	{
		m_gbuffer.init(gltexGbuffer, idaten::CudaGLRscRegisterType::ReadOnly);
	}

	static bool doneSetStackSize = false;

	void SVGFPathTracing::render(
		int width, int height,
		int maxSamples,
		int maxBounce)
	{
#ifdef __AT_DEBUG__
		if (!doneSetStackSize) {
			size_t val = 0;
			cudaThreadGetLimit(&val, cudaLimitStackSize);
			cudaThreadSetLimit(cudaLimitStackSize, val * 4);
			doneSetStackSize = true;
		}
#endif

		int bounce = 0;

		m_paths[Resolution::Hi].init(width * height);
		m_paths[Resolution::Low].init((width / 2) * (height / 2));

		m_rays[Resolution::Hi].init(width * height);
		m_rays[Resolution::Low].init((width / 2) * (height / 2));

		m_isects.init(width * height);

		m_shadowRays.init(width * height);

		cudaMemset(m_paths[Resolution::Hi].ptr(), 0, m_paths[Resolution::Hi].bytes());
		cudaMemset(m_paths[Resolution::Low].ptr(), 0, m_paths[Resolution::Low].bytes());

		CudaGLResourceMap rscmap(&m_glimg);
		auto outputSurf = m_glimg.bind();

		auto vtxTexPos = m_vtxparamsPos.bind();
		auto vtxTexNml = m_vtxparamsNml.bind();

		{
			std::vector<cudaTextureObject_t> tmp;
			for (int i = 0; i < m_nodeparam.size(); i++) {
				auto nodeTex = m_nodeparam[i].bind();
				tmp.push_back(nodeTex);
			}
			m_nodetex.writeByNum(&tmp[0], tmp.size());
		}

		if (!m_texRsc.empty())
		{
			std::vector<cudaTextureObject_t> tmp;
			for (int i = 0; i < m_texRsc.size(); i++) {
				auto cudaTex = m_texRsc[i].bind();
				tmp.push_back(cudaTex);
			}
			m_tex.writeByNum(&tmp[0], tmp.size());
		}

		cudaSurfaceObject_t aovExportBuffer = 0;
		if (m_aovGLBuffer.isValid()) {
			m_aovGLBuffer.map();
			aovExportBuffer = m_aovGLBuffer.bind();
		}

		static const int rrBounce = 3;

		// Set bounce count to 1 forcibly, aov render mode.
		maxBounce = (m_mode == Mode::AOVar ? 1 : maxBounce);

		auto time = AT_NAME::timer::getSystemTime();

		for (int i = 0; i < maxSamples; i++) {
			int seed = time.milliSeconds;
			//int seed = 0;

			onGenPath(
				width, height,
				i, maxSamples,
				seed,
				vtxTexPos,
				vtxTexNml);

			bounce = 0;

			// First bounce.
			{
				onHitTest(
					Resolution::Hi,
					width, height,
					bounce,
					vtxTexPos);

				onShadeMiss(
					Resolution::Hi,
					width, height, bounce, aovExportBuffer);

				int hitcount = 0;
				idaten::Compaction::compact(
					m_hitidx,
					m_hitbools,
					&hitcount);

				//AT_PRINTF("%d\n", hitcount);

				if (hitcount == 0) {
					break;
				}

				onShade(
					Resolution::Hi,
					outputSurf,
					aovExportBuffer,
					hitcount,
					width, height,
					bounce, rrBounce,
					vtxTexPos, vtxTexNml);

				bounce++;
			}

#if 1
			// Check if 2nd bounce ray hits light.
			if (bounce < maxBounce) {
				onHitTest(
					Resolution::Hi,
					width, height,
					bounce,
					vtxTexPos);

				onShadeMiss(
					Resolution::Hi,
					width, height, bounce, aovExportBuffer);

				int hitcount = 0;
				idaten::Compaction::compact(
					m_hitidx,
					m_hitbools,
					&hitcount);

				//AT_PRINTF("%d\n", hitcount);

				if (hitcount > 0) {
					onShadeDirectLight(
						hitcount,
						width, height,
						bounce, rrBounce,
						vtxTexPos, vtxTexNml);
				}
			}
#endif

			coarseBuffer(width, height);

#if 1
			// For specular.
			while (bounce < maxBounce) {
				onHitTest(
					Resolution::Hi,
					width, height,
					bounce,
					vtxTexPos);
				
				onShadeMiss(Resolution::Hi, width, height, bounce, aovExportBuffer);

				int hitcount = 0;
				idaten::Compaction::compact(
					m_hitidx,
					m_hitbools,
					&hitcount);

				//AT_PRINTF("%d\n", hitcount);

				if (hitcount == 0) {
					break;
				}

				onShade(
					Resolution::Hi,
					outputSurf,
					aovExportBuffer,
					hitcount,
					width, height,
					bounce, rrBounce,
					vtxTexPos, vtxTexNml);

				bounce++;
			}
#endif

			// Clear for low resolution.
			cudaMemset(m_hitbools.ptr(), 0, m_hitbools.bytes());

			// Reset.
			bounce = 1;

			// For low resolution.
			while (bounce < maxBounce) {
				onHitTest(
					Resolution::Low,
					width / 2, height / 2,
					bounce,
					vtxTexPos);

				onShadeMiss(Resolution::Low, width / 2, height / 2, bounce, aovExportBuffer);

				int hitcount = 0;
				idaten::Compaction::compact(
					m_hitidx,
					m_hitbools,
					&hitcount);

				//AT_PRINTF("%d\n", hitcount);

				if (hitcount == 0) {
					break;
				}

				onShade(
					Resolution::Low,
					outputSurf,
					aovExportBuffer,
					hitcount,
					width / 2, height / 2,
					bounce, rrBounce,
					vtxTexPos, vtxTexNml);

				bounce++;
			}
		}

		onGather(
			Resolution::Hi,
			outputSurf, width, height, maxSamples);
		onGather(
			Resolution::Low,
			outputSurf, width / 2, height / 2, maxSamples);

		if (m_mode == Mode::SVGF
			|| m_mode == Mode::ATrous)
		{
			onVarianceEstimation(
				Resolution::Hi,
				outputSurf, width, height);
			onAtrousFilter(
				Resolution::Hi,
				outputSurf, width, height);

			onVarianceEstimation(
				Resolution::Low,
				outputSurf, width / 2, height / 2);
			onAtrousFilter(
				Resolution::Low,
				outputSurf, width / 2, height / 2);

			if (m_mode == Mode::SVGF) {
				upsamplingAndMerge(outputSurf, width, height);
			}
		}
		else if (m_mode == Mode::VAR) {
			onVarianceEstimation(
				Resolution::Hi,
				outputSurf, width, height);

			onVarianceEstimation(
				Resolution::Low,
				outputSurf, width / 2, height / 2);
		}

		pick(
			m_pickedInfo.ix, m_pickedInfo.iy, 
			width, height,
			vtxTexPos);

		checkCudaErrors(cudaDeviceSynchronize());

		// Toggle aov buffer pos.
		m_curAOVPos = 1 - m_curAOVPos;

		m_frame++;

		{
			m_vtxparamsPos.unbind();
			m_vtxparamsNml.unbind();

			for (int i = 0; i < m_nodeparam.size(); i++) {
				m_nodeparam[i].unbind();
			}
			m_nodetex.reset();

			for (int i = 0; i < m_texRsc.size(); i++) {
				m_texRsc[i].unbind();
			}
			m_tex.reset();
		}

		if (m_aovGLBuffer.isValid()) {
			m_aovGLBuffer.unbind();
			m_aovGLBuffer.unmap();
		}
	}
}
