#include "svgf/svgf_pt.h"

#include "kernel/pt_common.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__global__ void coarseBuffers(
	cudaSurfaceObject_t dstNmlDepth,
	const idaten::SVGFPathTracing::Path* __restrict__ srcPaths,
	const aten::ray* __restrict__ srcRays,
	const float4* __restrict__ srcAovNormalDepth,
	const float4* __restrict__ srcAovTexclrTemporalWeight,
	const float4* __restrict__ srcAovMomentMeshid,
	idaten::SVGFPathTracing::Path* dstPaths,
	aten::ray* dstRays,
	float4* dstAovNormalDepth,
	float4* dstAovTexclrTemporalWeight,
	float4* dstAovMomentMeshid,
	int width, int height,
	int lowResWidth, int lowResHeight)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= lowResWidth || iy >= lowResHeight) {
		return;
	}

	int hiResX = ix * 2;
	int hiResY = iy * 2;

	int idx_0 = getIdx(hiResX + 0, hiResY + 0, width);
	int idx_1 = getIdx(hiResX + 1, hiResY + 0, width);
	int idx_2 = getIdx(hiResX + 0, hiResY + 1, width);
	int idx_3 = getIdx(hiResX + 1, hiResY + 1, width);

	idx_0 = min(idx_0, width * height - 1);
	idx_1 = min(idx_1, width * height - 1);
	idx_2 = min(idx_2, width * height - 1);
	idx_3 = min(idx_3, width * height - 1);

	const idaten::SVGFPathTracing::Path paths[4] = {
		srcPaths[idx_0],
		srcPaths[idx_1],
		srcPaths[idx_2],
		srcPaths[idx_3],
	};
	float4 nmlDepth[4] = {
		srcAovNormalDepth[idx_0],
		srcAovNormalDepth[idx_1],
		srcAovNormalDepth[idx_2],
		srcAovNormalDepth[idx_3],
	};
	int indices[4] = {
		idx_0, idx_1, idx_2, idx_3,
	};

	// Depthが一番大きいものを選ぶ.
	float maxDepth = -1.0f;
	int pos = -1;

#pragma unroll
	for (int i = 0; i < 4; i++) {
		if (nmlDepth[i].w > maxDepth
			&& !paths[i].isKill
			&& !paths[i].isTerminate)
		{
			maxDepth = nmlDepth[i].w;
			pos = i;
		}
	}

	int idx = getIdx(ix, iy, lowResWidth);

	if (pos >= 0) {
		dstPaths[idx] = paths[pos];
		dstAovNormalDepth[idx] = nmlDepth[pos];

		int srcIdx = indices[pos];

		dstRays[idx] = srcRays[srcIdx];
		dstAovTexclrTemporalWeight[idx] = srcAovTexclrTemporalWeight[srcIdx];
		dstAovMomentMeshid[idx] = srcAovMomentMeshid[srcIdx];
	}
	else {
		int srcIdx = getIdx(hiResX, hiResY, width);

		dstPaths[idx] = srcPaths[srcIdx];
		dstRays[idx] = srcRays[srcIdx];
		dstAovNormalDepth[idx] = srcAovNormalDepth[srcIdx];
		dstAovTexclrTemporalWeight[idx] = srcAovTexclrTemporalWeight[srcIdx];
		dstAovMomentMeshid[idx] = srcAovMomentMeshid[srcIdx];
	}

	surf2Dwrite(
		dstAovNormalDepth[idx],
		dstNmlDepth,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

__global__ void copyBackToOriginalBuffer(
	const idaten::SVGFPathTracing::Path* __restrict__ srcPaths,
	const aten::ray* __restrict__ srcRays,
	const float4* __restrict__ srcAovNormalDepth,
	const float4* __restrict__ srcAovTexclrTemporalWeight,
	const float4* __restrict__ srcAovMomentMeshid,
	idaten::SVGFPathTracing::Path* dstPaths,
	aten::ray* dstRays,
	float4* dstAovNormalDepth,
	float4* dstAovTexclrTemporalWeight,
	float4* dstAovMomentMeshid,
	int lowResWidth, int lowResHeight)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= lowResWidth || iy >= lowResHeight) {
		return;
	}

	int idx = getIdx(ix, iy, lowResWidth);

	dstPaths[idx] = srcPaths[idx];
	dstRays[idx] = srcRays[idx];
	dstAovNormalDepth[idx] = srcAovNormalDepth[idx];
	dstAovTexclrTemporalWeight[idx] = srcAovTexclrTemporalWeight[idx];
	dstAovMomentMeshid[idx] = srcAovMomentMeshid[idx];
}

namespace idaten
{
	void SVGFPathTracing::onCoarseBuffer(int width, int height)
	{
		int lowResWidth = width / 2;
		int lowResHeight = height / 2;

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(lowResWidth + block.x - 1) / block.x,
			(lowResHeight + block.y - 1) / block.y);

		int curaov = getCurAovs();

		m_tmpPaths.init(lowResWidth * lowResHeight);
		m_tmpRays.init(lowResWidth * lowResHeight);
		m_tmpAovBuffer[0].init(lowResWidth * lowResHeight);
		m_tmpAovBuffer[1].init(lowResWidth * lowResHeight);
		m_tmpAovBuffer[2].init(width * lowResHeight);

		m_aovLowResNmlDepth.map();
		auto aovLowResNmlDepthExportBuffer = m_aovLowResNmlDepth.bind();

		// Coarse buffers.
		coarseBuffers << <grid, block >> > (
			aovLowResNmlDepthExportBuffer,
			m_paths.ptr(),
			m_rays.ptr(),
			m_aovNormalDepth[curaov].ptr(),
			m_aovTexclrTemporalWeight[curaov].ptr(),
			m_aovMomentMeshid[curaov].ptr(),
			m_tmpPaths.ptr(),
			m_tmpRays.ptr(),
			m_tmpAovBuffer[0].ptr(),
			m_tmpAovBuffer[1].ptr(),
			m_tmpAovBuffer[2].ptr(),
			width, height,
			lowResWidth, lowResHeight);
		checkCudaKernel(coarseBuffers);

		m_aovLowResNmlDepth.unbind();
		m_aovLowResNmlDepth.unmap();

		// TODO
		// Copy back to orignal buffers.
		// 本来なら利用するバッファの切り替えをすれば、コピーは必要なく、速度も上がると思われる.
		// 一方でこちらの方法だとメモリ節約になる.
		copyBackToOriginalBuffer << <grid, block >> > (
			m_tmpPaths.ptr(),
			m_tmpRays.ptr(),
			m_tmpAovBuffer[0].ptr(),
			m_tmpAovBuffer[1].ptr(),
			m_tmpAovBuffer[2].ptr(),
			m_paths.ptr(),
			m_rays.ptr(),
			m_aovNormalDepth[curaov].ptr(),
			m_aovTexclrTemporalWeight[curaov].ptr(),
			m_aovMomentMeshid[curaov].ptr(),
			lowResWidth, lowResHeight);
		checkCudaKernel(copyBackToOriginalBuffer);
	}
}