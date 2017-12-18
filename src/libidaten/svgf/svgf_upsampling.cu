#include "svgf/svgf_pt.h"

#include "kernel/pt_common.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__global__ void checkIfSingular(
	idaten::SVGFPathTracing::Path* paths,
	int width, int height)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height) {
		return;
	}

	int idx = getIdx(ix, iy, width);

	if (!paths[idx].isSingular) {
		paths[idx].isKill = true;
		paths[idx].isTerminate = true;
	}
}

__global__ void coarseBuffers(
	const idaten::SVGFPathTracing::Path* __restrict__ srcPaths,
	const aten::ray* __restrict__ srcRays,
	const float4* __restrict__ srcAovNormalDepth,
	const float4* __restrict__ srcAovMomentMeshid,
	idaten::SVGFPathTracing::Path* dstPaths,
	aten::ray* dstRays,
	float4* dstAovNormalDepth,
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

	// Depth‚ªˆê”Ô‘å‚«‚¢‚à‚Ì‚ð‘I‚Ô.
	float maxDepth = -1.0f;
	int pos = -1;

#pragma unroll
	for (int i = 0; i < 4; i++) {
		if (nmlDepth[i].w > maxDepth
			&& !paths[i].isSingular
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
		dstAovMomentMeshid[idx] = srcAovMomentMeshid[srcIdx];
	}
	else {
		int srcIdx = getIdx(hiResX, hiResY, width);

		dstPaths[idx] = srcPaths[srcIdx];
		dstAovNormalDepth[idx] = srcAovNormalDepth[srcIdx];
		dstRays[idx] = srcRays[srcIdx];
		dstAovMomentMeshid[idx] = srcAovMomentMeshid[srcIdx];

		dstPaths[idx].isKill = true;
		dstPaths[idx].isTerminate = true;
	}

	// Reset contribution.
	dstPaths[idx].contrib = aten::vec3(0.0f);
}

inline __device__ float4 samplePoint(
	const float4* __restrict__ buffer,
	int w, int h,
	int x, int y,
	int offsetx, int offsety)
{
	x = clamp(x + offsetx, 0, w - 1);
	y = clamp(y + offsety, 0, h - 1);

	int idx = getIdx(x, y, w);

	return buffer[idx];
}

inline __device__ float4 sampleBilinear(
	const float4* __restrict__ buffer,
	int w, int h,
	int x, int y,
	int offsetx, int offsety)
{
	x = clamp(x + offsetx, 0, w);
	y = clamp(y + offsety, 0, h);

	float uvx = x / (float)w;
	float uvy = y / (float)h;

	return sampleBilinear(buffer, uvx, uvy, w, h);
}

__global__ void onUpsamplingAndMerge(
	cudaSurfaceObject_t dst,
	const float4* __restrict__ inLowResColor,
	const float4* __restrict__ inLowResNmlDepth,
	const float4* __restrict__ inHiResColor,
	const float4* __restrict__ inHiResNmlDepth,
	const float4* __restrict__ aovTexclrTemporalWeight,
	int lowResWidth, int lowResHeight,
	int hiResWidth, int hiResHeight)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= hiResWidth || iy >= hiResHeight) {
		return;
	}

	int hiResIdx = getIdx(ix, iy, hiResWidth);

	int pos = (iy & 0x01) * 2 + (ix & 0x01);

	float4 lowResNmlDepth[4];
	float4 lowResClr[4];

	int lx = ix / 2;
	int ly = iy / 2;

	int w = lowResWidth;
	int h = lowResHeight;

	switch (pos) {
	case 0:
		lowResNmlDepth[0] = samplePoint(inLowResNmlDepth, w, h, lx, ly, 0, 0);
		lowResNmlDepth[1] = samplePoint(inLowResNmlDepth, w, h, lx, ly, 1, 0);
		lowResNmlDepth[2] = samplePoint(inLowResNmlDepth, w, h, lx, ly, 0, 1);
		lowResNmlDepth[3] = samplePoint(inLowResNmlDepth, w, h, lx, ly, 1, 1);
		lowResClr[0] = sampleBilinear(inLowResColor, w, h, lx, ly, 0, 0);
		lowResClr[1] = sampleBilinear(inLowResColor, w, h, lx, ly, 1, 0);
		lowResClr[2] = sampleBilinear(inLowResColor, w, h, lx, ly, 0, 1);
		lowResClr[3] = sampleBilinear(inLowResColor, w, h, lx, ly, 1, 1);
		break;
	case 1:
		lowResNmlDepth[0] = samplePoint(inLowResNmlDepth, w, h, lx, ly, -1, 0);
		lowResNmlDepth[1] = samplePoint(inLowResNmlDepth, w, h, lx, ly, 0, 0);
		lowResNmlDepth[2] = samplePoint(inLowResNmlDepth, w, h, lx, ly, -1, 1);
		lowResNmlDepth[3] = samplePoint(inLowResNmlDepth, w, h, lx, ly, 0, 1);
		lowResClr[0] = sampleBilinear(inLowResColor, w, h, lx, ly, -1, 0);
		lowResClr[1] = sampleBilinear(inLowResColor, w, h, lx, ly, 1, 0);
		lowResClr[2] = sampleBilinear(inLowResColor, w, h, lx, ly, -1, 1);
		lowResClr[3] = sampleBilinear(inLowResColor, w, h, lx, ly, 0, 1);
		break;
	case 2:
		lowResNmlDepth[0] = samplePoint(inLowResNmlDepth, w, h, lx, ly, 0, -1);
		lowResNmlDepth[1] = samplePoint(inLowResNmlDepth, w, h, lx, ly, 1, -1);
		lowResNmlDepth[2] = samplePoint(inLowResNmlDepth, w, h, lx, ly, 0, 0);
		lowResNmlDepth[3] = samplePoint(inLowResNmlDepth, w, h, lx, ly, 1, 0);
		lowResClr[0] = sampleBilinear(inLowResColor, w, h, lx, ly, 0, -1);
		lowResClr[1] = sampleBilinear(inLowResColor, w, h, lx, ly, 1, -1);
		lowResClr[2] = sampleBilinear(inLowResColor, w, h, lx, ly, 0, 0);
		lowResClr[3] = sampleBilinear(inLowResColor, w, h, lx, ly, 1, 0);
		break;
	case 3:
		lowResNmlDepth[0] = samplePoint(inLowResNmlDepth, w, h, lx, ly, -1, -1);
		lowResNmlDepth[1] = samplePoint(inLowResNmlDepth, w, h, lx, ly, 0, -1);
		lowResNmlDepth[2] = samplePoint(inLowResNmlDepth, w, h, lx, ly, -1, 0);
		lowResNmlDepth[3] = samplePoint(inLowResNmlDepth, w, h, lx, ly, 0, 0);
		lowResClr[0] = sampleBilinear(inLowResColor, w, h, lx, ly, -1, -1);
		lowResClr[1] = sampleBilinear(inLowResColor, w, h, lx, ly, 0, -1);
		lowResClr[2] = sampleBilinear(inLowResColor, w, h, lx, ly, -1, 0);
		lowResClr[3] = sampleBilinear(inLowResColor, w, h, lx, ly, 0, 0);
		break;
	}

	static const float bilateralWeight[] = {
		9.0 / 16.0, 3.0 / 16.0, 3.0 / 16.0, 1.0 / 16.0,
		3.0 / 16.0, 9.0 / 16.0, 1.0 / 16.0, 3.0 / 16.0,
		3.0 / 16.0, 1.0 / 16.0, 9.0 / 16.0, 3.0 / 16.0,
		1.0 / 16.0, 3.0 / 16.0, 3.0 / 16.0, 9.0 / 16.0,
	};

	float4 hiResNmlDepth = inHiResNmlDepth[hiResIdx];

	float4 sum = make_float4(0.0f);
	float sumWeight = 0.0001f;

	for (int i = 0; i < 4; i++) {
		float depthWeight = clamp(1.0 / (0.0001f + abs(hiResNmlDepth.w - lowResNmlDepth[i].w)), 0.0f, 1.0f);

		// Disable depth.
		hiResNmlDepth.w = 0.0f;
		lowResNmlDepth[i].w = 0.0f;

		float nmlWeight = clamp(powf(dot(hiResNmlDepth, lowResNmlDepth[i]), 32), 0.0f, 1.0f);

		float weight = nmlWeight * depthWeight * bilateralWeight[pos * 4 + i];
		sum += lowResClr[i] * weight;
		sumWeight += weight;
	}

	sum /= sumWeight;

	// Merge.
	float4 hiResColor = inHiResColor[hiResIdx];
	//hiResColor = sum;

	hiResColor += inLowResColor[getIdx(lx, ly, lowResWidth)];

	// Multiply Albedo.
	hiResColor *= aovTexclrTemporalWeight[hiResIdx];

	surf2Dwrite(
		hiResColor,
		dst,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

namespace idaten
{
	void SVGFPathTracing::coarseBuffer(int width, int height)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		int curaov = getCurAovs();

		int lowResWidth = width / 2;
		int lowResHeight = height / 2;

		coarseBuffers << <grid, block >> > (
			m_paths[Resolution::Hi].ptr(),
			m_rays[Resolution::Hi].ptr(),
			m_aovNormalDepth[Resolution::Hi][curaov].ptr(),
			m_aovMomentMeshid[Resolution::Hi][curaov].ptr(),
			m_paths[Resolution::Low].ptr(),
			m_rays[Resolution::Low].ptr(),
			m_aovNormalDepth[Resolution::Low][curaov].ptr(),
			m_aovMomentMeshid[Resolution::Low][curaov].ptr(),
			width, height,
			lowResWidth, lowResHeight);
		checkCudaKernel(coarseBuffers);

		// Terminate path which is not singular.
		checkIfSingular << <grid, block >> > (
			m_paths[Resolution::Hi].ptr(),
			width, height);
		checkCudaKernel(checkIfSingular);
	}

	void SVGFPathTracing::upsamplingAndMerge(
		cudaSurfaceObject_t outputSurf,
		int width, int height)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		int curaov = getCurAovs();

		int lowResWidth = width / 2;
		int lowResHeight = height / 2;

		onUpsamplingAndMerge << <grid, block >> > (
			outputSurf,
			m_aovColorVariance[Resolution::Low][curaov].ptr(),
			m_aovNormalDepth[Resolution::Low][curaov].ptr(),
			m_aovColorVariance[Resolution::Hi][curaov].ptr(),
			m_aovNormalDepth[Resolution::Hi][curaov].ptr(),
			m_aovTexclrTemporalWeight[Resolution::Hi][curaov].ptr(),
			lowResWidth, lowResHeight,
			width, height);
		checkCudaKernel(onUpsamplingAndMerge);
	}
}