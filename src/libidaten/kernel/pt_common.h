#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"

#include "aten4idaten.h"

#define BLOCK_SIZE	(16)
#define BLOCK_SIZE2	(BLOCK_SIZE * BLOCK_SIZE)

inline AT_DEVICE_API int getIdx(int ix, int iy, int width)
{
#if 0
	int X = ix / BLOCK_SIZE;
	int Y = iy / BLOCK_SIZE;

	//int base = Y * BLOCK_SIZE2 * (width / BLOCK_SIZE) + X * BLOCK_SIZE2;

	int XB = X * BLOCK_SIZE;
	int YB = Y * BLOCK_SIZE;

	int base = YB * width + XB * BLOCK_SIZE;

	const auto idx = base + (iy - YB) * BLOCK_SIZE + (ix - XB);

	return idx;
#else
	return iy * width + ix;
#endif
}

inline __device__ int getLinearIdx(int x, int y, int w, int h)
{
	int max_buffer_size = w * h;
	return clamp(y * w + x, 0, max_buffer_size - 1);
}

// Bilinear sampler
inline __device__ float4 sampleBilinear(
	const float4* buffer,
	float uvx, float uvy,
	int w, int h)
{
	float2 uv = make_float2(uvx, uvy) * make_float2(w, h) - make_float2(0.5f, 0.5f);

	int x = floor(uv.x);
	int y = floor(uv.y);

	float2 uv_ratio = uv - make_float2(x, y);
	float2 uv_inv = make_float2(1.f, 1.f) - uv_ratio;

	int x1 = clamp(x + 1, 0, w - 1);
	int y1 = clamp(y + 1, 0, h - 1);

	float4 r = (buffer[getLinearIdx(x, y, w, h)] * uv_inv.x + buffer[getLinearIdx(x1, y, w, h)] * uv_ratio.x) * uv_inv.y +
		(buffer[getLinearIdx(x, y1, w, h)] * uv_inv.x + buffer[getLinearIdx(x1, y1, w, h)] * uv_ratio.x) * uv_ratio.y;

	return r;
}
