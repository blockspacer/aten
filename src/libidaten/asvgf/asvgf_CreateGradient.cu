#include "asvgf/asvgf.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__device__ inline bool isEqualInt2(const int2& a, const int2& b)
{
    return (a.x == b.x) && (a.y == b.y);
}

#define AT_IS_INBOUND(x, a, b)  (((a) <= (x)) && ((x) <= (b)))

__global__ void createGradientSample(
    idaten::TileDomain tileDomain,
    int tileSize,
    idaten::SVGFPathTracing::Path* paths,
    int4* gradientSample,
    int width, int height,
    int widthInRealRes, int heightInRealRes)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    ix += tileDomain.x;
    iy += tileDomain.y;

    const int idx = getIdx(ix, iy, width);

    // Keep previous sample.
    auto prevSample = gradientSample[idx];

    int xx = aten::clamp(ix * tileSize, 0, widthInRealRes - 1);
    int yy = aten::clamp(iy * tileSize, 0, heightInRealRes - 1);

    const int idxInRealRes = getIdx(xx, yy, widthInRealRes);

    // Sample random.
    aten::vec2 r = paths->sampler[idxInRealRes].nextSample2D();

    // Position in tile.
    gradientSample[idx].x = aten::clamp((int)(r.x * tileSize), 0, tileSize);
    gradientSample[idx].y = aten::clamp((int)(r.y * tileSize), 0, tileSize);

    int2 prevTilePos = make_int2(prevSample.x, prevSample.y);
    int2 prevPosInRealRes = make_int2(ix, iy) * tileSize + prevTilePos;

    bool isInRes = AT_IS_INBOUND(prevPosInRealRes.x, 0, widthInRealRes - 1);
    isInRes &= AT_IS_INBOUND(prevPosInRealRes.y, 0, heightInRealRes - 1);

    int prevIdx = getIdx(prevPosInRealRes.x, prevPosInRealRes.y, widthInRealRes);

    gradientSample[idx].z = isInRes ? prevIdx : -1;
}

__global__ void createGradient(
    idaten::TileDomain tileDomain,
    int tileSize,
    float4* outGradient,
    const float4* __restrict__ curAovColorUnfiltered,
    const float4* __restrict__ prevAovColorUnfiltered,
    const float4* __restrict__ curAovTexclrMeshid,
    const int4* __restrict__ gradientSample,
    int width, int height,
    int widthInRealRes, int heightInRealRes)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    ix += tileDomain.x;
    iy += tileDomain.y;

    const int idx = getIdx(ix, iy, width);

    int2 tilePos = make_int2(gradientSample[idx].x, gradientSample[idx].y);

    int2 posInRealRes = make_int2(ix, iy) * tileSize + tilePos;

    posInRealRes.x = aten::clamp(posInRealRes.x, 0, widthInRealRes - 1);
    posInRealRes.y = aten::clamp(posInRealRes.y, 0, heightInRealRes - 1);

    const int curIdxInRealRes = getIdx(posInRealRes.x, posInRealRes.y, widthInRealRes);

    auto curColor = curAovColorUnfiltered[curIdxInRealRes];
    float curLum = AT_NAME::color::luminance(curColor.x, curColor.y, curColor.z);

    outGradient[idx] = make_float4(0.0f);

    const int prevIdxInRealRes = gradientSample[idx].z;

    if (prevIdxInRealRes >= 0) {
        auto prevColor = prevAovColorUnfiltered[prevIdxInRealRes];
        float prevLum = AT_NAME::color::luminance(prevColor.x, prevColor.y, prevColor.z);

        outGradient[idx].x = max(curLum, prevLum);
        outGradient[idx].y = curLum - prevLum;
    }

    float2 moments = make_float2(curLum, curLum * curLum);
    float sumW = 1.0f;
    int centerMeshId = (int)curAovTexclrMeshid[curIdxInRealRes].w;

    for (int yy = 0; yy < tileSize; yy++) {
        for (int xx = 0; xx < tileSize; xx++) {
            int2 p = make_int2(ix, iy) * tileSize + make_int2(xx, yy);

            p.x = aten::clamp(p.x, 0, widthInRealRes - 1);
            p.y = aten::clamp(p.y, 0, heightInRealRes - 1);

            if (!isEqualInt2(posInRealRes, p)) {
                int pidx = getIdx(p.x, p.y, widthInRealRes);

                auto clr = curAovColorUnfiltered[pidx];
                int meshId = (int)curAovTexclrMeshid[pidx].w;

                float l = AT_NAME::color::luminance(clr.x, clr.y, clr.z);
                float w = (centerMeshId == meshId ? 1.0f : 0.0f);

                moments += make_float2(l, l * l) * w;
                sumW += w;
            }
        }
    }

    moments /= sumW;

    float variance = max(0.0f, moments.y - moments.x * moments.x);

    outGradient[idx].z = moments.x;
    outGradient[idx].w = variance;
}

namespace idaten
{

}