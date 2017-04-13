#include "kernel/raytracing.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten.h"

struct Ray {
	float3 org;
	float3 dir;

	__host__ __device__ Ray() {}
	__host__ __device__ Ray(float3 _org, float3 _dir)
	{
		org = _org;
		dir = normalize(_dir);
	}
};

// 直行ベクトルを計算.
__host__ __device__ float3 getOrthoVector(const float3& n)
{
	float3 p;

	// NOTE
	// dotを計算したときにゼロになるようなベクトル.
	// k は normalize 計算用.

	if (abs(n.z) > 0.0f) {
		float k = sqrtf(n.y * n.y + n.z * n.z);
		p.x = 0;
		p.y = -n.z / k;
		p.z = n.y / k;
	}
	else {
		float k = sqrtf(n.x * n.x + n.y * n.y);
		p.x = n.y / k;
		p.y = -n.x / k;
		p.z = 0;
	}

	return std::move(p);
}

struct hitrecord {
	float t{ AT_MATH_INF };

	float3 p;

	float3 normal;

	float3 color;

	float area{ 1.0f };
};

struct Sphere {
	float radius;
	float3 center;
	float3 color;

	Sphere() {}
	Sphere(const float3& c, float r, const float3& clr)
	{
		radius = r;
		center = c;
		color = clr;
	}
};

__host__ __device__ bool intersectSphere(
	const Sphere* sphere,
	const Ray* r, hitrecord* rec)
{
	const float3 p_o = sphere->center - r->org;
	const float b = dot(p_o, r->dir);

	// 判別式.
	const float D4 = b * b - dot(p_o, p_o) + sphere->radius * sphere->radius;

	if (D4 < 0.0f) {
		return false;
	}

	const float sqrt_D4 = sqrtf(D4);
	const float t1 = b - sqrt_D4;
	const float t2 = b + sqrt_D4;

	if (t1 > AT_MATH_EPSILON) {
		rec->t = t1;
	}
	else if (t2 > AT_MATH_EPSILON) {
		rec->t = t2;
	}
	else {
		return false;
	}

	rec->p = r->org + rec->t * r->dir;
	rec->normal = (rec->p - sphere->center) / sphere->radius; // 正規化して法線を得る

	rec->color = sphere->color;

	rec->area = 4 * AT_MATH_PI * sphere->radius * sphere->radius;

	return true;
}

struct CameraSampleResult {
	Ray r;
	float3 posOnLens;
	float3 nmlOnLens;

	__host__ __device__ CameraSampleResult() {}
};

struct Camera {
	float3 origin;

	float aspect;
	float3 center;

	float3 u;
	float3 v;

	float3 dir;
	float3 right;
	float3 up;

	float dist;
	int width;
	int height;
};

__host__ void initCamera(
	Camera& camera,
	const float3& origin,
	const float3& lookat,
	const float3& up,
	float vfov,	// vertical fov.
	uint32_t width, uint32_t height)
{
	float theta = Deg2Rad(vfov);

	camera.aspect = width / (float)height;

	float half_height = tanf(theta / 2);
	float half_width = camera.aspect * half_height;

	camera.origin = origin;

	// カメラ座標ベクトル.
	camera.dir = normalize(lookat - origin);
	camera.right = normalize(cross(camera.dir, up));
	camera.up = cross(camera.right, camera.dir);

	camera.center = origin + camera.dir;

	// スクリーンのUVベクトル.
	camera.u = half_width * camera.right;
	camera.v = half_height * camera.up;

	camera.dist = height / (2.0f * tanf(theta / 2));

	camera.width = width;
	camera.height = height;
}

__host__ __device__ void sampleCamera(
	CameraSampleResult* sample,
	Camera* camera,
	float s, float t)
{
	// [0, 1] -> [-1, 1]
	s = 2 * s - 1;
	t = 2 * t - 1;

	auto screenPos = s * camera->u + t * camera->v;

	screenPos = screenPos + camera->center;

	auto dirToScr = screenPos - camera->origin;

	sample->posOnLens = screenPos;
	sample->nmlOnLens = camera->dir;
	sample->r = Ray(camera->origin, dirToScr);
}

__host__ __device__ bool intersect(
	const Ray* r, hitrecord* rec,
	const Sphere* spheres, const int num)
{
	bool isHit = false;

	hitrecord tmp;

	for (int i = 0; i < num; i++) {
		if (intersectSphere(&spheres[i], r, &tmp)) {
			if (tmp.t < rec->t) {
				*rec = tmp;

				isHit = true;
			}
		}
	}

	return isHit;
}

#if 1
__global__ void raytracing(
	float4* p,
	int width, int height,
	Camera* camera,
	Sphere* spheres, int num)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;
#else
__host__ void raytracing(
	int ix, int iy,
	float4* p,
	int width, int height,
	Camera* camera,
	Sphere* spheres, int num)
{
#endif

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = iy * camera->width + ix;

	float s = ix / (float)camera->width;
	float t = iy / (float)camera->height;

	CameraSampleResult camsample;
	sampleCamera(&camsample, camera, s, t);

	hitrecord rec;

	if (intersect(&camsample.r, &rec, spheres, num)) {
		p[idx] = make_float4(1, 0, 0, 1);
	}
	else {
		p[idx] = make_float4(0, 0, 0, 1);
	}
}

static Sphere g_spheres[] = {
	Sphere(make_float3(0, 0, -10), 1.0, make_float3(1, 0, 0)),
};

void renderRayTracing(
	aten::vec4* image,
	int width, int height)
{
	Camera camera;
	initCamera(
		camera,
		make_float3(0, 0, 0),
		make_float3(0, 0, -1),
		make_float3(0, 1, 0),
		30,
		width, height);

#if 1
	aten::CudaMemory dst(sizeof(float4) * width * height);

	aten::CudaMemory cam(&camera, sizeof(Camera));
	
	aten::CudaMemory spheres(sizeof(Sphere) * AT_COUNTOF(g_spheres));
	spheres.write(g_spheres, sizeof(g_spheres));

	dim3 block(32, 32);
	dim3 grid(
		(width + block.x - 1) / block.x,
		(height + block.y - 1) / block.y);

	raytracing << <grid, block >> > (
		(float4*)dst.ptr(), 
		width, height, 
		(Camera*)cam.ptr(),
		(Sphere*)spheres.ptr(), AT_COUNTOF(g_spheres));

	checkCudaErrors(cudaDeviceSynchronize());

	dst.read(image, sizeof(aten::vec4) * width * height);
#else
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			raytracing(
				x, y,
				(float4*)image,
				width, height,
				&camera,
				g_spheres, AT_COUNTOF(g_spheres));
		}
	}
#endif
}
