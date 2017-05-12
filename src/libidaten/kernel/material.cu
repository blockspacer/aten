#include "kernel/material.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

typedef void(*FuncSampleMaterial)(
	AT_NAME::MaterialSampling*,
	const aten::MaterialParameter*,
	const aten::vec3&,
	const aten::vec3&,
	const aten::vec3&,
	aten::sampler*,
	float, float,
	bool);

__device__ void sampleMtrlNotSupported(
	AT_NAME::MaterialSampling*,
	const aten::MaterialParameter* param,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::vec3& orgnormal,
	aten::sampler* sampler,
	real u, real v,
	bool isLightPath)
{
	printf("Sample Material Not Supported[%d]\n", param->type);
}

__device__ FuncSampleMaterial funcSampleMaterial[aten::MaterialType::MaterialTypeMax];

__device__ void addMaterialFuncs()
{
	funcSampleMaterial[aten::MaterialType::Emissive] = AT_NAME::emissive::sample;			// Emissive
	funcSampleMaterial[aten::MaterialType::Lambert] = AT_NAME::lambert::sample;				// Lambert
	funcSampleMaterial[aten::MaterialType::OrneNayar] = AT_NAME::OrenNayar::sample;			// OrneNayar
	funcSampleMaterial[aten::MaterialType::Specular] = AT_NAME::specular::sample;			// Specular
	funcSampleMaterial[aten::MaterialType::Refraction] = AT_NAME::refraction::sample;		// Refraction
	funcSampleMaterial[aten::MaterialType::Blinn] = AT_NAME::MicrofacetBlinn::sample;		// MicrofacetBlinn
	funcSampleMaterial[aten::MaterialType::GGX] = AT_NAME::MicrofacetGGX::sample;			// MicrofacetGGX
	funcSampleMaterial[aten::MaterialType::Beckman] = AT_NAME::MicrofacetBeckman::sample;	// MicrofacetBeckman
	funcSampleMaterial[aten::MaterialType::Disney] = sampleMtrlNotSupported;	// DisneyBRDF
	funcSampleMaterial[aten::MaterialType::Toon] = sampleMtrlNotSupported;		// Toon
	funcSampleMaterial[aten::MaterialType::Layer] = sampleMtrlNotSupported;		// Layer
}

__device__ void sampleMaterial(
	AT_NAME::MaterialSampling* result,
	const aten::MaterialParameter* mtrl,
	const aten::vec3& normal,
	const aten::vec3& wi,
	const aten::vec3& orgnormal,
	aten::sampler* sampler,
	float u, float v)
{
	funcSampleMaterial[mtrl->type](result, mtrl, normal, wi, orgnormal, sampler, u, v, false);
}