#pragma once

#include "material/material.h"

namespace aten
{
	class MicrofacetBlinn : public material {
	public:
		MicrofacetBlinn() {}
		MicrofacetBlinn(const vec3& c, real shininess, real ior)
			: m_color(c), m_shininess(shininess), m_ior(ior)
		{}

		virtual ~MicrofacetBlinn() {}

	public:
		virtual vec3 color() const override final
		{
			return m_color;
		}

		virtual real pdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo) const override final;

		virtual vec3 sampleDirection(
			const vec3& in,
			const vec3& normal, 
			sampler* sampler) const override final;

		virtual vec3 bsdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo,
			real u, real v) const override final;

		virtual sampling sample(
			const vec3& in,
			const vec3& normal,
			const hitrecord& hitrec,
			sampler* sampler,
			real u, real v) const override final;

	private:
		vec3 m_color;
		texture* m_tex{ nullptr };
		real m_shininess{ real(0) };

		// ���̂̋��ܗ�.
		real m_ior;
	};
}
