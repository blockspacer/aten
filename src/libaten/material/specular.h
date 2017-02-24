#pragma once

#include "material/material.h"

namespace aten
{
	class specular : public material {
	public:
		specular() {}
		specular(const vec3& c)
			: m_color(c)
		{}

		virtual ~specular() {}

		virtual bool isSingular() const override final
		{
			return true;
		}

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
		vec3 m_color{ vec3(1, 1, 1) };
	};
}
