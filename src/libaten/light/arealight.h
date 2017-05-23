#pragma once

#include "light/light.h"

namespace AT_NAME {
	class AreaLight : public Light {
	public:
		AreaLight() 
			: Light(aten::LightType::Area, LightAttributeArea)
		{}
		AreaLight(aten::hitable* obj, const aten::vec3& le)
			: Light(aten::LightType::Area, LightAttributeArea)
		{
			m_param.object.ptr = obj;
			m_param.le = le;
		}

		AreaLight(aten::Values& val)
			: Light(aten::LightType::Area, LightAttributeArea, val)
		{
			m_param.object.ptr = (aten::hitable*)val.get("object", m_param.object.ptr);
			m_param.le = val.get("color", m_param.le);
		}

		virtual ~AreaLight() {}

	public:
		static AT_DEVICE_API void sample(
			aten::LightSampleResult* result,
			const aten::hitrecord* rec,
			const aten::LightParameter* param,
			const aten::vec3& org,
			aten::sampler* sampler)
		{
			auto obj = param->object.ptr;

			if (obj) {
				result->pos = rec->p;
				result->pdf = 1 / rec->area;
				result->dir = rec->p - org;
				result->nml = rec->normal;

				result->le = param->le;
				result->intensity = 1;
				result->finalColor = param->le;

				result->obj = obj;
			}
		}

		virtual aten::LightSampleResult sample(const aten::vec3& org, aten::sampler* sampler) const override final;

		virtual const aten::hitable* getLightObject() const override final
		{
			return (aten::hitable*)m_param.object.ptr;
		}

		virtual aten::hitable::SamplingPosNormalPdf getSamplePosNormalPdf(aten::sampler* sampler) const override final
		{
			if (m_param.object.ptr) {
				auto obj = getLightObject();
				return obj->getSamplePosNormalPdf(sampler);
			}
			return std::move(aten::hitable::SamplingPosNormalPdf(aten::make_float3(0), aten::make_float3(1), real(0)));
		}
	};
}