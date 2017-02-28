#pragma once

#include <vector>
#include "light/light.h"
#include "renderer/envmap.h"

namespace aten {
	class ImageBasedLight : public Light {
	public:
		ImageBasedLight() {}
		ImageBasedLight(envmap* envmap)
		{
			setEnvMap(envmap);
		}

		virtual ~ImageBasedLight() {}

	public:
		void setEnvMap(envmap* envmap)
		{
			if (m_envmap != envmap) {
				m_envmap = envmap;
				preCompute();
			}
		}

		envmap* getEnvMap()
		{
			return m_envmap;
		}

		virtual real samplePdf(const ray& r) const override final;

		virtual LightSampleResult sample(const vec3& org, sampler* sampler) const override final;

		virtual bool isSingular() const
		{
			return false;
		}

	private:
		void preCompute();

	private:
		envmap* m_envmap{ nullptr };
		real m_avgIllum{ real(0) };

		// v方向のcdf(cumulative distribution function = 累積分布関数 = sum of pdf).
		std::vector<real> m_cdfV;

		// u方向のcdf(cumulative distribution function = 累積分布関数 = sum of pdf).
		// 列ごとに保持する.
		std::vector<std::vector<real>> m_cdfU;
	};
}