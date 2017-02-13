#pragma once

#include "camera/camera.h"
#include "renderer/ray.h"
#include "math/vec3.h"
#include "math/math.h"

namespace aten {
	class PinholeCamera : public camera {
	public:
		PinholeCamera() {}
		virtual ~PinholeCamera() {}

		void init(
			vec3 origin, vec3 lookat, vec3 up,
			real vfov,
			uint32_t width, uint32_t height)
		{
			real theta = Deg2Rad(vfov);

			m_aspect = width / (real)height;

			real half_height = aten::tan(theta / 2);	// (h/2)/d
			real half_width = m_aspect * half_height;	// (h/2)/d * w/h = (w/2)/d

			real flocalLength = CONST_REAL(1.0);

			m_origin = origin;

			// カメラ座標ベクトル.
			m_dir = normalize(lookat - origin);
			m_right = normalize(cross(up, m_dir));
			m_up = cross(m_dir, m_right);

			// NOTE
			// half_width * screenDist = ((w/2)/d) * d = w/2
			// half_height * screenDist = ((h/2)/d) * d = h/2

			// スクリーンの左下位置.
			//m_LowerLeftCorner = origin - half_width * screenDist * m_right - half_height * screenDist * m_up + screenDist * m_dir;
			m_LowerLeftCorner = origin + half_width * flocalLength * m_right - half_height * flocalLength * m_up + flocalLength * m_dir;

			// 左下基準としたスクリーンのUVベクトル.
			m_u = 2 * half_width * flocalLength * -m_right;
			m_v = 2 * half_height * flocalLength * m_up;
		}

		virtual ray sample(real s, real t) override final
		{
			auto screenPos = s * m_u + t * m_v;
			screenPos = screenPos + m_LowerLeftCorner;

			auto dirToScr = screenPos - m_origin;

			ray ret(m_origin, dirToScr);

			return std::move(ret);
		}

	private:
		vec3 m_origin;

		real m_aspect;
		vec3 m_LowerLeftCorner;

		vec3 m_u;
		vec3 m_v;

		vec3 m_dir;
		vec3 m_right;
		vec3 m_up;
	};
}