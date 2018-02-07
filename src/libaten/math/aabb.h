#pragma once

#include "defs.h"
#include "math/vec4.h"
#include "math/mat4.h"
#include "math/ray.h"

namespace aten {
	class aabb {
	public:
		AT_DEVICE_API aabb()
		{
			empty();
		}
		AT_DEVICE_API aabb(const vec3& _min, const vec3& _max)
		{
			init(_min, _max);
		}
		AT_DEVICE_API ~aabb() {}

	public:
		AT_DEVICE_API void init(const vec3& _min, const vec3& _max)
		{
#if 0
			AT_ASSERT(_min.x <= _max.x);
			AT_ASSERT(_min.y <= _max.y);
			AT_ASSERT(_min.z <= _max.z);
#endif

			m_min = _min;
			m_max = _max;
		}

		void initBySize(const vec3& _min, const vec3& _size)
		{
			m_min = _min;
			m_max = m_min + _size;
		}

		vec3 size() const
		{
			vec3 size = m_max - m_min;
			return std::move(size);
		}

		AT_DEVICE_API bool hit(
			const ray& r,
			real t_min, real t_max,
			real* t_result = nullptr) const
		{
			return hit(
				r, 
				m_min, m_max,
				t_min, t_max, 
				t_result);
		}

		static AT_DEVICE_API bool hit(
			const ray& r,
			const aten::vec3& _min, const aten::vec3& _max,
			real t_min, real t_max,
			real* t_result = nullptr)
		{
#if 0
			bool isHit = false;

			for (uint32_t i = 0; i < 3; i++) {
				if (_min[i] == _max[i]) {
					continue;
				}

#if 0
				if (r.dir[i] == 0.0f) {
					continue;
				}

				auto inv = real(1) / r.dir[i];
#else
				auto inv = real(1) / (r.dir[i] + real(1e-6));
#endif

				// NOTE
				// ray : r = p + t * v
				// plane of AABB : x(t) = p(x) + t * v(x)
				//  t = (p(x) - x(t)) / v(x)
				// x軸の面は手前と奥があるので、それぞれの t を計算.
				// t がx軸の面の手前と奥の x の範囲内であれば、レイがAABBを通る.
				// これをxyz軸について計算する.

				auto t0 = (_min[i] - r.org[i]) * inv;
				auto t1 = (_max[i] - r.org[i]) * inv;

				if (inv < real(0)) {
#if 0
					std::swap(t0, t1);
#else
					real tmp = t0;
					t0 = t1;
					t1 = tmp;
#endif
				}

				t_min = (t0 > t_min ? t0 : t_min);
				t_max = (t1 < t_max ? t1 : t_max);

				if (t_max <= t_min) {
					return false;
				}

				if (t_result) {
					*t_result = t0;
				}

				isHit = true;
			}

			return isHit;
#else
			aten::vec3 invdir = real(1) / (r.dir + aten::vec3(real(1e-6)));
			aten::vec3 oxinvdir = -r.org * invdir;

			const auto f = _max * invdir + oxinvdir;
			const auto n = _min * invdir + oxinvdir;

			const auto tmax = max(f, n);
			const auto tmin = min(f, n);

			const auto t1 = aten::cmpMin(aten::cmpMin(aten::cmpMin(tmax.x, tmax.y), tmax.z), t_max);
			const auto t0 = aten::cmpMax(aten::cmpMax(aten::cmpMax(tmin.x, tmin.y), tmin.z), t_min);

			if (t_result) {
				*t_result = t0;
			}

			return t0 <= t1;
#endif
		}

		enum Face {
			None = -1,
			FaceX,
			FaceY,
			FaceZ,
		};

		static AT_DEVICE_API bool hit(
			const ray& r,
			const aten::vec3& _min, const aten::vec3& _max,
			real t_min, real t_max,
			real& t_result,
			Face& face)
		{
			aten::vec3 invdir = real(1) / (r.dir + aten::vec3(real(1e-6)));
			aten::vec3 oxinvdir = -r.org * invdir;

			const auto f = _max * invdir + oxinvdir;
			const auto n = _min * invdir + oxinvdir;

			const auto tmax = max(f, n);
			const auto tmin = min(f, n);

			const auto t1 = aten::cmpMin(aten::cmpMin(aten::cmpMin(tmax.x, tmax.y), tmax.z), t_max);
			const auto t0 = aten::cmpMax(aten::cmpMax(aten::cmpMax(tmin.x, tmin.y), tmin.z), t_min);

			t_result = t0;

			face = t_result == tmin.x
				? Face::FaceX
				: t_result == tmin.y
					? Face::FaceY
					: t_result == tmin.z
						? Face::FaceZ
						: Face::None;

			return t0 <= t1;
		}

		bool isIn(const vec3& p) const
		{
			bool isInX = (m_min.x <= p.x && p.x <= m_max.x);
			bool isInY = (m_min.y <= p.y && p.y <= m_max.y);
			bool isInZ = (m_min.z <= p.z && p.z <= m_max.z);

			return isInX && isInY && isInZ;
		}

		bool isIn(const aabb& bound) const
		{
			bool b0 = isIn(bound.m_min);
			bool b1 = isIn(bound.m_max);

			return b0 & b1;
		}

		const vec3& minPos() const
		{
			return m_min;
		}

		vec3& minPos()
		{
			return m_min;
		}

		const vec3& maxPos() const
		{
			return m_max;
		}

		vec3& maxPos()
		{
			return m_max;
		}

		vec3 getCenter() const
		{
			vec3 center = (m_min + m_max) * real(0.5);
			return std::move(center);
		}

		static vec3 computeFaceSurfaceArea(
			const vec3& vMin,
			const vec3& vMax)
		{
			auto dx = aten::abs(vMax.x - vMin.x);
			auto dy = aten::abs(vMax.y - vMin.y);
			auto dz = aten::abs(vMax.z - vMin.z);

			return std::move(vec3(dx * dy, dy * dz, dz * dx));
		}

		vec3 computeFaceSurfaceArea() const
		{
			return computeFaceSurfaceArea(m_max, m_min);
		}

		real computeSurfaceArea() const
		{
			auto dx = aten::abs(m_max.x - m_min.x);
			auto dy = aten::abs(m_max.y - m_min.y);
			auto dz = aten::abs(m_max.z - m_min.z);

			// ６面の面積を計算するが、AABBは対称なので、３面の面積を計算して２倍すればいい.
			auto area = dx * dy;
			area += dy * dz;
			area += dz * dx;
			area *= 2;

			return area;
		}

		AT_DEVICE_API void empty()
		{
			m_min.x = m_min.y = m_min.z = AT_MATH_INF;
			m_max.x = m_max.y = m_max.z = -AT_MATH_INF;
		}

		bool isValid() const
		{
			return (aten::cmpGEQ(m_min, m_max) & 0x07) == 0;
		}

		real getDiagonalLenght() const
		{
			auto ret = length(m_max - m_min);
			return ret;
		}

		void expand(const aabb& box)
		{
			*this = merge(*this, box);
		}

		void expand(const vec3& v)
		{
			vec3 _min = aten::min(m_min, v);
			vec3 _max = aten::max(m_max, v);

			m_min = _min;
			m_max = _max;
		}

		static aabb merge(const aabb& box0, const aabb& box1)
		{
			vec3 _min = aten::vec3(
				std::min(box0.m_min.x, box1.m_min.x),
				std::min(box0.m_min.y, box1.m_min.y),
				std::min(box0.m_min.z, box1.m_min.z));

			vec3 _max = aten::vec3(
				std::max(box0.m_max.x, box1.m_max.x),
				std::max(box0.m_max.y, box1.m_max.y),
				std::max(box0.m_max.z, box1.m_max.z));

			aabb _aabb(_min, _max);

			return std::move(_aabb);
		}

		static aabb transform(const aabb& box, const aten::mat4& mtxL2W)
		{
			vec3 center = box.getCenter();

			vec3 vMin = box.minPos() - center;
			vec3 vMax = box.maxPos() - center;

			vec3 pts[8] = {
				vec3(vMin.x, vMin.y, vMin.z),
				vec3(vMax.x, vMin.y, vMin.z),
				vec3(vMin.x, vMax.y, vMin.z),
				vec3(vMax.x, vMax.y, vMin.z),
				vec3(vMin.x, vMin.y, vMax.z),
				vec3(vMax.x, vMin.y, vMax.z),
				vec3(vMin.x, vMax.y, vMax.z),
				vec3(vMax.x, vMax.y, vMax.z),
			};

			vec3 newMin = vec3(AT_MATH_INF);
			vec3 newMax = vec3(-AT_MATH_INF);

			for (int i = 0; i < 8; i++) {
				vec3 v = mtxL2W.apply(pts[i]);

				newMin = vec3(
					std::min(newMin.x, v.x),
					std::min(newMin.y, v.y),
					std::min(newMin.z, v.z));
				newMax = vec3(
					std::max(newMax.x, v.x),
					std::max(newMax.y, v.y),
					std::max(newMax.z, v.z));
			}

			aabb ret(newMin + center, newMax + center);

			return std::move(ret);
		}

	private:
		vec3 m_min;
		vec3 m_max;
	};
}
