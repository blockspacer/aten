#pragma once

#include "defs.h"
#include "math/math.h"
#include "math/vec3.h"
#include "math/vec4.h"

namespace aten {
	class mat4 {
	public:
		static const mat4 Identity;
		static const mat4 Zero;

		union {
			real a[16];
			real m[4][4];
			vec4 v[4];
			struct {
				real m00, m01, m02, m03;
				real m10, m11, m12, m13;
				real m20, m21, m22, m23;
				real m30, m31, m32, m33;
			};
		};

		mat4()
		{
			identity();
		}
		mat4(const mat4& rhs)
		{
			*this = rhs;
		}
		mat4(
			real _m00, real _m01, real _m02, real _m03,
			real _m10, real _m11, real _m12, real _m13,
			real _m20, real _m21, real _m22, real _m23,
			real _m30, real _m31, real _m32, real _m33)
		{
			m00 = _m00; m01 = _m01; m02 = _m02; m03 = _m03;
			m10 = _m10; m11 = _m11; m12 = _m12; m13 = _m13;
			m20 = _m20; m21 = _m21; m22 = _m22; m23 = _m23;
			m30 = _m30; m31 = _m31; m32 = _m32; m33 = _m33;
		}

		inline mat4& identity()
		{
			*this = Identity;
			return *this;
		}

		inline mat4& zero()
		{
			*this = Zero;
			return *this;
		}

		inline const mat4& operator+() const
		{
			return *this;
		}

		inline mat4 operator-() const
		{
			mat4 ret;
			ret.m00 = -m00; ret.m01 = -m01; ret.m02 = -m02; ret.m03 = -m03;
			ret.m10 = -m10; ret.m11 = -m11; ret.m12 = -m12; ret.m13 = -m13;
			ret.m20 = -m20; ret.m21 = -m21; ret.m22 = -m22; ret.m23 = -m23;
			ret.m30 = -m30; ret.m31 = -m31; ret.m32 = -m32; ret.m33 = -m33;
			return std::move(ret);
		}

		inline real* operator[](int i)
		{
			return m[i];
		}
		inline real operator()(int i, int j) const
		{
			return m[i][j];
		}
		inline real& operator()(int i, int j)
		{
			return m[i][j];
		}

		inline mat4& operator+=(const mat4& mtx)
		{
			m00 += mtx.m00; m01 += mtx.m01; m02 += mtx.m02; m03 += mtx.m03;
			m10 += mtx.m10; m11 += mtx.m11; m12 += mtx.m12; m13 += mtx.m13;
			m20 += mtx.m20; m21 += mtx.m21; m22 += mtx.m22; m23 += mtx.m23;
			m30 += mtx.m30; m31 += mtx.m31; m32 += mtx.m32; m33 += mtx.m33;
			return *this;
		}
		inline mat4& operator-=(const mat4& mtx)
		{
			m00 -= mtx.m00; m01 -= mtx.m01; m02 -= mtx.m02; m03 -= mtx.m03;
			m10 -= mtx.m10; m11 -= mtx.m11; m12 -= mtx.m12; m13 -= mtx.m13;
			m20 -= mtx.m20; m21 -= mtx.m21; m22 -= mtx.m22; m23 -= mtx.m23;
			m30 -= mtx.m30; m31 -= mtx.m31; m32 -= mtx.m32; m33 -= mtx.m33;
			return *this;
		}
		inline mat4& operator*=(const mat4& mtx)
		{
			mat4 tmp;
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					tmp.m[i][j] = 0.0f;
					for (int k = 0; k < 4; ++k) {
						tmp.m[i][j] += this->m[i][k] * mtx.m[k][j];
					}
				}
			}
			
			*this = tmp;

			return *this;
		}

		inline mat4& operator*=(const real t)
		{
			m00 *= t; m01 *= t; m02 *= t; m03 *= t;
			m10 *= t; m11 *= t; m12 *= t; m13 *= t;
			m20 *= t; m21 *= t; m22 *= t; m23 *= t;
			m30 *= t; m31 *= t; m32 *= t; m33 *= t;
			return *this;
		}
		inline mat4& operator/=(const real t)
		{
			*this *= 1 / t;
			return *this;
		}

		inline vec3 apply(const vec3& v) const
		{
			vec4 t(v.x, v.y, v.z, 1);
			vec4 ret;

			for (int r = 0; r < 4; r++) {
				ret[r] = 0;
				for (int c = 0; c < 4; c++) {
					ret[r] += m[r][c] * t[c];
				}
			}

			return std::move(ret);
		}
		inline vec3 applyXYZ(const vec3& v) const
		{
			vec3 ret;

			for (int r = 0; r < 3; r++) {
				ret[r] = 0;
				for (int c = 0; c < 3; c++) {
					ret[r] += m[r][c] * v[c];
				}
			}

			return std::move(ret);
		}

		mat4& invert();

		inline mat4& transpose()
		{
			mat4 tmp;
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					tmp.m[i][j] = this->m[j][i];
				}
			}
			
			*this = tmp;

			return *this;
		}

		inline mat4& asTrans(const vec3& v)
		{
			identity();

			m03 = v.x;
			m13 = v.y;
			m23 = v.z;

			return *this;
		}

		inline mat4& asScale(real s)
		{
			identity();

			m00 = s;
			m11 = s;
			m22 = s;
			
			return *this;
		}

		inline mat4& asRotateByX(real r)
		{
			const real c = aten::cos(r);
			const real s = aten::sin(r);

			m00 = 1; m01 = 0; m02 = 0;  m03 = 0;
			m10 = 0; m11 = c; m12 = -s; m13 = 0;
			m20 = 0; m21 = s; m22 = c;  m23 = 0;
			m30 = 0; m31 = 0; m32 = 0;  m33 = 1;

			return *this;
		}

		inline mat4& asRotateByY(real r)
		{
			const real c = aten::cos(r);
			const real s = aten::sin(r);

			m00 = c;  m01 = 0; m02 = s; m03 = 0;
			m10 = 0;  m11 = 1; m12 = 0; m13 = 0;
			m20 = -s; m21 = 0; m22 = c; m23 = 0;
			m30 = 0;  m31 = 0; m32 = 0; m33 = 1;

			return *this;
		}

		inline mat4& asRotateByZ(real r)
		{
			const real c = aten::cos(r);
			const real s = aten::sin(r);

			m00 = c; m01 = -s; m02 = 0; m03 = 0;
			m10 = s; m11 = c;  m12 = 0; m13 = 0;
			m20 = 0; m21 = 0;  m22 = 1; m23 = 0;
			m30 = 0; m31 = 0;  m32 = 0; m33 = 1;

			return *this;
		}

		mat4& asRotateByAxis(real r, const vec3& axis);
	};

	inline mat4 operator+(const mat4& m1, const mat4& m2)
	{
		mat4 ret = m1;
		ret += m2;
		return std::move(ret);
	}

	inline mat4 operator-(const mat4& m1, const mat4& m2)
	{
		mat4 ret = m1;
		ret -= m2;
		return std::move(ret);
	}

	inline mat4 operator*(const mat4& m1, const mat4& m2)
	{
		mat4 ret = m1;
		ret *= m2;
		return std::move(ret);
	}

	inline vec3 operator*(const mat4& m, const vec3 v)
	{
		vec3 ret = m.apply(v);
		return std::move(ret);
	}

	inline mat4 operator*(real t, const mat4& m)
	{
		mat4 ret = m;
		ret *= t;
		return std::move(ret);
	}

	inline mat4 operator*(const mat4& m, real t)
	{
		mat4 ret = t * m;
		return std::move(ret);
	}

	inline mat4 operator/(const mat4& m, real t)
	{
		mat4 ret = m * (1 / t);
		return std::move(ret);
	}
}