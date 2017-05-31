#pragma once 

#include <vector>
#include "types.h"
#include "math/aabb.h"
#include "math/vec3.h"
#include "material/material.h"
#include "sampler/sampler.h"
#include "shape/shape.h"

//#define ENABLE_TANGENTCOORD_IN_HITREC

namespace aten {
	class hitable;

	struct hitrecord {
		vec3 p;

		vec3 normal;

#ifdef ENABLE_TANGENTCOORD_IN_HITREC
		// tangent coordinate.
		vec3 du;
		vec3 dv;
#endif

		// texture coordinate.
		real u{ real(0) };
		real v{ real(0) };

		real area{ real(1) };

		int objid{ -1 };
		int mtrlid{ -1 };
	};

	struct Intersection {
		real t{ AT_MATH_INF };

		int objid{ -1 };
		int mtrlid{ -1 };

		real area{ real(1) };

		union {
			// cube.
			struct {
				int face;
			};
			// triangle.
			struct {
				int idx[3];
				real a, b;	// barycentric
			};
		};
	};

	class hitable {
	public:
		hitable(const char* name = nullptr)
			: m_name(name)
		{}
		virtual ~hitable() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const = 0;

		virtual aabb getBoundingbox() const = 0;

		struct SamplePosNormalPdfResult {
			aten::vec3 pos;
			aten::vec3 nml;
			real area;

			real a;
			real b;
			int idx[3];
		};

		virtual void getSamplePosNormalArea(SamplePosNormalPdfResult* result, sampler* sampler) const
		{
			AT_ASSERT(false);
		}

		static void evalHitResult(
			const hitable* obj,
			const ray& r,
			hitrecord& rec,
			const Intersection& isect)
		{
			obj->evalHitResult(r, rec, isect);

			rec.objid = isect.objid;
			rec.mtrlid = isect.mtrlid;

#ifdef ENABLE_TANGENTCOORD_IN_HITREC
			// tangent coordinate.
			rec.du = normalize(getOrthoVector(rec.normal));
			rec.dv = normalize(cross(rec.normal, rec.du));
#endif
		}

	private:
		virtual void evalHitResult(
			const ray& r,
			hitrecord& rec,
			const Intersection& isect) const
		{
			AT_ASSERT(false);
		}

	private:
		const char* m_name;
	};
}
