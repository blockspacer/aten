#include "kernel/idatendefs.cuh"

template <idaten::IntersectType Type>
AT_CUDA_INLINE __device__ bool intersectSBVHTriangles(
	cudaTextureObject_t nodes,
	const Context* ctxt,
	const aten::ray r,
	float t_min, float t_max,
	aten::Intersection* isect)
{
	aten::Intersection isectTmp;

	int nodeid = 0;
	
	float4 node0;	// xyz: boxmin, z: hit
	float4 node1;	// xyz: boxmax, z: hit
	float4 attrib;	// x:shapeid, y:primid, z:exid,	w:meshid

	float4 boxmin;
	float4 boxmax;

	float t = AT_MATH_INF;

	isect->t = t_max;

	while (nodeid >= 0) {
		node0 = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);	// xyz : boxmin, z: hit
		node1 = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 1);	// xyz : boxmax, z: miss
		attrib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 2);	// x : parent, y : triid, z : padding, w : padding

		boxmin = make_float4(node0.x, node0.y, node0.z, 1.0f);
		boxmax = make_float4(node1.x, node1.y, node1.z, 1.0f);

		bool isHit = false;

		if (attrib.y >= 0) {
			int primidx = (int)attrib.y;
			aten::PrimitiveParamter prim;
			prim.v0 = ((aten::vec4*)ctxt->prims)[primidx * aten::PrimitiveParamter_float4_size + 0];
			prim.v1 = ((aten::vec4*)ctxt->prims)[primidx * aten::PrimitiveParamter_float4_size + 1];

			isectTmp.t = AT_MATH_INF;
			isHit = hitTriangle(&prim, ctxt, r, &isectTmp);

			bool isIntersect = (Type == idaten::IntersectType::Any
				? isHit
				: isectTmp.t < isect->t);

			if (isIntersect) {
				*isect = isectTmp;
				isect->objid = -1;
				isect->primid = primidx;
				isect->mtrlid = prim.mtrlid;

				//isect->meshid = (int)attrib.w;
				isect->meshid = prim.gemoid;

				t_max = isect->t;

				if (Type == idaten::IntersectType::Closer
					|| Type == idaten::IntersectType::Any)
				{
					return true;
				}
			}
		}
		else {
			isHit = hitAABB(r.org, r.dir, boxmin, boxmax, t_min, t_max, &t);
		}

		if (isHit) {
			nodeid = (int)node0.w;
		}
		else {
			nodeid = (int)node1.w;
		}
	}

	return (isect->objid >= 0);
}

#define ENABLE_PLANE_LOOP_SBVH

template <idaten::IntersectType Type>
AT_CUDA_INLINE __device__ bool intersectSBVH(
	const Context* ctxt,
	const aten::ray r,
	float t_min, float t_max,
	aten::Intersection* isect)
{
	aten::Intersection isectTmp;

	isect->t = t_max;

	int nodeid = 0;

	float4 node0;	// xyz: boxmin, z: hit
	float4 node1;	// xyz: boxmax, z: hit
	float4 attrib;	// x:shapeid, y:primid, z:exid,	w:meshid

	float4 boxmin;
	float4 boxmax;

	real t = AT_MATH_INF;

	cudaTextureObject_t node = ctxt->nodes[0];
	aten::ray transformedRay = r;

	int toplayerHit = -1;
	int toplayerMiss = -1;
	int objid = 0;
	int meshid = 0;

	while (nodeid >= 0) {
		node0 = tex1Dfetch<float4>(node, aten::GPUBvhNodeSize * nodeid + 0);	// xyz : boxmin, z: hit
		node1 = tex1Dfetch<float4>(node, aten::GPUBvhNodeSize * nodeid + 1);	// xyz : boxmin, z: hit
		attrib = tex1Dfetch<float4>(node, aten::GPUBvhNodeSize * nodeid + 2);	// x : shapeid, y : primid, z : exid, w : meshid

		boxmin = make_float4(node0.x, node0.y, node0.z, 1.0f);
		boxmax = make_float4(node1.x, node1.y, node1.z, 1.0f);

		bool isHit = false;

#ifdef ENABLE_PLANE_LOOP_SBVH
		if (attrib.x >= 0) {
			// Leaf.
			const auto* s = &ctxt->shapes[(int)attrib.x];

			if (attrib.z >= 0) {	// exid
				//if (hitAABB(r.org, r.dir, boxmin, boxmax, t_min, t_max, &t))
				{
					if (s->mtxid >= 0) {
						auto mtxW2L = ctxt->matrices[s->mtxid * 2 + 1];
						transformedRay.dir = mtxW2L.applyXYZ(r.dir);
						transformedRay.dir = normalize(transformedRay.dir);
						transformedRay.org = mtxW2L.apply(r.org) + AT_MATH_EPSILON * transformedRay.dir;
					}
					else {
						transformedRay = r;
					}

					node = ctxt->nodes[(int)attrib.z];

					objid = (int)attrib.x;
					meshid = (int)attrib.w;

					toplayerHit = (int)node0.w;
					toplayerMiss = (int)node1.w;

					isHit = true;
					node0.w = 0.0f;
				}
			}
			else if (attrib.y >= 0) {
				int primidx = (int)attrib.y;
				aten::PrimitiveParamter prim;
				prim.v0 = ((aten::vec4*)ctxt->prims)[primidx * aten::PrimitiveParamter_float4_size + 0];
				prim.v1 = ((aten::vec4*)ctxt->prims)[primidx * aten::PrimitiveParamter_float4_size + 1];

				isectTmp.t = AT_MATH_INF;
				isHit = hitTriangle(&prim, ctxt, transformedRay, &isectTmp);

				bool isIntersect = (Type == idaten::IntersectType::Any
					? isHit
					: isectTmp.t < isect->t);

				if (isIntersect) {
					*isect = isectTmp;
					isect->objid = objid;
					isect->primid = primidx;
					isect->mtrlid = prim.mtrlid;

					isect->meshid = prim.gemoid;
					isect->meshid = (isect->meshid < 0 ? meshid : isect->meshid);

					t_max = isect->t;

					if (Type == idaten::IntersectType::Closer
						|| Type == idaten::IntersectType::Any)
					{
						return true;
					}
				}
			}
		}
		else {
			//isHit = aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t);
			isHit = hitAABB(transformedRay.org, transformedRay.dir, boxmin, boxmax, t_min, t_max, &t);
		}

		if (isHit) {
			nodeid = (int)node0.w;
		}
		else {
			nodeid = (int)node1.w;
		}

		if (nodeid < 0 && toplayerHit >= 0) {
			nodeid = isHit ? toplayerHit : toplayerMiss;
			toplayerHit = -1;
			toplayerMiss = -1;
			node = ctxt->nodes[0];
			transformedRay = r;
		}
#else
		if (attrib.x >= 0) {
			// Leaf.
			const auto* s = &ctxt->shapes[(int)attrib.x];

			if (attrib.z >= 0) {	// exid
									//if (aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t)) {
				if (hitAABB(r.org, r.dir, boxmin, boxmax, t_min, t_max, &t)) {
					if (s->mtxid >= 0) {
						auto mtxW2L = ctxt->matrices[s->mtxid * 2 + 1];
						transformedRay.dir = mtxW2L.applyXYZ(r.dir);
						transformedRay.dir = normalize(transformedRay.dir);
						transformedRay.org = mtxW2L.apply(r.org) + AT_MATH_EPSILON * transformedRay.dir;
					}
					else {
						transformedRay = r;
					}

					isHit = intersectSBVHTriangles<Type>(ctxt->nodes[(int)attrib.z], ctxt, transformedRay, t_min, t_max, &isectTmp);
				}
			}
			else {
				// TODO
				// Only sphere...
				//isHit = intersectShape(s, nullptr, ctxt, r, t_min, t_max, &recTmp, &recOptTmp);
				isectTmp.t = AT_MATH_INF;
				isHit = hitSphere(s, r, t_min, t_max, &isectTmp);
				isectTmp.mtrlid = s->mtrl.idx;
			}

			bool isIntersect = (Type == idaten::IntersectType::Any
				? isHit
				: isectTmp.t < isect->t);

			if (isIntersect) {
				*isect = isectTmp;
				isect->objid = (int)attrib.x;
				isect->meshid = (isect->meshid < 0 ? (int)attrib.w : isect->meshid);

				t_max = isect->t;

				if (Type == idaten::IntersectType::Closer
					|| Type == idaten::IntersectType::Any)
				{
					return true;
				}
			}
	}
		else {
			//isHit = aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t);
			isHit = hitAABB(r.org, r.dir, boxmin, boxmax, t_min, t_max, &t);
		}

		if (isHit) {
			nodeid = (int)node0.w;
		}
		else {
			nodeid = (int)node1.w;
		}
#endif
	}

	return (isect->objid >= 0);
}

AT_CUDA_INLINE __device__ bool intersectClosestSBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect,
	float t_max/*= AT_MATH_INF*/)
{
	float t_min = AT_MATH_EPSILON;

	bool isHit = intersectSBVH<idaten::IntersectType::Closest>(
		ctxt,
		r,
		t_min, t_max,
		isect);

	return isHit;
}

AT_CUDA_INLINE __device__ bool intersectCloserSBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect,
	const float t_max)
{
	float t_min = AT_MATH_EPSILON;

	bool isHit = intersectSBVH<idaten::IntersectType::Closer>(
		ctxt,
		r,
		t_min, t_max,
		isect);

	return isHit;
}

AT_CUDA_INLINE __device__ bool intersectAnySBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect)
{
	float t_min = AT_MATH_EPSILON;
	float t_max = AT_MATH_INF;

	bool isHit = intersectSBVH<idaten::IntersectType::Any>(
		ctxt,
		r,
		t_min, t_max,
		isect);

	return isHit;
}