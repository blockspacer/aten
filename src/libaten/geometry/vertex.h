#pragma once

#include <vector>
#include "types.h"
#include "math/vec4.h"
#include "visualizer/GeomDataBuffer.h"

namespace aten
{
	struct vertex {
		vec4 pos;
		vec3 nml;
		vec3 uv;	// z is used for checking to compute plane normal in real-time.
	};

	class VertexManager {
		static std::vector<int> s_indices;
		static std::vector<vertex> s_vertices;

		static GeomVertexBuffer s_vb;

	private:
		VertexManager() {}
		~VertexManager() {}

	public:
		static void addIndex(int idx)
		{
			s_indices.push_back(idx);
		}
		static void addVertex(const vertex& vtx)
		{
			s_vertices.push_back(vtx);
		}

		static int getIndex(int pos)
		{
			AT_ASSERT(pos < s_indices.size());
			return s_indices[pos];
		}
		static vertex& getVertex(int idx)
		{
			return s_vertices[idx];
		}

		static const std::vector<int>& getIndices()
		{
			return s_indices;
		}
		static const std::vector<vertex>& getVertices()
		{
			return s_vertices;
		}

		static uint32_t getVertexNum()
		{
			return (uint32_t)s_vertices.size();
		}

		static void build();
		static GeomVertexBuffer& getVB()
		{
			return s_vb;
		}
	};
}