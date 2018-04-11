#include "deformable/DeformMesh.h"
#include "misc/stream.h"

namespace aten
{
	bool DeformMesh::read(
		FileInputStream* stream,
		IDeformMeshReadHelper* helper)
	{
		AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_header, sizeof(m_header)));

		// TODO
		// Not support multi group (aka. LOD).
		m_header.numMeshGroup = 1;

		bool isGPUSkinning = m_header.isGPUSkinning;

		m_groups.resize(m_header.numMeshGroup);

		for (uint32_t i = 0; i < m_header.numMeshGroup; i++) {
			AT_VRETURN_FALSE(m_groups[i].read(stream, helper, isGPUSkinning));
		}

		return true;
	}

	void DeformMesh::render(
		const SkeletonController& skeleton,
		IDeformMeshRenderHelper* helper)
	{
		bool isGPUSkinning = m_header.isGPUSkinning;
		m_groups[0].render(skeleton, helper, isGPUSkinning);
	}

	void DeformMesh::getGeometryData(
		std::vector<SkinningVertex>& vtx,
		std::vector<uint32_t>& idx,
		std::vector<aten::PrimitiveParamter>& tris) const
	{
		m_groups[0].getGeometryData(vtx, idx, tris);
	}
}
