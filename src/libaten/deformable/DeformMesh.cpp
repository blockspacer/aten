#include "deformable/DeformMesh.h"
#include "misc/stream.h"

namespace aten
{
	bool DeformMesh::read(
		FileInputStream* stream,
		IDeformMeshReadHelper* helper)
	{
		AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_header, sizeof(m_header)));

		m_header.numMeshGroup = 1;

		bool needKeepGeometryData = m_header.isGPUSkinning;

		m_groups.resize(m_header.numMeshGroup);

		for (uint32_t i = 0; i < m_header.numMeshGroup; i++) {
			AT_VRETURN_FALSE(m_groups[i].read(stream, helper, needKeepGeometryData));
		}

		return true;
	}

	void DeformMesh::render(
		const SkeletonController& skeleton,
		IDeformMeshRenderHelper* helper)
	{
		for (auto& group : m_groups) {
			group.render(skeleton, helper);
		}
	}

	void DeformMesh::getGeometryData(
		std::vector<SkinningVertex>& vtx,
		std::vector<uint32_t>& idx) const
	{
		for (uint32_t i = 0; i < m_header.numMeshGroup; i++) {
			m_groups[i].getGeometryData(vtx, idx);
		}
	}
}
