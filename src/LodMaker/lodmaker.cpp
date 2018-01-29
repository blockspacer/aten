#include "lodmaker.h"

struct QVertex {
	aten::vertex v;
	uint32_t idx;
	uint32_t grid[3];
	uint64_t hash;

	QVertex() {}

	QVertex(
		aten::vertex _v, 
		uint32_t i, 
		uint64_t h,
		uint32_t gx, uint32_t gy, uint32_t gz)
		: v(_v), idx(i), hash(h)
	{
		grid[0] = gx;
		grid[1] = gy;
		grid[2] = gz;
	}

	const QVertex& operator=(const QVertex& rhs)
	{
		v = rhs.v;
		idx = rhs.idx;
		grid[0] = rhs.grid[0];
		grid[1] = rhs.grid[1];
		grid[2] = rhs.grid[2];
		hash = rhs.hash;

		return *this;
	}
};

// NOTE
// gridX * gridY * gridZ �� uint64_t �𒴂��Ȃ��悤�ɂ���

using qit = std::vector<QVertex>::iterator;

static void computeAverage(qit start, qit end)
{
	aten::vec4 pos(start->v.pos);
	aten::vec3 nml(start->v.nml);
	aten::vec3 uv(start->v.uv);

	uint32_t cnt = 1;

	for (auto q = start + 1; q != end; q++) {
		pos += q->v.pos;
		nml += q->v.nml;
		uv += q->v.uv;

		cnt++;
	}

	real div = real(1) / cnt;

	pos *= div;
	nml *= div;
	uv *= div;

	pos.w = real(1);
	nml = normalize(nml);

	// NOTE
	// z is used for checking to compute plane normal in real-time.
	uv.z = uv.z >= 0 ? 1 : 0;

	// �v�Z���ʂ�߂�...
	for (auto q = start; q != end; q++) {
		q->v.pos = pos;
		q->v.nml = nml;
		q->v.uv = uv;
	}
}

static void computeClosestFromAvg(qit start, qit end)
{
	aten::vec4 pos(start->v.pos);

	uint32_t cnt = 1;

	for (auto q = start + 1; q != end; q++) {
		pos += q->v.pos;

		cnt++;
	}

	real div = real(1) / cnt;

	pos *= div;

	real distMin = AT_MATH_INF;

	qit closest = start;

	for (auto q = start; q != end; q++) {
		auto d = (pos - q->v.pos).length();
		if (d < distMin) {
			distMin = d;
			closest = q;
		}
	}

	// �v�Z���ʂ�߂�...
	for (auto q = start; q != end; q++) {
		q->v = closest->v;
	}
}

void LodMaker::make(
	std::vector<aten::vertex>& dstVertices,
	std::vector<std::vector<int>>& dstIndices,
	const aten::aabb& bound,
	const std::vector<aten::vertex>& vertices,
	const std::vector<std::vector<aten::face*>>& triGroups,
	int gridX,
	int gridY,
	int gridZ)
{
	auto bmin = bound.minPos();
	auto range = bound.size();

	aten::vec3 scale(
		(gridX - 1) / range.x,
		(gridY - 1) / range.y,
		(gridZ - 1) / range.z);

	std::vector<std::vector<QVertex>> qvtxs(triGroups.size());
	std::vector<std::vector<uint32_t>> sortedIndices(triGroups.size());

	for (uint32_t i = 0; i < triGroups.size(); i++) {
		const auto tris = triGroups[i];

		qvtxs[i].reserve(tris.size());

		for (uint32_t n = 0; n < tris.size(); n++) {
			const auto tri = tris[n];

			for (int t = 0; t < 3; t++) {
				uint32_t idx = tri->param.idx[t];
				const auto& v = vertices[idx];

				auto grid = ((aten::vec3)v.pos - bmin) * scale + real(0.5);

				uint32_t gx = (uint32_t)grid.x;
				uint32_t gy = (uint32_t)grid.y;
				uint32_t gz = (uint32_t)grid.z;

				uint64_t hash = gz * (gridX * gridY) + gy * gridX + gx;

				qvtxs[i].push_back(QVertex(v, idx, hash, gx, gy, gz));
			}
		}
	}

	// �����O���b�h�ɓ����Ă��钸�_�̏��ɂȂ�悤�Ƀ\�[�g����.

	for (uint32_t i = 0; i < qvtxs.size(); i++)
	{
		std::sort(
			qvtxs[i].begin(), qvtxs[i].end(),
			[](const QVertex& q0, const QVertex& q1)
		{
			return q0.hash > q1.hash;
		});

		uint32_t num = (uint32_t)qvtxs[i].size();

		sortedIndices[i].resize(num);

		// �C���f�b�N�X�����_�ɂ��킹�ĕ��בւ���.
		for (uint32_t n = 0; n < num; n++) {
			const auto& q = qvtxs[i][n];

			// ���X�̃C���f�b�N�X�̈ʒu�ɐV�����C���f�b�N�X�l������.
			sortedIndices[i][q.idx] = n;
		}
	}

	// �����O���b�h���ɓ����Ă��钸�_�̕��ϒl���v�Z���āA�P�̒��_�ɂ��Ă��܂�.
	// UV�͕���邪�A���F�Z�J���_���o�E���X�Ɏg���̂ŁA�����͋C�ɂ��Ȃ�.

	for (uint32_t i = 0; i < qvtxs.size(); i++)
	{
		auto start = qvtxs[i].begin();

		while (start != qvtxs[i].end())
		{
			auto end = start;

			// �O���b�h���قȂ钸�_�ɂȂ�܂ŒT��.
			while (end != qvtxs[i].end() && start->hash == end->hash) {
				end++;
			}

			// TODO
			// ���όv�Z.
			//computeAverage(start, end);
			computeClosestFromAvg(start, end);

			for (auto q = start; q != end; q++) {
				if (q == start) {
					dstVertices.push_back(q->v);
				}
				else {
					// �قȂ�ʒu�̓_�̏ꍇ.

					// �O�̓_�Ɣ�r����.
					auto prev = q - 1;

					auto v0 = q->v.pos;
					auto v1 = prev->v.pos;

					bool isEqual = (memcmp(&v0, &v1, sizeof(v0)) == 0);

					if (!isEqual) {
						dstVertices.push_back(q->v);
					}
				}

				// �C���f�b�N�X�X�V.
				q->idx = (uint32_t)dstVertices.size() - 1;
			}

			// ���̊J�n�ʒu���X�V
			start = end;
		}
	}

	// LOD���ꂽ���ʂ̃C���f�b�N�X���i�[.

	dstIndices.resize(triGroups.size());

	for (uint32_t i = 0; i < triGroups.size(); i++) {
		const auto tris = triGroups[i];

		dstIndices[i].reserve(tris.size() * 3);

		for (uint32_t n = 0; n < tris.size(); n++) {
			const auto tri = tris[n];

			for (int t = 0; t < 3; t++) {
				uint32_t idx = tri->param.idx[t];

				auto sortedIdx = sortedIndices[i][idx];
				auto newIdx = qvtxs[i][sortedIdx].idx;

				dstIndices[i].push_back(newIdx);
			}
		}
	}
}

void LodMaker::removeCollapsedTriangles(
	std::vector<std::vector<int>>& dstIndices,
	const std::vector<aten::vertex>& vertices,
	const std::vector<std::vector<int>>& indices)
{
	dstIndices.resize(indices.size());

	for (int i = 0; i < indices.size(); i++) {
		const auto& idxs = indices[i];

		for (int n = 0; n < idxs.size(); n += 3) {
			auto id0 = idxs[n + 0];
			auto id1 = idxs[n + 1];
			auto id2 = idxs[n + 2];

			const auto& v0 = vertices[id0];
			const auto& v1 = vertices[id1];
			const auto& v2 = vertices[id2];

			// �O�p�`�̖ʐ� = �Q�ӂ̊O�ς̒��� / 2;
			auto e0 = v1.pos - v0.pos;
			auto e1 = v2.pos - v0.pos;
			auto area = real(0.5) * cross(e0, e1).length();

			if (area > real(0)) {
				dstIndices[i].push_back(id0);
				dstIndices[i].push_back(id1);
				dstIndices[i].push_back(id2);
			}
		}
	}
}