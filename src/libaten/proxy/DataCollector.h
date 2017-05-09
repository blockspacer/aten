#pragma once

#include <vector>
#include "shape/shape.h"
#include "light/light.h"
#include "material/material.h"
#include "object/vertex.h"

namespace aten {
	class DataCollector {
	private:
		DataCollector() {}
		~DataCollector() {}

	public:
		static void collect(
			std::vector<aten::ShapeParameter>& shapeparams,
			std::vector<aten::PrimitiveParamter>& primparams,
			std::vector<aten::LightParameter>& lightparams,
			std::vector<aten::MaterialParameter>& mtrlparms,
			std::vector<aten::vertex>& vtxparams);
	};
}