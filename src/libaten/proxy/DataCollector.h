#pragma once

#include <vector>
#include "geometry/geomparam.h"
#include "light/light.h"
#include "material/material.h"
#include "geometry/vertex.h"
#include "scene/scene.h"

namespace aten {
    class DataCollector {
    private:
        DataCollector() {}
        ~DataCollector() {}

    public:
        static void collect(
            scene& scene,
            std::vector<aten::GeomParameter>& shapeparams,
            std::vector<aten::PrimitiveParamter>& primparams,
            std::vector<aten::LightParameter>& lightparams,
            std::vector<aten::MaterialParameter>& mtrlparms,
            std::vector<aten::vertex>& vtxparams);

        static void collectTriangles(
            std::vector<std::vector<aten::PrimitiveParamter>>& triangles,
            std::vector<int>& triIdOffsets);
    };
}