#pragma once

#include "SelectorConfig.h"
#include "SelectorTypes.h"

#include <span>

namespace gaussian_selector {

WeightedGraph buildSpatialGraph(std::span<const Vec3> positions,
    std::span<const CacFeature> cac, std::span<const float> opacity,
    const SelectorConfig& config);

} // namespace gaussian_selector
