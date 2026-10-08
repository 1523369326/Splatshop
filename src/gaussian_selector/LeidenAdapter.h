#pragma once

#include "SelectorConfig.h"
#include "SelectorTypes.h"

#include <string>

namespace gaussian_selector {

struct LeidenResult {
    std::vector<std::uint32_t> membership;
    std::uint32_t clusterCount = 0;
    double quality = 0;
    std::string error;
    explicit operator bool() const { return error.empty(); }
};

LeidenResult runWeightedLeiden(const WeightedGraph& graph,const SelectorConfig& config);

} // namespace gaussian_selector
