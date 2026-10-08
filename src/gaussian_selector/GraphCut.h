#pragma once

#include "SelectorTypes.h"

#include <span>

namespace gaussian_selector {

struct CutResult { std::vector<std::uint8_t> foreground; double flow=0; double energy=0; double capacityScale=0; };
double evaluateEnergy(const WeightedGraph& graph,std::span<const double> foregroundCost,
    std::span<const double> backgroundCost,std::span<const std::uint8_t> foreground,double lambda);
CutResult solveGraphCut(const WeightedGraph& graph,std::span<const double> foregroundCost,
    std::span<const double> backgroundCost,double lambda);
CutResult solveIntegerGraphCut(const WeightedGraph& graph,std::span<const double> foregroundCost,
    std::span<const double> backgroundCost,double lambda,std::uint32_t precision=1'000'000);

} // namespace gaussian_selector
