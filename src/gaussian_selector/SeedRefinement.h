#pragma once

#include "SelectorConfig.h"
#include "SelectorTypes.h"
#include <optional>
#include <span>

namespace gaussian_selector {
struct SeedRefinement {
    std::optional<SceneEncoding> encoding;
    std::size_t conflicts = 0;
    std::size_t split = 0;
    std::size_t unresolved = 0;
};

// A request-local partition; the cached, stroke-independent encoding is never
// changed. Re-cluster only mixed FG/BG superpoints, with explicit work limits.
SeedRefinement splitSeedConflicts(const SceneEncoding& base,
    std::span<const SeedLabel> seeds, const SelectorConfig& config);
SeedRefinement splitUncertainSuperpoints(const SceneEncoding& base,
    std::span<const double> calibratedLogits,const SelectorConfig& config);
} // namespace gaussian_selector
