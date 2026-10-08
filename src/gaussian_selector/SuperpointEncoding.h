#pragma once

#include "SelectorConfig.h"
#include "SelectorTypes.h"

#include <span>

namespace gaussian_selector {

SceneEncoding poolAndBuildSuperpoints(SnapshotKey key,
    std::span<const GaussianSnapshotRow> rows,std::span<const CacFeature> cac,
    std::span<const std::uint32_t> membership,const SelectorConfig& config);

SceneEncoding poolAndBuildSuperpoints(SnapshotKey key,
    std::span<const Vec3> positions, std::span<const CacFeature> cac,
    std::span<const float> opacity, std::span<const std::uint32_t> membership,
    const SelectorConfig& config);

// Exact Gaussian graph below budget; adaptive voxel mean nodes above it.
SceneEncoding encodeScene(SnapshotKey key,std::span<const GaussianSnapshotRow> rows,
    const SelectorConfig& config);

} // namespace gaussian_selector
