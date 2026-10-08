#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace gaussian_selector {

constexpr std::uint32_t kInvalidRow = std::numeric_limits<std::uint32_t>::max();
constexpr std::size_t kCacDimensions = 18;
constexpr std::size_t kAppearanceDimensions = 19;

struct Vec3 { float x = 0, y = 0, z = 0; };
struct Quaternion { float w = 1, x = 0, y = 0, z = 0; };
using CacFeature = std::array<float, kCacDimensions>;
using AppearanceFeature = std::array<double, kAppearanceDimensions>;

struct SnapshotKey {
    std::uint64_t sceneRevision = 0;
    std::uint64_t encodingRevision = 0;
    std::uint64_t indexEpoch = 0;
    auto operator<=>(const SnapshotKey&) const = default;
};

struct JobKey {
    SnapshotKey snapshot;
    std::uint64_t evidenceRevision = 0;
    std::uint64_t roiRevision = 0;
    std::uint64_t requestId = 0;
    auto operator<=>(const JobKey&) const = default;
};

struct GaussianIdentity {
    std::uint64_t nodeIdentity = 0;
    std::uint64_t indexEpoch = 0;
    std::uint32_t localIndex = 0;
};

struct GaussianSnapshotRow {
    GaussianIdentity identity;
    Vec3 position;
    Vec3 physicalScale{1, 1, 1};
    Quaternion rotation;
    std::array<float, 3> baseColor{};
    float opacity = 0;
    int shDegree = 0;
    // RGB-interleaved higher-order coefficients; DC is intentionally absent.
    std::vector<float> shRest;
};

enum class SeedLabel : std::uint8_t { Unknown, Foreground, Background };

struct WeightedEdge {
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    double weight = 0;
};

struct WeightedGraph {
    std::uint32_t nodeCount = 0;
    std::vector<WeightedEdge> edges;
};

struct Superpoint {
    Vec3 position;
    CacFeature cac{};
    float opacity = 0;
    std::uint32_t memberCount = 0;
};

struct SceneEncoding {
    SnapshotKey key;
    std::vector<CacFeature> gaussianCac;
    // Only attributes needed for bounded local re-clustering; no duplicate SH.
    std::vector<Vec3> gaussianPositions;
    std::vector<float> gaussianOpacity;
    std::vector<float> gaussianMaxScale;
    std::size_t encodingNodeCount = 0;
    bool voxelPooled = false;
    std::vector<std::uint32_t> gaussianToSuperpoint;
    std::vector<Superpoint> superpoints;
    WeightedGraph superpointGraph;
    std::vector<AppearanceFeature> standardizedAppearance;
};

struct CoverageMass {
    double all = 0;
    double foreground = 0;
    double background = 0;
    double foregroundSupport = -1, backgroundSupport = -1;
    double foregroundCenter = -1, backgroundCenter = -1;
};

struct SeedConfidence { double foreground=0, background=0; };
struct ViewEvidence {
    std::vector<SeedLabel> labels;
    std::vector<SeedConfidence> confidence;
};

struct SolveDiagnostics {
    bool degradedAppearance = false;
    bool experimentalRefinement = false;
    std::size_t refinementIterations = 0;
    std::size_t foregroundSeeds = 0;
    std::size_t backgroundSeeds = 0;
    std::size_t conflictingSuperpoints = 0;
    std::size_t splitSuperpoints = 0;
    std::size_t unresolvedConflicts = 0;
    std::size_t solvedSuperpoints = 0;
    std::size_t solvedEdges = 0;
    std::size_t uncertainSplits = 0;
    std::size_t removedUnseededComponents = 0;
    std::size_t removedOversizedGaussians = 0;
    bool integerCut = false;
    double capacityScale = 0;
    std::size_t pseudoBackgroundSamples = 0;
    double cutEnergy = 0;
    std::string message;
};

struct SelectionResult {
    JobKey key;
    std::vector<std::uint64_t> maskWords;
    std::vector<std::uint8_t> superpointForeground;
    SolveDiagnostics diagnostics;
};

inline bool maskGet(const std::vector<std::uint64_t>& words, std::size_t i) {
    return i / 64 < words.size() && ((words[i / 64] >> (i % 64)) & 1u) != 0;
}
inline void maskSet(std::vector<std::uint64_t>& words, std::size_t i, bool value) {
    if (words.size() <= i / 64) words.resize(i / 64 + 1);
    const auto bit = std::uint64_t{1} << (i % 64);
    if (value) words[i / 64] |= bit; else words[i / 64] &= ~bit;
}

} // namespace gaussian_selector
