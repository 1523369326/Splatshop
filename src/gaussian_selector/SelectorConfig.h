#pragma once

#include <cstddef>
#include <cstdint>

namespace gaussian_selector {

enum class RefinementMode { InteractiveFullGraph, ForegroundRestrictedExperimental };

struct SelectorConfig {
    std::uint32_t knn = 8;
    double gateQuantile = 0.95;
    double spatialWeight = 0.35;
    double colorWeight = 0.4;
    double opacityWeight = 0.25;
    double leidenResolution = 0.05;
    double leidenBeta = 0.01;
    std::uint32_t leidenIterations = 2;
    std::uint64_t randomSeed = 42;
    std::uint32_t gmmComponents = 3;
    std::uint32_t emIterations = 80;
    double emRelativeTolerance = 1e-4;
    double varianceFloor = 1e-3;
    double standardDeviationFloor = 1e-6;
    double targetSeedConfidence = 0.95;
    double seedPenalty = 4.0;
    double pairwiseLambda = 1.0;
    double coverageThreshold = 0.10;
    double minimumVisibleMass = 0.01;
    bool adaptiveCoverageThreshold = true;
    double coverageQuantile = 0.10;
    double centerDepthTolerance = 0.02;
    std::uint32_t minimumScribbleSupport = 1; // E: notes do not publish gate cutoff.
    bool requireCenterEvidence = false; // E: measurements exposed; combining rule unpublished.
    // Author notes: Spark evidence visibility stops below T=0.9.
    // 0 disables the extra evidence cutoff (standard raster replay).
    float evidenceTransmittanceStop = 0.9f;
    std::uint32_t evidenceOwnerBudget = 16;
    bool splitConflictingSuperpoints = true;
    std::uint32_t maximumConflictSplits = 64;
    std::uint32_t maximumConflictMembers = 20'000;
    bool refineUncertainSuperpoints = true;
    std::uint32_t maximumUncertainSplits = 4;
    std::uint32_t maximumUncertainMembers = 20'000;
    double uncertainProbabilityGap = 0.25;
    bool useSoftKnnContrast = true;
    std::uint32_t softKnn = 8;
    // The notes do not publish unary_mix, budgets or filter defaults. E values.
    double unaryGmmMix = 0.5;
    bool allowPseudoBackground = true;
    std::uint32_t maximumPseudoBackground = 32;
    double pseudoBackgroundConfidence = 0.1;
    std::uint32_t graphNodeTarget = 40'000;
    std::uint32_t graphNodeBudget = 50'000;
    std::uint32_t voxelSearchIterations = 20;
    bool integerGraphCut = true;
    std::uint32_t integerCapacityPrecision = 1'000'000;
    bool keepSeededComponents = true;
    bool rejectOversizedGaussians = true;
    double oversizedMedianMultiplier = 8.0;
    std::size_t cpuBudgetBytes = std::size_t{1} << 30;
    std::size_t gpuBudgetBytes = std::size_t{256} << 20;
    std::size_t fullScopeWarningCount = 500'000;
    float roiForegroundPaddingFraction = 0.20f;
    float roiScenePaddingFraction = 0.02f;
    RefinementMode refinementMode = RefinementMode::InteractiveFullGraph;
};

} // namespace gaussian_selector
