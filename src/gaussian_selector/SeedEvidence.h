#pragma once

#include "SelectorConfig.h"
#include "SelectorTypes.h"

#include <span>

namespace gaussian_selector {

struct ReferenceFragment {
    std::uint32_t gaussian=0;
    double intrinsicAlpha=0;
    float depth=0;
    bool writesDepth=true;
};

std::vector<CoverageMass> referenceCoverage(
    std::size_t gaussianCount,
    std::span<const std::vector<ReferenceFragment>> pixels,
    std::span<const std::uint8_t> foregroundMask,
    std::span<const std::uint8_t> backgroundMask,
    double transmittanceStop=4.0/255.0,
    float evidenceTransmittanceStop=0,
    std::span<const float> initialOpaqueDepth={});

std::vector<SeedLabel> classifyCoverage(std::span<const CoverageMass> mass,const SelectorConfig& config);
ViewEvidence classifyEvidence(std::span<const CoverageMass> mass,const SelectorConfig& config);
std::vector<SeedConfidence> averageConfidence(std::span<const std::vector<SeedConfidence>> views);
std::vector<SeedLabel> majorityVote(std::span<const std::vector<SeedLabel>> views);
std::vector<SeedLabel> broadcastSeeds(std::span<const SeedLabel> gaussianSeeds,
    std::span<const std::uint32_t> gaussianToSuperpoint,std::size_t superpointCount,
    std::size_t* conflicts=nullptr);

} // namespace gaussian_selector
