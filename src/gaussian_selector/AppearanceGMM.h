#pragma once

#include "SelectorConfig.h"
#include "SelectorTypes.h"

#include <span>

namespace gaussian_selector {

struct DiagonalGmm {
    std::vector<double> weights;
    std::vector<AppearanceFeature> means;
    std::vector<AppearanceFeature> variances;
    bool degraded = false;
    double logLikelihood(const AppearanceFeature& x) const;
};

DiagonalGmm fitDiagonalGmm(std::span<const AppearanceFeature> samples,const SelectorConfig& config);

struct UnaryCosts { std::vector<double> foreground,background,calibrated; bool degraded=false; };
UnaryCosts buildAppearanceUnaries(std::span<const AppearanceFeature> features,
    std::span<const SeedLabel> seeds,const SelectorConfig& config);
UnaryCosts buildAppearanceUnariesWithReservoir(std::span<const AppearanceFeature> features,
    std::span<const SeedLabel> seeds,
    std::span<const AppearanceFeature> foregroundReservoir,
    std::span<const AppearanceFeature> backgroundReservoir,
    const SelectorConfig& config);

UnaryCosts buildBlendedAppearanceUnaries(std::span<const AppearanceFeature> features,
    std::span<const SeedLabel> seeds,std::span<const SeedConfidence> confidence,
    const SelectorConfig& config);

} // namespace gaussian_selector
