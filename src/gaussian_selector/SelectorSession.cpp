#include "SelectorSession.h"
#include "AppearanceGMM.h"
#include "GraphCut.h"
#include "SeedEvidence.h"
#include "SeedRefinement.h"
#include "SpatialGraph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gaussian_selector {
SelectionResult SelectorSession::solve(JobKey key,std::span<const SeedLabel> gaussianSeeds,
    std::span<const SeedConfidence> gaussianConfidence)const{
    if(!encoding_)throw std::invalid_argument("missing selector encoding");
    if(key.snapshot!=encoding_->key)throw std::invalid_argument("stale selector snapshot");
    if(gaussianSeeds.size()!=encoding_->gaussianToSuperpoint.size())throw std::invalid_argument("seed count mismatch");
    if(!gaussianConfidence.empty()&&gaussianConfidence.size()!=gaussianSeeds.size())throw std::invalid_argument("confidence count mismatch");
    auto conflicts=splitSeedConflicts(*encoding_,gaussianSeeds,config_);
    if(config_.splitConflictingSuperpoints&&conflicts.unresolved)
        throw std::runtime_error("conflicting FG/BG superpoints exceed local split budget; increase Leiden resolution or use a smaller ROI");
    const SceneEncoding* encoding=conflicts.encoding?&*conflicts.encoding:encoding_.get();
    SelectionResult out;out.key=key;out.diagnostics.conflictingSuperpoints=conflicts.conflicts;
    out.diagnostics.splitSuperpoints=conflicts.split;out.diagnostics.unresolvedConflicts=conflicts.unresolved;
    std::vector<SeedLabel> spSeeds;
    std::vector<SeedConfidence> spConfidence;
    auto aggregate=[&]{
        spSeeds=broadcastSeeds(gaussianSeeds,encoding->gaussianToSuperpoint,encoding->superpoints.size());
        spConfidence.assign(spSeeds.size(),{});std::vector<std::size_t> counts(spSeeds.size());
        for(std::size_t i=0;i<gaussianSeeds.size();++i){const auto sp=encoding->gaussianToSuperpoint[i];
            const auto c=gaussianConfidence.empty()?SeedConfidence{gaussianSeeds[i]==SeedLabel::Foreground?1.0:0.0,gaussianSeeds[i]==SeedLabel::Background?1.0:0.0}:gaussianConfidence[i];
            if(c.foreground>0||c.background>0){spConfidence[sp].foreground+=c.foreground;spConfidence[sp].background+=c.background;++counts[sp];}}
        for(std::size_t i=0;i<counts.size();++i)if(counts[i]){spConfidence[i].foreground/=counts[i];spConfidence[i].background/=counts[i];}
        if(std::none_of(spSeeds.begin(),spSeeds.end(),[](auto s){return s==SeedLabel::Foreground;}))
            throw std::invalid_argument("foreground scribble evidence required");
        out.diagnostics.pseudoBackgroundSamples=0;
        if(std::none_of(spSeeds.begin(),spSeeds.end(),[](auto s){return s==SeedLabel::Background;})){
            if(!config_.allowPseudoBackground||!config_.useSoftKnnContrast)throw std::invalid_argument("background scribble required with pseudo-BG disabled");
            AppearanceFeature mean{};std::size_t count=0;for(std::size_t i=0;i<spSeeds.size();++i)if(spSeeds[i]==SeedLabel::Foreground){for(std::size_t d=0;d<mean.size();++d)mean[d]+=encoding->standardizedAppearance[i][d];++count;}for(auto& x:mean)x/=count;
            std::vector<std::pair<double,std::uint32_t>> candidates;
            for(std::uint32_t i=0;i<spSeeds.size();++i)if(spSeeds[i]==SeedLabel::Unknown&&spConfidence[i].foreground==0){double distance=0;for(std::size_t d=0;d<mean.size();++d){const double delta=encoding->standardizedAppearance[i][d]-mean[d];distance+=delta*delta;}candidates.emplace_back(-distance,i);}
            std::sort(candidates.begin(),candidates.end());if(candidates.size()>config_.maximumPseudoBackground)candidates.resize(config_.maximumPseudoBackground);
            // Notes publish a 32-sample weak floor but not its semantic sample
            // ranking. E: farthest appearance without FG support, soft only.
            for(auto [_,i]:candidates)spConfidence[i].background=std::max(spConfidence[i].background,config_.pseudoBackgroundConfidence);
            out.diagnostics.pseudoBackgroundSamples=candidates.size();
            if(candidates.empty())throw std::invalid_argument("no pseudo-BG candidates; draw a background stroke");
        }
    };
    auto solveCut=[&](const WeightedGraph& graph,const UnaryCosts& unary){
        return config_.integerGraphCut?solveIntegerGraphCut(graph,unary.foreground,unary.background,config_.pairwiseLambda,config_.integerCapacityPrecision):solveGraphCut(graph,unary.foreground,unary.background,config_.pairwiseLambda);
    };
    aggregate();
    auto unary=buildBlendedAppearanceUnaries(encoding->standardizedAppearance,spSeeds,spConfidence,config_);
    auto cut=solveCut(encoding->superpointGraph,unary);
    out.diagnostics.refinementIterations=1;out.diagnostics.degradedAppearance=unary.degraded;
    // Author notes: one post-cut round on up to four low-confidence SPs.
    auto uncertain=splitUncertainSuperpoints(*encoding,unary.calibrated,config_);
    if(uncertain.encoding){
        encoding=&*uncertain.encoding;aggregate();
        unary=buildBlendedAppearanceUnaries(encoding->standardizedAppearance,spSeeds,spConfidence,config_);
        cut=solveCut(encoding->superpointGraph,unary);++out.diagnostics.refinementIterations;
        out.diagnostics.degradedAppearance|=unary.degraded;
    }
    out.diagnostics.uncertainSplits=uncertain.split;
    out.superpointForeground=std::move(cut.foreground);out.diagnostics.cutEnergy=cut.energy;
    out.diagnostics.capacityScale=cut.capacityScale;out.diagnostics.integerCut=config_.integerGraphCut;
    out.diagnostics.solvedSuperpoints=encoding->superpoints.size();out.diagnostics.solvedEdges=encoding->superpointGraph.edges.size();
    for(auto s:spSeeds){out.diagnostics.foregroundSeeds+=s==SeedLabel::Foreground;out.diagnostics.backgroundSeeds+=s==SeedLabel::Background;}

    // Keep the foreground-restricted research mode distinct from the two-stage
    // SP splitting described in the author notes.
    if(config_.refinementMode==RefinementMode::ForegroundRestrictedExperimental){
        if(out.diagnostics.pseudoBackgroundSamples)throw std::invalid_argument("experimental FG restriction requires real BG strokes");
        out.diagnostics.experimentalRefinement=true;
        std::vector<AppearanceFeature> fg,bg;for(std::size_t i=0;i<spSeeds.size();++i){if(spSeeds[i]==SeedLabel::Foreground)fg.push_back(encoding->standardizedAppearance[i]);if(spSeeds[i]==SeedLabel::Background)bg.push_back(encoding->standardizedAppearance[i]);}
        for(std::size_t iteration=0;iteration<2;++iteration){
            std::vector<std::uint32_t> active;for(std::uint32_t i=0;i<out.superpointForeground.size();++i)if(out.superpointForeground[i])active.push_back(i);if(active.empty())break;
            std::vector<Vec3> positions(active.size());std::vector<CacFeature> cac(active.size());std::vector<float> opacity(active.size());std::vector<AppearanceFeature> features(active.size());std::vector<SeedLabel> seeds(active.size());
            for(std::size_t i=0;i<active.size();++i){const auto old=active[i];positions[i]=encoding->superpoints[old].position;cac[i]=encoding->superpoints[old].cac;opacity[i]=encoding->superpoints[old].opacity;features[i]=encoding->standardizedAppearance[old];seeds[i]=spSeeds[old];}
            auto graph=buildSpatialGraph(positions,cac,opacity,config_);auto restricted=buildAppearanceUnariesWithReservoir(features,seeds,fg,bg,config_);auto result=solveCut(graph,restricted);
            std::vector<std::uint8_t> next(out.superpointForeground.size(),0);for(std::size_t i=0;i<active.size();++i)next[active[i]]=result.foreground[i];
            out.diagnostics.degradedAppearance|=restricted.degraded;out.diagnostics.cutEnergy=result.energy;++out.diagnostics.refinementIterations;
            if(next==out.superpointForeground)break;out.superpointForeground=std::move(next);
        }
        out.diagnostics.message="ForegroundRestrictedExperimental remains a separate engineering variant; original BG reservoir retained";
    }

    if(config_.keepSeededComponents){
        std::vector<std::uint32_t> parent(spSeeds.size());std::iota(parent.begin(),parent.end(),0);
        auto root=[&](std::uint32_t v){while(parent[v]!=v){parent[v]=parent[parent[v]];v=parent[v];}return v;};
        for(const auto& e:encoding->superpointGraph.edges)if(e.weight>0&&out.superpointForeground[e.a]&&out.superpointForeground[e.b])parent[root(e.a)]=root(e.b);
        std::vector<bool> anchored(spSeeds.size());for(std::uint32_t i=0;i<spSeeds.size();++i)if(out.superpointForeground[i]&&spSeeds[i]==SeedLabel::Foreground)anchored[root(i)]=true;
        std::vector<bool> removed(spSeeds.size());for(std::uint32_t i=0;i<spSeeds.size();++i)if(out.superpointForeground[i]&&!anchored[root(i)]){out.superpointForeground[i]=0;removed[root(i)]=true;}
        out.diagnostics.removedUnseededComponents=std::count(removed.begin(),removed.end(),true);
    }
    double scaleLimit=std::numeric_limits<double>::infinity();
    if(config_.rejectOversizedGaussians&&!gaussianSeeds.empty()&&encoding->gaussianMaxScale.size()==gaussianSeeds.size()){
        auto scales=encoding->gaussianMaxScale;const auto mid=scales.begin()+scales.size()/2;
        std::nth_element(scales.begin(),mid,scales.end());scaleLimit=std::max(double(*mid),1e-12)*config_.oversizedMedianMultiplier;
    }
    out.maskWords.resize((gaussianSeeds.size()+63)/64);
    for(std::size_t i=0;i<gaussianSeeds.size();++i){bool selected=out.superpointForeground[encoding->gaussianToSuperpoint[i]]!=0;
        if(selected&&std::isfinite(scaleLimit)&&encoding->gaussianMaxScale[i]>scaleLimit){selected=false;++out.diagnostics.removedOversizedGaussians;}
        maskSet(out.maskWords,i,selected);}
    return out;
}
} // namespace gaussian_selector
