#include "SeedRefinement.h"
#include "LeidenAdapter.h"
#include "SpatialGraph.h"
#include "SuperpointEncoding.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>

namespace gaussian_selector {
SeedRefinement splitSeedConflicts(const SceneEncoding& base,
    std::span<const SeedLabel> seeds, const SelectorConfig& config) {
    if(seeds.size()!=base.gaussianToSuperpoint.size())throw std::invalid_argument("seed count mismatch");
    const auto k=base.superpoints.size();
    std::vector<unsigned char> labels(k,0);
    for(std::size_t i=0;i<seeds.size();++i){
        const auto sp=base.gaussianToSuperpoint[i];
        if(sp>=k)throw std::out_of_range("superpoint id");
        if(seeds[i]==SeedLabel::Foreground)labels[sp]|=1;
        if(seeds[i]==SeedLabel::Background)labels[sp]|=2;
    }
    SeedRefinement result;
    result.conflicts=std::count(labels.begin(),labels.end(),3);
    if(!result.conflicts || !config.splitConflictingSuperpoints){result.unresolved=result.conflicts;return result;}
    if(base.gaussianPositions.size()!=seeds.size() || base.gaussianCac.size()!=seeds.size() || base.gaussianOpacity.size()!=seeds.size())
        throw std::invalid_argument("conflict refinement needs per-Gaussian encoding attributes");

    std::vector<std::vector<std::uint32_t>> members(k);
    for(std::uint32_t i=0;i<seeds.size();++i)if(labels[base.gaussianToSuperpoint[i]]==3)
        members[base.gaussianToSuperpoint[i]].push_back(i);
    auto membership=base.gaussianToSuperpoint;
    std::uint32_t next=static_cast<std::uint32_t>(k);
    for(std::uint32_t sp=0;sp<k;++sp){
        const auto& ids=members[sp];
        if(ids.empty())continue;
        if(result.split>=config.maximumConflictSplits || ids.size()>config.maximumConflictMembers){++result.unresolved;continue;}
        std::vector<Vec3> positions(ids.size());
        std::vector<CacFeature> cac(ids.size());
        std::vector<float> opacity(ids.size());
        for(std::size_t i=0;i<ids.size();++i){positions[i]=base.gaussianPositions[ids[i]];cac[i]=base.gaussianCac[ids[i]];opacity[i]=base.gaussianOpacity[ids[i]];}
        auto graph=buildSpatialGraph(positions,cac,opacity,config);
        auto partition=runWeightedLeiden(graph,config);
        if(!partition)throw std::runtime_error(partition.error);
        std::vector<unsigned char> communityLabels(partition.clusterCount,0);
        for(std::size_t i=0;i<ids.size();++i){
            if(seeds[ids[i]]==SeedLabel::Foreground)communityLabels[partition.membership[i]]|=1;
            if(seeds[ids[i]]==SeedLabel::Background)communityLabels[partition.membership[i]]|=2;
        }
        // The notes do not publish the constrained splitting rule. Engineering
        // guard: if local Leiden still merges opposing seeds, separate FG, BG
        // and unknown members inside that community. Unknowns are NOT seeds.
        std::map<std::pair<std::uint32_t,int>,std::uint32_t> childIds;
        for(std::size_t i=0;i<ids.size();++i){
            const auto community=partition.membership[i];
            const int label=communityLabels[community]==3?int(seeds[ids[i]]):-1;
            auto [it,inserted]=childIds.emplace(std::make_pair(community,label),0);
            if(inserted)it->second=childIds.size()==1?sp:next++;
            membership[ids[i]]=it->second;
        }
        ++result.split;
    }
    if(result.split)result.encoding=poolAndBuildSuperpoints(base.key,base.gaussianPositions,
        base.gaussianCac,base.gaussianOpacity,membership,config);
    if(result.encoding){result.encoding->gaussianMaxScale=base.gaussianMaxScale;result.encoding->voxelPooled=base.voxelPooled;result.encoding->encodingNodeCount=base.encodingNodeCount;}
    return result;
}

SeedRefinement splitUncertainSuperpoints(const SceneEncoding& base,
    std::span<const double> logits,const SelectorConfig& config){
    SeedRefinement result;if(!config.refineUncertainSuperpoints)return result;
    if(logits.size()!=base.superpoints.size())throw std::invalid_argument("uncertainty size mismatch");
    if(base.gaussianPositions.size()!=base.gaussianToSuperpoint.size())return result;
    std::vector<std::pair<double,std::uint32_t>> candidates;
    for(std::uint32_t i=0;i<logits.size();++i){const double gap=std::abs(std::tanh(logits[i]));const auto size=base.superpoints[i].memberCount;if(gap<config.uncertainProbabilityGap&&size>1&&size<=config.maximumUncertainMembers)candidates.emplace_back(gap,i);}
    std::sort(candidates.begin(),candidates.end());if(candidates.size()>config.maximumUncertainSplits)candidates.resize(config.maximumUncertainSplits);
    if(candidates.empty())return result;
    std::map<std::uint32_t,std::vector<std::uint32_t>> members;for(auto [_,sp]:candidates)members[sp];
    for(std::uint32_t i=0;i<base.gaussianToSuperpoint.size();++i){auto it=members.find(base.gaussianToSuperpoint[i]);if(it!=members.end())it->second.push_back(i);}
    auto membership=base.gaussianToSuperpoint;std::uint32_t next=std::uint32_t(base.superpoints.size());
    for(const auto& [sp,ids]:members){
        std::vector<Vec3> positions(ids.size());std::vector<CacFeature> cac(ids.size());std::vector<float> opacity(ids.size());
        for(std::size_t i=0;i<ids.size();++i){positions[i]=base.gaussianPositions[ids[i]];cac[i]=base.gaussianCac[ids[i]];opacity[i]=base.gaussianOpacity[ids[i]];}
        auto local=config;local.leidenResolution=std::max(.2,4*config.leidenResolution); // E: finer local resolution, absent from notes.
        auto partition=runWeightedLeiden(buildSpatialGraph(positions,cac,opacity,local),local);if(!partition)throw std::runtime_error(partition.error);
        if(partition.clusterCount<2)continue;
        std::vector<std::uint32_t> child(partition.clusterCount);child[0]=sp;for(std::size_t i=1;i<child.size();++i)child[i]=next++;
        for(std::size_t i=0;i<ids.size();++i)membership[ids[i]]=child[partition.membership[i]];
        ++result.split;
    }
    if(result.split){result.encoding=poolAndBuildSuperpoints(base.key,base.gaussianPositions,base.gaussianCac,base.gaussianOpacity,membership,config);result.encoding->gaussianMaxScale=base.gaussianMaxScale;result.encoding->voxelPooled=base.voxelPooled;result.encoding->encodingNodeCount=base.encodingNodeCount;}
    return result;
}
} // namespace gaussian_selector
