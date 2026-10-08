#include "SuperpointEncoding.h"
#include "CanonicalAxisColor.h"
#include "SpatialGraph.h"
#include "LeidenAdapter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace gaussian_selector {
namespace {
struct Cell {std::int64_t x,y,z;bool operator==(const Cell&)const=default;};
struct CellHash {
    std::size_t operator()(const Cell& c)const {
        return std::size_t(std::uint64_t(c.x)*73856093ull ^ std::uint64_t(c.y)*19349663ull ^ std::uint64_t(c.z)*83492791ull);
    }
};
struct Bucket {std::array<double,3> position{};std::array<double,18> cac{};double opacity=0;std::size_t count=0;};
}

SceneEncoding encodeScene(SnapshotKey key,std::span<const GaussianSnapshotRow> rows,const SelectorConfig& config){
    if(config.graphNodeBudget<2 || !config.graphNodeTarget || config.graphNodeTarget>config.graphNodeBudget)
        throw std::invalid_argument("invalid adaptive graph budget");
    std::vector<Vec3> positions(rows.size());std::vector<CacFeature> cac(rows.size());std::vector<float> opacity(rows.size());
    for(std::size_t i=0;i<rows.size();++i){positions[i]=rows[i].position;cac[i]=computeCanonicalAxisColor(rows[i]);opacity[i]=rows[i].opacity;}
    std::vector<std::uint32_t> membership(rows.size());
    std::size_t nodes=rows.size();
    const bool pooled=rows.size()>config.graphNodeBudget;
    if(!pooled){
        auto partition=runWeightedLeiden(buildSpatialGraph(positions,cac,opacity,config),config);
        if(!partition)throw std::runtime_error(partition.error);
        membership=std::move(partition.membership);
    }else{
        Vec3 lo=positions.front(),hi=lo;
        for(const auto& p:positions){lo.x=std::min(lo.x,p.x);lo.y=std::min(lo.y,p.y);lo.z=std::min(lo.z,p.z);hi.x=std::max(hi.x,p.x);hi.y=std::max(hi.y,p.y);hi.z=std::max(hi.z,p.z);}
        const double extent=std::max({double(hi.x)-lo.x,double(hi.y)-lo.y,double(hi.z)-lo.z,1e-12});
        auto cell=[&](const Vec3& p,std::uint32_t resolution){const double edge=extent/resolution;return Cell{std::int64_t(std::floor((double(p.x)-lo.x)/edge)),std::int64_t(std::floor((double(p.y)-lo.y)/edge)),std::int64_t(std::floor((double(p.z)-lo.z)/edge))};};
        auto count=[&](std::uint32_t resolution){std::unordered_map<Cell,char,CellHash> cells;cells.reserve(config.graphNodeBudget+1);for(const auto& p:positions){cells.emplace(cell(p,resolution),0);if(cells.size()>config.graphNodeBudget)break;}return cells.size();};
        std::uint32_t resolution=std::max(2u,std::uint32_t(std::ceil(std::cbrt(double(config.graphNodeTarget))))),best=1;
        std::size_t bestCount=0;
        for(std::uint32_t probe=0;probe<config.voxelSearchIterations;++probe){
            const auto n=count(resolution);
            if(n<=config.graphNodeBudget && n>bestCount){best=resolution;bestCount=n;}
            if(n>=.75*config.graphNodeTarget && n<=config.graphNodeBudget){best=resolution;break;}
            const auto next=n<.75*config.graphNodeTarget?std::uint32_t(std::ceil(1.5*resolution)):std::max(1u,std::uint32_t(std::floor(.8*resolution)));
            if(next==resolution)break;resolution=next;
        }
        // Deterministic coarse fallback is still voxel pooling, never sampling.
        if(!bestCount){while(best>1 && count(best)>config.graphNodeBudget)--best;if(count(best)>config.graphNodeBudget)throw std::runtime_error("voxel graph budget cannot be met");}
        std::unordered_map<Cell,std::uint32_t,CellHash> ids;ids.reserve(config.graphNodeBudget);
        std::vector<Bucket> buckets;
        for(std::uint32_t i=0;i<positions.size();++i){auto [it,inserted]=ids.emplace(cell(positions[i],best),std::uint32_t(buckets.size()));if(inserted)buckets.emplace_back();membership[i]=it->second;auto& b=buckets[it->second];b.position[0]+=positions[i].x;b.position[1]+=positions[i].y;b.position[2]+=positions[i].z;b.opacity+=opacity[i];for(std::size_t d=0;d<18;++d)b.cac[d]+=cac[i][d];++b.count;}
        if(buckets.size()>config.graphNodeBudget)throw std::runtime_error("adaptive voxel graph exceeds budget");
        nodes=buckets.size();std::vector<Vec3> vp(nodes);std::vector<CacFeature> vc(nodes);std::vector<float> vo(nodes);
        for(std::size_t i=0;i<nodes;++i){const auto& b=buckets[i];vp[i]={float(b.position[0]/b.count),float(b.position[1]/b.count),float(b.position[2]/b.count)};vo[i]=float(b.opacity/b.count);for(std::size_t d=0;d<18;++d)vc[i][d]=float(b.cac[d]/b.count);}
        auto partition=runWeightedLeiden(buildSpatialGraph(vp,vc,vo,config),config);if(!partition)throw std::runtime_error(partition.error);
        for(auto& m:membership)m=partition.membership[m];
    }
    auto out=poolAndBuildSuperpoints(key,rows,cac,membership,config);out.encodingNodeCount=nodes;out.voxelPooled=pooled;return out;
}
} // namespace gaussian_selector
