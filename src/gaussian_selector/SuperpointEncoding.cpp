#include "SuperpointEncoding.h"
#include "SpatialGraph.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gaussian_selector {

SceneEncoding poolAndBuildSuperpoints(SnapshotKey key,std::span<const GaussianSnapshotRow> rows,
    std::span<const CacFeature> cac,std::span<const std::uint32_t> membership,const SelectorConfig& config){
    if(rows.size()!=cac.size()||rows.size()!=membership.size())throw std::invalid_argument("membership size mismatch");
    std::vector<Vec3> positions(rows.size());
    std::vector<float> opacity(rows.size());
    for(std::size_t i=0;i<rows.size();++i){positions[i]=rows[i].position;opacity[i]=rows[i].opacity;}
    auto out=poolAndBuildSuperpoints(key,positions,cac,opacity,membership,config);
    out.gaussianMaxScale.resize(rows.size());
    for(std::size_t i=0;i<rows.size();++i)out.gaussianMaxScale[i]=std::max({rows[i].physicalScale.x,rows[i].physicalScale.y,rows[i].physicalScale.z});
    return out;
}

SceneEncoding poolAndBuildSuperpoints(SnapshotKey key,std::span<const Vec3> positions,
    std::span<const CacFeature> cac,std::span<const float> opacity,
    std::span<const std::uint32_t> membership,const SelectorConfig& config){
    if(positions.size()!=cac.size()||positions.size()!=opacity.size()||positions.size()!=membership.size())throw std::invalid_argument("membership size mismatch");
    SceneEncoding out;out.key=key;out.gaussianCac.assign(cac.begin(),cac.end());out.gaussianToSuperpoint.assign(membership.begin(),membership.end());
    out.gaussianPositions.assign(positions.begin(),positions.end());out.gaussianOpacity.assign(opacity.begin(),opacity.end());
    std::uint32_t k=0;for(auto m:membership)k=std::max(k,m+1);out.superpoints.resize(k);
    for(std::size_t i=0;i<positions.size();i++){auto&s=out.superpoints[membership[i]];s.position.x+=positions[i].x;s.position.y+=positions[i].y;s.position.z+=positions[i].z;s.opacity+=opacity[i];for(std::size_t d=0;d<kCacDimensions;d++)s.cac[d]+=cac[i][d];s.memberCount++;}
    for(auto&s:out.superpoints){if(!s.memberCount)continue;const float r=1.0f/s.memberCount;s.position.x*=r;s.position.y*=r;s.position.z*=r;s.opacity*=r;for(auto&v:s.cac)v*=r;}
    std::vector<Vec3> p(k);std::vector<CacFeature> c(k);std::vector<float> o(k);for(std::size_t i=0;i<k;i++){p[i]=out.superpoints[i].position;c[i]=out.superpoints[i].cac;o[i]=out.superpoints[i].opacity;}
    out.superpointGraph=buildSpatialGraph(p,c,o,config);
    out.standardizedAppearance.resize(k);if(k==0)return out;
    AppearanceFeature mean{},sd{};for(std::size_t i=0;i<k;i++){for(std::size_t d=0;d<18;d++)mean[d]+=c[i][d];mean[18]+=o[i];}for(auto&v:mean)v/=double(k);
    for(std::size_t i=0;i<k;i++){for(std::size_t d=0;d<18;d++){double x=c[i][d]-mean[d];sd[d]+=x*x;}double x=o[i]-mean[18];sd[18]+=x*x;}
    for(auto&v:sd)v=std::max(std::sqrt(v/double(k)),config.standardDeviationFloor);
    for(std::size_t i=0;i<k;i++){for(std::size_t d=0;d<18;d++)out.standardizedAppearance[i][d]=(c[i][d]-mean[d])/sd[d];out.standardizedAppearance[i][18]=(o[i]-mean[18])/sd[18];}
    return out;
}

} // namespace gaussian_selector
