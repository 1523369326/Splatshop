#include "SeedEvidence.h"
#include "CoverageReplay.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gaussian_selector {

std::vector<CoverageMass> referenceCoverage(std::size_t n,std::span<const std::vector<ReferenceFragment>> pixels,
    std::span<const std::uint8_t> fg,std::span<const std::uint8_t> bg,double stop,
    float evidenceStop,std::span<const float> opaqueDepth){
    if(pixels.size()!=fg.size()||pixels.size()!=bg.size())throw std::invalid_argument("coverage mask size mismatch");
    if(!opaqueDepth.empty()&&opaqueDepth.size()!=pixels.size())throw std::invalid_argument("coverage depth size mismatch");
    std::vector<CoverageMass> out(n);
    for(std::size_t p=0;p<pixels.size();p++){
        CoverageReplay replay;
        const float depth=opaqueDepth.empty()?std::numeric_limits<float>::infinity():opaqueDepth[p];
        for(const auto&f:pixels[p]){
            if(!replay.acceptsDepth(f.depth,depth))break;
            if(f.gaussian!=kInvalidRow && f.gaussian>=n)throw std::out_of_range("fragment identity");
            const float a=replay.contribution(float(std::clamp(f.intrinsicAlpha,0.0,1.0)));
            if(f.gaussian!=kInvalidRow){out[f.gaussian].all+=a;if(fg[p])out[f.gaussian].foreground+=a;if(bg[p])out[f.gaussian].background+=a;}
            replay.advance(a,f.writesDepth,evidenceStop,float(stop));
            if(replay.stopped)break;
        }
    }
    return out;
}

std::vector<SeedLabel> classifyCoverage(std::span<const CoverageMass> mass,const SelectorConfig& c){
    return classifyEvidence(mass,c).labels;
}

ViewEvidence classifyEvidence(std::span<const CoverageMass> mass,const SelectorConfig& c){
    ViewEvidence out;out.labels.assign(mass.size(),SeedLabel::Unknown);out.confidence.resize(mass.size());
    std::vector<double> positives,negatives;
    for(std::size_t i=0;i<mass.size();++i){
        const auto& m=mass[i];if(!std::isfinite(m.all+m.foreground+m.background)||m.all<c.minimumVisibleMass||m.all<=0)continue;
        auto& v=out.confidence[i];v.foreground=std::clamp(m.foreground/m.all,0.0,1.0);v.background=std::clamp(m.background/m.all,0.0,1.0);
        if((m.foregroundSupport>=0&&m.foregroundSupport<c.minimumScribbleSupport)||(c.requireCenterEvidence&&m.foregroundCenter==0))v.foreground=0;
        if((m.backgroundSupport>=0&&m.backgroundSupport<c.minimumScribbleSupport)||(c.requireCenterEvidence&&m.backgroundCenter==0))v.background=0;
        if(v.foreground>0)positives.push_back(v.foreground);if(v.background>0)negatives.push_back(v.background);
    }
    auto threshold=[&](std::vector<double>& v){if(!c.adaptiveCoverageThreshold||v.empty())return c.coverageThreshold;const auto i=std::size_t(std::clamp(c.coverageQuantile,0.0,1.0)*(v.size()-1));std::nth_element(v.begin(),v.begin()+i,v.end());return std::max(c.coverageThreshold,v[i]);};
    const double ft=threshold(positives),bt=threshold(negatives);
    for(std::size_t i=0;i<mass.size();++i){const auto v=out.confidence[i];const bool f=v.foreground>0 && v.foreground>=ft,b=v.background>0 && v.background>=bt;if(f!=b)out.labels[i]=f?SeedLabel::Foreground:SeedLabel::Background;}
    return out;
}

std::vector<SeedConfidence> averageConfidence(std::span<const std::vector<SeedConfidence>> views){
    if(views.empty())return{};std::vector<SeedConfidence> out(views.front().size());std::vector<std::size_t> count(out.size());
    for(const auto& view:views){if(view.size()!=out.size())throw std::invalid_argument("confidence view size mismatch");for(std::size_t i=0;i<out.size();++i)if(view[i].foreground>0||view[i].background>0){out[i].foreground+=view[i].foreground;out[i].background+=view[i].background;++count[i];}}
    for(std::size_t i=0;i<out.size();++i)if(count[i]){out[i].foreground/=count[i];out[i].background/=count[i];}return out;
}

std::vector<SeedLabel> majorityVote(std::span<const std::vector<SeedLabel>> views){
    if(views.empty())return{};const auto n=views.front().size();std::vector<SeedLabel> out(n,SeedLabel::Unknown);for(auto&v:views)if(v.size()!=n)throw std::invalid_argument("vote size mismatch");for(std::size_t i=0;i<n;i++){int f=0,b=0;for(auto&v:views){f+=v[i]==SeedLabel::Foreground;b+=v[i]==SeedLabel::Background;}if(f>b)out[i]=SeedLabel::Foreground;else if(b>f)out[i]=SeedLabel::Background;}return out;
}

std::vector<SeedLabel> broadcastSeeds(std::span<const SeedLabel> seeds,std::span<const std::uint32_t> map,std::size_t k,std::size_t* conflicts){
    if(seeds.size()!=map.size())throw std::invalid_argument("seed map size mismatch");std::vector<std::size_t> f(k),b(k);for(std::size_t i=0;i<seeds.size();i++){if(map[i]>=k)throw std::out_of_range("superpoint id");f[map[i]]+=seeds[i]==SeedLabel::Foreground;b[map[i]]+=seeds[i]==SeedLabel::Background;}
    std::vector<SeedLabel> out(k,SeedLabel::Unknown);std::size_t c=0;for(std::size_t i=0;i<k;i++){if(f[i]&&b[i])c++;if(f[i]>b[i])out[i]=SeedLabel::Foreground;else if(b[i]>f[i])out[i]=SeedLabel::Background;}if(conflicts)*conflicts=c;return out;
}

} // namespace gaussian_selector
