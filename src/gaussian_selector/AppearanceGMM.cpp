#include "AppearanceGMM.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace gaussian_selector {
namespace {
constexpr double pi=3.14159265358979323846;
double logsum(std::span<const double> v){double m=*std::max_element(v.begin(),v.end());double s=0;for(double x:v)s+=std::exp(x-m);return m+std::log(s);}
double softplus(double x){return std::max(x,0.0)+std::log1p(std::exp(-std::abs(x)));}
double median(std::vector<double> v){auto i=v.begin()+v.size()/2;std::nth_element(v.begin(),i,v.end());double m=*i;if(v.size()%2==0){auto j=std::max_element(v.begin(),i);m=(m+*j)*.5;}return m;}
}

double DiagonalGmm::logLikelihood(const AppearanceFeature& x)const{
    std::vector<double> terms(weights.size());for(std::size_t c=0;c<weights.size();c++){double q=std::log(std::max(weights[c],1e-300));for(std::size_t d=0;d<x.size();d++){const double v=variances[c][d],z=x[d]-means[c][d];q-=.5*(std::log(2*pi*v)+z*z/v);}terms[c]=q;}return logsum(terms);
}

DiagonalGmm fitDiagonalGmm(std::span<const AppearanceFeature> x,const SelectorConfig& cfg){
    if(x.empty()||!cfg.gmmComponents)throw std::invalid_argument("cannot fit empty GMM");const std::size_t k=std::min<std::size_t>(cfg.gmmComponents,std::max<std::size_t>(1,x.size()-1));DiagonalGmm g;g.degraded=x.size()<=1||k<cfg.gmmComponents;g.weights.assign(k,1.0/k);g.means.resize(k);g.variances.resize(k);
    for(std::size_t c=0;c<k;c++)g.means[c]=x[(c*x.size())/k];AppearanceFeature global{};for(auto&s:x)for(std::size_t d=0;d<global.size();d++)global[d]+=s[d];for(auto&v:global)v/=x.size();AppearanceFeature var{};for(auto&s:x)for(std::size_t d=0;d<var.size();d++){double z=s[d]-global[d];var[d]+=z*z;}for(auto&v:var)v=std::max(v/x.size(),cfg.varianceFloor);for(auto&v:g.variances)v=var;
    std::vector<double> resp(x.size()*k),logs(k);double previous=-std::numeric_limits<double>::infinity();
    for(std::uint32_t it=0;it<cfg.emIterations;it++){double ll=0;for(std::size_t i=0;i<x.size();i++){for(std::size_t c=0;c<k;c++){double q=std::log(std::max(g.weights[c],1e-300));for(std::size_t d=0;d<19;d++){double z=x[i][d]-g.means[c][d],v=g.variances[c][d];q-=.5*(std::log(2*pi*v)+z*z/v);}logs[c]=q;}double z=logsum(logs);ll+=z;for(std::size_t c=0;c<k;c++)resp[i*k+c]=std::exp(logs[c]-z);}
        if(it&&std::abs(ll-previous)<=cfg.emRelativeTolerance*(1+std::abs(previous)))break;previous=ll;
        for(std::size_t c=0;c<k;c++){double n=0;for(std::size_t i=0;i<x.size();i++)n+=resp[i*k+c];if(n<1e-9){g.degraded=true;g.weights[c]=1e-9;g.means[c]=x[c%x.size()];g.variances[c]=var;continue;}g.weights[c]=n/x.size();for(std::size_t d=0;d<19;d++){double m=0;for(std::size_t i=0;i<x.size();i++)m+=resp[i*k+c]*x[i][d];g.means[c][d]=m/n;double v=0;for(std::size_t i=0;i<x.size();i++){double q=x[i][d]-g.means[c][d];v+=resp[i*k+c]*q*q;}g.variances[c][d]=std::max(v/n,cfg.varianceFloor);}}
        double sum=std::accumulate(g.weights.begin(),g.weights.end(),0.0);for(auto&w:g.weights)w/=sum;
    }return g;
}

UnaryCosts buildAppearanceUnariesWithReservoir(std::span<const AppearanceFeature> features,std::span<const SeedLabel> seeds,std::span<const AppearanceFeature> fs,std::span<const AppearanceFeature> bs,const SelectorConfig& cfg){
    if(features.size()!=seeds.size())throw std::invalid_argument("feature seed mismatch");if(fs.empty()||bs.empty())throw std::invalid_argument("both FG and BG evidence reservoirs are required");
    auto fg=fitDiagonalGmm(fs,cfg),bg=fitDiagonalGmm(bs,cfg);std::vector<double> delta(features.size()),df,db;for(std::size_t i=0;i<features.size();i++)delta[i]=fg.logLikelihood(features[i])-bg.logLikelihood(features[i]);for(const auto& x:fs)df.push_back(fg.logLikelihood(x)-bg.logLikelihood(x));for(const auto& x:bs)db.push_back(fg.logLikelihood(x)-bg.logLikelihood(x));
    const double mf=median(df),mb=median(db),gap=mf-mb;UnaryCosts out;out.degraded=fg.degraded||bg.degraded||!std::isfinite(gap)||gap<=1e-9;out.foreground.resize(features.size());out.background.resize(features.size());out.calibrated.resize(features.size());const double midpoint=(mf+mb)*.5;const double scale=out.degraded?0.0:std::log(cfg.targetSeedConfidence/(1-cfg.targetSeedConfidence))/gap;
    for(std::size_t i=0;i<features.size();i++){double a=scale*(delta[i]-midpoint);out.calibrated[i]=a;out.foreground[i]=softplus(-2*a)+(seeds[i]==SeedLabel::Background?cfg.seedPenalty:0);out.background[i]=softplus(2*a)+(seeds[i]==SeedLabel::Foreground?cfg.seedPenalty:0);}return out;
}

UnaryCosts buildAppearanceUnaries(std::span<const AppearanceFeature> features,std::span<const SeedLabel> seeds,const SelectorConfig& cfg){
    if(features.size()!=seeds.size())throw std::invalid_argument("feature seed mismatch");std::vector<AppearanceFeature> fs,bs;for(std::size_t i=0;i<features.size();i++){if(seeds[i]==SeedLabel::Foreground)fs.push_back(features[i]);else if(seeds[i]==SeedLabel::Background)bs.push_back(features[i]);}
    return buildAppearanceUnariesWithReservoir(features,seeds,fs,bs,cfg);
}

UnaryCosts buildBlendedAppearanceUnaries(std::span<const AppearanceFeature> features,
    std::span<const SeedLabel> seeds,std::span<const SeedConfidence> confidence,const SelectorConfig& cfg){
    if(features.size()!=seeds.size()||features.size()!=confidence.size())throw std::invalid_argument("contrast evidence size mismatch");
    if(!cfg.useSoftKnnContrast)return buildAppearanceUnaries(features,seeds,cfg);
    std::vector<AppearanceFeature> fs,bs;
    std::vector<std::pair<std::size_t,double>> sf,sb;
    for(std::size_t i=0;i<features.size();++i){
        if(seeds[i]==SeedLabel::Foreground)fs.push_back(features[i]);
        if(seeds[i]==SeedLabel::Background)bs.push_back(features[i]);
        // Human hard seeds provide a reliable floor to each nonparametric model.
        const double f=std::max(confidence[i].foreground,seeds[i]==SeedLabel::Foreground?1.0:0.0);
        const double b=std::max(confidence[i].background,seeds[i]==SeedLabel::Background?1.0:0.0);
        if(f>0)sf.emplace_back(i,f);if(b>0)sb.emplace_back(i,b);
    }
    if(fs.empty()||sb.empty())throw std::invalid_argument("FG and BG model evidence required");
    auto fg=fitDiagonalGmm(fs,cfg);
    auto bg=bs.empty()?DiagonalGmm{}:fitDiagonalGmm(bs,cfg);
    auto logDensity=[&](const AppearanceFeature& x,const auto& samples){
        std::vector<std::pair<double,double>> neighbors;neighbors.reserve(samples.size());double total=0;
        for(const auto& [i,weight]:samples){double distance=0;for(std::size_t d=0;d<x.size();++d){const double z=x[d]-features[i][d];distance+=z*z;}neighbors.emplace_back(distance,weight);total+=weight;}
        const auto k=std::min<std::size_t>(std::max(1u,cfg.softKnn),neighbors.size());
        std::nth_element(neighbors.begin(),neighbors.begin()+k-1,neighbors.end());
        double support=0,radius=0;for(std::size_t i=0;i<k;++i){support+=neighbors[i].second;radius=std::max(radius,neighbors[i].first);}
        // Confidence-weighted kNN ball density in standardized 19D. Kernel,
        // radius floor and mixing weight are E, not published author defaults.
        return std::log(std::max(support/total,1e-300))-.5*x.size()*std::log(std::max(radius,cfg.varianceFloor));
    };
    std::vector<double> delta(features.size()),df,db;
    // Singleton GMMs cannot estimate covariance: rely on the kNN estimator.
    const double mix=(fs.size()<2||bs.size()<2)?0.0:std::clamp(cfg.unaryGmmMix,0.0,1.0);
    for(std::size_t i=0;i<features.size();++i){
        const double gd=mix>0?fg.logLikelihood(features[i])-bg.logLikelihood(features[i]):0;
        const double kd=logDensity(features[i],sf)-logDensity(features[i],sb);
        delta[i]=mix*gd+(1-mix)*kd;
        if(seeds[i]==SeedLabel::Foreground)df.push_back(delta[i]);if(seeds[i]==SeedLabel::Background)db.push_back(delta[i]);
    }
    if(db.empty())for(const auto& [i,_]:sb)db.push_back(delta[i]);
    const double mf=median(df),mb=median(db),gap=mf-mb,midpoint=(mf+mb)*.5;
    const bool bad=!std::isfinite(gap)||gap<=1e-9;
    UnaryCosts out;out.degraded=fg.degraded||bg.degraded||bad;out.foreground.resize(features.size());out.background.resize(features.size());out.calibrated.resize(features.size());
    const double scale=bad?0:std::log(cfg.targetSeedConfidence/(1-cfg.targetSeedConfidence))/gap;
    for(std::size_t i=0;i<features.size();++i){const double a=scale*(delta[i]-midpoint);out.calibrated[i]=a;out.foreground[i]=softplus(-2*a)+(seeds[i]==SeedLabel::Background?cfg.seedPenalty:0);out.background[i]=softplus(2*a)+(seeds[i]==SeedLabel::Foreground?cfg.seedPenalty:0);}
    return out;
}

} // namespace gaussian_selector
