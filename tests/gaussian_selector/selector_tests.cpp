#include "AppearanceGMM.h"
#include "CanonicalAxisColor.h"
#include "GraphCut.h"
#include "LeidenAdapter.h"
#include "SeedEvidence.h"
#include "SelectorSelectionCommand.h"
#include "SelectorSession.h"
#include "SpatialGraph.h"
#include "ScreenRoi.h"
#include "CoverageReplay.h"
#include "SeedRefinement.h"
#include "SuperpointEncoding.h"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace gaussian_selector;
namespace {
void require(bool x,const char*m){if(!x)throw std::runtime_error(m);}
bool near(double a,double b,double e=1e-6){return std::abs(a-b)<=e;}

void testScreenRoi() {
    ScreenRoi roi{75,75,25,25};
    require(roi.containsClip(0,0,0,1,100,100),"ROI reversed drag contains center");
    require(roi.containsClip(-.5f,.5f,0,1,100,100),"ROI includes boundary");
    require(!roi.containsClip(.8f,0,0,1,100,100),"ROI rejects outside center");
    require(!roi.containsClip(0,0,0,-1,100,100),"ROI rejects behind camera");
    require(!roi.containsClip(0,0,2,1,100,100),"ROI rejects clipped depth");
    for(auto mode : {ApplyMode::Replace,ApplyMode::Add,ApplyMode::Subtract,ApplyMode::Intersect}) {
        std::vector<std::uint32_t> flags={9,8,9,1}, before=flags;
        std::vector<std::uint8_t> domain={0,1,1,0};
        std::vector<std::uint64_t> preview={2};
        auto delta=makeSelectionDelta({},flags,preview,domain,mode,1);
        applySelectionDelta(flags,delta,true,1);
        require(flags[0]==before[0] && flags[3]==before[3],"ROI outside selection preserved");
        require((flags[1]&8) && (flags[2]&8),"ROI unrelated flags preserved");
        auto after=flags;
        applySelectionDelta(flags,delta,false,1); require(flags==before,"ROI exact undo");
        applySelectionDelta(flags,delta,true,1); require(flags==after,"ROI exact redo");
    }
}

void testCac(){GaussianSnapshotRow r;r.baseColor={.2f,.4f,.6f};r.rotation={1,0,0,0};auto a=computeCanonicalAxisColor(r);for(int i=0;i<6;i++){require(near(a[i*3],.2),"SH0 r");require(near(a[i*3+1],.4),"SH0 g");require(near(a[i*3+2],.6),"SH0 b");}r.shDegree=1;r.shRest.resize(9);r.shRest[6]=.1f;auto p=computeCanonicalAxisColor(r);r.rotation={-1,0,0,0};auto n=computeCanonicalAxisColor(r);require(p==n,"q and -q");require(p[0]!=p[3],"SH1 signed axis");}

void testGraph(){std::vector<Vec3>p={{0,0,0},{0,0,0},{1,0,0}};std::vector<CacFeature>c(3);std::vector<float>o(3,.5f);SelectorConfig cfg;cfg.knn=2;auto g=buildSpatialGraph(p,c,o,cfg);for(auto&e:g.edges){require(e.a<e.b,"unique canonical edge");require(std::isfinite(e.weight),"finite edge");}for(std::size_t i=1;i<g.edges.size();i++)require(!(g.edges[i-1].a==g.edges[i].a&&g.edges[i-1].b==g.edges[i].b),"dedup edge");}

void testLeiden(){WeightedGraph g{6,{{0,1,1},{1,2,1},{0,2,1},{3,4,1},{4,5,1},{3,5,1},{2,3,.001}}};SelectorConfig c;c.leidenResolution=.05;auto r=runWeightedLeiden(g,c);
#ifdef GAUSSIAN_SELECTOR_HAS_IGRAPH
require(bool(r),"igraph Leiden smoke");require(r.membership.size()==6&&r.clusterCount>=2,"Leiden membership");
#else
require(!r&&r.error.find("requires igraph")!=std::string::npos,"explicit missing Leiden dependency");
#endif
}

void testCut(){
    {WeightedGraph g{1,{}};std::vector<double>f={0},b={3};auto r=solveGraphCut(g,f,b,1);require(r.foreground[0],"one-node terminal orientation");}
    {WeightedGraph g{2,{{0,1,.3}}};std::vector<double>f={0,3},b={3,0};auto r=solveGraphCut(g,f,b,1);require(r.foreground[0]&&!r.foreground[1],"terminal orientation");}
    std::mt19937 rng(42);std::uniform_real_distribution<double> cost(0.0,3.0),coin(0.0,1.0);
    for(std::uint32_t n=1;n<=10;n++)for(int trial=0;trial<8;trial++){
        WeightedGraph g;g.nodeCount=n;for(std::uint32_t a=0;a<n;a++)for(std::uint32_t b=a+1;b<n;b++)if(coin(rng)<.3)g.edges.push_back({a,b,cost(rng)});
        std::vector<double>f(n),b(n);for(std::uint32_t i=0;i<n;i++){f[i]=cost(rng);b[i]=cost(rng);}const double lambda=.7;auto r=solveGraphCut(g,f,b,lambda);double best=1e100;
        for(std::uint32_t bits=0;bits<(1u<<n);bits++){std::vector<std::uint8_t>x(n);for(std::uint32_t i=0;i<n;i++)x[i]=(bits>>i)&1u;best=std::min(best,evaluateEnergy(g,f,b,x,lambda));}
        require(near(r.energy,best,1e-8),"random mincut oracle");
    }
}

void testCoverage(){std::vector<std::vector<ReferenceFragment>>pixels={{{0,.8},{1,.5}},{{0,.8}}};std::vector<std::uint8_t>fg={1,0},bg={0,0};auto m=referenceCoverage(2,pixels,fg,bg,0);require(near(m[0].foreground,.8),"front mass");require(near(m[1].foreground,.1),"transmittance mass");require(near(m[0].foreground/m[0].all,.5),"full-frame denominator");}

void testCoverageVisibility(){
    std::vector<std::vector<ReferenceFragment>> pixels={{{0,.8},{1,.5}},{{0,.8},{1,.5}}};
    std::vector<std::uint8_t> fg={1,0},bg={0,1};
    auto standard=referenceCoverage(2,pixels,fg,bg,4.0/255,0);
    auto shallow=referenceCoverage(2,pixels,fg,bg,4.0/255,.5f);
    require(standard[1].all>0 && shallow[1].all==0,"shallow evidence excludes rear contributor");
    require(near(shallow[0].foreground/shallow[0].all,.5),"cutoff denominator includes unpainted pixels");
    require(near(shallow[0].background,.8),"cutoff symmetric FG/BG evidence");
    pixels={{{kInvalidRow,.8},{0,.5}}};fg={1};bg={0};
    auto outside=referenceCoverage(1,pixels,fg,bg,4.0/255,.5f);
    require(outside[0].all==0,"outside-domain occluder stops evidence");
    pixels={{{0,.2,1},{1,.5,3}}};std::vector<float> opaque={2};
    auto depth=referenceCoverage(2,pixels,fg,bg,4.0/255,0,opaque);
    require(depth[0].all>0 && depth[1].all==0,"pre-GS opaque depth rejects rear layer");
    for(auto evidenceStop:{0.0f,.5f,.9f}) {
        CoverageReplay replay;
        replay.advance(replay.contribution(.995f),true,evidenceStop);
        require(replay.stopped,"front opacity terminates first batch");
        for(int batch=1;batch<5;++batch)require(!replay.acceptsDepth(float(batch),100),"termination persists across batches");
    }
    CoverageReplay noDepth;
    noDepth.advance(noDepth.contribution(.8f),false,.5f);
    require(noDepth.stopped,"evidence cutoff independent of depthwrite");
    CoverageReplay rasterOnly;
    rasterOnly.advance(rasterOnly.contribution(.995f),false,0);
    require(!rasterOnly.stopped,"standard replay respects disabled depthwrite");
}

void testConflictRefinement(){
#ifdef GAUSSIAN_SELECTOR_HAS_IGRAPH
    SelectorConfig cfg;cfg.knn=3;cfg.pairwiseLambda=0;
    std::vector<Vec3> positions={{0,0,0},{.01f,0,0},{1,0,0},{1.01f,0,0}};
    std::vector<CacFeature> cac(4);for(std::size_t i=0;i<4;++i)cac[i].fill(i<2?.9f:.1f);
    std::vector<float> opacity(4,.8f);std::vector<std::uint32_t> membership(4,0);
    auto base=std::make_shared<SceneEncoding>(poolAndBuildSuperpoints({1,2,3},positions,cac,opacity,membership,cfg));
    std::vector<SeedLabel> seeds={SeedLabel::Foreground,SeedLabel::Unknown,SeedLabel::Background,SeedLabel::Unknown};
    auto split=splitSeedConflicts(*base,seeds,cfg);
    require(split.conflicts==1 && split.split==1 && !split.unresolved,"mixed superpoint refined");
    const auto& refined=*split.encoding;
    require(refined.gaussianToSuperpoint[0]!=refined.gaussianToSuperpoint[2],"opposing seeds cannot share refined SP");
    require(refined.gaussianToSuperpoint.size()==seeds.size(),"refinement preserves Gaussian row order");
    SelectorSession session(cfg);session.setEncoding(base);
    const auto key=JobKey{base->key,1,0,1};
    auto result=session.solve(key,seeds);
    require(maskGet(result.maskWords,0) && !maskGet(result.maskWords,2),"cut separates FG/BG of formerly merged SP");
    require(result.maskWords==session.solve(key,seeds).maskWords,"cached repeated solve equivalent");
    require(base->superpoints.size()==1 && base->gaussianToSuperpoint==membership,"cached base partition immutable");
    std::swap(seeds[0],seeds[2]);auto revised=session.solve(key,seeds);
    require(!maskGet(revised.maskWords,0) && maskGet(revised.maskWords,2),"new strokes use fresh conflict split");
    cfg.maximumConflictMembers=2;
    auto capped=splitSeedConflicts(*base,seeds,cfg);require(capped.unresolved==1 && !capped.encoding,"local split bounded");
    SelectorSession limited(cfg);limited.setEncoding(base);bool refused=false;
    try{limited.solve(key,seeds);}catch(const std::runtime_error&){refused=true;}
    require(refused,"unresolved mixed labels refuse misleading preview");
#endif
}

void testVotes(){std::vector<std::vector<SeedLabel>>v={{SeedLabel::Foreground},{SeedLabel::Background}};require(majorityVote(v)[0]==SeedLabel::Unknown,"vote tie");}

void testCalibration(){SelectorConfig c;c.gmmComponents=1;std::vector<AppearanceFeature>x(4);for(auto&v:x)v.fill(0);x[0][0]=2;x[1][0]=2.1;x[2][0]=-2;x[3][0]=-2.1;std::vector<SeedLabel>s={SeedLabel::Foreground,SeedLabel::Foreground,SeedLabel::Background,SeedLabel::Background};auto u=buildAppearanceUnaries(x,s,c);double af=(u.calibrated[0]+u.calibrated[1])*.5,ab=(u.calibrated[2]+u.calibrated[3])*.5;auto sigmoid=[](double z){return 1/(1+std::exp(-z));};require(std::abs(sigmoid(2*af)-.95)<.03,"FG calibration");require(std::abs(sigmoid(2*ab)-.05)<.03,"BG calibration");}

void testIntegerCut(){
    std::mt19937 rng(17);std::uniform_real_distribution<double> sample(0,4);
    for(std::uint32_t n=1;n<=9;++n)for(int trial=0;trial<12;++trial){
        WeightedGraph g;g.nodeCount=n;std::vector<double> f(n),b(n);
        for(std::uint32_t i=0;i<n;++i){f[i]=sample(rng);b[i]=sample(rng);for(std::uint32_t j=i+1;j<n;++j)if(sample(rng)>2)g.edges.push_back({i,j,sample(rng)});}
        auto result=solveIntegerGraphCut(g,f,b,.7,10000);const auto scale=result.capacityScale;
        auto q=[](double value,double scale){return std::floor(static_cast<long double>(value)*scale+.5L);};
        auto quantizedEnergy=[&](const auto& mask){long double energy=0;for(std::uint32_t i=0;i<n;++i)energy+=q(mask[i]?f[i]:b[i],scale);for(const auto& e:g.edges)if(mask[e.a]!=mask[e.b])energy+=q(.7*e.weight,scale);return energy;};
        long double best=1e30L;for(std::uint32_t bits=0;bits<(1u<<n);++bits){std::vector<std::uint8_t> mask(n);for(std::uint32_t i=0;i<n;++i)mask[i]=(bits>>i)&1;best=std::min(best,quantizedEnergy(mask));}
        require(quantizedEnergy(result.foreground)==best,"integer push-relabel exact quantized optimum");
    }
    WeightedGraph graph{2,{{0,1,.1}}};std::vector<double> f={0,1e100},b={1e100,0};
    auto extreme=solveIntegerGraphCut(graph,f,b,1);require(extreme.foreground[0]&&!extreme.foreground[1],"dynamic scaling extreme capacities");
    bool rejected=false;f[0]=-1;try{solveIntegerGraphCut(graph,f,b,1);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"integer invalid capacity refused");
}

void testNotesContrast(){
    SelectorConfig cfg;require(cfg.emIterations==80&&near(cfg.evidenceTransmittanceStop,.9f),"author notes defaults");
    std::vector<AppearanceFeature> samples(4);for(std::size_t i=0;i<samples.size();++i)samples[i].fill(double(i));
    for(std::size_t n=1;n<=4;++n){auto model=fitDiagonalGmm(std::span(samples).first(n),cfg);require(model.weights.size()==std::min<std::size_t>(3,std::max<std::size_t>(1,n-1)),"GMM min(3,n-1) with singleton guard");}
    std::vector<AppearanceFeature> features(3);features[0].fill(2);features[1].fill(-2);features[2].fill(1.8);
    std::vector<SeedLabel> seeds={SeedLabel::Foreground,SeedLabel::Background,SeedLabel::Unknown};
    std::vector<SeedConfidence> evidence={{1,0},{0,1},{.8,0}};
    auto unary=buildBlendedAppearanceUnaries(features,seeds,evidence,cfg);
    require(unary.calibrated[0]>0&&unary.calibrated[1]<0,"singleton kNN fallback retains contrast");
    for(auto x:unary.foreground)require(std::isfinite(x),"finite blended GMM/kNN unary");
    std::vector<CoverageMass> masses={{1,.05,0},{1,.3,0},{1,.6,0},{1,0,.5},{1,0,0}};
    auto view=classifyEvidence(masses,cfg);require(view.labels[0]==SeedLabel::Unknown&&view.labels[2]==SeedLabel::Foreground,"adaptive Q10 uses coverage floor");
    std::vector<std::vector<SeedConfidence>> views={view.confidence,view.confidence};require(near(averageConfidence(views)[2].foreground,.6),"multi-view soft confidence averaging");
    masses={{1,.8,0,0,0,0,0}};require(classifyEvidence(masses,cfg).labels[0]==SeedLabel::Unknown,"support gate excludes unsupported scribble evidence");
    masses[0].foregroundSupport=2;cfg.requireCenterEvidence=true;require(classifyEvidence(masses,cfg).labels[0]==SeedLabel::Unknown,"optional center visibility gate");
    masses[0].foregroundCenter=1;require(classifyEvidence(masses,cfg).labels[0]==SeedLabel::Foreground,"visible supported center is lifted");
}

void testAdaptiveEncoding(){
#ifdef GAUSSIAN_SELECTOR_HAS_IGRAPH
    SelectorConfig cfg;cfg.graphNodeBudget=8;cfg.graphNodeTarget=6;
    std::vector<GaussianSnapshotRow> rows(80);
    for(std::size_t i=0;i<rows.size();++i){rows[i].position={float(i%4),float((i/4)%4),float(i/16)};rows[i].baseColor={.2f,.4f,.6f};rows[i].opacity=.8f;rows[i].identity.localIndex=uint32_t(100+i*2);}
    auto encoding=encodeScene({1,2,3},rows,cfg);require(encoding.voxelPooled&&encoding.encodingNodeCount<=8,"adaptive voxel graph obeys budget");
    require(encoding.gaussianToSuperpoint.size()==rows.size(),"voxel broadcast preserves all Gaussian rows");
    std::size_t members=0;for(const auto& sp:encoding.superpoints)members+=sp.memberCount;require(members==rows.size(),"voxel pooling never drops Gaussian members");
    auto repeat=encodeScene({1,2,3},rows,cfg);require(repeat.gaussianToSuperpoint==encoding.gaussianToSuperpoint,"deterministic voxel encoding");
    auto exact=encodeScene({1,2,3},std::span(rows).first(4),cfg);require(!exact.voxelPooled&&exact.encodingNodeCount==4,"below budget exact Gaussian kNN");
    std::vector<double> logits(encoding.superpoints.size(),0);
    auto refined=splitUncertainSuperpoints(encoding,logits,cfg);require(refined.split<=4,"post-cut at most four SP splits");
    if(refined.encoding)require(refined.encoding->gaussianToSuperpoint.size()==rows.size(),"post-cut refinement preserves row mapping");
    std::vector<Vec3> p(20);std::vector<CacFeature> c(20);std::vector<float> o(20,.8f);std::vector<std::uint32_t> groups(20);
    for(std::size_t i=0;i<20;++i){groups[i]=uint32_t(i/4);p[i]={float((i/4)*100+(i%4)/2*10)+float(i%2)*.001f,0,0};}
    cfg.knn=1;auto grouped=poolAndBuildSuperpoints({1,2,3},p,c,o,groups,cfg);
    std::vector<double> lowConfidence(5,0);auto post=splitUncertainSuperpoints(grouped,lowConfidence,cfg);
    require(post.split==4&&post.encoding&&post.encoding->superpoints.size()>=9,"four genuine post-cut SP splits and graph rebuild");
    cfg.maximumUncertainMembers=3;require(!splitUncertainSuperpoints(grouped,lowConfidence,cfg).encoding,"post-cut member budget enforced");
#endif
}

void testNotesPostfilters(){
    SelectorConfig cfg;cfg.refineUncertainSuperpoints=false;cfg.pairwiseLambda=0;
    auto encoding=std::make_shared<SceneEncoding>();encoding->key={1,2,3};encoding->gaussianToSuperpoint={0,1,2,3};encoding->superpoints.resize(4);encoding->standardizedAppearance.resize(4);encoding->gaussianMaxScale={1,1,1,100};
    for(std::size_t i=0;i<4;++i){encoding->standardizedAppearance[i].fill(i==1?-2:2);encoding->superpoints[i].memberCount=1;}
    encoding->superpointGraph={4,{{0,3,.1}}};std::vector<SeedLabel> seeds={SeedLabel::Foreground,SeedLabel::Background,SeedLabel::Unknown,SeedLabel::Unknown};
    SelectorSession session(cfg);session.setEncoding(encoding);auto result=session.solve({encoding->key,1,0,1},seeds);
    require(maskGet(result.maskWords,0)&&!maskGet(result.maskWords,1)&&!maskGet(result.maskWords,2)&&!maskGet(result.maskWords,3),"seeded component and oversized result filters");
    require(result.diagnostics.removedUnseededComponents==1&&result.diagnostics.removedOversizedGaussians==1,"postfilter diagnostics");
    cfg.keepSeededComponents=false;cfg.rejectOversizedGaussians=false;SelectorSession unfiltered(cfg);unfiltered.setEncoding(encoding);
    auto all=unfiltered.solve({encoding->key,1,0,1},seeds);require(maskGet(all.maskWords,2)&&maskGet(all.maskWords,3),"postfilters reversible by config");
    seeds[1]=SeedLabel::Unknown;auto pseudo=unfiltered.solve({encoding->key,2,0,2},seeds);
    require(pseudo.diagnostics.pseudoBackgroundSamples>0&&pseudo.diagnostics.pseudoBackgroundSamples<=32,"weak pseudo-BG floor declared");
}

void testUndo(){constexpr std::uint32_t selected=1,deleted=8;std::vector<std::uint32_t>flags={selected|deleted,0,selected};std::vector<std::uint64_t>preview(1);maskSet(preview,1,true);std::vector<std::uint8_t>domain={1,1,0};auto d=makeSelectionDelta({},flags,preview,domain,ApplyMode::Replace,selected);applySelectionDelta(flags,d,true,selected);require(flags[0]==deleted&&flags[1]==selected&&flags[2]==selected,"replace scope and other bits");applySelectionDelta(flags,d,false,selected);require(flags[0]==(selected|deleted)&&flags[1]==0&&flags[2]==selected,"exact undo");applySelectionDelta(flags,d,true,selected);require(flags[0]==deleted&&flags[1]==selected,"redo");}

void testRestrictedRefinement(){SelectorConfig c;c.gmmComponents=1;c.knn=2;c.refinementMode=RefinementMode::ForegroundRestrictedExperimental;SceneEncoding e;e.key={1,2,3};e.gaussianToSuperpoint={0,1,2,3};e.superpoints.resize(4);e.standardizedAppearance.resize(4);for(std::uint32_t i=0;i<4;i++){e.superpoints[i].position={float(i),0,0};e.superpoints[i].opacity=.5f;e.superpoints[i].memberCount=1;e.standardizedAppearance[i].fill(0);e.standardizedAppearance[i][0]=i<2?2.0:-2.0;}std::vector<Vec3>p(4);std::vector<CacFeature>cac(4);std::vector<float>o(4,.5f);for(std::size_t i=0;i<4;i++)p[i]=e.superpoints[i].position;e.superpointGraph=buildSpatialGraph(p,cac,o,c);SelectorSession s(c);s.setEncoding(e);std::vector<SeedLabel>seeds={SeedLabel::Foreground,SeedLabel::Unknown,SeedLabel::Unknown,SeedLabel::Background};auto r=s.solve({e.key,1,0,1},seeds);require(r.diagnostics.experimentalRefinement,"restricted mode diagnostics");require(r.diagnostics.refinementIterations>=1&&r.diagnostics.refinementIterations<=3,"restricted iteration bound");require(r.diagnostics.message.find("engineering variant")!=std::string::npos,"restricted mode disclosed");}
}

int main(){try{testScreenRoi();testCac();testGraph();testLeiden();testCut();testIntegerCut();testCoverage();testCoverageVisibility();testConflictRefinement();testVotes();testCalibration();testNotesContrast();testAdaptiveEncoding();testNotesPostfilters();testUndo();testRestrictedRefinement();std::cout<<"GaussianSelector core tests passed\n";return 0;}catch(const std::exception&e){std::cerr<<"FAILED: "<<e.what()<<'\n';return 1;}}
