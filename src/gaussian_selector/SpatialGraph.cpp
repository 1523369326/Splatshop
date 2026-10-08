#include "SpatialGraph.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace gaussian_selector {
namespace {

float coord(const Vec3& p,int a){return a==0?p.x:(a==1?p.y:p.z);}
double dist2(const Vec3&a,const Vec3&b){double x=a.x-b.x,y=a.y-b.y,z=a.z-b.z;return x*x+y*y+z*z;}

class KdTree {
    struct Node{std::uint32_t point=0;int axis=0;int left=-1,right=-1;};
    std::span<const Vec3> points_;
    std::vector<Node> nodes_;
    int build(std::vector<std::uint32_t>& ids,std::size_t lo,std::size_t hi,int depth){
        if(lo>=hi)return -1; const int axis=depth%3; const auto mid=(lo+hi)/2;
        std::nth_element(ids.begin()+lo,ids.begin()+mid,ids.begin()+hi,[&](auto a,auto b){return coord(points_[a],axis)<coord(points_[b],axis);});
        const int n=static_cast<int>(nodes_.size()); nodes_.push_back({ids[mid],axis,-1,-1});
        nodes_[n].left=build(ids,lo,mid,depth+1); nodes_[n].right=build(ids,mid+1,hi,depth+1); return n;
    }
    using Item=std::pair<double,std::uint32_t>;
    void search(int n,const Vec3&q,std::uint32_t self,std::size_t k,std::priority_queue<Item>& heap)const{
        if(n<0)return; const auto& node=nodes_[n];
        const double d=dist2(q,points_[node.point]);
        if(node.point!=self&&(heap.size()<k||d<heap.top().first)){heap.push({d,node.point});if(heap.size()>k)heap.pop();}
        const double delta=double(coord(q,node.axis))-coord(points_[node.point],node.axis);
        const int near=delta<0?node.left:node.right,far=delta<0?node.right:node.left;
        search(near,q,self,k,heap);
        if(heap.size()<k||delta*delta<=heap.top().first)search(far,q,self,k,heap);
    }
public:
    explicit KdTree(std::span<const Vec3> p):points_(p){std::vector<std::uint32_t> ids(p.size());std::iota(ids.begin(),ids.end(),0);nodes_.reserve(p.size());build(ids,0,ids.size(),0);}
    std::vector<std::uint32_t> query(std::uint32_t i,std::size_t k)const{std::priority_queue<Item> h;search(nodes_.empty()?-1:0,points_[i],i,k,h);std::vector<std::uint32_t> r;while(!h.empty()){r.push_back(h.top().second);h.pop();}std::reverse(r.begin(),r.end());return r;}
};

double quantile(std::vector<double> v,double q){
    if(v.empty())return 1.0; q=std::clamp(q,0.0,1.0); const auto i=std::size_t(q*double(v.size()-1));std::nth_element(v.begin(),v.begin()+i,v.end());return v[i];
}
double median(std::vector<double> v){return quantile(std::move(v),0.5);}
double cacDistance(const CacFeature&a,const CacFeature&b){double s=0;for(std::size_t i=0;i<a.size();i++){double d=a[i]-b[i];s+=d*d;}return std::sqrt(s);}
struct Candidate{std::uint32_t a,b;double dx,dc,dop,d=0;};
std::uint64_t edgeKey(std::uint32_t a,std::uint32_t b){if(a>b)std::swap(a,b);return(std::uint64_t(a)<<32)|b;}

} // namespace

WeightedGraph buildSpatialGraph(std::span<const Vec3> positions,
    std::span<const CacFeature> cac,std::span<const float> opacity,
    const SelectorConfig& cfg){
    if(positions.size()!=cac.size()||positions.size()!=opacity.size())throw std::invalid_argument("graph attribute sizes differ");
    if(positions.size()>std::numeric_limits<std::uint32_t>::max())throw std::length_error("Gaussian graph exceeds 32-bit index range");
    WeightedGraph out{static_cast<std::uint32_t>(positions.size()),{}}; if(positions.size()<2)return out;
    const std::size_t k=std::min<std::size_t>(cfg.knn,positions.size()-1); KdTree tree(positions);
    std::unordered_map<std::uint64_t,Candidate> unique; unique.reserve(positions.size()*k);
    std::vector<double> xs,cs,os;xs.reserve(positions.size()*k);cs.reserve(xs.capacity());os.reserve(xs.capacity());
    for(std::uint32_t i=0;i<positions.size();i++)for(auto j:tree.query(i,k)){
        auto a=std::min(i,j),b=std::max(i,j);auto key=edgeKey(a,b);if(unique.contains(key))continue;
        Candidate c{a,b,std::sqrt(dist2(positions[a],positions[b])),cacDistance(cac[a],cac[b]),std::abs(double(opacity[a])-opacity[b])};
        xs.push_back(c.dx);cs.push_back(c.dc);os.push_back(c.dop);unique.emplace(key,c);
    }
    const double nx=std::max(quantile(xs,0.95),1e-12),nc=std::max(quantile(cs,0.95),1e-12),no=std::max(quantile(os,0.95),1e-12);
    const double gx=quantile(xs,cfg.gateQuantile)/nx,gc=quantile(cs,cfg.gateQuantile)/nc,go=quantile(os,cfg.gateQuantile)/no;
    std::vector<Candidate> kept;kept.reserve(unique.size());
    std::vector<std::vector<double>> local(positions.size());
    for(auto&[_,c]:unique){c.dx/=nx;c.dc/=nc;c.dop/=no;if(c.dx>gx||c.dc>gc||c.dop>go)continue;c.d=cfg.spatialWeight*c.dx+cfg.colorWeight*c.dc+cfg.opacityWeight*c.dop;kept.push_back(c);local[c.a].push_back(c.d);local[c.b].push_back(c.d);}
    std::vector<double> gamma(positions.size(),1e-12);for(std::size_t i=0;i<local.size();i++)gamma[i]=std::max(median(local[i]),1e-12);
    out.edges.reserve(kept.size());for(const auto&c:kept){const double denom=std::max(gamma[c.a]*gamma[c.b],1e-24);const double w=c.d==0?1.0:std::exp(-(c.d*c.d)/denom);out.edges.push_back({c.a,c.b,std::isfinite(w)?w:0.0});}
    std::sort(out.edges.begin(),out.edges.end(),[](auto&a,auto&b){return a.a==b.a?a.b<b.b:a.a<b.a;});return out;
}

} // namespace gaussian_selector
