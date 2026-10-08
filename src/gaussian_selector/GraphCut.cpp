#include "GraphCut.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace gaussian_selector {
namespace {
struct Arc{int to,rev;double capacity;};
class Dinic{
    std::vector<std::vector<Arc>> g;std::vector<int> level,it;
public:
    explicit Dinic(int n):g(n),level(n),it(n){}
    void add(int a,int b,double c){Arc x{b,int(g[b].size()),c},y{a,int(g[a].size()),0};g[a].push_back(x);g[b].push_back(y);}
    void undirected(int a,int b,double c){add(a,b,c);add(b,a,c);}
    bool bfs(int s,int t){std::fill(level.begin(),level.end(),-1);std::queue<int>q;q.push(s);level[s]=0;while(!q.empty()){int v=q.front();q.pop();for(auto&e:g[v])if(e.capacity>1e-12&&level[e.to]<0){level[e.to]=level[v]+1;q.push(e.to);}}return level[t]>=0;}
    double dfs(int v,int t,double f){if(v==t)return f;for(int&i=it[v];i<int(g[v].size());i++){auto&e=g[v][i];if(e.capacity<=1e-12||level[e.to]!=level[v]+1)continue;double p=dfs(e.to,t,std::min(f,e.capacity));if(p>1e-12){e.capacity-=p;g[e.to][e.rev].capacity+=p;return p;}}return 0;}
    double flow(int s,int t){double total=0;while(bfs(s,t)){std::fill(it.begin(),it.end(),0);while(double f=dfs(s,t,std::numeric_limits<double>::infinity()))total+=f;}return total;}
    std::vector<std::uint8_t> sourceSide(int s){std::vector<std::uint8_t>v(g.size());std::queue<int>q;q.push(s);v[s]=1;while(!q.empty()){int a=q.front();q.pop();for(auto&e:g[a])if(e.capacity>1e-12&&!v[e.to]){v[e.to]=1;q.push(e.to);}}return v;}
};
}

double evaluateEnergy(const WeightedGraph& g,std::span<const double> fc,std::span<const double> bc,std::span<const std::uint8_t> f,double lambda){if(fc.size()!=g.nodeCount||bc.size()!=g.nodeCount||f.size()!=g.nodeCount)throw std::invalid_argument("energy size mismatch");double e=0;for(std::size_t i=0;i<f.size();i++)e+=f[i]?fc[i]:bc[i];for(auto&a:g.edges){if(a.a>=g.nodeCount||a.b>=g.nodeCount)throw std::out_of_range("edge endpoint");if(f[a.a]!=f[a.b])e+=lambda*a.weight;}return e;}

CutResult solveGraphCut(const WeightedGraph& g,std::span<const double> fc,std::span<const double> bc,double lambda){if(fc.size()!=g.nodeCount||bc.size()!=g.nodeCount)throw std::invalid_argument("cut size mismatch");if(g.nodeCount>std::uint32_t(std::numeric_limits<int>::max()-2))throw std::length_error("cut graph exceeds solver index range");const int s=static_cast<int>(g.nodeCount),t=s+1;Dinic d(t+1);for(std::uint32_t i=0;i<g.nodeCount;i++){if(fc[i]<0||bc[i]<0||!std::isfinite(fc[i]+bc[i]))throw std::invalid_argument("invalid unary");d.add(s,i,bc[i]);d.add(i,t,fc[i]);}for(auto&e:g.edges){if(e.a>=g.nodeCount||e.b>=g.nodeCount)throw std::out_of_range("edge endpoint");const double c=lambda*e.weight;if(c<0||!std::isfinite(c))throw std::invalid_argument("invalid pairwise");d.undirected(e.a,e.b,c);}CutResult out;out.flow=d.flow(s,t);auto side=d.sourceSide(s);out.foreground.assign(side.begin(),side.begin()+g.nodeCount);out.energy=evaluateEnergy(g,fc,bc,out.foreground,lambda);return out;}

} // namespace gaussian_selector
