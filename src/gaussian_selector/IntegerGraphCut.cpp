#include "GraphCut.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>

namespace gaussian_selector {
namespace {
struct Arc {int to,reverse;std::int64_t capacity;};
class PushRelabel {
    std::vector<std::vector<Arc>> graph_;
public:
    explicit PushRelabel(int n):graph_(n){}
    void add(int a,int b,std::int64_t capacity){Arc f{b,int(graph_[b].size()),capacity},r{a,int(graph_[a].size()),0};graph_[a].push_back(f);graph_[b].push_back(r);}
    std::int64_t solve(int source,int sink){
        const int n=int(graph_.size());std::vector<int> height(n),current(n);std::vector<std::int64_t> excess(n);std::vector<bool> queued(n);std::queue<int> active;
        auto enqueue=[&](int v){if(v!=source&&v!=sink&&excess[v]>0&&!queued[v]){active.push(v);queued[v]=true;}};
        height[source]=n;
        for(auto& edge:graph_[source]){const auto amount=edge.capacity;if(!amount)continue;edge.capacity=0;graph_[edge.to][edge.reverse].capacity+=amount;excess[edge.to]+=amount;excess[source]-=amount;enqueue(edge.to);}
        while(!active.empty()){
            const int v=active.front();active.pop();queued[v]=false;
            while(excess[v]>0){
                if(current[v]==int(graph_[v].size())){
                    int h=std::numeric_limits<int>::max();for(const auto& e:graph_[v])if(e.capacity>0)h=std::min(h,height[e.to]);
                    if(h==std::numeric_limits<int>::max())throw std::runtime_error("integer solver has stranded excess");
                    height[v]=h+1;current[v]=0;continue;
                }
                auto& e=graph_[v][current[v]];
                if(e.capacity>0 && height[v]==height[e.to]+1){const auto amount=std::min(excess[v],e.capacity);e.capacity-=amount;graph_[e.to][e.reverse].capacity+=amount;excess[v]-=amount;excess[e.to]+=amount;enqueue(e.to);}else ++current[v];
            }
        }
        return excess[sink];
    }
    std::vector<std::uint8_t> sourceSide(int source)const{std::vector<std::uint8_t> seen(graph_.size());std::queue<int> queue;seen[source]=1;queue.push(source);while(!queue.empty()){const int v=queue.front();queue.pop();for(const auto& e:graph_[v])if(e.capacity>0&&!seen[e.to]){seen[e.to]=1;queue.push(e.to);}}return seen;}
};
}

CutResult solveIntegerGraphCut(const WeightedGraph& graph,std::span<const double> fg,
    std::span<const double> bg,double lambda,std::uint32_t precision){
    if(fg.size()!=graph.nodeCount||bg.size()!=graph.nodeCount)throw std::invalid_argument("integer cut size mismatch");
    if(!precision||!std::isfinite(lambda)||lambda<0)throw std::invalid_argument("invalid integer cut settings");
    if(graph.nodeCount>std::uint32_t(std::numeric_limits<int>::max()/2-2))throw std::length_error("integer cut exceeds solver index range");
    double maximum=0;
    auto inspect=[&](double value){if(!std::isfinite(value)||value<0)throw std::invalid_argument("invalid cut capacity");maximum=std::max(maximum,value);};
    for(std::size_t i=0;i<fg.size();++i){inspect(fg[i]);inspect(bg[i]);}
    for(const auto& e:graph.edges){if(e.a>=graph.nodeCount||e.b>=graph.nodeCount||e.a==e.b)throw std::out_of_range("integer cut edge endpoint");inspect(lambda*e.weight);}
    const long double budget=std::numeric_limits<std::int64_t>::max()/8;
    const long double count=2.0L*(graph.nodeCount+static_cast<long double>(graph.edges.size())+1);
    const double scale=maximum>0?double(std::min<long double>(precision,budget/count)/maximum):1;
    if(!std::isfinite(scale)||scale<=0)throw std::overflow_error("integer capacity scaling is not representable");
    std::int64_t total=0;
    auto quantize=[&](double value){const long double scaled=static_cast<long double>(value)*scale;if(scaled>budget||!std::isfinite(scaled))throw std::overflow_error("integer capacity overflow");const auto capacity=std::int64_t(std::floor(scaled+.5L));if(capacity>budget-total)throw std::overflow_error("total integer capacity overflow");total+=capacity;return capacity;};
    const int source=int(graph.nodeCount),sink=source+1;PushRelabel solver(sink+1);
    for(std::uint32_t i=0;i<graph.nodeCount;++i){solver.add(source,i,quantize(bg[i]));solver.add(i,sink,quantize(fg[i]));}
    for(const auto& edge:graph.edges){const auto capacity=quantize(lambda*edge.weight);solver.add(edge.a,edge.b,capacity);solver.add(edge.b,edge.a,quantize(lambda*edge.weight));}
    CutResult result;result.capacityScale=scale;result.flow=double(solver.solve(source,sink))/scale;auto side=solver.sourceSide(source);result.foreground.assign(side.begin(),side.begin()+graph.nodeCount);result.energy=evaluateEnergy(graph,fg,bg,result.foreground,lambda);return result;
}
} // namespace gaussian_selector
