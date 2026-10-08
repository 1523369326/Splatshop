#include "LeidenAdapter.h"

#include <mutex>

#ifdef GAUSSIAN_SELECTOR_HAS_IGRAPH
#include <igraph.h>
#endif

namespace gaussian_selector {

LeidenResult runWeightedLeiden(const WeightedGraph& input,const SelectorConfig& config){
#ifndef GAUSSIAN_SELECTOR_HAS_IGRAPH
    (void)input;(void)config;
    return {{},0,0,"GaussianSelector encoding requires igraph 1.0.x; no fallback clustering is permitted"};
#else
    static std::mutex igraphMutex;
    static std::once_flag igraphSetup;
    std::scoped_lock lock(igraphMutex); // default RNG and error handler are global
    std::call_once(igraphSetup, [] {
        igraph_setup();
        // Production code must receive an error code instead of allowing the
        // C library's default handler to abort the editor process.
        igraph_set_error_handler(igraph_error_handler_ignore);
    });
    LeidenResult out;
    if(input.nodeCount==0)return out;
    if(input.edges.empty()){
        out.membership.resize(input.nodeCount);for(std::uint32_t i=0;i<input.nodeCount;i++)out.membership[i]=i;out.clusterCount=input.nodeCount;return out;
    }
    igraph_rng_seed(igraph_rng_default(),static_cast<igraph_uint_t>(config.randomSeed));
    igraph_vector_int_t endpoints;igraph_vector_t weights;igraph_vector_int_t membership;
    igraph_vector_int_init(&endpoints,static_cast<igraph_int_t>(input.edges.size()*2));
    igraph_vector_init(&weights,static_cast<igraph_int_t>(input.edges.size()));
    igraph_vector_int_init(&membership,0);
    for(std::size_t i=0;i<input.edges.size();i++){
        VECTOR(endpoints)[2*i]=input.edges[i].a;VECTOR(endpoints)[2*i+1]=input.edges[i].b;VECTOR(weights)[i]=input.edges[i].weight;
    }
    igraph_t graph;igraph_error_t e=igraph_create(&graph,&endpoints,input.nodeCount,IGRAPH_UNDIRECTED);const bool graphInitialized=e==IGRAPH_SUCCESS;
    igraph_int_t clusters=0;igraph_real_t quality=0;
    if(e==IGRAPH_SUCCESS)e=igraph_community_leiden_simple(&graph,&weights,IGRAPH_LEIDEN_OBJECTIVE_CPM,
        config.leidenResolution,config.leidenBeta,false,config.leidenIterations,&membership,&clusters,&quality);
    if(e==IGRAPH_SUCCESS){out.membership.resize(input.nodeCount);for(std::uint32_t i=0;i<input.nodeCount;i++)out.membership[i]=static_cast<std::uint32_t>(VECTOR(membership)[i]);out.clusterCount=static_cast<std::uint32_t>(clusters);out.quality=quality;}
    else out.error=std::string("igraph Leiden failed: ")+igraph_strerror(e);
    if(graphInitialized)igraph_destroy(&graph);
    igraph_vector_int_destroy(&membership);igraph_vector_destroy(&weights);igraph_vector_int_destroy(&endpoints);return out;
#endif
}

} // namespace gaussian_selector
