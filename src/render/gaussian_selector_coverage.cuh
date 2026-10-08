#pragma once

#include "../gaussian_selector/CoverageReplay.h"

// On-demand replay kernel for GaussianSelector Eq. (6).  It consumes the
// exact standard-path staged/tile arrays plus an optional stage-index identity
// sidecar.  Callers must provide the opaque depth image captured immediately
// before the Gaussian pass, never the final framebuffer depth.
extern "C" __global__
void kernel_gaussian_selector_capture_depth(RenderTarget target, float* opaqueDepth)
{
    const uint32_t pixel = cg::this_grid().thread_rank();
    const uint32_t count = uint32_t(target.width) * uint32_t(target.height);
    if(pixel >= count) return;
    opaqueDepth[pixel] = __int_as_float(target.framebuffer[pixel] >> 32);
}

extern "C" __global__
void kernel_gaussian_selector_coverage(
    CommonLaunchArgs args, RenderTarget target, Tile* tiles, const uint32_t* tileIds,
    uint32_t* indices, StageData* stagedatas, uint32_t numFragments,
    const uint32_t* stageSnapshotRows, const uint8_t* foregroundMask,
    const uint8_t* backgroundMask, const float* initialOpaqueDepth,
    float* allMass, float* foregroundMass, float* backgroundMass,
    float evidenceTransmittanceStop, uint32_t maximumOwners,
    float* foregroundSupport, float* backgroundSupport,
    float* foregroundCenter, float* backgroundCenter, float centerDepthTolerance)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    const int tileID = grid.block_rank();
    const int tilePixelIndex = threadIdx.x;
    const int tilesX = (target.width + TILE_SIZE_3DGS - 1) / TILE_SIZE_3DGS;
    const int px = (tileID % tilesX) * TILE_SIZE_3DGS + tilePixelIndex % TILE_SIZE_3DGS;
    const int py = (tileID / tilesX) * TILE_SIZE_3DGS + tilePixelIndex / TILE_SIZE_3DGS;
    const bool inside = px < target.width && py < target.height;
    const int pixelID = px + py * target.width;
    const Tile tile = tiles[tileID];
    // Empty Tile entries are zero-filled by the host and are otherwise
    // indistinguishable from the real range [0,0]. Validate the sorted key at
    // both ends before touching indices[0].
    const bool validTile = tile.firstIndex <= tile.lastIndex &&
        tile.lastIndex < numFragments && tileIds[tile.firstIndex] == uint32_t(tileID) &&
        tileIds[tile.lastIndex] == uint32_t(tileID);
    const int count = validTile
        ? int(min(tile.lastIndex - tile.firstIndex + 1, 1'000'000u)) : 0;

    __shared__ RasterizationData staged[PREFETCH_COUNT];
    __shared__ uint32_t rows[PREFETCH_COUNT];
    gaussian_selector::CoverageReplay replay;
    uint32_t owners = 0;
    float frontContributorDepth = 0;
    const float opaqueDepth = inside ? initialOpaqueDepth[pixelID] : 0.0f;
    const int iterations = (count + PREFETCH_COUNT - 1) / PREFETCH_COUNT;
    for(int batch=0; batch<iterations; ++batch){
        const int q=PREFETCH_COUNT*batch+tilePixelIndex;
        if(q<count && tilePixelIndex<PREFETCH_COUNT){
            const uint32_t stageIndex=indices[tile.firstIndex+q];
            const StageData sd=stagedatas[stageIndex];
            vec2 b1,b2;float depth;decode_stagedata(sd,b1,b2,depth);
            RasterizationData r;
            r.dir1=b1/dot(b1,b1);r.dir2=b2/dot(b2,b2);
            r.imgpos=vec2(sd.imgPos_encoded)/10.0f;r.depth=depth;r.flags=sd.flags;
            const uint32_t C=sd.color;
            r.colors=vec4((C>>0)&255,(C>>8)&255,(C>>16)&255,((C>>24)&255)/255.0f);
            staged[tilePixelIndex]=r;rows[tilePixelIndex]=stageSnapshotRows?stageSnapshotRows[stageIndex]:0xffffffffu;
        }
        block.sync();
        if(inside && !replay.stopped){
            const int batchCount=min(PREFETCH_COUNT,count-PREFETCH_COUNT*batch);
            for(int i=0;i<batchCount;i++){
                const auto r=staged[i];
                if(!replay.acceptsDepth(r.depth, opaqueDepth))break;
                const vec2 d=vec2(float(px),float(py))-r.imgpos;
                const float a=dot(r.dir1,d),b=dot(r.dir2,d),radius=a*a+b*b;
                if(radius>=1.0f)continue;
                float weight=expf(-4.0f*radius);
                float opacity=r.colors.a;
                if(args.uniforms.showRing && weight<0.025f){opacity=1.0f;weight+=0.5f;}
                const float contribution=replay.contribution(weight*opacity);
                if(contribution<=0)continue;
                if(owners==0)frontContributorDepth=r.depth;
                const uint32_t row=rows[i];
                if(row!=0xffffffffu){
                    atomicAdd(allMass+row,contribution);
                    if(foregroundMask[pixelID])atomicAdd(foregroundMass+row,contribution);
                    if(backgroundMask[pixelID])atomicAdd(backgroundMass+row,contribution);
                    const bool center=fabsf(d.x)<=.5f && fabsf(d.y)<=.5f &&
                        r.depth<=frontContributorDepth*(1.0f+centerDepthTolerance);
                    if(foregroundMask[pixelID]){atomicAdd(foregroundSupport+row,1.0f);if(center)atomicAdd(foregroundCenter+row,1.0f);}
                    if(backgroundMask[pixelID]){atomicAdd(backgroundSupport+row,1.0f);if(center)atomicAdd(backgroundCenter+row,1.0f);}
                }
                replay.advance(contribution, (r.flags&FLAGS_DISABLE_DEPTHWRITE)==0,
                    evidenceTransmittanceStop);
                // One native raster contributor per Gaussian/pixel; unlike
                // Spark peeling there are no repeated layer owners to merge.
                ++owners;if(maximumOwners && owners>=maximumOwners)replay.stopped=true;
                if(replay.stopped)break;
            }
        }
        block.sync();
    }
}
