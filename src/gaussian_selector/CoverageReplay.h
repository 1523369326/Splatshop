#pragma once

// Shared by the CPU oracle and the CUDA replay. A terminated pixel stays
// terminated across staging batches; threads still participate in block sync.
#ifdef __CUDACC__
#define GS_REPLAY_HD __host__ __device__
#else
#define GS_REPLAY_HD
#endif

namespace gaussian_selector {
struct CoverageReplay {
    float accumulatedAlpha = 0;
    bool stopped = false;

    GS_REPLAY_HD bool acceptsDepth(float depth, float opaqueDepth) {
        if(depth > opaqueDepth) stopped = true;
        return !stopped;
    }
    GS_REPLAY_HD float contribution(float intrinsicAlpha) const {
        return intrinsicAlpha * (1.0f - accumulatedAlpha);
    }
    GS_REPLAY_HD void advance(float mass, bool writesDepth, float evidenceStop,
                              float rasterStop = 4.0f / 255.0f) {
        accumulatedAlpha += mass;
        const float remaining = 1.0f - accumulatedAlpha;
        // Evidence visibility is independent of a node's depth-write setting.
        // Zero disables the additional evidence cutoff.
        stopped = stopped || (evidenceStop > 0 && remaining < evidenceStop) ||
            (writesDepth && remaining < rasterStop);
    }
};
} // namespace gaussian_selector

#undef GS_REPLAY_HD
