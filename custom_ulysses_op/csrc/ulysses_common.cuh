#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace ulysses {

// Shared host/device dim descriptor (int32 fields; kernels use int64 for addressing).
struct Ulysses4DDims {
    int32_t b, s_local, s_global, n_local, n_global, d, rank;
};

struct A2AConfig;  // a2a_config.cuh

template<int WS>
struct PeerPtrs {
    void* p[WS];
};

// Baseline is a raw byte copy; Identity is a no-op. Future RoPE/RMSNorm specialize this hook.
struct EpilogueIdentity {
    __device__ __forceinline__ void operator()(uint8_t* /*row_bytes*/, int /*row_off*/) const {}
};

enum class KernelArch {
    kPlain80,
    kPlain90,
    kPlain100
};

// Varlen (uneven s/n, need not divide world_size): prefix-offset arrays (world_size <= 8 -> <= 9 entries).
// The uniform case is a special case (s_off[r]=r*s_local, n_off[r]=r*n_local). Caller supplies the split (no runtime
// gather).
struct SplitInfo {
    int32_t world_size, rank, b, d;
    int32_t s_off[9];  // s_off[r] = seq start of rank r; s_off[world_size] = S (total seq)
    int32_t n_off[9];  // n_off[r] = head start of rank r; n_off[world_size] = N (total heads)
};

// Free function: pure-CUDA vectorized direct write to peer symmetric memory (host already
// resolved peer_ptrs via nvshmem_ptr). Launch with the given config (no env, no autotune).
void launch_a2a(const void*                  src,
                const std::vector<uint64_t>& peer_ptrs,
                const Ulysses4DDims&         dims,
                int                          mode,
                int                          elem_size,
                const A2AConfig&             cfg,
                cudaStream_t                 stream);

// non-TMA config resolution: default config + pick_blocks measured block count (pick_blocks has
// its own process-level static cache; the result is then held by the group's cfg_cache_).
A2AConfig resolve_config_nontma(const void*                  src,
                                const std::vector<uint64_t>& peer_ptrs,
                                const Ulysses4DDims&         dims,
                                int                          mode,
                                int                          elem_size,
                                cudaStream_t                 stream);

// Varlen version: source-side routing (scan local input, route each head/seq slot to its owning peer via split
// offsets).
void launch_a2a_varlen(const void*                  src,
                       const std::vector<uint64_t>& peer_ptrs,
                       const SplitInfo&             sp,
                       int                          mode,
                       int                          elem_size,
                       cudaStream_t                 stream);

// Varlen TMA version (sm90+): one src map + a separate dst map per peer, grid.y=peer. See a2a_tma.cu.
void launch_a2a_tma_varlen(const void*                  src,
                           const std::vector<uint64_t>& peer_ptrs,
                           const SplitInfo&             sp,
                           int                          mode,
                           int                          elem_size,
                           cudaStream_t                 stream);

// TMA version (fewer SMs, better comm/compute overlap); uniform mode0/mode1. See all_to_all_tma.cu.
// Launch with the given config (build maps + launch only; no team, no autotune, no env).
void launch_a2a_tma(const void*                  src,
                    const std::vector<uint64_t>& peer_ptrs,
                    const Ulysses4DDims&         dims,
                    int                          mode,
                    int                          elem_size,
                    const A2AConfig&             cfg,
                    cudaStream_t                 stream);

// TMA config resolution: pick the best candidate by micro-benchmark (result cached in the group's
// cfg_cache_; a hit skips the micro-benchmark).
// team: nvshmem team handle, used for cross-rank lockstep barrier during candidate micro-benchmarks. verbose: debug
// printing.
A2AConfig resolve_config_tma(const void*                  src,
                             const std::vector<uint64_t>& peer_ptrs,
                             const Ulysses4DDims&         dims,
                             int                          mode,
                             int                          elem_size,
                             int                          team,
                             bool                         verbose,
                             cudaStream_t                 stream);

}  // namespace ulysses
