#pragma once
#include "a2a_config.cuh"
#include "symmetric_pool.cuh"
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <nvshmem.h>
#include <torch/custom_class.h>
#include <vector>

namespace ulysses {

// Per-group CE (copy-engine) transfer resources: one stream per peer for the memcpy
// fan-out. Created lazily by UlyssesGroup::ce_resources(), released in destroy(). Serial
// use only (same contract as the config caches). Join events are deliberately NOT pooled
// here -- see the fresh-event note in launch_a2a_ce.
struct CEResources {
    std::vector<cudaStream_t> streams;
};

// CE transfer path (all_to_all_ce.cu): per-peer cudaMemcpy2DAsync fan-out over ce.streams,
// joined back to `stream` with events. Pure DMA -- no SM usage, no launch config, no
// autotune. The caller appends the flag barrier (no nvshmem quiet needed: these are not
// NVSHMEM proxy writes).
void launch_a2a_ce(const void*                  src,
                   const std::vector<uint64_t>& peer_ptrs,
                   const Ulysses4DDims&         dims,
                   int                          mode,
                   int                          elem_size,
                   const CEResources&           ce,
                   cudaStream_t                 stream);

class UlyssesGroup: public torch::CustomClassHolder {
public:
    static int64_t              uniqueid_nints();  // ceil(sizeof(nvshmemx_uniqueid_t)/8)
    static std::vector<int64_t> get_uniqueid();    // rank0 only
    static void init_world(std::vector<int64_t> uid_ints, int64_t global_rank, int64_t global_nranks);  // idempotent

    UlyssesGroup(std::vector<int64_t> peer_global_pes, int64_t my_rank, int64_t device_id, int64_t reserved_bytes);
    ~UlyssesGroup() override;

    int64_t rank() const
    {
        return my_rank_;
    }
    int64_t world_size() const
    {
        return world_size_;
    }
    // This rank's CUDA device ordinal: entry points without an input tensor (the signal ops)
    // guard on it, since nothing else pins the current device.
    int64_t device_id() const
    {
        return device_id_;
    }
    void destroy();

    SymmetricHeapPool& pool()
    {
        return *pool_;
    }
    nvshmem_team_t team() const
    {
        return team_;
    }

    // Device compute capability major (cached via cudaDeviceGetAttribute at construction). resolve_config uses
    // it for the auto path choice (sm<9 has no TMA).
    int sm_major() const
    {
        return sm_major_;
    }

    struct PathConfig {
        bool      tma;
        A2AConfig cfg;
    };

    // Resolve the launch path + config for (dims,mode). use_tma_i is the tri-state -1 auto / 0 non-TMA /
    // 1 TMA (the sm<9 error for an explicit 1 is enforced by the caller). Returns the path to launch and
    // its config; a cache hit returns directly.
    //
    // Explicit 0/1: micro-benchmark that path's candidates, keep the fastest. Auto (-1): on sm<9 -> non-TMA;
    // on sm90+ micro-benchmark BOTH paths and pick the faster (the runtime replacement of the old static DiT
    // table -- "tune the best path+config and cache it"), memoised in auto_path_cache_.
    //
    // The microbench times the REAL per-call op (launch + quiet + fast_barrier) so its ranking matches steady
    // state, and the two paths' times are directly comparable. The fast_barrier makes the miss branch
    // cross-rank, but it is HANG-SAFE: pure-lazy SPMD (no tune()) means all ranks issue the same
    // (shape,mode,use_tma) sequence and miss the same entry on the first call together -> equal barrier calls
    // in lockstep (the candidate count is a function of rank-invariant dims, so it is identical on every rank).
    // A cache-hit rank and a miss rank never coexist for the same call, so no rank blocks alone.
    //
    // The auto path choice is a per-rank local timing decision, so on a near-tie shape different ranks may pick
    // different kernels. This is harmless: the two paths are functionally equivalent P2P writes (each rank
    // writes its own region correctly either way), and per-call barrier counts stay equal regardless of path.
    //
    // Thread safety: resolve_config/all_to_all on the same group instance must be called serially (SPMD
    // single-threaded; the caches are lock-free, so concurrent multi-stream use is not thread-safe).
    PathConfig resolve_config(const Ulysses4DDims&         dims,
                              int                          mode,
                              int                          use_tma_i,
                              const void*                  src,
                              const std::vector<uint64_t>& peers,
                              int                          elem,
                              cudaStream_t                 stream);

    // Generic cfg_cache_ front for extra paths (e.g. the QK-fused scatter): hit returns the cached
    // config, miss runs tune() once and caches. Same hang-safety contract as resolve_config: under
    // SPMD all ranks miss the same key on the first call together (tune's microbench barriers stay
    // in lockstep). Serial access only (see resolve_config comment).
    A2AConfig resolve_config_cached(const ConfigKey& key, const std::function<A2AConfig()>& tune);

    // Custom single-node NVLink flag barrier: replaces the slow nvshmem sync (~280us) that falls back on
    // hardware without NVLS fabric. Call nvshmemx_quiet_on_stream first (so this rank's writes are globally
    // visible). No-op when world_size==1.
    void fast_barrier(cudaStream_t stream);

    // Lazily create (world_size streams + events) and return the CE transfer resources.
    const CEResources& ce_resources();

    // Consumer-signal handshake (DeepEP-style split of arrive and wait) for grouped CE
    // collectives. signal_arrive enqueues one 1-byte cudaMemsetAsync per rank (epoch low
    // byte, CE-executed -- zero SM) on the given stream, AFTER the group's data copies in
    // stream order. signal_wait launches a 1-block poll kernel on the given (consumer)
    // stream that ld.acquire-spins until every rank's signal byte matches the epoch --
    // the consumer proceeds the moment the last peer's data lands, with no barrier
    // kernel competing for an SM slot on the comm stream and no extra event hop.
    // Byte matching (current epoch or the next one -- a peer may run at most one group
    // ahead, see the poll-kernel comment) keeps the protocol wrap- and reset-free; epochs
    // whose low byte is 0 are skipped identically on every rank. Same rank-uniform
    // call-sequence contract as fast_barrier; arrive/wait pairs and fast_barrier calls may
    // be mixed across call sites as long as the pattern is identical on all ranks.
    void signal_arrive(cudaStream_t stream);
    void signal_wait(cudaStream_t stream);

private:
    int                                my_rank_, world_size_, device_id_;
    int                                sm_major_ = 0;  // cached cudaDeviceGetAttribute(major) at construction
    std::vector<int>                   peer_global_pes_;
    nvshmem_team_t                     team_;
    bool                               owns_team_ = false;
    bool                               destroyed_ = false;
    std::unique_ptr<SymmetricHeapPool> pool_;

    // cfg_cache_ holds the best config per (ws,mode,tma,n_local,s_local,d) (key excludes b/elem -- 2B path
    // only; the tma bit distinguishes the two paths). Lock-free std::map, must be accessed serially (see
    // resolve_config comment).
    std::map<ConfigKey, A2AConfig> cfg_cache_;
    // auto_path_cache_ memoises the best path for the auto (use_tma=None) case per (mode,n_local,s_local,d)
    // -- true=TMA -- so a repeat auto call skips the two-path micro-benchmark.
    std::map<std::tuple<int, int, int, int>, bool> auto_path_cache_;

    // CE transfer resources (lazy; see ce_resources()).
    CEResources ce_;
    bool        ce_ready_ = false;

    // fast_barrier state: symmetric flag buffer (uint64[ws]) + monotonic epoch (incremented lockstep per rank).
    bool                  bar_ready_ = false;
    uint64_t              bar_epoch_ = 0;
    void*                 bar_local_ = nullptr;  // this rank's flag base
    std::vector<uint64_t> bar_peers_;            // per-peer flag base (including self)

    void ensure_bar_init_(cudaStream_t stream);      // shared flag-buffer init
    void fast_barrier_kernel_(cudaStream_t stream);  // launch the spin-kernel barrier at bar_epoch_

    // consumer-signal state: symmetric byte flags (uint8[ws]) + monotonic epoch.
    bool                  csig_ready_ = false;
    uint64_t              csig_epoch_ = 0;
    void*                 csig_local_ = nullptr;
    std::vector<uint64_t> csig_peers_;

    void ensure_csig_init_(cudaStream_t stream);
};

}  // namespace ulysses
