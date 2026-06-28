#pragma once
#include "a2a_config.cuh"
#include "symmetric_pool.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <nvshmem.h>
#include <torch/custom_class.h>
#include <vector>

namespace ulysses {

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
    void destroy();

    SymmetricHeapPool& pool()
    {
        return *pool_;
    }
    nvshmem_team_t team() const
    {
        return team_;
    }

    // Device compute capability major (cached via cudaDeviceGetAttribute at construction). decide_tma uses it for auto
    // resolution.
    int sm_major() const
    {
        return sm_major_;
    }

    // Resolve the launch config for (dims,mode,use_tma): hit in cfg_cache_ returns directly; miss runs a
    // microbenchmark to pick the best and caches it. The use_tma path is passed by the caller after
    // decide_tma resolution (true->TMA, false->non-TMA) and must be identical across all ranks.
    //
    // [Collective invariant] The miss branch is a collective call: resolve_config_tma runs a lockstep
    // microbenchmark with nvshmemx_team_sync_on_stream; the non-TMA path has no nvshmem collective of its
    // own (pick_blocks times locally), its collective comes from the upstream symmetric allocation in
    // pool().acquire.
    // So all ranks must call all_to_all / tune with the exact same (shape, mode) sequence; a hit rank skips
    // the collective while a miss rank blocks -> whole group hangs. tune() and lazy fallback are mutually
    // exclusive: either all ranks pre-tune the same shape set, or all ranks rely on lazy; never mix tune on
    // some ranks with lazy on others.
    //
    // Thread safety: resolve_config/all_to_all on the same group instance must be called serially (SPMD
    // single-threaded; cfg_cache_ is lock-free, so concurrent multi-stream use is not thread-safe).
    A2AConfig resolve_config(const Ulysses4DDims&         dims,
                             int                          mode,
                             bool                         use_tma,
                             const void*                  src,
                             const std::vector<uint64_t>& peers,
                             int                          elem,
                             cudaStream_t                 stream);

    // Warm up the config for a shape (collective call): builds a temp src + symmetric output, runs the
    // microbenchmark to pick the best and writes cfg_cache_, so later all_to_all with the same (shape,mode)
    // hits the cache and skips the collective microbenchmark.
    //
    // [Collective invariant] All ranks must call with the same (b,s_local,n_local,d,mode,use_tma) sequence;
    // mutually exclusive with lazy fallback (see resolve_config comment). Params are the per-rank LOCAL logical
    // dims (b, s_local, n_local, d), identical for mode0/mode1 (s_local=S_global/ws, n_local=H/ws) -- no
    // ambiguity over which middle dim is local vs global. The mode-dependent physical input shape is
    // reconstructed inside (global = local*ws). use_tma:-1 auto/0/1, resolved by decide_tma_path into the
    // actual path, which is the only path warmed up.
    void tune(int64_t b, int64_t s_local, int64_t n_local, int64_t d, int64_t mode, int64_t use_tma, bool verbose);

    // Varlen counterpart of tune(): warm up the non-TMA varlen launch config for the given per-rank splits
    // (collective). seq_lens/head_splits are the per-rank sequence lengths / head counts (len == world_size),
    // identical on every rank (as in all_to_all_single_4d). The config is keyed by this rank's local total, so
    // each rank tunes its own launch params. use_tma:-1 auto/0/1 -> non-TMA varlen warmed (auto/false); explicit
    // True selects TMA-varlen, which has no tunable launch config (no-op). Mutually exclusive with lazy fallback.
    void tune_varlen(int64_t              b,
                     std::vector<int64_t> seq_lens,
                     std::vector<int64_t> head_splits,
                     int64_t              d,
                     int64_t              mode,
                     int64_t              use_tma,
                     bool                 verbose);

    // Custom single-node NVLink flag barrier: replaces the slow nvshmem sync (~280us) that falls back on
    // hardware without NVLS fabric. Call nvshmemx_quiet_on_stream first (so this rank's writes are globally
    // visible). No-op when world_size==1.
    void fast_barrier(cudaStream_t stream);

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

    // fast_barrier state: symmetric flag buffer (uint64[ws]) + monotonic epoch (incremented lockstep per rank).
    bool                  bar_ready_ = false;
    uint64_t              bar_epoch_ = 0;
    void*                 bar_local_ = nullptr;  // this rank's flag base
    std::vector<uint64_t> bar_peers_;            // per-peer flag base (including self)
};

}  // namespace ulysses
