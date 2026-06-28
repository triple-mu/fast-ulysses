#include "a2a_config.cuh"
#include "ulysses_common.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <functional>
#include <map>
#include <mutex>
#include <tuple>

namespace ulysses {

// Each (peer, b, s) unit copies one contiguous n_local*d block (contiguous in both src and dst under
// mode0/mode1): inner (nl,v) is contiguous, so consecutive threads fill a whole block, giving coalesced
// large (typically 4KB) remote bursts. Far better NVLink efficiency than the old scattered writes
// ("write a 256B d-row then jump 4KB"). UNROLL register prefetch pipelines read/write (local reads hidden behind remote
// writes).
template<int WORLD_SIZE, int MODE, int UNROLL, typename Epilogue>
__global__ void a2a_copy_generic(
    const uint8_t* __restrict__ src, PeerPtrs<WORLD_SIZE> peers, Ulysses4DDims dims, int elem_size, Epilogue)
{
    const int     row_bytes = dims.d * elem_size;   // 16B aligned (guaranteed by Global Constraints)
    const int     vecs      = row_bytes >> 4;       // uint4 count per d-row
    const int     blk_vecs  = dims.n_local * vecs;  // uint4 count per contiguous n_local*d block
    const int64_t units     = static_cast<int64_t>(WORLD_SIZE) * dims.b * dims.s_local;
    const int64_t total     = units * blk_vecs;
    const int64_t stride    = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t tid       = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;

    for (int64_t base = tid; base < total; base += stride * UNROLL) {
        const uint4* sp[UNROLL];
        uint4*       dp[UNROLL];
        uint4        reg[UNROLL];
        // Read phase (local HBM): prefetch UNROLL uint4s into registers
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            int64_t idx = base + static_cast<int64_t>(k) * stride;
            sp[k]       = nullptr;
            if (idx >= total)
                continue;
            int     inner = static_cast<int>(idx % blk_vecs);  // uint4 offset within block (over nl and d)
            int64_t u     = idx / blk_vecs;                    // unit index over [WS, b, s_local]
            int     s     = static_cast<int>(u % dims.s_local);
            u /= dims.s_local;
            int b_idx = static_cast<int>(u % dims.b);
            u /= dims.b;
            int peer = static_cast<int>(u);  // 0..WS-1

            int64_t src_base_row, dst_base_row;  // block base address in units of d-rows
            if (MODE == 0) {
                src_base_row = (static_cast<int64_t>(b_idx) * dims.s_local + s) * dims.n_global + peer * dims.n_local;
                dst_base_row =
                    (static_cast<int64_t>(b_idx) * dims.s_global + (dims.rank * dims.s_local + s)) * dims.n_local;
            }
            else {
                src_base_row = (static_cast<int64_t>(b_idx) * dims.s_global + (peer * dims.s_local + s)) * dims.n_local;
                dst_base_row =
                    (static_cast<int64_t>(b_idx) * dims.s_local + s) * dims.n_global + dims.rank * dims.n_local;
            }
            sp[k]  = reinterpret_cast<const uint4*>(src + src_base_row * row_bytes) + inner;
            dp[k]  = reinterpret_cast<uint4*>(static_cast<uint8_t*>(peers.p[peer]) + dst_base_row * row_bytes) + inner;
            reg[k] = *sp[k];
        }
        // Write phase (remote NVLink): issued in bulk, local read latency hidden behind it
#pragma unroll
        for (int k = 0; k < UNROLL; ++k)
            if (sp[k])
                *dp[k] = reg[k];
    }
    __threadfence_system();  // system-scope visibility of P2P direct writes to other GPUs
}

template<int WS, int UNROLL>
static void launch_ws_u(const PeerPtrs<WS>&  pp,
                        const uint8_t*       src,
                        const Ulysses4DDims& dims,
                        int                  mode,
                        int                  elem_size,
                        int                  blocks,
                        int                  threads,
                        cudaStream_t         stream)
{
    if (mode == 0)
        a2a_copy_generic<WS, 0, UNROLL, EpilogueIdentity>
            <<<blocks, threads, 0, stream>>>(src, pp, dims, elem_size, EpilogueIdentity{});
    else
        a2a_copy_generic<WS, 1, UNROLL, EpilogueIdentity>
            <<<blocks, threads, 0, stream>>>(src, pp, dims, elem_size, EpilogueIdentity{});
}

template<int WS>
static void launch_ws(const uint8_t*               src,
                      const std::vector<uint64_t>& peer_ptrs,
                      const Ulysses4DDims&         dims,
                      int                          mode,
                      int                          elem_size,
                      int                          blocks,
                      int                          threads,
                      int                          unroll,
                      cudaStream_t                 stream)
{
    PeerPtrs<WS> pp;
    for (int i = 0; i < WS; ++i)
        pp.p[i] = reinterpret_cast<void*>(peer_ptrs[i]);
    switch (unroll) {
        case 2:
            launch_ws_u<WS, 2>(pp, src, dims, mode, elem_size, blocks, threads, stream);
            break;
        case 8:
            launch_ws_u<WS, 8>(pp, src, dims, mode, elem_size, blocks, threads, stream);
            break;
        case 16:
            launch_ws_u<WS, 16>(pp, src, dims, mode, elem_size, blocks, threads, stream);
            break;
        default:
            launch_ws_u<WS, 4>(pp, src, dims, mode, elem_size, blocks, threads, stream);
            break;
    }
}

// Dispatch by ws to the matching launch_ws<WS> (folds the duplicated switch(ws) in launch_a2a /
// resolve_config_nontma). Caller already TORCH_CHECKs ws in {1,2,4,8}, so default is a no-op.
static void nontma_dispatch(int                          ws,
                            const uint8_t*               s,
                            const std::vector<uint64_t>& peers,
                            const Ulysses4DDims&         dims,
                            int                          mode,
                            int                          elem,
                            int                          blocks,
                            int                          threads,
                            int                          unroll,
                            cudaStream_t                 stream)
{
    switch (ws) {
        case 1:
            launch_ws<1>(s, peers, dims, mode, elem, blocks, threads, unroll, stream);
            break;
        case 2:
            launch_ws<2>(s, peers, dims, mode, elem, blocks, threads, unroll, stream);
            break;
        case 4:
            launch_ws<4>(s, peers, dims, mode, elem, blocks, threads, unroll, stream);
            break;
        case 8:
            launch_ws<8>(s, peers, dims, mode, elem, blocks, threads, unroll, stream);
            break;
        default:
            break;
    }
}

// SM count of the current device: queried once per process and cached (one device bound per process).
static int sm_count_cached()
{
    static const int sm = [] {
        int d = 0, s = 0;
        cudaGetDevice(&d);
        cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, d);
        return s > 0 ? s : 132;
    }();
    return sm;
}

static int clamp_blocks(int64_t needed, int64_t want)
{
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, want)));
}

// Auto-pick grid: micro-benchmark candidate block counts (x sm_count, capped to needed), return the fastest.
// launch(blocks) runs the actual copy kernel with the given block count (remote writes are real; autotune
// trial runs are overwritten by the final launch afterwards, so correctness is unaffected).
// Process-level static cache (key=(ws,mode,total,kind)): a local memo of block micro-benchmark results
// (not collective, pure local cudaEvent timing). This is the only autotune cache for varlen non-TMA
// (varlen calls this directly, bypassing UlyssesGroup::cfg_cache_); the uniform path also skips reruns on
// a key hit. Distinct from and coexists with group cfg_cache_ (the uniform A2AConfig config cache).
// kind distinguishes call paths (0=uniform direct-write a2a_copy_generic, 1=varlen source-routed
// a2a_copy_varlen): the two paths use different total formulas with a tiny chance of a total collision,
// so kind fully isolates the two caches.
static int pick_blocks(int                             ws,
                       int                             mode,
                       int64_t                         total,
                       int64_t                         needed,
                       const std::function<void(int)>& launch,
                       cudaStream_t                    stream,
                       int                             kind)
{
    const int sm = sm_count_cached();

    static std::mutex                                        mu;
    static std::map<std::tuple<int, int, int64_t, int>, int> cache;
    const std::tuple<int, int, int64_t, int>                 key{ws, mode, total, kind};
    {
        std::lock_guard<std::mutex> lk(mu);
        auto                        it = cache.find(key);
        if (it != cache.end())
            return it->second;
    }

    const double factors[] = {2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0};
    cudaEvent_t  s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    int   best   = clamp_blocks(needed, static_cast<int64_t>(sm * 16));
    float best_t = 1e30f;
    for (double f : factors) {
        int blocks = clamp_blocks(needed, static_cast<int64_t>(sm * f));
        for (int i = 0; i < 3; ++i)
            launch(blocks);  // warm up
        cudaEventRecord(s, stream);
        for (int i = 0; i < 10; ++i)
            launch(blocks);
        cudaEventRecord(e, stream);
        cudaEventSynchronize(e);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, s, e);
        if (ms < best_t) {
            best_t = ms;
            best   = blocks;
        }
    }
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    {
        std::lock_guard<std::mutex> lk(mu);
        cache[key] = best;
    }
    return best;
}

// Launch the non-TMA direct-write kernel with the given config. Use blocks>0 directly; otherwise pick_blocks
// auto-selects. threads/unroll set the number of in-flight remote write transactions (Little's law); 512 threads
// measured to raise NVLink BW from ~210-260 (256 threads) to a steady ~310 across all N (close to TMA), hence default
// 512.
void launch_a2a(const void*                  src,
                const std::vector<uint64_t>& peer_ptrs,
                const Ulysses4DDims&         dims,
                int                          mode,
                int                          elem_size,
                const A2AConfig&             cfg,
                cudaStream_t                 stream)
{
    const int      ws        = static_cast<int>(peer_ptrs.size());
    const int      threads   = cfg.threads;
    const int      unroll    = cfg.unroll;
    const int64_t  row_bytes = static_cast<int64_t>(dims.d) * elem_size;
    const int      vecs      = static_cast<int>(row_bytes >> 4);
    const int64_t  total     = static_cast<int64_t>(ws) * dims.b * dims.n_local * dims.s_local * vecs;
    const int64_t  needed    = (total + threads - 1) / threads;
    const uint8_t* s         = static_cast<const uint8_t*>(src);
    // grid-stride, multiple rows per thread; block count auto-tuned (autotune cache per (ws,mode,total)).
    // pick_blocks's 5th arg needs a callable, so keep this outer lambda; the switch(ws) body folds into
    // nontma_dispatch.
    auto launch = [&](int blocks) {
        nontma_dispatch(ws, s, peer_ptrs, dims, mode, elem_size, blocks, threads, unroll, stream);
    };
    const int blocks = cfg.blocks > 0 ? cfg.blocks : pick_blocks(ws, mode, total, needed, launch, stream, 0);
    launch(blocks);
}

// non-TMA config resolution: default config plus a live block sweep written into cfg.blocks (completes the config).
// No cache of its own: the returned cfg (with blocks) is held by UlyssesGroup::cfg_cache_.
A2AConfig resolve_config_nontma(const void*                  src,
                                const std::vector<uint64_t>& peer_ptrs,
                                const Ulysses4DDims&         dims,
                                int                          mode,
                                int                          elem_size,
                                cudaStream_t                 stream)
{
    const int      ws        = static_cast<int>(peer_ptrs.size());
    A2AConfig      cfg       = default_config(mode, dims.n_local);
    const int      threads   = cfg.threads;
    const int      unroll    = cfg.unroll;
    const int64_t  row_bytes = static_cast<int64_t>(dims.d) * elem_size;
    const int      vecs      = static_cast<int>(row_bytes >> 4);
    const int64_t  total     = static_cast<int64_t>(ws) * dims.b * dims.n_local * dims.s_local * vecs;
    const int64_t  needed    = (total + threads - 1) / threads;
    const uint8_t* s         = static_cast<const uint8_t*>(src);
    // Outer lambda feeds pick_blocks (5th arg needs a callable); the switch(ws) body folds into nontma_dispatch.
    auto launch = [&](int blocks) {
        nontma_dispatch(ws, s, peer_ptrs, dims, mode, elem_size, blocks, threads, unroll, stream);
    };
    cfg.blocks = pick_blocks(ws, mode, total, needed, launch, stream, 0);
    return cfg;
}

// ---- varlen (uneven s/n) source-routed kernel: scan local input, route each element to its owning peer by split
// offsets and write directly ----
template<int WORLD_SIZE, int MODE, typename Epilogue>
__global__ void
a2a_copy_varlen(const uint8_t* __restrict__ src, PeerPtrs<WORLD_SIZE> peers, SplitInfo sp, int elem_size, Epilogue)
{
    const int row_bytes = sp.d * elem_size;
    const int vecs      = row_bytes >> 4;
    const int me        = sp.rank;
    const int S         = sp.s_off[WORLD_SIZE];
    const int N         = sp.n_off[WORLD_SIZE];

    if constexpr (MODE == 0) {
        // Local input [b, s_me, N, d]; me's sequence block has global start s_base, landing in each peer's output [b,
        // S, nchunk, d].
        const int     s_me   = sp.s_off[me + 1] - sp.s_off[me];
        const int     s_base = sp.s_off[me];
        const int64_t total  = static_cast<int64_t>(sp.b) * s_me * N * vecs;
        for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
             idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
            int     v = static_cast<int>(idx % vecs);
            int64_t r = idx / vecs;
            int     h = static_cast<int>(r % N);
            r /= N;
            int s = static_cast<int>(r % s_me);
            r /= s_me;
            int b_idx = static_cast<int>(r);
            int peer  = 0;  // owning peer of head h: n_off[peer] <= h < n_off[peer+1]
#pragma unroll
            for (int t = 0; t < WORLD_SIZE; ++t)
                if (h >= sp.n_off[t + 1])
                    peer = t + 1;
            int          nchunk  = sp.n_off[peer + 1] - sp.n_off[peer];
            int          hl      = h - sp.n_off[peer];
            int64_t      src_row = (static_cast<int64_t>(b_idx) * s_me + s) * N + h;
            int64_t      dst_row = (static_cast<int64_t>(b_idx) * S + (s_base + s)) * nchunk + hl;
            const uint4* spp     = reinterpret_cast<const uint4*>(src + src_row * row_bytes);
            uint4*       dp      = reinterpret_cast<uint4*>(static_cast<uint8_t*>(peers.p[peer]) + dst_row * row_bytes);
            dp[v]                = spp[v];
        }
    }
    else {
        // Local input [b, S, n_me, d]; me's head block has global start n_base, landing in each peer's output [b,
        // schunk, N, d].
        const int     n_me   = sp.n_off[me + 1] - sp.n_off[me];
        const int     n_base = sp.n_off[me];
        const int64_t total  = static_cast<int64_t>(sp.b) * S * n_me * vecs;
        for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
             idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
            int     v  = static_cast<int>(idx % vecs);
            int64_t r  = idx / vecs;
            int     hl = static_cast<int>(r % n_me);
            r /= n_me;
            int sg = static_cast<int>(r % S);
            r /= S;
            int b_idx = static_cast<int>(r);
            int peer  = 0;  // owning peer of sequence sg: s_off[peer] <= sg < s_off[peer+1]
#pragma unroll
            for (int t = 0; t < WORLD_SIZE; ++t)
                if (sg >= sp.s_off[t + 1])
                    peer = t + 1;
            int          schunk  = sp.s_off[peer + 1] - sp.s_off[peer];
            int          sl      = sg - sp.s_off[peer];
            int64_t      src_row = (static_cast<int64_t>(b_idx) * S + sg) * n_me + hl;
            int64_t      dst_row = (static_cast<int64_t>(b_idx) * schunk + sl) * N + (n_base + hl);
            const uint4* spp     = reinterpret_cast<const uint4*>(src + src_row * row_bytes);
            uint4*       dp      = reinterpret_cast<uint4*>(static_cast<uint8_t*>(peers.p[peer]) + dst_row * row_bytes);
            dp[v]                = spp[v];
        }
    }
    __threadfence_system();
}

template<int WS>
static void launch_ws_varlen(const uint8_t*               src,
                             const std::vector<uint64_t>& peer_ptrs,
                             const SplitInfo&             sp,
                             int                          mode,
                             int                          elem_size,
                             int                          blocks,
                             int                          threads,
                             cudaStream_t                 stream)
{
    PeerPtrs<WS> pp;
    for (int i = 0; i < WS; ++i)
        pp.p[i] = reinterpret_cast<void*>(peer_ptrs[i]);
    if (mode == 0)
        a2a_copy_varlen<WS, 0, EpilogueIdentity>
            <<<blocks, threads, 0, stream>>>(src, pp, sp, elem_size, EpilogueIdentity{});
    else
        a2a_copy_varlen<WS, 1, EpilogueIdentity>
            <<<blocks, threads, 0, stream>>>(src, pp, sp, elem_size, EpilogueIdentity{});
}

void launch_a2a_varlen(const void*                  src,
                       const std::vector<uint64_t>& peer_ptrs,
                       const SplitInfo&             sp,
                       int                          mode,
                       int                          elem_size,
                       cudaStream_t                 stream)
{
    const int      ws        = static_cast<int>(peer_ptrs.size());
    const int      threads   = 512;  // as in uniform: more in-flight remote writes -> higher NVLink BW
    const int64_t  row_bytes = static_cast<int64_t>(sp.d) * elem_size;
    const int      vecs      = static_cast<int>(row_bytes >> 4);
    const int      me        = sp.rank;
    const int      S         = sp.s_off[ws];
    const int      N         = sp.n_off[ws];
    const int64_t  rows      = (mode == 0) ? static_cast<int64_t>(sp.b) * (sp.s_off[me + 1] - sp.s_off[me]) * N :
                                             static_cast<int64_t>(sp.b) * S * (sp.n_off[me + 1] - sp.n_off[me]);
    const int64_t  total     = rows * vecs;
    const int64_t  needed    = (total + threads - 1) / threads;
    const uint8_t* s         = static_cast<const uint8_t*>(src);
    auto           launch    = [&](int blocks) {
        switch (ws) {
            case 1:
                launch_ws_varlen<1>(s, peer_ptrs, sp, mode, elem_size, blocks, threads, stream);
                break;
            case 2:
                launch_ws_varlen<2>(s, peer_ptrs, sp, mode, elem_size, blocks, threads, stream);
                break;
            case 4:
                launch_ws_varlen<4>(s, peer_ptrs, sp, mode, elem_size, blocks, threads, stream);
                break;
            case 8:
                launch_ws_varlen<8>(s, peer_ptrs, sp, mode, elem_size, blocks, threads, stream);
                break;
            default:
                break;
        }
    };
    launch(pick_blocks(ws, mode, total, needed, launch, stream, 1));
}

}  // namespace ulysses
