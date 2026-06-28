// TMA (cp.async.bulk.tensor) A2A: few SMs issue, TMA engine moves data (src gmem->smem->peer gmem).
// One launch covers all peers with many blocks (one tile per block) -> high concurrency saturates NVLink. mode0/mode1
// unified. Mechanism/coordinates verified by tma_p2p_probe / tma_a2a_test. Raw PTX (mbar/TMA/wait_group/fence)
// extracted into named device functions in tma_ptx.cuh, behavior-equivalent.
#include "a2a_config.cuh"
#include "tma_ptx.cuh"
#include "ulysses_common.cuh"
#include <algorithm>
#include <c10/util/Exception.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <nvshmemx.h>
#include <vector>

namespace ulysses {

struct TmaMaps {
    CUtensorMap m[8];  // per-peer dst tensormap (world_size <= 8)
};

// Global tile index -> (peer, n, s, bi).
__device__ __forceinline__ void
tma_decode(int g, int per_peer, int n_ntiles, int n_stiles, int tile_n, int tile_s, int& peer, int& n, int& s, int& bi)
{
    peer    = g / per_peer;
    int t   = g % per_peer;
    int nt  = t % n_ntiles;
    int tmp = t / n_ntiles;
    int st  = tmp % n_stiles;
    bi      = tmp / n_stiles;
    n       = nt * tile_n;
    s       = st * tile_s;
}

// Software-pipelined: single thread issues, B smem buffers rotate, prefetch-1-ahead + wait_group(B-1)
// keeps B-1 TMA stores in flight -> remote NVLink write pipeline stays full (removes the single-stage serial bubble).
// grid covers the full (peer, b, s-tile, n-tile) set; each block grid-strides over its own run of tiles.
__global__ void a2a_tma_kernel(const __grid_constant__ CUtensorMap src_map,
                               const __grid_constant__ TmaMaps     dst,
                               int                                 ws,
                               int                                 mode,
                               int                                 rank,
                               int                                 s_local,
                               int                                 n_local,
                               int                                 b,
                               int                                 tile_s,
                               int                                 tile_n,
                               uint32_t                            tile_bytes,
                               int                                 stages)
{
    if (threadIdx.x != 0)
        return;

    extern __shared__ uint8_t smem_raw[];
    const uint32_t            tb_al = (tile_bytes + 127u) & ~127u;  // 128B-align each buffer
    uintptr_t                 base  = (reinterpret_cast<uintptr_t>(smem_raw) + 127) & ~static_cast<uintptr_t>(127);
    uint64_t*                 mbar  = reinterpret_cast<uint64_t*>(base + (uintptr_t)stages * tb_al);

    uint32_t buf_a[8], mbar_a[8];
    int      parity[8];
    for (int k = 0; k < stages; ++k) {
        buf_a[k]  = (uint32_t)__cvta_generic_to_shared(reinterpret_cast<void*>(base + (uintptr_t)k * tb_al));
        mbar_a[k] = (uint32_t)__cvta_generic_to_shared(mbar + k);
        parity[k] = 0;
        mbar_init(mbar_a[k]);
    }

    const int n_ntiles = (n_local + tile_n - 1) / tile_n;
    const int n_stiles = (s_local + tile_s - 1) / tile_s;
    const int per_peer = b * n_stiles * n_ntiles;
    const int total    = ws * per_peer;

    // mode0: src head-dim offset peer*n_local / dst seq-dim offset rank*s_local; mode1 swaps them.
    const int src_n_pp  = (mode == 0) ? n_local : 0;  // multiplied by peer
    const int src_s_pp  = (mode == 0) ? 0 : s_local;
    const int dst_n_off = (mode == 0) ? 0 : rank * n_local;
    const int dst_s_off = (mode == 0) ? rank * s_local : 0;

    // Tiles owned by this block (grid-stride run): g = blockIdx.x + j*gridDim.x, M total
    const int M = (total - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
    if (M <= 0)
        return;

    // prologue: issue loads for the first min(stages, M) tiles
    for (int k = 0; k < stages && k < M; ++k) {
        int g = blockIdx.x + k * gridDim.x;
        int peer, n, s, bi;
        tma_decode(g, per_peer, n_ntiles, n_stiles, tile_n, tile_s, peer, n, s, bi);
        mbar_arrive_expect(mbar_a[k], tile_bytes);
        tma_load_4d(buf_a[k], &src_map, 0, n + peer * src_n_pp, s + peer * src_s_pp, bi, mbar_a[k]);
    }

    for (int j = 0; j < M; ++j) {
        int cur = j % stages;
        int g   = blockIdx.x + j * gridDim.x;
        int peer, n, s, bi;
        tma_decode(g, per_peer, n_ntiles, n_stiles, tile_n, tile_s, peer, n, s, bi);

        // wait for this tile's load to complete
        mbar_wait(mbar_a[cur], parity[cur]);
        parity[cur] ^= 1;
        async_proxy_fence();

        // store this tile -> peer dst
        tma_store_4d(&dst.m[peer], 0, n + dst_n_off, s + dst_s_off, bi, buf_a[cur]);
        tma_commit_group();

        // prefetch next tile: first drain to <= stages-1 stores in flight so the target buffer was read by an old store
        int nl = j + stages;
        if (nl < M) {
            tma_wait_group(
                stages
                - 1);  // equivalent to the original switch only for stages in {2,3,4,6} (-> wait_group {1,2,3,5})
            int slot = nl % stages;
            int g2   = blockIdx.x + nl * gridDim.x;
            int peer2, n2, s2, bi2;
            tma_decode(g2, per_peer, n_ntiles, n_stiles, tile_n, tile_s, peer2, n2, s2, bi2);
            mbar_arrive_expect(mbar_a[slot], tile_bytes);
            tma_load_4d(buf_a[slot], &src_map, 0, n2 + peer2 * src_n_pp, s2 + peer2 * src_s_pp, bi2, mbar_a[slot]);
        }
    }
    tma_wait_group(0);  // drain all stores before exit (correctness)
}

// 4D tensormap: dims(innermost first)=[d, ndim, sdim, b]; box=[d, tile_n, tile_s, 1].
static CUtensorMap tma_make_map(void* base, int d, int ndim, int sdim, int b, int tile_n, int tile_s, int elem_size)
{
    CUtensorMap         m;
    uint64_t            es       = (uint64_t)elem_size;
    uint64_t            gdims[4] = {(uint64_t)d, (uint64_t)ndim, (uint64_t)sdim, (uint64_t)b};
    uint64_t            gstr[3]  = {(uint64_t)d * es, (uint64_t)ndim * d * es, (uint64_t)sdim * ndim * d * es};
    uint32_t            box[4]   = {(uint32_t)d, (uint32_t)tile_n, (uint32_t)tile_s, 1u};
    uint32_t            estr[4]  = {1, 1, 1, 1};
    CUtensorMapDataType dt       = (elem_size == 2) ? CU_TENSOR_MAP_DATA_TYPE_UINT16 : CU_TENSOR_MAP_DATA_TYPE_UINT8;
    CUresult            r        = cuTensorMapEncodeTiled(&m,
                                        dt,
                                        4,
                                        base,
                                        gdims,
                                        gstr,
                                        box,
                                        estr,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE,
                                        CU_TENSOR_MAP_SWIZZLE_NONE,
                                        CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return m;
}

// Build src + per-peer dst tensormaps for the uniform path. mode0/mode1 src/dst layouts are dual:
// mode0 src=[b,s_local,n_global,d], dst=[b,s_global,n_local,d]; mode1 swaps n/s in both.
static void build_tma_maps(int                          mode,
                           const Ulysses4DDims&         dims,
                           const void*                  src,
                           const std::vector<uint64_t>& peers,
                           int                          elem,
                           int                          tile_n,
                           int                          tile_s,
                           CUtensorMap&                 src_map,
                           TmaMaps&                     dst)
{
    const int ws = (int)peers.size();
    const int d = dims.d, b = dims.b;
    const int s_local = dims.s_local, n_local = dims.n_local;
    const int s_global = dims.s_global, n_global = dims.n_global;
    if (mode == 0) {
        src_map = tma_make_map((void*)src, d, n_global, s_local, b, tile_n, tile_s, elem);
        for (int p = 0; p < ws; ++p)
            dst.m[p] = tma_make_map((void*)peers[p], d, n_local, s_global, b, tile_n, tile_s, elem);
    }
    else {
        src_map = tma_make_map((void*)src, d, n_local, s_global, b, tile_n, tile_s, elem);
        for (int p = 0; p < ws; ++p)
            dst.m[p] = tma_make_map((void*)peers[p], d, n_global, s_local, b, tile_n, tile_s, elem);
    }
}

// Launch the TMA kernel for a given config (build maps + launch only). mode0/mode1 use different src/dst tensormap
// layouts.
void launch_a2a_tma(const void*                  src,
                    const std::vector<uint64_t>& peer_ptrs,
                    const Ulysses4DDims&         dims,
                    int                          mode,
                    int                          elem_size,
                    const A2AConfig&             cfg,
                    cudaStream_t                 stream)
{
    const int      ws = (int)peer_ptrs.size();
    const int      d = dims.d, b = dims.b, rank = dims.rank;
    const int      s_local = dims.s_local, n_local = dims.n_local;
    const int      tile_n     = std::min({cfg.tile_n, n_local, 256});
    const int      tile_s     = std::min({cfg.tile_s, s_local, 256});
    const int      stages     = cfg.stages;
    const uint32_t tile_bytes = (uint32_t)tile_s * tile_n * d * elem_size;
    const uint32_t tb_al      = (tile_bytes + 127u) & ~127u;
    const int      smem       = (int)(tb_al * (uint32_t)stages) + 128 + 8 * stages;

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(a2a_tma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
        attr_set = true;
    }

    const int n_stiles = (s_local + tile_s - 1) / tile_s;
    const int n_ntiles = (n_local + tile_n - 1) / tile_n;
    const int total    = ws * b * n_stiles * n_ntiles;
    const int bdiv     = std::max(cfg.bdiv, 1);
    const int blocks = std::max(std::min(total, 65535) / bdiv, 1);  // bdiv>1: more tiles per block -> enable pipelining

    CUtensorMap src_map;
    TmaMaps     dst;
    build_tma_maps(mode, dims, src, peer_ptrs, elem_size, tile_n, tile_s, src_map, dst);
    a2a_tma_kernel<<<blocks, 1, smem, stream>>>(
        src_map, dst, ws, mode, rank, s_local, n_local, b, tile_s, tile_n, tile_bytes, stages);
}

// Candidate configs: cover small N (small tile_n, shallow pipeline) to large N (large tile_n, deep pipeline).
// A non-divisor tile_n (e.g. n_local-1) often beats a whole block -- it forms multiple n-tiles for higher
// TMA concurrency (a single whole-block tile is actually slower). Seed includes default_config (mode0); rest are
// empirically tuned.
static std::vector<A2AConfig> tma_candidates(int mode, int n_local, int s_local)
{
    std::vector<A2AConfig> v;
    auto                   add = [&](int tn, int ts, int st, int bd) {
        tn = std::max(1, std::min(tn, n_local));
        ts = std::max(1, std::min(ts, s_local));
        for (auto& c : v)
            if (c.tile_n == tn && c.tile_s == ts && c.stages == st && c.bdiv == bd)
                return;
        A2AConfig c{};
        c.tile_n = tn;
        c.tile_s = ts;
        c.stages = st;
        c.bdiv   = bd;
        v.push_back(c);
    };
    const A2AConfig def = default_config(mode, n_local);  // seed: default config enters candidates first
    add(def.tile_n, def.tile_s, def.stages, def.bdiv);
    const int nl1 = std::max(1, n_local - 1), nlh = std::max(1, n_local / 2);
    // bdiv=1: many blocks (large N naturally pipelines via multiple tiles per block).
    add(nl1, 1, 6, 1);  // strong config for large N
    add(nl1, 1, 4, 1);
    add(8, 1, 4, 1);
    add(nlh, 1, 4, 1);
    add(n_local, 1, 4, 1);
    // bdiv>1: fewer blocks -> more tiles per block -> enable per-block software pipeline (key for small/medium N).
    add(8, 1, 4, 8);
    add(nl1, 1, 4, 8);
    add(8, 1, 4, 16);
    add(nl1, 1, 6, 16);
    add(8, 1, 6, 32);
    add(nl1, 1, 4, 4);  // small n_local (DiT) measured optimum: small tile_n + medium bdiv
    add(nlh, 1, 4, 4);
    return v;
}

// Microbench one candidate: return median per-call time (us) of run_once.
// The timed region only includes run_once (= real data-movement cost of kernel + quiet); the team_sync lockstep
// is kept outside the timed region (otherwise its ~fixed large overhead compresses candidate differences and makes
// configs indistinguishable at small N). 10 iters per batch, median of three batches for noise robustness.
static float microbench_config(const std::function<void()>& run_once, int team, cudaStream_t stream)
{
    auto        sync = [&] { nvshmemx_team_sync_on_stream((nvshmem_team_t)team, stream); };
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    float med[3];
    for (int rep = 0; rep < 3; ++rep) {
        sync();  // lockstep start (outside timed region)
        for (int i = 0; i < 3; ++i)
            run_once();  // warmup
        sync();
        cudaEventRecord(s, stream);
        for (int i = 0; i < 10; ++i)
            run_once();
        cudaEventRecord(e, stream);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&med[rep], s, e);
        sync();  // re-align all ranks before the next batch
    }
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    // median of three batches
    if (med[0] > med[1])
        std::swap(med[0], med[1]);
    if (med[1] > med[2])
        std::swap(med[1], med[2]);
    if (med[0] > med[1])
        std::swap(med[0], med[1]);
    return med[1] * 100.f;  // ms(10 iters) -> us/call: /10 iters *1000 (ms->us)
}

// autotune: microbench all candidates, return the fastest config (remote writes are real but overwritten by the
// subsequent final launch, so correctness is unaffected).
// The cache lives in the caller (UlyssesGroup::cfg_cache_); this function has no cache of its own, so it is a pure
// collective microbench -- the caller must guarantee the cache-hit branch skips this function (see the collective-call
// hard invariant in ulysses_group.cuh). verbose replaces the old TMA_DEBUG.
A2AConfig resolve_config_tma(const void*                  src,
                             const std::vector<uint64_t>& peer_ptrs,
                             const Ulysses4DDims&         dims,
                             int                          mode,
                             int                          elem_size,
                             int                          team,
                             bool                         verbose,
                             cudaStream_t                 stream)
{
    const int  n_local = dims.n_local, s_local = dims.s_local, ws = (int)peer_ptrs.size();
    const auto cands = tma_candidates(mode, n_local, s_local);

    A2AConfig best   = cands[0];
    float     best_t = 1e30f;
    for (const auto& c : cands) {
        float us = microbench_config(
            [&] {
                launch_a2a_tma(src, peer_ptrs, dims, mode, elem_size, c, stream);
                nvshmemx_quiet_on_stream(stream);
            },
            team,
            stream);
        if (verbose)
            fprintf(stderr,
                    "[tma-at] ws=%d mode=%d nl=%d sl=%d | tn=%d ts=%d st=%d bd=%d -> %.1f us/call\n",
                    ws,
                    mode,
                    n_local,
                    s_local,
                    c.tile_n,
                    c.tile_s,
                    c.stages,
                    c.bdiv,
                    us);
        if (us < best_t) {
            best_t = us;
            best   = c;
        }
    }
    return best;
}

// ============================ Varlen (uneven splits) TMA path ============================
// Per-peer chunk sizes differ: use one src map (full local input) + an independent dst map per peer.
// grid.y = peer (each block fixed to one peer, avoiding a tile prefix sum); grid.x grid-strides over that peer's tiles.
// If a trailing tile crosses the peer's chunk boundary, the dst map's OOB clipping discards it (no padding needed).
// Software pipeline same as uniform.
__global__ void a2a_tma_varlen_kernel(const __grid_constant__ CUtensorMap src_map,
                                      const __grid_constant__ TmaMaps     dst,
                                      SplitInfo                           sp,
                                      int                                 mode,
                                      int                                 tile_s,
                                      int                                 tile_n,
                                      uint32_t                            tile_bytes,
                                      int                                 stages)
{
    if (threadIdx.x != 0)
        return;
    const int peer = blockIdx.y;
    const int me   = sp.rank;

    // This peer's tile dims (ntiles_dim/stiles_dim) and src/dst coordinate offsets:
    // mode0: peer owns head chunk [n_off[peer],n_off[peer+1]); this rank writes its own seq chunk (offset s_off[me]).
    // mode1: peer owns seq chunk [s_off[peer],s_off[peer+1]); this rank writes its own head chunk (offset n_off[me]).
    int ntiles_dim, stiles_dim, n_src_off, s_src_off, n_dst_off, s_dst_off;
    if (mode == 0) {
        ntiles_dim = sp.n_off[peer + 1] - sp.n_off[peer];
        stiles_dim = sp.s_off[me + 1] - sp.s_off[me];
        n_src_off  = sp.n_off[peer];
        s_src_off  = 0;
        n_dst_off  = 0;
        s_dst_off  = sp.s_off[me];
    }
    else {
        ntiles_dim = sp.n_off[me + 1] - sp.n_off[me];
        stiles_dim = sp.s_off[peer + 1] - sp.s_off[peer];
        n_src_off  = 0;
        s_src_off  = sp.s_off[peer];
        n_dst_off  = sp.n_off[me];
        s_dst_off  = 0;
    }
    const int n_ntiles = (ntiles_dim + tile_n - 1) / tile_n;
    const int n_stiles = (stiles_dim + tile_s - 1) / tile_s;
    const int per_b    = n_stiles * n_ntiles;
    const int total    = sp.b * per_b;  // total tiles for this peer

    extern __shared__ uint8_t smem_raw[];
    const uint32_t            tb_al = (tile_bytes + 127u) & ~127u;
    uintptr_t                 base  = (reinterpret_cast<uintptr_t>(smem_raw) + 127) & ~static_cast<uintptr_t>(127);
    uint64_t*                 mbar  = reinterpret_cast<uint64_t*>(base + (uintptr_t)stages * tb_al);
    uint32_t                  buf_a[8], mbar_a[8];
    int                       parity[8];
    for (int k = 0; k < stages; ++k) {
        buf_a[k]  = (uint32_t)__cvta_generic_to_shared(reinterpret_cast<void*>(base + (uintptr_t)k * tb_al));
        mbar_a[k] = (uint32_t)__cvta_generic_to_shared(mbar + k);
        parity[k] = 0;
        mbar_init(mbar_a[k]);
    }

    const int M = (total - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
    if (M <= 0)
        return;

        // In-block tile index g -> (n, s, bi) (peer fixed to blockIdx.y)
#define VL_DECODE(g, NN, SS, BI)                                                                                       \
    do {                                                                                                               \
        int t_  = (g) % per_b;                                                                                         \
        (BI)    = (g) / per_b;                                                                                         \
        int nt_ = t_ % n_ntiles;                                                                                       \
        int st_ = t_ / n_ntiles;                                                                                       \
        (NN)    = nt_ * tile_n;                                                                                        \
        (SS)    = st_ * tile_s;                                                                                        \
    } while (0)

    for (int k = 0; k < stages && k < M; ++k) {
        int g = blockIdx.x + k * gridDim.x;
        int n, s, bi;
        VL_DECODE(g, n, s, bi);
        mbar_arrive_expect(mbar_a[k], tile_bytes);
        tma_load_4d(buf_a[k], &src_map, 0, n + n_src_off, s + s_src_off, bi, mbar_a[k]);
    }
    for (int j = 0; j < M; ++j) {
        int cur = j % stages;
        int g   = blockIdx.x + j * gridDim.x;
        int n, s, bi;
        VL_DECODE(g, n, s, bi);
        mbar_wait(mbar_a[cur], parity[cur]);
        parity[cur] ^= 1;
        async_proxy_fence();
        tma_store_4d(&dst.m[peer], 0, n + n_dst_off, s + s_dst_off, bi, buf_a[cur]);
        tma_commit_group();
        int nl = j + stages;
        if (nl < M) {
            tma_wait_group(
                stages
                - 1);  // equivalent to the original switch only for stages in {2,3,4,6} (-> wait_group {1,2,3,5})
            int slot = nl % stages;
            int g2   = blockIdx.x + nl * gridDim.x;
            int n2, s2, bi2;
            VL_DECODE(g2, n2, s2, bi2);
            mbar_arrive_expect(mbar_a[slot], tile_bytes);
            tma_load_4d(buf_a[slot], &src_map, 0, n2 + n_src_off, s2 + s_src_off, bi2, mbar_a[slot]);
        }
    }
    tma_wait_group(0);
#undef VL_DECODE
}

// Varlen launch: use a fixed set of reasonable defaults (varlen shapes vary widely, no autotune).
void launch_a2a_tma_varlen(const void*                  src,
                           const std::vector<uint64_t>& peer_ptrs,
                           const SplitInfo&             sp,
                           int                          mode,
                           int                          elem_size,
                           cudaStream_t                 stream)
{
    const int ws = (int)peer_ptrs.size();
    const int d = sp.d, b = sp.b, me = sp.rank;
    const int S = sp.s_off[ws], N = sp.n_off[ws];

    const int s_me = sp.s_off[me + 1] - sp.s_off[me];
    const int n_me = sp.n_off[me + 1] - sp.n_off[me];

    // varlen uses a fixed default config (shapes vary, no shape autotune).
    A2AConfig cfg;
    cfg.tile_n = 8;  // tile_n=8 splits the peer-chunk + shallow pipeline + bdiv enables pipelining
    cfg.tile_s = 1;
    cfg.stages = 4;
    cfg.bdiv   = 8;

    // Key: the "locally-full" dim must use tile=1 (or whole block), else a trailing OOB tile would write into another
    // rank's region in dst (dst only clips at that dim's full length, not at this rank's sub-segment). The peer-chunk
    // dim is correctly clipped by dst's ndim/sdim.
    // mode0: locally-full = s_me -> tile_s=1, tile_n splits the peer head chunk nchunk.
    // mode1: locally-full = n_me -> tile_n=n_me (whole block, n_ntiles=1, no trailing OOB), tile_s splits the peer seq
    // chunk schunk.
    int tile_n, tile_s;
    if (mode == 0) {
        tile_n = std::max(1, std::min(cfg.tile_n, 256));
        tile_s = 1;
    }
    else {
        tile_n = std::max(1, std::min(n_me, 256));
        tile_s = std::max(1, std::min(cfg.tile_s, 256));
    }
    const int      stages     = cfg.stages;
    const int      bdiv       = std::max(cfg.bdiv, 1);
    const uint32_t tile_bytes = (uint32_t)tile_s * tile_n * d * elem_size;
    const uint32_t tb_al      = (tile_bytes + 127u) & ~127u;
    const int      smem       = (int)(tb_al * (uint32_t)stages) + 128 + 8 * stages;

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(a2a_tma_varlen_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
        attr_set = true;
    }

    // Per-peer tile count varies with chunk size -> take the max to set grid.x; grid.y=ws, per-peer computed in-kernel
    // from blockIdx.y.
    int max_total = 1;
    for (int p = 0; p < ws; ++p) {
        int nd    = (mode == 0) ? (sp.n_off[p + 1] - sp.n_off[p]) : n_me;
        int sd    = (mode == 0) ? s_me : (sp.s_off[p + 1] - sp.s_off[p]);
        int tiles = b * ((sd + tile_s - 1) / tile_s) * ((nd + tile_n - 1) / tile_n);
        max_total = std::max(max_total, tiles);
    }
    const int blocks_x = std::max(std::min(max_total, 65535) / bdiv, 1);
    dim3      grid(blocks_x, ws, 1);

    TmaMaps dst;
    if (mode == 0) {
        CUtensorMap src_map = tma_make_map((void*)src, d, N, s_me, b, tile_n, tile_s, elem_size);  // local [b,s_me,N,d]
        for (int p = 0; p < ws; ++p) {
            int nchunk = sp.n_off[p + 1] - sp.n_off[p];
            dst.m[p] =
                tma_make_map((void*)peer_ptrs[p], d, nchunk, S, b, tile_n, tile_s, elem_size);  // peer [b,S,nchunk,d]
        }
        a2a_tma_varlen_kernel<<<grid, 1, smem, stream>>>(src_map, dst, sp, 0, tile_s, tile_n, tile_bytes, stages);
    }
    else {
        CUtensorMap src_map = tma_make_map((void*)src, d, n_me, S, b, tile_n, tile_s, elem_size);  // local [b,S,n_me,d]
        for (int p = 0; p < ws; ++p) {
            int schunk = sp.s_off[p + 1] - sp.s_off[p];
            dst.m[p] =
                tma_make_map((void*)peer_ptrs[p], d, N, schunk, b, tile_n, tile_s, elem_size);  // peer [b,schunk,N,d]
        }
        a2a_tma_varlen_kernel<<<grid, 1, smem, stream>>>(src_map, dst, sp, 1, tile_s, tile_n, tile_bytes, stages);
    }
}

}  // namespace ulysses
