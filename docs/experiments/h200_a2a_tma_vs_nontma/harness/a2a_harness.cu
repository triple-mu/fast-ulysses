// Standalone A2A harness: replicates fast_ulysses a2a_copy_generic (non-TMA) and a2a_tma_kernel (TMA)
// on raw cudaMalloc + P2P (no torch / NVSHMEM), for ncu profiling and hypothesis isolation on 8xH200.
//
// Verbs:
//   verify --mode M --ws W --N n --H h --D d            byte-exact check of both kernels vs reference
//   bench  --path {nontma|tma} --concurrent {0|1} ...    a2a timing (all ranks or single sender), optional --sweep
//   pair   --path {nontma|tma|memcpy} --mb M --local{0|1} GPU0->GPU1 flat copy BW (ws=1 trick)
//   variant --which {wo|localdst} ...                    non-TMA write-only / local-dst isolation
//
// Build: nvcc -O3 -arch=sm_90a -lineinfo -lcuda -o a2a_harness a2a_harness.cu
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <new>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_));           \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// ---------------- structs (copied from ulysses_common.cuh) ----------------
struct Ulysses4DDims {
    int32_t b, s_local, s_global, n_local, n_global, d, rank;
};
template<int WS>
struct PeerPtrs {
    void* p[WS];
};

// ---------------- TMA PTX helpers (copied verbatim from tma_ptx.cuh) ----------------
__device__ __forceinline__ void mbar_init(uint32_t mbar)
{
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" ::"r"(mbar));
}
__device__ __forceinline__ void mbar_arrive_expect(uint32_t mbar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(mbar), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint32_t mbar, int phase)
{
    asm volatile(
        "{\n .reg .pred p;\n L_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n @!p bra L_%=;\n }\n" ::"r"(mbar),
        "r"(phase));
}
__device__ __forceinline__ void
tma_load_4d(uint32_t smem, const void* map, int c0, int c1, int c2, int c3, uint32_t mbar)
{
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(smem),
                 "l"(map),
                 "r"(c0),
                 "r"(c1),
                 "r"(c2),
                 "r"(c3),
                 "r"(mbar)
                 : "memory");
}
__device__ __forceinline__ void tma_store_4d(const void* map, int c0, int c1, int c2, int c3, uint32_t smem)
{
    asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "
                 "[%0, {%1, %2, %3, %4}], [%5];" ::"l"(map),
                 "r"(c0),
                 "r"(c1),
                 "r"(c2),
                 "r"(c3),
                 "r"(smem)
                 : "memory");
}
__device__ __forceinline__ void tma_commit_group()
{
    asm volatile("cp.async.bulk.commit_group;");
}
template<int N>
__device__ __forceinline__ void tma_wait_group()
{
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void async_proxy_fence()
{
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void st_release_sys_u64(uint64_t* addr, uint64_t v)
{
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(addr), "l"(v) : "memory");
}
__device__ __forceinline__ uint64_t ld_acquire_sys_u64(const uint64_t* addr)
{
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(addr) : "memory");
    return v;
}

// ---------------- non-TMA kernel (copied verbatim from all_to_all.cu; epilogue dropped=identity) ----
// VARIANT: 0 = faithful copy; 1 = write-only (no src read, constant payload, same address stream);
//          (local-dst is a host-side change: peers point at local buffers, kernel identical.)
template<int WORLD_SIZE, int MODE, int UNROLL, int VARIANT>
__global__ void
a2a_copy_generic(const uint8_t* __restrict__ src, PeerPtrs<WORLD_SIZE> peers, Ulysses4DDims dims, int elem_size)
{
    const int     row_bytes = dims.d * elem_size;
    const int     vecs      = row_bytes >> 4;
    const int     blk_vecs  = dims.n_local * vecs;
    const int64_t units     = static_cast<int64_t>(WORLD_SIZE) * dims.b * dims.s_local;
    const int64_t total     = units * blk_vecs;
    const int64_t stride    = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t tid       = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;

    for (int64_t base = tid; base < total; base += stride * UNROLL) {
        const uint4* sp[UNROLL];
        uint4*       dp[UNROLL];
        uint4        reg[UNROLL];
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            int64_t idx = base + static_cast<int64_t>(k) * stride;
            sp[k]       = nullptr;
            if (idx >= total)
                continue;
            int     inner = static_cast<int>(idx % blk_vecs);
            int64_t u     = idx / blk_vecs;
            int     s     = static_cast<int>(u % dims.s_local);
            u /= dims.s_local;
            int b_idx = static_cast<int>(u % dims.b);
            u /= dims.b;
            int peer = static_cast<int>(u);

            int64_t src_base_row, dst_base_row;
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
            sp[k] = reinterpret_cast<const uint4*>(src + src_base_row * row_bytes) + inner;
            dp[k] = reinterpret_cast<uint4*>(static_cast<uint8_t*>(peers.p[peer]) + dst_base_row * row_bytes) + inner;
            if (VARIANT == 1)
                reg[k] = make_uint4(0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u);
            else
                reg[k] = *sp[k];
        }
#pragma unroll
        for (int k = 0; k < UNROLL; ++k)
            if (sp[k])
                *dp[k] = reg[k];
    }
    __threadfence_system();
}

// ---------------- TMA kernel (copied verbatim from all_to_all_tma.cu) ----------------
struct TmaMaps {
    CUtensorMap m[8];
};

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

template<int STAGES>
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
                               uint32_t                            tile_bytes)
{
    if (threadIdx.x != 0)
        return;

    extern __shared__ uint8_t smem_raw[];
    const uint32_t            tb_al = (tile_bytes + 127u) & ~127u;
    uintptr_t                 base  = (reinterpret_cast<uintptr_t>(smem_raw) + 127) & ~static_cast<uintptr_t>(127);
    uint64_t*                 mbar  = reinterpret_cast<uint64_t*>(base + (uintptr_t)STAGES * tb_al);

    uint32_t buf_a[STAGES], mbar_a[STAGES];
    int      parity[STAGES];
    for (int k = 0; k < STAGES; ++k) {
        buf_a[k]  = (uint32_t)__cvta_generic_to_shared(reinterpret_cast<void*>(base + (uintptr_t)k * tb_al));
        mbar_a[k] = (uint32_t)__cvta_generic_to_shared(mbar + k);
        parity[k] = 0;
        mbar_init(mbar_a[k]);
    }

    const int n_ntiles = (n_local + tile_n - 1) / tile_n;
    const int n_stiles = (s_local + tile_s - 1) / tile_s;
    const int per_peer = b * n_stiles * n_ntiles;
    const int total    = ws * per_peer;

    const int src_n_pp  = (mode == 0) ? n_local : 0;
    const int src_s_pp  = (mode == 0) ? 0 : s_local;
    const int dst_n_off = (mode == 0) ? 0 : rank * n_local;
    const int dst_s_off = (mode == 0) ? rank * s_local : 0;

    const int M = (total - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
    if (M <= 0)
        return;

    for (int k = 0; k < STAGES && k < M; ++k) {
        int g = blockIdx.x + k * gridDim.x;
        int peer, n, s, bi;
        tma_decode(g, per_peer, n_ntiles, n_stiles, tile_n, tile_s, peer, n, s, bi);
        mbar_arrive_expect(mbar_a[k], tile_bytes);
        tma_load_4d(buf_a[k], &src_map, 0, n + peer * src_n_pp, s + peer * src_s_pp, bi, mbar_a[k]);
    }

    for (int j = 0; j < M; ++j) {
        int cur = j % STAGES;
        int g   = blockIdx.x + j * gridDim.x;
        int peer, n, s, bi;
        tma_decode(g, per_peer, n_ntiles, n_stiles, tile_n, tile_s, peer, n, s, bi);

        mbar_wait(mbar_a[cur], parity[cur]);
        parity[cur] ^= 1;
        async_proxy_fence();

        tma_store_4d(&dst.m[peer], 0, n + dst_n_off, s + dst_s_off, bi, buf_a[cur]);
        tma_commit_group();

        int nl = j + STAGES;
        if (nl < M) {
            tma_wait_group<STAGES - 1>();
            int slot = nl % STAGES;
            int g2   = blockIdx.x + nl * gridDim.x;
            int peer2, n2, s2, bi2;
            tma_decode(g2, per_peer, n_ntiles, n_stiles, tile_n, tile_s, peer2, n2, s2, bi2);
            mbar_arrive_expect(mbar_a[slot], tile_bytes);
            tma_load_4d(buf_a[slot], &src_map, 0, n2 + peer2 * src_n_pp, s2 + peer2 * src_s_pp, bi2, mbar_a[slot]);
        }
    }
    tma_wait_group<0>();
}

// ---------------- barrier kernel (copied from ulysses_group.cu) ----------------
struct BarPeers {
    uint64_t p[8];
};
__global__ void ulysses_barrier_kernel(uint64_t* local, BarPeers peers, int ws, int rank, uint64_t epoch)
{
    int t = threadIdx.x;
    if (t >= ws)
        return;
    uint64_t* remote = reinterpret_cast<uint64_t*>(peers.p[t]) + rank;
    st_release_sys_u64(remote, epoch);
    uint64_t  v;
    uint64_t* mine = local + t;
    do {
        v = ld_acquire_sys_u64(mine);
    } while (v < epoch);
}

// ---------------- pattern fill / check ----------------
// value at (rank, flat idx) = 16-bit mix -> verifiable on any dst rank
__global__ void fill_pattern(uint16_t* p, int64_t n, int rank)
{
    int64_t i  = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t st = (int64_t)gridDim.x * blockDim.x;
    for (; i < n; i += st)
        p[i] = (uint16_t)((i * 2654435761u) ^ (rank * 40503u));
}
__device__ __forceinline__ uint16_t pat(int64_t i, int rank)
{
    return (uint16_t)((i * 2654435761u) ^ (rank * 40503u));
}
// mode0: dst_rank p holds out[b, r*s_local+s, j, :] = src_r[b, s, p*n_local+j, :]
// mode1: dst_rank p holds out[b, s, r*n_local+j, :] = src_r[b, p*s_local+s, j, :]   (r = src rank)
__global__ void check_mode0(const uint16_t* dst, Ulysses4DDims dims, int dst_rank, int ws, unsigned long long* errs)
{
    // dst shape (b, s_global, n_local, d)
    int64_t n  = (int64_t)dims.b * dims.s_global * dims.n_local * dims.d;
    int64_t i  = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t st = (int64_t)gridDim.x * blockDim.x;
    for (; i < n; i += st) {
        int64_t t  = i;
        int     dd = (int)(t % dims.d);
        t /= dims.d;
        int j = (int)(t % dims.n_local);
        t /= dims.n_local;
        int sg = (int)(t % dims.s_global);
        t /= dims.s_global;
        int bb = (int)t;
        int r  = sg / dims.s_local;  // src rank
        int s  = sg % dims.s_local;
        // src_r layout (b, s_local, n_global, d), row = dst_rank*n_local + j
        int64_t si =
            (((int64_t)bb * dims.s_local + s) * dims.n_global + (int64_t)dst_rank * dims.n_local + j) * dims.d + dd;
        if (dst[i] != pat(si, r))
            atomicAdd(errs, 1ull);
    }
}
__global__ void check_mode1(const uint16_t* dst, Ulysses4DDims dims, int dst_rank, int ws, unsigned long long* errs)
{
    // dst shape (b, s_local, n_global, d)
    int64_t n  = (int64_t)dims.b * dims.s_local * dims.n_global * dims.d;
    int64_t i  = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t st = (int64_t)gridDim.x * blockDim.x;
    for (; i < n; i += st) {
        int64_t t  = i;
        int     dd = (int)(t % dims.d);
        t /= dims.d;
        int ng = (int)(t % dims.n_global);
        t /= dims.n_global;
        int s = (int)(t % dims.s_local);
        t /= dims.s_local;
        int bb = (int)t;
        int r  = ng / dims.n_local;  // src rank
        int j  = ng % dims.n_local;
        // src_r layout (b, s_global, n_local, d), seq row = dst_rank*s_local + s
        int64_t si =
            (((int64_t)bb * dims.s_global + (int64_t)dst_rank * dims.s_local + s) * dims.n_local + j) * dims.d + dd;
        if (dst[i] != pat(si, r))
            atomicAdd(errs, 1ull);
    }
}

// ---------------- host-side launch glue ----------------
struct Ctx {
    int                       ws;
    int                       mode;
    Ulysses4DDims             dims;      // rank filled per launch
    std::vector<uint16_t*>    src, dst;  // per device
    std::vector<uint64_t*>    barflags;  // per device flags[8]
    std::vector<cudaStream_t> stream;
    int64_t                   numel;
    uint64_t                  epoch = 0;
};

template<int WS, int UNROLL, int VARIANT>
static void launch_nontma_wu(int dev, const Ctx& c, int blocks, int threads, bool localdst, cudaStream_t s)
{
    PeerPtrs<WS> pp;
    for (int i = 0; i < WS; ++i)
        pp.p[i] = localdst ? (void*)c.dst[dev] : (void*)c.dst[i];
    Ulysses4DDims d = c.dims;
    d.rank          = dev;
    if (c.mode == 0)
        a2a_copy_generic<WS, 0, UNROLL, VARIANT><<<blocks, threads, 0, s>>>((const uint8_t*)c.src[dev], pp, d, 2);
    else
        a2a_copy_generic<WS, 1, UNROLL, VARIANT><<<blocks, threads, 0, s>>>((const uint8_t*)c.src[dev], pp, d, 2);
}

template<int WS>
static void
launch_nontma_ws(int dev, const Ctx& c, int blocks, int threads, int unroll, int variant, bool localdst, cudaStream_t s)
{
    if (variant == 1) {
        if (unroll == 8)
            launch_nontma_wu<WS, 8, 1>(dev, c, blocks, threads, localdst, s);
        else
            launch_nontma_wu<WS, 4, 1>(dev, c, blocks, threads, localdst, s);
    }
    else {
        if (unroll == 8)
            launch_nontma_wu<WS, 8, 0>(dev, c, blocks, threads, localdst, s);
        else
            launch_nontma_wu<WS, 4, 0>(dev, c, blocks, threads, localdst, s);
    }
}

static void
launch_nontma(int dev, const Ctx& c, int blocks, int threads, int unroll, int variant, bool localdst, cudaStream_t s)
{
    switch (c.ws) {
        case 1:
            launch_nontma_ws<1>(dev, c, blocks, threads, unroll, variant, localdst, s);
            break;
        case 2:
            launch_nontma_ws<2>(dev, c, blocks, threads, unroll, variant, localdst, s);
            break;
        case 4:
            launch_nontma_ws<4>(dev, c, blocks, threads, unroll, variant, localdst, s);
            break;
        case 8:
            launch_nontma_ws<8>(dev, c, blocks, threads, unroll, variant, localdst, s);
            break;
        default:
            fprintf(stderr, "ws must be 1/2/4/8\n");
            exit(1);
    }
}

static CUtensorMap tma_make_map(void* base, int d, int ndim, int sdim, int b, int tile_n, int tile_s, int elem_size)
{
    CUtensorMap m;
    uint64_t    es       = (uint64_t)elem_size;
    uint64_t    gdims[4] = {(uint64_t)d, (uint64_t)ndim, (uint64_t)sdim, (uint64_t)b};
    uint64_t    gstr[3]  = {(uint64_t)d * es, (uint64_t)ndim * d * es, (uint64_t)sdim * ndim * d * es};
    uint32_t    box[4]   = {(uint32_t)d, (uint32_t)tile_n, (uint32_t)tile_s, 1u};
    uint32_t    estr[4]  = {1, 1, 1, 1};
    CUresult    r        = cuTensorMapEncodeTiled(&m,
                                        CU_TENSOR_MAP_DATA_TYPE_UINT16,
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
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %d\n", (int)r);
        exit(1);
    }
    return m;
}

// stages fixed at 4 (repo) unless overridden 2/8 for experiments
template<int STAGES>
static void
launch_tma_st(int dev, const Ctx& c, int tile_n, int tile_s, int bdiv, int smempad, bool localdst, cudaStream_t s)
{
    const auto& dm = c.dims;
    const int   d = dm.d, b = dm.b;
    const int   s_local = dm.s_local, n_local = dm.n_local;
    const int   s_global = dm.s_global, n_global = dm.n_global;
    tile_n                    = std::min({tile_n, n_local, 256});
    tile_s                    = std::min({tile_s, s_local, 256});
    const uint32_t tile_bytes = (uint32_t)tile_s * tile_n * d * 2;
    const uint32_t tb_al      = (tile_bytes + 127u) & ~127u;
    const int      smem       = (int)(tb_al * (uint32_t)STAGES) + 128 + 8 * STAGES + smempad;

    static bool attr_set[3] = {false, false, false};
    int         ai          = (STAGES == 2) ? 0 : (STAGES == 4 ? 1 : 2);
    if (!attr_set[ai]) {
        CK(cudaFuncSetAttribute(a2a_tma_kernel<STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024));
        attr_set[ai] = true;
    }

    const int n_stiles = (s_local + tile_s - 1) / tile_s;
    const int n_ntiles = (n_local + tile_n - 1) / tile_n;
    const int total    = c.ws * b * n_stiles * n_ntiles;
    const int blocks   = std::max(std::min(total, 65535) / bdiv, 1);

    CUtensorMap src_map;
    TmaMaps     dst;
    if (c.mode == 0) {
        src_map = tma_make_map((void*)c.src[dev], d, n_global, s_local, b, tile_n, tile_s, 2);
        for (int p = 0; p < c.ws; ++p)
            dst.m[p] =
                tma_make_map((void*)(localdst ? c.dst[dev] : c.dst[p]), d, n_local, s_global, b, tile_n, tile_s, 2);
    }
    else {
        src_map = tma_make_map((void*)c.src[dev], d, n_local, s_global, b, tile_n, tile_s, 2);
        for (int p = 0; p < c.ws; ++p)
            dst.m[p] =
                tma_make_map((void*)(localdst ? c.dst[dev] : c.dst[p]), d, n_global, s_local, b, tile_n, tile_s, 2);
    }
    a2a_tma_kernel<STAGES>
        <<<blocks, 1, smem, s>>>(src_map, dst, c.ws, c.mode, dev, s_local, n_local, b, tile_s, tile_n, tile_bytes);
}

static void launch_tma(
    int dev, const Ctx& c, int tile_n, int tile_s, int stages, int bdiv, int smempad, bool localdst, cudaStream_t s)
{
    if (stages == 2)
        launch_tma_st<2>(dev, c, tile_n, tile_s, bdiv, smempad, localdst, s);
    else if (stages == 8)
        launch_tma_st<8>(dev, c, tile_n, tile_s, bdiv, smempad, localdst, s);
    else
        launch_tma_st<4>(dev, c, tile_n, tile_s, bdiv, smempad, localdst, s);
}

static void barrier_all(Ctx& c)
{
    ++c.epoch;
    for (int i = 0; i < c.ws; ++i) {
        CK(cudaSetDevice(i));
        BarPeers bp;
        for (int j = 0; j < c.ws; ++j)
            bp.p[j] = (uint64_t)c.barflags[j];
        ulysses_barrier_kernel<<<1, 32, 0, c.stream[i]>>>(c.barflags[i], bp, c.ws, i, c.epoch);
    }
}

// ---------------- arg parsing ----------------
static int argi(int argc, char** argv, const char* k, int defv)
{
    for (int i = 0; i < argc - 1; ++i)
        if (!strcmp(argv[i], k))
            return atoi(argv[i + 1]);
    return defv;
}
static const char* args(int argc, char** argv, const char* k, const char* defv)
{
    for (int i = 0; i < argc - 1; ++i)
        if (!strcmp(argv[i], k))
            return argv[i + 1];
    return defv;
}

#define CUCHECK(x)                                                                                                     \
    do {                                                                                                               \
        CUresult r_ = (x);                                                                                             \
        if (r_ != CUDA_SUCCESS) {                                                                                      \
            const char* s = nullptr;                                                                                   \
            cuGetErrorString(r_, &s);                                                                                  \
            fprintf(stderr, "CU error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, s ? s : "?");                        \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// VMM allocation on device `dev`, mapped R/W for all `ndev` devices (single process, no IPC needed).
// handle_type: 0 = no export handle, 1 = POSIX FD, 2 = FABRIC (nvshmem's path on NVSwitch systems).
static void* vmm_alloc(int dev, size_t bytes, int ndev, int handle_type)
{
    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id         = dev;
    if (handle_type == 1)
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    if (handle_type == 2)
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    size_t gran = 0;
    CUCHECK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    size_t                       sz = (bytes + gran - 1) / gran * gran;
    CUmemGenericAllocationHandle h;
    CUresult                     cr = cuMemCreate(&h, sz, &prop, 0);
    if (cr != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(cr, &s);
        fprintf(stderr, "cuMemCreate(handle_type=%d) failed: %s\n", handle_type, s ? s : "?");
        exit(1);
    }
    CUdeviceptr va = 0;
    CUCHECK(cuMemAddressReserve(&va, sz, gran, 0, 0));
    CUCHECK(cuMemMap(va, sz, 0, h, 0));
    CUmemAccessDesc desc[8] = {};
    for (int i = 0; i < ndev; ++i) {
        desc[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc[i].location.id   = i;
        desc[i].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    CUCHECK(cuMemSetAccess(va, sz, desc, ndev));
    return (void*)va;
}

static int g_alloc_mode = 0;  // 0 = cudaMalloc, 1 = vmm(no handle), 2 = vmm posix-fd, 3 = vmm fabric

static Ctx setup(int ws, int mode, int64_t N, int H, int D, int b)
{
    Ctx c;
    c.ws    = ws;
    c.mode  = mode;
    c.dims  = {b, (int32_t)(N / ws), (int32_t)N, H / ws, H, D, 0};
    c.numel = (int64_t)b * (N / ws) * H * D;  // same numel per rank both modes
    c.src.resize(ws);
    c.dst.resize(ws);
    c.barflags.resize(ws);
    c.stream.resize(ws);
    for (int i = 0; i < ws; ++i) {
        CK(cudaSetDevice(i));
        for (int j = 0; j < ws; ++j)
            if (j != i)
                cudaDeviceEnablePeerAccess(j, 0);  // may return already-enabled; ignore
        cudaGetLastError();
        CK(cudaMalloc(&c.src[i], c.numel * 2));  // src is a plain torch-like tensor in the real bench too
        if (g_alloc_mode == 0)
            CK(cudaMalloc(&c.dst[i], c.numel * 2));
        else
            c.dst[i] = (uint16_t*)vmm_alloc(i, c.numel * 2, ws, g_alloc_mode - 1);
        CK(cudaMalloc(&c.barflags[i], 8 * sizeof(uint64_t)));
        CK(cudaMemset(c.barflags[i], 0, 8 * sizeof(uint64_t)));
        CK(cudaStreamCreate(&c.stream[i]));
        fill_pattern<<<1024, 256, 0, c.stream[i]>>>(c.src[i], c.numel, i);
        CK(cudaMemset(c.dst[i], 0xEE, c.numel * 2));
    }
    for (int i = 0; i < ws; ++i) {
        CK(cudaSetDevice(i));
        CK(cudaDeviceSynchronize());
    }
    return c;
}

struct KCfg {
    // nontma
    int threads = 512, unroll = 4, factor = 16;
    // tma
    int tile_n = 0, tile_s = 1, stages = 4, bdiv = 4;
    int smempad = 0;  // extra smem bytes per block: throttles resident blocks/SM without changing the code path
};

static int g_mixn   = 0;  // if >0: odd-numbered ranks use this tile_n instead (heterogeneous-config experiment)
int        g_queued = 0;  // 1 = enqueue all timed iters without host syncs (lib-like phase-locked lockstep)

static bool launch_path(const char* path, int dev, Ctx& c, const KCfg& k_in, int variant, bool localdst)
{
    KCfg k = k_in;
    if (g_mixn > 0 && (dev & 1))
        k.tile_n = g_mixn;
    CK(cudaSetDevice(dev));
    if (!strcmp(path, "tma")) {
        // host-side smem guard (200KB cap as in repo)
        int      tn = std::min({k.tile_n, c.dims.n_local, 256}), ts = std::min({k.tile_s, c.dims.s_local, 256});
        uint32_t tb_al = ((uint32_t)ts * tn * c.dims.d * 2 + 127u) & ~127u;
        if ((int)(tb_al * (uint32_t)k.stages) + 128 + 8 * k.stages + k.smempad > 200 * 1024)
            return false;
        launch_tma(dev, c, k.tile_n, k.tile_s, k.stages, k.bdiv, k.smempad, localdst, c.stream[dev]);
    }
    else {
        int sm = 0;
        CK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev));
        int64_t total  = (int64_t)c.ws * c.dims.b * c.dims.n_local * c.dims.s_local * (c.dims.d * 2 / 16);
        int64_t needed = (total + k.threads - 1) / k.threads;
        int     blocks = (int)std::max<int64_t>(1, std::min<int64_t>(needed, (int64_t)sm * k.factor));
        launch_nontma(dev, c, blocks, k.threads, k.unroll, variant, localdst, c.stream[dev]);
    }
    return cudaGetLastError() == cudaSuccess;  // OOR probe: mimic repo tuner's skip
}

// timing: warmup then iters; concurrent=1 launches on all ranks with the repo barrier between iters
// (matches bench_uniform's per-iter kernel+barrier cadence); events wrap the a2a kernel only.
static double bench_once(const char* path,
                         Ctx&        c,
                         const KCfg& k,
                         int         variant,
                         bool        localdst,
                         bool        concurrent,
                         int         iters,
                         double*     kernel_us_out)
{
    int                      nact = concurrent ? c.ws : 1;
    std::vector<cudaEvent_t> ev0(nact), ev1(nact);
    for (int i = 0; i < nact; ++i) {
        CK(cudaSetDevice(i));
        CK(cudaEventCreate(&ev0[i]));
        CK(cudaEventCreate(&ev1[i]));
    }
    // probe launch (bare): OOR configs are skipped, mirroring the repo tuner
    if (!launch_path(path, 0, c, k, variant, localdst))
        return -1.0;
    // warmup
    for (int w = 0; w < 3; ++w) {
        for (int i = 0; i < nact; ++i)
            launch_path(path, i, c, k, variant, localdst);
        if (concurrent && c.ws > 1)
            barrier_all(c);
    }
    for (int i = 0; i < nact; ++i) {
        CK(cudaSetDevice(i));
        CK(cudaStreamSynchronize(c.stream[i]));
    }
    // queued mode (replicates bench_uniform's timed loop): enqueue ALL iters without host syncs --
    // the barrier chain phase-locks every rank's per-iter start. Reports avg us/iter incl barrier.
    extern int g_queued;
    if (g_queued) {
        CK(cudaSetDevice(0));
        CK(cudaEventRecord(ev0[0], c.stream[0]));
        for (int it = 0; it < iters; ++it) {
            for (int i = 0; i < nact; ++i)
                launch_path(path, i, c, k, variant, localdst);
            if (concurrent && c.ws > 1)
                barrier_all(c);
        }
        CK(cudaSetDevice(0));
        CK(cudaEventRecord(ev1[0], c.stream[0]));
        for (int i = 0; i < nact; ++i) {
            CK(cudaSetDevice(i));
            CK(cudaStreamSynchronize(c.stream[i]));
        }
        float ms;
        CK(cudaEventElapsedTime(&ms, ev0[0], ev1[0]));
        for (int i = 0; i < nact; ++i) {
            CK(cudaSetDevice(i));
            CK(cudaEventDestroy(ev0[i]));
            CK(cudaEventDestroy(ev1[i]));
        }
        return ms * 1000.0 / iters;
    }
    // timed: per-iter kernel-only events on rank 0; total wall includes barrier
    std::vector<float> per_iter;
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < nact; ++i) {
            if (i == 0) {
                CK(cudaSetDevice(0));
                CK(cudaEventRecord(ev0[0], c.stream[0]));
            }
            launch_path(path, i, c, k, variant, localdst);
            if (i == 0) {
                CK(cudaSetDevice(0));
                CK(cudaEventRecord(ev1[0], c.stream[0]));
            }
        }
        if (concurrent && c.ws > 1)
            barrier_all(c);
        for (int i = 0; i < nact; ++i) {
            CK(cudaSetDevice(i));
            CK(cudaStreamSynchronize(c.stream[i]));
        }
        float ms;
        CK(cudaEventElapsedTime(&ms, ev0[0], ev1[0]));
        per_iter.push_back(ms * 1000.f);
    }
    std::sort(per_iter.begin(), per_iter.end());
    double med = per_iter[per_iter.size() / 2];
    if (kernel_us_out)
        *kernel_us_out = med;
    // Distribution tail matters: with the lockstep barrier the group runs at max-over-ranks each iter,
    // so a jittery config's effective speed is its tail, not its median.
    fprintf(stderr,
            "DIST min=%.1f p50=%.1f p90=%.1f max=%.1f us (n=%zu)\n",
            per_iter.front(),
            med,
            per_iter[(size_t)(per_iter.size() * 0.9)],
            per_iter.back(),
            per_iter.size());
    for (int i = 0; i < nact; ++i) {
        CK(cudaSetDevice(i));
        CK(cudaEventDestroy(ev0[i]));
        CK(cudaEventDestroy(ev1[i]));
    }
    return med;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s {verify|bench|pair|variant} [--k v ...]\n", argv[0]);
        return 1;
    }
    const char* verb  = argv[1];
    int         ws    = argi(argc, argv, "--ws", 8);
    int         mode  = argi(argc, argv, "--mode", 0);
    int64_t     N     = argi(argc, argv, "--N", 32768);
    int         H     = argi(argc, argv, "--H", 128);
    int         D     = argi(argc, argv, "--D", 128);
    int         b     = argi(argc, argv, "--b", 1);
    int         iters = argi(argc, argv, "--iters", 20);
    KCfg        k;
    k.threads              = argi(argc, argv, "--threads", 512);
    k.unroll               = argi(argc, argv, "--unroll", 4);
    k.factor               = argi(argc, argv, "--factor", 16);
    k.tile_n               = argi(argc, argv, "--tile_n", std::max(1, H / ws - 1));
    k.tile_s               = argi(argc, argv, "--tile_s", 1);
    k.stages               = argi(argc, argv, "--stages", 4);
    k.bdiv                 = argi(argc, argv, "--bdiv", 4);
    k.smempad              = argi(argc, argv, "--smempad", 0);
    const char* am         = args(argc, argv, "--alloc", "malloc");
    g_alloc_mode           = !strcmp(am, "vmm") ? 1 : (!strcmp(am, "posix") ? 2 : (!strcmp(am, "fabric") ? 3 : 0));
    g_mixn                 = argi(argc, argv, "--mixn", 0);
    g_queued               = argi(argc, argv, "--queued", 0);
    const char* path       = args(argc, argv, "--path", "nontma");
    bool        concurrent = argi(argc, argv, "--concurrent", 1) != 0;
    bool        localdst   = argi(argc, argv, "--localdst", 0) != 0;
    int         variant    = argi(argc, argv, "--variant", 0);

    if (!strcmp(verb, "verify")) {
        Ctx c = setup(ws, mode, N, H, D, b);
        for (const char* p : {"nontma", "tma"}) {
            for (int i = 0; i < ws; ++i) {
                CK(cudaSetDevice(i));
                CK(cudaMemset(c.dst[i], 0xEE, c.numel * 2));
            }
            for (int i = 0; i < ws; ++i) {
                CK(cudaSetDevice(i));
                CK(cudaDeviceSynchronize());  // all dst cleared before any rank writes
            }
            for (int i = 0; i < ws; ++i)
                launch_path(p, i, c, k, 0, false);
            for (int i = 0; i < ws; ++i) {
                CK(cudaSetDevice(i));
                CK(cudaDeviceSynchronize());
            }
            unsigned long long  h_err = 0;
            unsigned long long* d_err;
            for (int i = 0; i < ws; ++i) {
                CK(cudaSetDevice(i));
                CK(cudaMalloc(&d_err, 8));
                CK(cudaMemset(d_err, 0, 8));
                Ulysses4DDims dm = c.dims;
                dm.rank          = i;
                if (mode == 0)
                    check_mode0<<<1024, 256>>>(c.dst[i], dm, i, ws, d_err);
                else
                    check_mode1<<<1024, 256>>>(c.dst[i], dm, i, ws, d_err);
                unsigned long long e;
                CK(cudaMemcpy(&e, d_err, 8, cudaMemcpyDeviceToHost));
                h_err += e;
                CK(cudaFree(d_err));
            }
            printf("verify %s mode%d ws=%d N=%ld H=%d D=%d: %s (%llu errors)\n",
                   p,
                   mode,
                   ws,
                   (long)N,
                   H,
                   D,
                   h_err ? "FAIL" : "OK",
                   h_err);
        }
        return 0;
    }

    if (!strcmp(verb, "bench") || !strcmp(verb, "variant")) {
        Ctx    c         = setup(ws, mode, N, H, D, b);
        double remote_gb = (double)c.numel * 2.0 * (ws - 1) / ws;
        if (localdst)
            remote_gb = (double)c.numel * 2.0;  // all bytes go to local HBM; report total moved
        bool sweep = argi(argc, argv, "--sweep", 0) != 0;
        if (sweep && !strcmp(path, "nontma")) {
            for (int th : {256, 512, 1024})
                for (int un : {4, 8})
                    for (int f : {8, 12, 16, 24, 32}) {
                        KCfg kk    = k;
                        kk.threads = th;
                        kk.unroll  = un;
                        kk.factor  = f;
                        double us  = bench_once(path, c, kk, variant, localdst, concurrent, iters, nullptr);
                        if (us < 0)
                            printf("SWEEP nontma mode%d ws=%d N=%ld H=%d th=%d un=%d f=%d | SKIP (launch OOR)\n",
                                   mode,
                                   ws,
                                   (long)N,
                                   H,
                                   th,
                                   un,
                                   f);
                        else
                            printf("SWEEP nontma mode%d ws=%d N=%ld H=%d th=%d un=%d f=%d | %8.1f us %6.0f GB/s\n",
                                   mode,
                                   ws,
                                   (long)N,
                                   H,
                                   th,
                                   un,
                                   f,
                                   us,
                                   remote_gb / (us * 1e3));
                        fflush(stdout);
                    }
        }
        else if (sweep) {
            int                              nl = H / ws;
            std::vector<std::pair<int, int>> tiles;
            for (int tn : {std::max(1, nl - 1), std::max(1, nl / 2), nl, 8, 16, std::min(nl * 2, 256)})
                for (int ts : {1, 2, 4, 8})
                    tiles.push_back({tn, ts});
            std::sort(tiles.begin(), tiles.end());
            tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());
            for (auto [tn, ts] : tiles)
                for (int st : {2, 4, 8})
                    for (int bd : {2, 4, 8}) {
                        KCfg kk   = k;
                        kk.tile_n = tn;
                        kk.tile_s = ts;
                        kk.stages = st;
                        kk.bdiv   = bd;
                        double us = bench_once(path, c, kk, variant, localdst, concurrent, iters, nullptr);
                        if (us < 0)
                            printf("SWEEP tma mode%d ws=%d N=%ld H=%d tn=%d ts=%d stg=%d bdiv=%d | SKIP (smem/OOR)\n",
                                   mode,
                                   ws,
                                   (long)N,
                                   H,
                                   tn,
                                   ts,
                                   st,
                                   bd);
                        else
                            printf("SWEEP tma mode%d ws=%d N=%ld H=%d tn=%d ts=%d stg=%d bdiv=%d | %8.1f us %6.0f "
                                   "GB/s\n",
                                   mode,
                                   ws,
                                   (long)N,
                                   H,
                                   tn,
                                   ts,
                                   st,
                                   bd,
                                   us,
                                   remote_gb / (us * 1e3));
                        fflush(stdout);
                    }
        }
        else {
            double us = bench_once(path, c, k, variant, localdst, concurrent, iters, nullptr);
            printf("BENCH %s mode%d ws=%d N=%ld H=%d D=%d conc=%d var=%d localdst=%d th=%d un=%d f=%d tn=%d ts=%d "
                   "stg=%d bdiv=%d | %8.1f us %6.0f GB/s\n",
                   path,
                   mode,
                   ws,
                   (long)N,
                   H,
                   D,
                   (int)concurrent,
                   variant,
                   (int)localdst,
                   k.threads,
                   k.unroll,
                   k.factor,
                   k.tile_n,
                   k.tile_s,
                   k.stages,
                   k.bdiv,
                   us,
                   remote_gb / (us * 1e3));
        }
        return 0;
    }

    if (!strcmp(verb, "mp")) {
        // Multi-process replica of the lib deployment shape: ws forked processes, one GPU each,
        // peer dst/flag buffers exchanged via cudaIpc. Isolates {multi-process + IPC mappings}.
        struct Shm {
            cudaIpcMemHandle_t dst_h[8], flag_h[8];
            std::atomic<int>   ready, opened, done;
            double             us[8];
        };
        Shm* shm = (Shm*)mmap(nullptr, sizeof(Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        new (shm) Shm();
        shm->ready = shm->opened = shm->done = 0;
        for (int r = 1; r < ws; ++r)
            if (fork() == 0) {
                goto child;  // children fall through with their own rank
            }
        // parent = rank 0
    child:;
        int rank = 0;
        {
            // recover rank: count how many pids are children == fork order not tracked; use ready counter
            rank = shm->ready.fetch_add(1);
        }
        {
            CK(cudaSetDevice(rank));
            int64_t   numel = (int64_t)b * (N / ws) * H * D;
            uint16_t *src, *dst;
            uint64_t* flags;
            CK(cudaMalloc(&src, numel * 2));
            CK(cudaMalloc(&dst, numel * 2));
            CK(cudaMalloc(&flags, 8 * sizeof(uint64_t)));
            CK(cudaMemset(flags, 0, 8 * sizeof(uint64_t)));
            fill_pattern<<<1024, 256>>>(src, numel, rank);
            CK(cudaDeviceSynchronize());
            CK(cudaIpcGetMemHandle(&shm->dst_h[rank], dst));
            CK(cudaIpcGetMemHandle(&shm->flag_h[rank], flags));
            shm->opened.fetch_add(1);
            while (shm->opened.load() < ws)
                usleep(100);
            Ctx c;
            c.ws    = ws;
            c.mode  = mode;
            c.dims  = {b, (int32_t)(N / ws), (int32_t)N, H / ws, H, D, 0};
            c.numel = numel;
            c.src.assign(ws, nullptr);
            c.dst.assign(ws, nullptr);
            c.barflags.assign(ws, nullptr);
            c.stream.assign(ws, nullptr);
            c.src[rank] = src;
            for (int p = 0; p < ws; ++p) {
                if (p == rank) {
                    c.dst[p]      = dst;
                    c.barflags[p] = flags;
                    continue;
                }
                void *dp, *fp;
                CK(cudaIpcOpenMemHandle(&dp, shm->dst_h[p], cudaIpcMemLazyEnablePeerAccess));
                CK(cudaIpcOpenMemHandle(&fp, shm->flag_h[p], cudaIpcMemLazyEnablePeerAccess));
                c.dst[p]      = (uint16_t*)dp;
                c.barflags[p] = (uint64_t*)fp;
            }
            cudaStream_t st;
            CK(cudaStreamCreate(&st));
            c.stream[rank] = st;
            uint64_t epoch = 0;
            BarPeers bp;
            for (int j = 0; j < ws; ++j)
                bp.p[j] = (uint64_t)c.barflags[j];
            auto one_iter = [&] {
                launch_path(path, rank, c, k, variant, false);
                ++epoch;
                ulysses_barrier_kernel<<<1, 32, 0, st>>>(c.barflags[rank], bp, ws, rank, epoch);
            };
            for (int w = 0; w < 3; ++w)
                one_iter();
            CK(cudaStreamSynchronize(st));
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            CK(cudaEventRecord(e0, st));
            for (int it = 0; it < iters; ++it)
                one_iter();  // fully enqueued, no host sync: lib-like phase lock
            CK(cudaEventRecord(e1, st));
            CK(cudaStreamSynchronize(st));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            shm->us[rank] = ms * 1000.0 / iters;
            shm->done.fetch_add(1);
            while (shm->done.load() < ws)
                usleep(100);
            if (rank == 0) {
                double remote_gb = (double)numel * 2.0 * (ws - 1) / ws;
                double mx        = 0;
                for (int r2 = 0; r2 < ws; ++r2)
                    mx = std::max(mx, shm->us[r2]);
                printf("MP %s mode%d ws=%d N=%ld tn=%d ts=%d stg=%d bdiv=%d th=%d f=%d | rank0 %.1f us max %.1f us "
                       "%5.0f GB/s\n",
                       path,
                       mode,
                       ws,
                       (long)N,
                       k.tile_n,
                       k.tile_s,
                       k.stages,
                       k.bdiv,
                       k.threads,
                       k.factor,
                       shm->us[0],
                       mx,
                       remote_gb / (mx * 1e3));
                for (int r2 = 1; r2 < ws; ++r2)
                    wait(nullptr);
            }
            else {
                _exit(0);
            }
        }
        return 0;
    }

    if (!strcmp(verb, "seq")) {
        // Hysteresis probe: same process, config A (13 iters, like one tune candidate) then config B.
        // If B measures worse than its fresh-process time, the congested equilibrium persists across
        // config switches (tune-order contamination).
        Ctx    c         = setup(ws, mode, N, H, D, b);
        double remote_gb = (double)c.numel * 2.0 * (ws - 1) / ws;
        KCfg   ka = k, kb = k;
        ka.tile_n = argi(argc, argv, "--a_tn", 8);
        ka.tile_s = argi(argc, argv, "--a_ts", 1);
        kb.tile_n = argi(argc, argv, "--b_tn", 16);
        kb.tile_s = argi(argc, argv, "--b_ts", 4);
        for (int rep = 0; rep < 3; ++rep) {
            double ua = bench_once("tma", c, ka, 0, false, true, 13, nullptr);
            double ub = bench_once("tma", c, kb, 0, false, true, 13, nullptr);
            printf("SEQ rep%d A(tn%d,ts%d)=%.1fus %.0fGB/s -> B(tn%d,ts%d)=%.1fus %.0fGB/s\n",
                   rep,
                   ka.tile_n,
                   ka.tile_s,
                   ua,
                   remote_gb / (ua * 1e3),
                   kb.tile_n,
                   kb.tile_s,
                   ub,
                   remote_gb / (ub * 1e3));
            fflush(stdout);
        }
        return 0;
    }

    if (!strcmp(verb, "pair")) {
        // ws=1 trick: full-copy a2a, dst pointer optionally on device 1 (remote) or device 0 (local)
        int64_t mb = argi(argc, argv, "--mb", 256);
        N          = mb * 1024 * 1024 / (2 * (int64_t)H * D);  // pick s to hit requested MB
        Ctx c      = setup(1, 0, N, H, D, 1);
        // move dst to device 1 unless --local 1
        if (!localdst) {
            CK(cudaSetDevice(1));
            for (int j = 0; j < 2; ++j)
                if (j != 1)
                    cudaDeviceEnablePeerAccess(j, 0);
            cudaGetLastError();
            uint16_t* d1;
            CK(cudaMalloc(&d1, c.numel * 2));
            CK(cudaSetDevice(0));
            cudaDeviceEnablePeerAccess(1, 0);
            cudaGetLastError();
            c.dst[0] = d1;  // rank0's "peer 0" now lives on GPU1
        }
        double bytes = (double)c.numel * 2.0;
        if (!strcmp(path, "memcpy")) {
            // copy-engine DMA reference
            std::vector<float> ts;
            cudaEvent_t        e0, e1;
            CK(cudaSetDevice(0));
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            for (int i = 0; i < 3; ++i)
                CK(cudaMemcpyAsync(c.dst[0], c.src[0], c.numel * 2, cudaMemcpyDefault, c.stream[0]));
            CK(cudaStreamSynchronize(c.stream[0]));
            for (int i = 0; i < iters; ++i) {
                CK(cudaEventRecord(e0, c.stream[0]));
                CK(cudaMemcpyAsync(c.dst[0], c.src[0], c.numel * 2, cudaMemcpyDefault, c.stream[0]));
                CK(cudaEventRecord(e1, c.stream[0]));
                CK(cudaStreamSynchronize(c.stream[0]));
                float ms;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                ts.push_back(ms * 1000.f);
            }
            std::sort(ts.begin(), ts.end());
            printf("PAIR memcpy local=%d %ldMB | %8.1f us %6.0f GB/s\n",
                   (int)localdst,
                   (long)mb,
                   ts[ts.size() / 2],
                   bytes / (ts[ts.size() / 2] * 1e3));
        }
        else {
            double us = bench_once(path, c, k, variant, false, false, iters, nullptr);
            printf("PAIR %s local=%d %ldMB th=%d un=%d f=%d tn=%d ts=%d stg=%d bdiv=%d | %8.1f us %6.0f GB/s\n",
                   path,
                   (int)localdst,
                   (long)mb,
                   k.threads,
                   k.unroll,
                   k.factor,
                   k.tile_n,
                   k.tile_s,
                   k.stages,
                   k.bdiv,
                   us,
                   bytes / (us * 1e3));
        }
        return 0;
    }

    fprintf(stderr, "unknown verb %s\n", verb);
    return 1;
}
