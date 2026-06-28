#include "ulysses_common.cuh"
#include <algorithm>
#include <cuda_runtime.h>

namespace ulysess {

// grid-stride，每线程拷一个 uint4（16B）。连续线程拷连续 uint4 → 合并访存。
template<int WORLD_SIZE, int MODE, typename Epilogue>
__global__ void a2a_copy_generic(
    const uint8_t* __restrict__ src, PeerPtrs<WORLD_SIZE> peers, Ulysses4DDims dims, int elem_size, Epilogue)
{
    const int     row_bytes     = dims.d * elem_size;  // 16B 对齐（Global Constraints 保证）
    const int     vecs          = row_bytes >> 4;      // 每行 uint4 个数
    const int64_t rows_per_peer = static_cast<int64_t>(dims.b) * dims.n_local * dims.s_local;
    const int64_t total         = static_cast<int64_t>(WORLD_SIZE) * rows_per_peer * vecs;

    for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int     v = static_cast<int>(idx % vecs);
        int64_t r = idx / vecs;  // 行号 ∈ [WS, b, n_local, s_local]
        int     s = static_cast<int>(r % dims.s_local);
        r /= dims.s_local;
        int nl = static_cast<int>(r % dims.n_local);
        r /= dims.n_local;
        int b_idx = static_cast<int>(r % dims.b);
        r /= dims.b;
        int peer = static_cast<int>(r);  // 0..WS-1

        int64_t src_row, dst_row;
        if (MODE == 0) {
            int src_n_global = peer * dims.n_local + nl;
            src_row          = (static_cast<int64_t>(b_idx) * dims.s_local + s) * dims.n_global + src_n_global;
            int dst_s_global = dims.rank * dims.s_local + s;
            dst_row          = (static_cast<int64_t>(b_idx) * dims.s_global + dst_s_global) * dims.n_local + nl;
        }
        else {
            int src_s_global = peer * dims.s_local + s;
            src_row          = (static_cast<int64_t>(b_idx) * dims.s_global + src_s_global) * dims.n_local + nl;
            int dst_n_global = dims.rank * dims.n_local + nl;
            dst_row          = (static_cast<int64_t>(b_idx) * dims.s_local + s) * dims.n_global + dst_n_global;
        }
        const uint4* sp = reinterpret_cast<const uint4*>(src + src_row * row_bytes);
        uint4*       dp = reinterpret_cast<uint4*>(static_cast<uint8_t*>(peers.p[peer]) + dst_row * row_bytes);
        dp[v]           = sp[v];
    }
    __threadfence_system();  // P2P 直写对其他 GPU 的 system-scope 可见性
}

template<int WS>
static void launch_ws(const uint8_t*               src,
                      const std::vector<uint64_t>& peer_ptrs,
                      const Ulysses4DDims&         dims,
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
        a2a_copy_generic<WS, 0, EpilogueIdentity>
            <<<blocks, threads, 0, stream>>>(src, pp, dims, elem_size, EpilogueIdentity{});
    else
        a2a_copy_generic<WS, 1, EpilogueIdentity>
            <<<blocks, threads, 0, stream>>>(src, pp, dims, elem_size, EpilogueIdentity{});
}

// 当前设备的 SM 数：进程内只查询一次并缓存（每进程绑定单一设备）。
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

void launch_a2a(const void*                  src,
                const std::vector<uint64_t>& peer_ptrs,
                const Ulysses4DDims&         dims,
                int                          mode,
                int                          elem_size,
                cudaStream_t                 stream)
{
    const int     ws        = static_cast<int>(peer_ptrs.size());
    const int     threads   = 256;
    const int64_t row_bytes = static_cast<int64_t>(dims.d) * elem_size;
    const int     vecs      = static_cast<int>(row_bytes >> 4);
    const int64_t total     = static_cast<int64_t>(ws) * dims.b * dims.n_local * dims.s_local * vecs;
    const int64_t needed    = (total + threads - 1) / threads;
    // 按 SM 数封顶 grid：每线程 grid-stride 处理多行/多个 uint4（1-CTA-多行），
    // 把 __threadfence_system 调用次数从 ~needed 降到 ~cap，并摊薄索引开销。
    const int64_t cap    = static_cast<int64_t>(sm_count_cached()) * 8;
    int           blocks = static_cast<int>(std::min<int64_t>(needed, cap));
    blocks               = std::max(blocks, 1);
    const uint8_t* s     = static_cast<const uint8_t*>(src);
    switch (ws) {
        case 1:
            launch_ws<1>(s, peer_ptrs, dims, mode, elem_size, blocks, threads, stream);
            break;
        case 2:
            launch_ws<2>(s, peer_ptrs, dims, mode, elem_size, blocks, threads, stream);
            break;
        case 4:
            launch_ws<4>(s, peer_ptrs, dims, mode, elem_size, blocks, threads, stream);
            break;
        case 8:
            launch_ws<8>(s, peer_ptrs, dims, mode, elem_size, blocks, threads, stream);
            break;
        default:
            break;  // 调用方已 TORCH_CHECK(ws ∈ {1,2,4,8})
    }
}

// ---- 变长（uneven s/n）源路由 kernel：遍历本地输入，每元素按 split 偏移找归属 peer 直写 ----
template<int WORLD_SIZE, int MODE, typename Epilogue>
__global__ void
a2a_copy_varlen(const uint8_t* __restrict__ src, PeerPtrs<WORLD_SIZE> peers, SplitInfo sp, int elem_size, Epilogue)
{
    const int row_bytes = sp.d * elem_size;
    const int vecs      = row_bytes >> 4;
    const int me        = sp.rank;
    const int S         = sp.s_off[WORLD_SIZE];
    const int N         = sp.n_off[WORLD_SIZE];

    if (MODE == 0) {
        // 本地输入 [b, s_me, N, d]；me 序列块全局起点 s_base，落到各 peer 输出 [b, S, nchunk, d]。
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
            int peer  = 0;  // 头 h 的归属 peer：n_off[peer] ≤ h < n_off[peer+1]
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
        // 本地输入 [b, S, n_me, d]；me 头块全局起点 n_base，落到各 peer 输出 [b, schunk, N, d]。
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
            int peer  = 0;  // 序列 sg 的归属 peer：s_off[peer] ≤ sg < s_off[peer+1]
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
    const int     ws        = static_cast<int>(peer_ptrs.size());
    const int     threads   = 256;
    const int64_t row_bytes = static_cast<int64_t>(sp.d) * elem_size;
    const int     vecs      = static_cast<int>(row_bytes >> 4);
    const int     me        = sp.rank;
    const int     S         = sp.s_off[ws];
    const int     N         = sp.n_off[ws];
    const int64_t rows      = (mode == 0) ? static_cast<int64_t>(sp.b) * (sp.s_off[me + 1] - sp.s_off[me]) * N :
                                            static_cast<int64_t>(sp.b) * S * (sp.n_off[me + 1] - sp.n_off[me]);
    const int64_t total     = rows * vecs;
    const int64_t needed    = (total + threads - 1) / threads;
    const int64_t cap       = static_cast<int64_t>(sm_count_cached()) * 8;
    int           blocks    = static_cast<int>(std::min<int64_t>(needed, cap));
    blocks                  = std::max(blocks, 1);
    const uint8_t* s        = static_cast<const uint8_t*>(src);
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
}

}  // namespace ulysess
