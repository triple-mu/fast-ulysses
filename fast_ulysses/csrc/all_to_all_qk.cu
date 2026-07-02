// mode0 Ulysses all-to-all with QK RMSNorm + RoPE FUSED into the scatter, restructured to be isomorphic
// to the fast non-TMA a2a kernel (all_to_all.cu): grid-stride over uint4 PAIRS, UNROLL register-prefetch
// pipeline (local reads hidden behind remote NVLink writes), autotuned threads/unroll/blocks. Each
// thread-iteration owns the pair (vec j, vec j+vecs/2) of one source d-row -- 32B -- so both RoPE
// pairings resolve in registers: NeoX partners sit d/2 apart = exactly the other vec of the pair; GPT-J
// partners are adjacent inside a vec. cross-head norm (Wan default) reads a precomputed per-token
// inv-rms (vectorized pre-pass below) -> the scatter is pure elementwise; per-head norm reduces its
// d-row in-kernel with segmented warp shuffles (a row = vecs/2 consecutive lanes; no smem, no
// __syncthreads). Semantics match Wan (fp32 RMSNorm, eps inside rsqrt, per-channel weight; RoPE over
// full d, interleaved/non). Replaces the old one-block-per-d-row scalar kernel whose 2B/thread remote
// writes collapsed NVLink efficiency.
#include "qk_norm_rope.cuh"
#include "ulysses_common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

namespace ulysses {

namespace {

constexpr int kVecElems = 8;  // elems per uint4 (fp16/bf16 only; 16B / 2B)

// 2-wide hardware converts (halves the cvt instruction count vs scalar static_cast; both are RNE,
// matching the torch reference).
template<typename scalar_t>
struct V2;
template<>
struct V2<at::Half> {
    using t2 = __half2;
    static __device__ __forceinline__ float2 tof(t2 v) { return __half22float2(v); }
    static __device__ __forceinline__ t2     fromf(float2 f) { return __float22half2_rn(f); }
};
template<>
struct V2<at::BFloat16> {
    using t2 = __nv_bfloat162;
    static __device__ __forceinline__ float2 tof(t2 v) { return __bfloat1622float2(v); }
    static __device__ __forceinline__ t2     fromf(float2 f) { return __float22bfloat162_rn(f); }
};

// fp32 unpack/pack of one uint4 (8 elems = 4 packed pairs).
template<typename scalar_t>
__device__ __forceinline__ void unpack8(const uint4& v, float* f)
{
    using t2     = typename V2<scalar_t>::t2;
    const t2* e  = reinterpret_cast<const t2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 p = V2<scalar_t>::tof(e[i]);
        f[2 * i]     = p.x;
        f[2 * i + 1] = p.y;
    }
}

template<typename scalar_t>
__device__ __forceinline__ uint4 pack8(const float* f)
{
    using t2 = typename V2<scalar_t>::t2;
    uint4 v;
    t2*   e = reinterpret_cast<t2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        e[i] = V2<scalar_t>::fromf(make_float2(f[2 * i], f[2 * i + 1]));
    return v;
}

// Vectorized fp32 loads for weight/cos/sin. All call sites are 16B-aligned by construction:
// offsets are multiples of 4 (c0=8j, half>=8, f0=4j) on fp32 arrays the caller checks contiguous.
__device__ __forceinline__ void load_f4(float* dst, const float* __restrict__ p)
{
    const float4 a = *reinterpret_cast<const float4*>(p);
    dst[0] = a.x, dst[1] = a.y, dst[2] = a.z, dst[3] = a.w;
}

__device__ __forceinline__ void load_f8(float* dst, const float* __restrict__ p)
{
    load_f4(dst, p);
    load_f4(dst + 4, p + 4);
}

}  // namespace

// Per-token cross-head inv-RMS pre-pass: one block per (b,s_local) token, uint4-vectorized read of the
// token's n*d elements, warp+block shuffle reduction. Output inv_rms[b*s_local] = rsqrt(mean+eps).
template<typename scalar_t>
__global__ void token_inv_rms_kernel(const scalar_t* __restrict__ x, float* __restrict__ inv_rms, int nd_vecs, int nd, float eps)
{
    const int64_t token = blockIdx.x;
    const uint4*  p     = reinterpret_cast<const uint4*>(x) + token * nd_vecs;
    float         local = 0.f;
    for (int v = threadIdx.x; v < nd_vecs; v += blockDim.x) {
        const uint4     q = p[v];
        const scalar_t* e = reinterpret_cast<const scalar_t*>(&q);
#pragma unroll
        for (int i = 0; i < kVecElems; ++i) {
            const float f = static_cast<float>(e[i]);
            local += f * f;
        }
    }
    for (int o = 16; o > 0; o >>= 1)
        local += __shfl_down_sync(0xffffffffu, local, o);
    __shared__ float wsum[32];
    const int        lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0)
        wsum[wid] = local;
    __syncthreads();
    if (wid == 0) {
        const int nw = blockDim.x >> 5;
        float     t  = (lane < nw) ? wsum[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1)
            t += __shfl_down_sync(0xffffffffu, t, o);
        if (lane == 0)
            inv_rms[token] = rsqrtf(t / nd + eps);
    }
}

// Fast path for the Wan default (cross-head norm + GPT-J interleaved RoPE): the epilogue is pure
// elementwise (inv-rms precomputed, GPT-J pairs live INSIDE each uint4), so the index space and memory
// pattern are IDENTICAL to a2a_copy_generic -- one uint4 per thread-iteration, contiguous within the
// (peer,b,s) unit, per-lane grid-stride loop, UNROLL up to 8. Roughly half the instructions of the
// generic pair kernel below (no second stream, no cross-vec coupling).
template<typename scalar_t, int UNROLL>
__global__ void a2a_qk_chil_kernel(const scalar_t* __restrict__ x,
                                   const float* __restrict__    inv_rms,  // [b*s_local]
                                   PeerPtrs<8>                  peers,
                                   const float* __restrict__    weight,   // [n_global*d]
                                   const float* __restrict__    cosb,
                                   const float* __restrict__    sinb,
                                   Ulysses4DDims                dims,
                                   int                          ws)
{
    const int     vecs     = dims.d >> 3;            // uint4 per d-row (power of two)
    const int     vshift   = 31 - __clz(vecs);       // log2(vecs)
    const int     blk_vecs = dims.n_local * vecs;
    const int     half     = dims.d >> 1;
    const int64_t total    = static_cast<int64_t>(ws) * dims.b * dims.s_local * blk_vecs;
    const int64_t stride   = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t tid      = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;

    for (int64_t base = tid; base < total; base += stride * UNROLL) {
        uint4  reg[UNROLL];
        uint4* dp[UNROLL];
        float  inv[UNROLL], w[UNROLL][kVecElems], cs[UNROLL][4], sn[UNROLL][4];
        // Read phase (local HBM + cached params): issue ALL loads -- data AND epilogue params -- up
        // front so their latencies overlap; the write phase then runs on registers only and the remote
        // writes issue back-to-back (a param L2 miss no longer sits on the write path).
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            const int64_t idx = base + static_cast<int64_t>(k) * stride;
            dp[k]             = nullptr;
            if (idx >= total)
                continue;
            const int inner = static_cast<int>(idx % blk_vecs);
            int64_t   u     = idx / blk_vecs;
            const int s     = static_cast<int>(u % dims.s_local);
            u /= dims.s_local;
            const int     bi      = static_cast<int>(u % dims.b);
            const int     peer    = static_cast<int>(u / dims.b);
            const int     nl      = inner >> vshift;
            const int     v       = inner & (vecs - 1);
            const int64_t src_row = (static_cast<int64_t>(bi) * dims.s_local + s) * dims.n_global + peer * dims.n_local + nl;
            const int64_t dst_row = (static_cast<int64_t>(bi) * dims.s_global + (dims.rank * dims.s_local + s)) * dims.n_local + nl;
            reg[k]                = reinterpret_cast<const uint4*>(x)[src_row * vecs + v];
            dp[k]                 = reinterpret_cast<uint4*>(peers.p[peer]) + dst_row * vecs + v;
            inv[k]                = __ldg(inv_rms + static_cast<int64_t>(bi) * dims.s_local + s);
            load_f8(w[k], weight + static_cast<int64_t>(peer * dims.n_local + nl) * dims.d + (v << 3));
            const int64_t fb = static_cast<int64_t>(s) * half + (v << 2);  // freq base = (8v)/2
            load_f4(cs[k], cosb + fb);
            load_f4(sn[k], sinb + fb);
        }
        // Compute + write phase: register-only epilogue, remote NVLink write per uint4
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            if (!dp[k])
                continue;
            float xv[kVecElems];
            unpack8<scalar_t>(reg[k], xv);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float a = xv[2 * t] * inv[k] * w[k][2 * t];
                const float b = xv[2 * t + 1] * inv[k] * w[k][2 * t + 1];
                xv[2 * t]     = a * cs[k][t] - b * sn[k][t];
                xv[2 * t + 1] = a * sn[k][t] + b * cs[k][t];
            }
            *dp[k] = pack8<scalar_t>(xv);
        }
    }
    __threadfence_system();  // system-scope visibility of P2P direct writes to other GPUs
}

// Generic fused mode0 scatter (per-head norm and/or NeoX rope). Index space: [ws, b, s_local, n_local,
// vh] where vh = (d/8)/2 = uint4 pairs per d-row; one thread-iteration = one pair (vec j, vec j+vh),
// norm(+rope)'d in fp32 registers and P2P-written to the peer. The loop condition is warp-uniform
// (base - lane stays constant across the warp) so every lane reaches the per-head __shfl_xor_sync in
// step; out-of-range slots carry zeroes and skip memory. A row's vh lanes are warp-aligned (vh is a
// power of two <= 32; blockDim and stride are multiples of vh) and total % vh == 0, so a row's lanes
// are all-valid or all-invalid.
template<typename scalar_t, int UNROLL, bool CROSS_HEAD, bool INTERLEAVED>
__global__ void a2a_qk_fused_kernel(const scalar_t* __restrict__ x,
                                    const float* __restrict__    inv_rms,  // [b*s_local] or null
                                    PeerPtrs<8>                  peers,
                                    const float* __restrict__    weight,
                                    const float* __restrict__    cosb,
                                    const float* __restrict__    sinb,
                                    Ulysses4DDims                dims,
                                    int                          ws,
                                    float                        eps)
{
    constexpr int E      = kVecElems;
    const int     d      = dims.d;
    const int     half   = d >> 1;
    const int     vh     = half / E;         // >= 1 (caller checks d >= 16), power of two
    const int     vhshift = 31 - __clz(vh);  // log2(vh)
    const int     row_pairs = dims.n_local * vh;
    const int64_t total  = static_cast<int64_t>(ws) * dims.b * dims.s_local * row_pairs;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t tid    = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int     lane   = threadIdx.x & 31;

    for (int64_t base = tid; base - lane < total; base += stride * UNROLL) {
        uint4   lo[UNROLL], hi[UNROLL];
        uint4*  dp[UNROLL];  // dst pair base in peer memory; null = out of range
        int     sj[UNROLL], hj[UNROLL], jj[UNROLL];
        int64_t tok[UNROLL];
        // Read phase (local HBM): prefetch UNROLL pairs into registers
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            const int64_t idx = base + static_cast<int64_t>(k) * stride;
            dp[k]             = nullptr;
            sj[k] = hj[k] = jj[k] = 0;
            tok[k]                = 0;
            lo[k] = hi[k] = uint4{0u, 0u, 0u, 0u};
            if (idx >= total)
                continue;
            const int ip = static_cast<int>(idx % row_pairs);
            int64_t   u  = idx / row_pairs;
            const int nl = ip >> vhshift, j = ip & (vh - 1);
            const int s  = static_cast<int>(u % dims.s_local);
            u /= dims.s_local;
            const int     bi      = static_cast<int>(u % dims.b);
            const int     peer    = static_cast<int>(u / dims.b);
            const int64_t src_row = (static_cast<int64_t>(bi) * dims.s_local + s) * dims.n_global + peer * dims.n_local + nl;
            const int64_t dst_row = (static_cast<int64_t>(bi) * dims.s_global + (dims.rank * dims.s_local + s)) * dims.n_local + nl;
            const uint4*  sp      = reinterpret_cast<const uint4*>(x) + src_row * (2 * vh) + j;
            dp[k]                 = reinterpret_cast<uint4*>(peers.p[peer]) + dst_row * (2 * vh) + j;
            lo[k]                 = sp[0];
            hi[k]                 = sp[vh];
            sj[k]                 = s;
            hj[k]                 = peer * dims.n_local + nl;
            jj[k]                 = j;
            tok[k]                = static_cast<int64_t>(bi) * dims.s_local + s;
        }
        // Compute + write phase: epilogue in registers, remote NVLink write per pair
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            float xl[E], xh[E];
            unpack8<scalar_t>(lo[k], xl);
            unpack8<scalar_t>(hi[k], xh);

            float inv;
            if (CROSS_HEAD) {
                inv = dp[k] ? __ldg(inv_rms + tok[k]) : 0.f;
            }
            else {
                float sq = 0.f;
#pragma unroll
                for (int e = 0; e < E; ++e)
                    sq += xl[e] * xl[e] + xh[e] * xh[e];
                // segmented reduce over the row's vh lanes (invalid rows are all-invalid -> harmless)
                for (int o = vh >> 1; o > 0; o >>= 1)
                    sq += __shfl_xor_sync(0xffffffffu, sq, o);
                inv = rsqrtf(sq / d + eps);
            }

            const int    c0   = jj[k] * E;
            const float* wrow = weight + (CROSS_HEAD ? static_cast<int64_t>(hj[k]) * d : 0);
            float        wl[E], wh[E];
            load_f8(wl, wrow + c0);
            load_f8(wh, wrow + c0 + half);
#pragma unroll
            for (int e = 0; e < E; ++e) {
                xl[e] *= inv * wl[e];
                xh[e] *= inv * wh[e];
            }

            const float* cb = cosb + static_cast<int64_t>(sj[k]) * half;
            const float* sb = sinb + static_cast<int64_t>(sj[k]) * half;
            if (INTERLEAVED) {
                // pairs (2i, 2i+1) live inside each vec; freq index i = element/2
                const int f0 = c0 >> 1, f1 = (c0 + half) >> 1;
                float     cl[E / 2], sl[E / 2], ch[E / 2], sh[E / 2];
                load_f4(cl, cb + f0);
                load_f4(sl, sb + f0);
                load_f4(ch, cb + f1);
                load_f4(sh, sb + f1);
#pragma unroll
                for (int t = 0; t < E / 2; ++t) {
                    const float a = xl[2 * t], b = xl[2 * t + 1];
                    xl[2 * t]      = a * cl[t] - b * sl[t];
                    xl[2 * t + 1]  = a * sl[t] + b * cl[t];
                    const float a2 = xh[2 * t], b2 = xh[2 * t + 1];
                    xh[2 * t]      = a2 * ch[t] - b2 * sh[t];
                    xh[2 * t + 1]  = a2 * sh[t] + b2 * ch[t];
                }
            }
            else {
                // NeoX: pair (c, c+half) = (xl[e], xh[e]) of this very vec pair; freq index = c
                float cs[E], sn[E];
                load_f8(cs, cb + c0);
                load_f8(sn, sb + c0);
#pragma unroll
                for (int e = 0; e < E; ++e) {
                    const float a = xl[e], b = xh[e];
                    xl[e] = a * cs[e] - b * sn[e];
                    xh[e] = b * cs[e] + a * sn[e];
                }
            }

            if (dp[k]) {
                dp[k][0]  = pack8<scalar_t>(xl);
                dp[k][vh] = pack8<scalar_t>(xh);
            }
        }
    }
    __threadfence_system();  // system-scope visibility of P2P direct writes to other GPUs
}

template<typename scalar_t, int UNROLL>
static void launch_qk_u(const scalar_t*      x,
                        const float*         inv_rms,
                        const PeerPtrs<8>&   peers,
                        const float*         weight,
                        const float*         cosb,
                        const float*         sinb,
                        const Ulysses4DDims& dims,
                        int                  ws,
                        bool                 cross_head,
                        bool                 interleaved,
                        float                eps,
                        int                  blocks,
                        int                  threads,
                        cudaStream_t         stream)
{
    if (cross_head && interleaved)  // Wan default -> plain-isomorphic fast path (eps folded in pre-pass)
        a2a_qk_chil_kernel<scalar_t, UNROLL><<<blocks, threads, 0, stream>>>(x, inv_rms, peers, weight, cosb, sinb, dims, ws);
    else if (cross_head)
        a2a_qk_fused_kernel<scalar_t, UNROLL, true, false><<<blocks, threads, 0, stream>>>(x, inv_rms, peers, weight, cosb, sinb, dims, ws, eps);
    else if (interleaved)
        a2a_qk_fused_kernel<scalar_t, UNROLL, false, true><<<blocks, threads, 0, stream>>>(x, inv_rms, peers, weight, cosb, sinb, dims, ws, eps);
    else
        a2a_qk_fused_kernel<scalar_t, UNROLL, false, false><<<blocks, threads, 0, stream>>>(x, inv_rms, peers, weight, cosb, sinb, dims, ws, eps);
}

template<typename scalar_t>
static void launch_typed_qk(const scalar_t*              x,
                            const float*                 inv_rms,
                            const std::vector<uint64_t>& peer_ptrs,
                            const float*                 weight,
                            const float*                 cosb,
                            const float*                 sinb,
                            const Ulysses4DDims&         dims,
                            bool                         cross_head,
                            bool                         interleaved,
                            float                        eps,
                            const A2AConfig&             cfg,
                            cudaStream_t                 stream)
{
    PeerPtrs<8> peers{};
    const int   ws = static_cast<int>(peer_ptrs.size());
    for (int i = 0; i < ws; ++i)
        peers.p[i] = reinterpret_cast<void*>(peer_ptrs[i]);
    // unroll candidates are {2, 4, 8} (resolve_config_qk); only these three are instantiated.
    if (cfg.unroll >= 8)
        launch_qk_u<scalar_t, 8>(x, inv_rms, peers, weight, cosb, sinb, dims, ws, cross_head, interleaved, eps, cfg.blocks, cfg.threads, stream);
    else if (cfg.unroll >= 4)
        launch_qk_u<scalar_t, 4>(x, inv_rms, peers, weight, cosb, sinb, dims, ws, cross_head, interleaved, eps, cfg.blocks, cfg.threads, stream);
    else
        launch_qk_u<scalar_t, 2>(x, inv_rms, peers, weight, cosb, sinb, dims, ws, cross_head, interleaved, eps, cfg.blocks, cfg.threads, stream);
}

// cross-head only: fill inv_rms[b*s_local] from the source [b,s_local,n_global,d].
void launch_token_inv_rms(const void* x, float* inv_rms, const Ulysses4DDims& dims, float eps, at::ScalarType dtype, cudaStream_t stream)
{
    const long blocks  = static_cast<long>(dims.b) * dims.s_local;
    const int  nd      = dims.n_global * dims.d;
    const int  nd_vecs = nd / kVecElems;  // d >= 16 -> nd divisible by 8
    if (dtype == at::kHalf)
        token_inv_rms_kernel<at::Half><<<blocks, 256, 0, stream>>>(static_cast<const at::Half*>(x), inv_rms, nd_vecs, nd, eps);
    else
        token_inv_rms_kernel<at::BFloat16><<<blocks, 256, 0, stream>>>(static_cast<const at::BFloat16*>(x), inv_rms, nd_vecs, nd, eps);
    ULYSSES_CUDA_CHECK(cudaGetLastError());
}

// dtype dispatch WITHOUT an error check: the autotune probe launches this bare and inspects
// cudaGetLastError itself (an OOR candidate must be skipped, not thrown).
static void dispatch_qk(const void*                  src,
                        const float*                 inv_rms,
                        const std::vector<uint64_t>& peer_ptrs,
                        const float*                 weight,
                        const float*                 cosb,
                        const float*                 sinb,
                        const Ulysses4DDims&         dims,
                        bool                         cross_head,
                        bool                         interleaved,
                        float                        eps,
                        at::ScalarType               dtype,
                        const A2AConfig&             cfg,
                        cudaStream_t                 stream)
{
    if (dtype == at::kHalf)
        launch_typed_qk<at::Half>(static_cast<const at::Half*>(src), inv_rms, peer_ptrs, weight, cosb, sinb, dims, cross_head, interleaved, eps, cfg, stream);
    else
        launch_typed_qk<at::BFloat16>(static_cast<const at::BFloat16*>(src), inv_rms, peer_ptrs, weight, cosb, sinb, dims, cross_head, interleaved, eps, cfg, stream);
}

void launch_a2a_qk(const void*                  src,
                   const float*                 inv_rms,
                   const std::vector<uint64_t>& peer_ptrs,
                   const float*                 weight,
                   const float*                 cosb,
                   const float*                 sinb,
                   const Ulysses4DDims&         dims,
                   bool                         cross_head,
                   bool                         interleaved,
                   float                        eps,
                   at::ScalarType               dtype,
                   const A2AConfig&             cfg,
                   cudaStream_t                 stream)
{
    dispatch_qk(src, inv_rms, peer_ptrs, weight, cosb, sinb, dims, cross_head, interleaved, eps, dtype, cfg, stream);
    ULYSSES_CUDA_CHECK(cudaGetLastError());
}

// Fused-scatter autotune: same converged sweep skeleton as resolve_config_nontma, with unroll {2,4,8}
// (the chil fast path is uint4-per-thread like plain; the pair kernel's 32B/thread makes 8 heavy, but
// the OOR probe / timing prunes it). The index space is the chil uint4 count (the pair kernel's total
// is exactly half, so blocks derived from it stay valid for both). Each timed run is the REAL per-call
// op -- (cross-head pre-pass +) scatter + finish -- so the ranking matches steady state; the pre-pass
// is a constant term across candidates. inv_rms must be filled before entry (probe launches
// dereference it).
A2AConfig resolve_config_qk(const void*                  src,
                            const float*                 inv_rms,
                            const std::vector<uint64_t>& peer_ptrs,
                            const float*                 weight,
                            const float*                 cosb,
                            const float*                 sinb,
                            const Ulysses4DDims&         dims,
                            bool                         cross_head,
                            bool                         interleaved,
                            float                        eps,
                            at::ScalarType               dtype,
                            cudaStream_t                 stream,
                            bool                         verbose,
                            const std::function<void()>& finish)
{
    const int     ws    = static_cast<int>(peer_ptrs.size());
    const bool    chil  = cross_head && interleaved;
    const int64_t units = static_cast<int64_t>(ws) * dims.b * dims.s_local * dims.n_local;
    const int64_t total = units * (dims.d / kVecElems) / (chil ? 1 : 2);  // uint4s (chil) or pairs
    const int     sm    = sm_count_cached();

    const int    threads_cand[] = {256, 512, 1024};
    const int    unroll_cand[]  = {2, 4, 8};
    const double factors[]      = {8.0, 12.0, 16.0, 24.0, 32.0};

    auto clampb = [&](int64_t needed, int64_t want) {
        return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, want)));
    };

    A2AConfig best;
    best.threads  = 512;
    best.unroll   = 2;
    best.blocks   = clampb((total + 511) / 512, static_cast<int64_t>(sm) * 16);
    float best_us = 1e30f;
    for (int threads : threads_cand) {
        const int64_t needed = (total + threads - 1) / threads;
        for (int unroll : unroll_cand) {
            for (double f : factors) {
                const int blocks = clampb(needed, static_cast<int64_t>(sm * f));
                A2AConfig cand;
                cand.threads = threads;
                cand.unroll  = unroll;
                cand.blocks  = blocks;
                // Probe (bare scatter launch, not timed): skip candidates that fail to launch (OOR),
                // same as resolve_config_nontma. inv_rms is already filled, so the dereference is safe.
                dispatch_qk(src, inv_rms, peer_ptrs, weight, cosb, sinb, dims, cross_head, interleaved, eps, dtype, cand, stream);
                if (cudaGetLastError() != cudaSuccess)
                    continue;
                const float us = microbench_us(
                    [&] {
                        if (cross_head)
                            launch_token_inv_rms(src, const_cast<float*>(inv_rms), dims, eps, dtype, stream);
                        dispatch_qk(src, inv_rms, peer_ptrs, weight, cosb, sinb, dims, cross_head, interleaved, eps, dtype, cand, stream);
                        finish();
                    },
                    stream);
                if (us < best_us) {
                    best_us      = us;
                    best.threads = threads;
                    best.unroll  = unroll;
                    best.blocks  = blocks;
                }
            }
        }
    }
    const A2AConfig dc = default_config(0, dims.n_local);
    best.tile_n        = dc.tile_n;
    best.tile_s        = dc.tile_s;
    if (verbose) {  // all ranks: config divergence across ranks is exactly what this diagnoses
        const double remote_bytes = static_cast<double>(units) * dims.d * 2.0 * (ws - 1) / ws;
        const double gbps         = remote_bytes / (best_us * 1e3);
        std::cout << "[ulysses qk-fused tune] rank=" << dims.rank << " ws=" << ws << " cross=" << cross_head
                  << " n_local=" << dims.n_local << " s_local=" << dims.s_local << " d=" << dims.d
                  << " -> threads=" << best.threads << " unroll=" << best.unroll << " blocks=" << best.blocks
                  << " | " << std::fixed << std::setprecision(1) << best_us << " us/iter " << std::setprecision(0)
                  << gbps << " GB/s" << std::endl;
    }
    return best;
}

}  // namespace ulysses
