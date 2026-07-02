// mode0 Ulysses all-to-all with QK RMSNorm + RoPE FUSED into the scatter itself: each source d-row is
// normed+roped in registers/smem and written straight to its peer destination (single kernel, no separate
// transform pass). per-head norm reduces within the d-row (fully in-kernel); cross-head norm needs the whole
// token, so a tiny [b*s_local] inv-rms pre-pass runs first and the scatter reads it. Semantics match Wan
// (fp32 RMSNorm, eps inside rsqrt, per-channel weight; RoPE over full d, interleaved/non).
#include "qk_norm_rope.cuh"
#include "ulysses_common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace ulysses {

// Per-token cross-head inv-RMS: one block per (b,s_local) token, reduce over n*d. Output inv_rms[b*s_local].
template<typename scalar_t>
__global__ void token_inv_rms_kernel(const scalar_t* __restrict__ x, float* __restrict__ inv_rms, int n, int d, float eps)
{
    const long token = blockIdx.x;
    const int  nd    = n * d;
    const long base  = token * static_cast<long>(nd);
    const int  tid   = threadIdx.x;
    float      local = 0.f;
    for (int j = tid; j < nd; j += blockDim.x) {
        const float v = static_cast<float>(x[base + j]);
        local += v * v;
    }
    for (int o = 16; o > 0; o >>= 1)
        local += __shfl_down_sync(0xffffffffu, local, o);
    __shared__ float wsum[32];
    const int        lane = tid & 31, wid = tid >> 5;
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

// Fused mode0 scatter: block per SOURCE d-row (blockDim.x == d). Source layout [b,s_local,n_global,d],
// dest layout [b,s_global,n_local,d]. Applies norm(+rope) then P2P-writes to peer = h/n_local.
template<typename scalar_t, bool CROSS_HEAD, bool INTERLEAVED>
__global__ void a2a_qk_scatter_kernel(const scalar_t* __restrict__ x,
                                      const float* __restrict__    inv_rms,  // [b*s_local] or null
                                      PeerPtrs<8>                  peers,
                                      const float* __restrict__    weight,
                                      const float* __restrict__    cosb,
                                      const float* __restrict__    sinb,
                                      int                          s_local,
                                      int                          s_global,
                                      int                          n_global,
                                      int                          n_local,
                                      int                          d,
                                      int                          rank,
                                      float                        eps)
{
    const long row  = blockIdx.x;  // over [b*s_local*n_global)
    const int  c    = threadIdx.x;
    const int  half = d >> 1;
    const int  h    = static_cast<int>(row % n_global);
    long       tmp  = row / n_global;
    const int  s    = static_cast<int>(tmp % s_local);
    const int  bidx = static_cast<int>(tmp / s_local);
    const int  peer = h / n_local;
    const int  lh   = h - peer * n_local;  // local head at dest

    const long src_row = (static_cast<long>(bidx) * s_local + s) * n_global + h;
    const long dst_row = (static_cast<long>(bidx) * s_global + (rank * s_local + s)) * n_local + lh;
    const scalar_t* sp = x + src_row * d;
    scalar_t*       dp = reinterpret_cast<scalar_t*>(peers.p[peer]) + dst_row * d;

    extern __shared__ float sm[];  // d floats
    float                   v = static_cast<float>(sp[c]);

    if (CROSS_HEAD) {
        v = v * inv_rms[bidx * s_local + s] * weight[h * d + c];
    }
    else {
        sm[c] = v * v;
        __syncthreads();
        for (int stride = half; stride > 0; stride >>= 1) {
            if (c < stride)
                sm[c] += sm[c + stride];
            __syncthreads();
        }
        const float inv = rsqrtf(sm[0] / d + eps);
        __syncthreads();
        v = v * inv * weight[c];
    }

    // RoPE (per d-row) on the normed value.
    sm[c] = v;
    __syncthreads();
    float o;
    if (INTERLEAVED) {
        const int   i  = c >> 1;
        const float cs = cosb[s * half + i], sn = sinb[s * half + i];
        o = ((c & 1) == 0) ? (sm[c] * cs - sm[c + 1] * sn) : (sm[c] * cs + sm[c - 1] * sn);
    }
    else {
        if (c < half) {
            const float cs = cosb[s * half + c], sn = sinb[s * half + c];
            o = sm[c] * cs - sm[c + half] * sn;
        }
        else {
            const int   i  = c - half;
            const float cs = cosb[s * half + i], sn = sinb[s * half + i];
            o = sm[c] * cs + sm[i] * sn;
        }
    }
    dp[c] = static_cast<scalar_t>(o);
    __threadfence_system();  // P2P direct write visibility to the peer GPU
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
                            cudaStream_t                 stream)
{
    PeerPtrs<8> peers{};
    for (size_t i = 0; i < peer_ptrs.size(); ++i)
        peers.p[i] = reinterpret_cast<void*>(peer_ptrs[i]);
    const long blocks = static_cast<long>(dims.b) * dims.s_local * dims.n_global;
    const int  smem   = dims.d * static_cast<int>(sizeof(float));
    const int  d = dims.d, sl = dims.s_local, sg = dims.s_global, ng = dims.n_global, nl = dims.n_local, rk = dims.rank;
    if (cross_head && interleaved)
        a2a_qk_scatter_kernel<scalar_t, true, true><<<blocks, d, smem, stream>>>(x, inv_rms, peers, weight, cosb, sinb, sl, sg, ng, nl, d, rk, eps);
    else if (cross_head)
        a2a_qk_scatter_kernel<scalar_t, true, false><<<blocks, d, smem, stream>>>(x, inv_rms, peers, weight, cosb, sinb, sl, sg, ng, nl, d, rk, eps);
    else if (interleaved)
        a2a_qk_scatter_kernel<scalar_t, false, true><<<blocks, d, smem, stream>>>(x, inv_rms, peers, weight, cosb, sinb, sl, sg, ng, nl, d, rk, eps);
    else
        a2a_qk_scatter_kernel<scalar_t, false, false><<<blocks, d, smem, stream>>>(x, inv_rms, peers, weight, cosb, sinb, sl, sg, ng, nl, d, rk, eps);
}

// cross-head only: fill inv_rms[b*s_local] from the source [b,s_local,n_global,d].
void launch_token_inv_rms(const void* x, float* inv_rms, const Ulysses4DDims& dims, float eps, at::ScalarType dtype, cudaStream_t stream)
{
    const long blocks = static_cast<long>(dims.b) * dims.s_local;
    if (dtype == at::kHalf)
        token_inv_rms_kernel<at::Half><<<blocks, 256, 0, stream>>>(static_cast<const at::Half*>(x), inv_rms, dims.n_global, dims.d, eps);
    else
        token_inv_rms_kernel<at::BFloat16><<<blocks, 256, 0, stream>>>(static_cast<const at::BFloat16*>(x), inv_rms, dims.n_global, dims.d, eps);
    ULYSSES_CUDA_CHECK(cudaGetLastError());
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
                   cudaStream_t                 stream)
{
    if (dtype == at::kHalf)
        launch_typed_qk<at::Half>(static_cast<const at::Half*>(src), inv_rms, peer_ptrs, weight, cosb, sinb, dims, cross_head, interleaved, eps, stream);
    else
        launch_typed_qk<at::BFloat16>(static_cast<const at::BFloat16*>(src), inv_rms, peer_ptrs, weight, cosb, sinb, dims, cross_head, interleaved, eps, stream);
    ULYSSES_CUDA_CHECK(cudaGetLastError());
}

}  // namespace ulysses
