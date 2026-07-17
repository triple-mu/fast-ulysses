// Fused QK RMSNorm + RoPE kernels (standalone building blocks; also reused by the a2a-fused op).
// Semantics mirror Wan exactly: RMSNorm in fp32 (eps inside rsqrt, per-channel weight, no bias), then
// RoPE over the full head_dim. See qk_norm_rope.cuh for the API contract.
#include "qk_norm_rope.cuh"
#include "ulysses_common.cuh"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace ulysses {

// One block per d-row (blockDim.x == d, d a power of two). Handles per-head RMSNorm and/or RoPE.
// x/out: [b,seq,n,d]. weight: [d] (fp32) when DO_NORM. cos/sin: [seq,d/2] (fp32) when DO_ROPE.
template<typename scalar_t, bool DO_NORM, bool DO_ROPE, bool INTERLEAVED>
__global__ void per_head_kernel(const scalar_t* __restrict__ x,
                                scalar_t* __restrict__ out,
                                const float* __restrict__ weight,
                                const float* __restrict__ cosb,
                                const float* __restrict__ sinb,
                                int   seq,
                                int   n,
                                int   d,
                                float eps)
{
    const long row  = blockIdx.x;   // over [b*seq*n)
    const int  c    = threadIdx.x;  // 0..d-1
    const int  s    = static_cast<int>((row / n) % seq);
    const long base = row * d;
    const int  half = d >> 1;

    extern __shared__ float sm[];  // d floats
    float                   v = static_cast<float>(x[base + c]);

    if (DO_NORM) {
        sm[c] = v * v;
        __syncthreads();
        for (int stride = half; stride > 0; stride >>= 1) {
            if (c < stride)
                sm[c] += sm[c + stride];
            __syncthreads();
        }
        const float inv = rsqrtf(sm[0] / d + eps);
        __syncthreads();  // all threads have read sm[0] before it is overwritten below
        v = v * inv * weight[c];
    }

    if (DO_ROPE) {
        sm[c] = v;
        __syncthreads();
        float o;
        if (INTERLEAVED) {
            const int   i  = c >> 1;  // pair (2i, 2i+1)
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
        v = o;
    }
    out[base + c] = static_cast<scalar_t>(v);
}

// One block per token (b,seq); reduces sum-of-squares over the whole n*d vector (cross-head RMS), then
// applies the per-channel scale and (optionally) per-head RoPE. blockDim.x = 256. When DO_ROPE, dynamic
// shared holds the token's n*d normed values so RoPE can read its pair within each head.
template<typename scalar_t, bool DO_ROPE, bool INTERLEAVED>
__global__ void cross_head_kernel(const scalar_t* __restrict__ x,
                                  scalar_t* __restrict__ out,
                                  const float* __restrict__ weight,  // [n*d]
                                  const float* __restrict__ cosb,
                                  const float* __restrict__ sinb,
                                  int   seq,
                                  int   n,
                                  int   d,
                                  float eps)
{
    const long token = blockIdx.x;  // over [b*seq)
    const int  s     = static_cast<int>(token % seq);
    const int  nd    = n * d;
    const int  half  = d >> 1;
    const long base  = token * static_cast<long>(nd);
    const int  tid   = threadIdx.x;
    const int  nthr  = blockDim.x;

    float local = 0.f;
    for (int j = tid; j < nd; j += nthr) {
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
        const int nw = nthr >> 5;
        float     t  = (lane < nw) ? wsum[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1)
            t += __shfl_down_sync(0xffffffffu, t, o);
        if (lane == 0)
            wsum[0] = t;
    }
    __syncthreads();
    const float inv = rsqrtf(wsum[0] / nd + eps);

    if (!DO_ROPE) {
        for (int j = tid; j < nd; j += nthr)
            out[base + j] = static_cast<scalar_t>(static_cast<float>(x[base + j]) * inv * weight[j]);
        return;
    }

    extern __shared__ float sm[];  // n*d floats: normed values
    for (int j = tid; j < nd; j += nthr)
        sm[j] = static_cast<float>(x[base + j]) * inv * weight[j];
    __syncthreads();
    for (int j = tid; j < nd; j += nthr) {
        const int  h  = j / d;
        const int  c  = j - h * d;
        const long bh = static_cast<long>(h) * d;
        float      o;
        if (INTERLEAVED) {
            const int   i  = c >> 1;
            const float cs = cosb[s * half + i], sn = sinb[s * half + i];
            o = ((c & 1) == 0) ? (sm[bh + c] * cs - sm[bh + c + 1] * sn) : (sm[bh + c] * cs + sm[bh + c - 1] * sn);
        }
        else {
            if (c < half) {
                const float cs = cosb[s * half + c], sn = sinb[s * half + c];
                o = sm[bh + c] * cs - sm[bh + c + half] * sn;
            }
            else {
                const int   i  = c - half;
                const float cs = cosb[s * half + i], sn = sinb[s * half + i];
                o = sm[bh + c] * cs + sm[bh + i] * sn;
            }
        }
        out[base + j] = static_cast<scalar_t>(o);
    }
}

// Core launcher: dispatch dtype + (do_norm, do_rope, mode, interleaved) to a kernel instantiation.
// x/out are raw device pointers; weight/cos/sin may be null when the corresponding step is off.
template<typename scalar_t>
static void launch_typed(const scalar_t* x,
                         scalar_t*       out,
                         const float*    weight,
                         const float*    cosb,
                         const float*    sinb,
                         int             b,
                         int             seq,
                         int             n,
                         int             d,
                         bool            do_norm,
                         bool            do_rope,
                         int             mode,  // 0 per-head, 1 cross-head
                         bool            interleaved,
                         float           eps,
                         cudaStream_t    stream)
{
    if (mode == 1) {
        // cross-head always normalizes (reduce over n*d). Pure cross-head RoPE is meaningless.
        const int     threads = 256;
        const long    blocks  = static_cast<long>(b) * seq;
        const int64_t smem =
            do_rope ? static_cast<int64_t>(n) * d * sizeof(float) : 0;  // int64: n*d*4 can overflow int
        if (do_rope) {
            // The n*d fp32 staging exceeds the default 48KB dynamic-smem limit for larger n*d: opt the
            // two RoPE instantiations in up to the device cap (once per scalar_t), fail loudly beyond it.
            // The settable dynamic max is the opt-in cap MINUS the kernel's static smem (wsum[32]).
            static const int smem_cap = [] {
                int dev = 0, m = 0;
                ULYSSES_CUDA_CHECK(cudaGetDevice(&dev));
                ULYSSES_CUDA_CHECK(cudaDeviceGetAttribute(&m, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
                int cap = m;
                for (const void* fn : {reinterpret_cast<const void*>(cross_head_kernel<scalar_t, true, true>),
                                       reinterpret_cast<const void*>(cross_head_kernel<scalar_t, true, false>)}) {
                    cudaFuncAttributes fa{};
                    ULYSSES_CUDA_CHECK(cudaFuncGetAttributes(&fa, fn));
                    const int dyn = m - static_cast<int>(fa.sharedSizeBytes);
                    ULYSSES_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
                    cap = std::min(cap, dyn);
                }
                return cap;
            }();
            TORCH_CHECK(smem <= smem_cap,
                        "cross_head norm+rope stages n*d fp32 in shared memory: needs ",
                        smem,
                        " B > device cap ",
                        smem_cap,
                        " B (n*d too large)");
        }
        if (do_rope && interleaved)
            cross_head_kernel<scalar_t, true, true>
                <<<blocks, threads, smem, stream>>>(x, out, weight, cosb, sinb, seq, n, d, eps);
        else if (do_rope)
            cross_head_kernel<scalar_t, true, false>
                <<<blocks, threads, smem, stream>>>(x, out, weight, cosb, sinb, seq, n, d, eps);
        else
            cross_head_kernel<scalar_t, false, false>
                <<<blocks, threads, 0, stream>>>(x, out, weight, cosb, sinb, seq, n, d, eps);
        return;
    }
    // per-head: one block per d-row.
    const long blocks = static_cast<long>(b) * seq * n;
    const int  smem   = d * static_cast<int>(sizeof(float));
    if (do_norm && do_rope && interleaved)
        per_head_kernel<scalar_t, true, true, true>
            <<<blocks, d, smem, stream>>>(x, out, weight, cosb, sinb, seq, n, d, eps);
    else if (do_norm && do_rope)
        per_head_kernel<scalar_t, true, true, false>
            <<<blocks, d, smem, stream>>>(x, out, weight, cosb, sinb, seq, n, d, eps);
    else if (do_norm)
        per_head_kernel<scalar_t, true, false, false>
            <<<blocks, d, smem, stream>>>(x, out, weight, cosb, sinb, seq, n, d, eps);
    else if (interleaved)
        per_head_kernel<scalar_t, false, true, true>
            <<<blocks, d, smem, stream>>>(x, out, weight, cosb, sinb, seq, n, d, eps);
    else
        per_head_kernel<scalar_t, false, true, false>
            <<<blocks, d, smem, stream>>>(x, out, weight, cosb, sinb, seq, n, d, eps);
}

static void run_qk(const void*    x,
                   void*          out,
                   const float*   weight,
                   const float*   cosb,
                   const float*   sinb,
                   int            b,
                   int            seq,
                   int            n,
                   int            d,
                   bool           do_norm,
                   bool           do_rope,
                   int            mode,
                   bool           interleaved,
                   float          eps,
                   at::ScalarType dtype,
                   cudaStream_t   stream)
{
    if (dtype == at::kHalf)
        launch_typed<at::Half>(static_cast<const at::Half*>(x),
                               static_cast<at::Half*>(out),
                               weight,
                               cosb,
                               sinb,
                               b,
                               seq,
                               n,
                               d,
                               do_norm,
                               do_rope,
                               mode,
                               interleaved,
                               eps,
                               stream);
    else
        launch_typed<at::BFloat16>(static_cast<const at::BFloat16*>(x),
                                   static_cast<at::BFloat16*>(out),
                                   weight,
                                   cosb,
                                   sinb,
                                   b,
                                   seq,
                                   n,
                                   d,
                                   do_norm,
                                   do_rope,
                                   mode,
                                   interleaved,
                                   eps,
                                   stream);
    ULYSSES_CUDA_CHECK(cudaGetLastError());
}

// ---- shared validation ----
static void check_x(const at::Tensor& x, int& b, int& seq, int& n, int& d)
{
    TORCH_CHECK(x.is_cuda() && x.dim() == 4, "x must be a 4D CUDA tensor [b,seq,n,d]");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x dtype must be float16 or bfloat16");
    b = x.size(0), seq = x.size(1), n = x.size(2), d = x.size(3);
    TORCH_CHECK(d > 0 && d <= 1024 && (d & (d - 1)) == 0, "d must be a power of two in (0,1024], got ", d);
}

static const float* fptr(const at::Tensor& t)
{
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous(),
                "weight/cos/sin must be contiguous fp32 CUDA tensors");
    return t.data_ptr<float>();
}

at::Tensor rms_norm(at::Tensor x, at::Tensor weight, int64_t mode, double eps)
{
    int b, seq, n, d;
    check_x(x, b, seq, n, d);
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 (per-head) or 1 (cross-head)");
    TORCH_CHECK(weight.numel() == (mode == 0 ? d : n * d), "weight numel mismatch for the chosen mode");
    x                             = x.contiguous();
    at::Tensor                out = at::empty_like(x);
    const at::cuda::CUDAGuard guard(x.device());
    run_qk(x.data_ptr(),
           out.data_ptr(),
           fptr(weight),
           nullptr,
           nullptr,
           b,
           seq,
           n,
           d,
           true,
           false,
           static_cast<int>(mode),
           false,
           static_cast<float>(eps),
           x.scalar_type(),
           at::cuda::getCurrentCUDAStream());
    return out;
}

// Kernels index cos/sin by row up to seq-1 -- an under-sized table is a GPU OOB read.
static void check_cos_sin(const at::Tensor& cos, const at::Tensor& sin, int seq, int d)
{
    TORCH_CHECK(cos.size(-1) == d / 2 && sin.size(-1) == d / 2, "cos/sin last dim must be d/2");
    TORCH_CHECK(cos.numel() >= static_cast<int64_t>(seq) * (d / 2)
                    && sin.numel() >= static_cast<int64_t>(seq) * (d / 2),
                "cos/sin must cover seq rows: numel >= seq*d/2");
}

at::Tensor rope(at::Tensor x, at::Tensor cos, at::Tensor sin, bool interleaved)
{
    int b, seq, n, d;
    check_x(x, b, seq, n, d);
    check_cos_sin(cos, sin, seq, d);
    x                             = x.contiguous();
    at::Tensor                out = at::empty_like(x);
    const at::cuda::CUDAGuard guard(x.device());
    run_qk(x.data_ptr(),
           out.data_ptr(),
           nullptr,
           fptr(cos),
           fptr(sin),
           b,
           seq,
           n,
           d,
           false,
           true,
           0,
           interleaved,
           0.f,
           x.scalar_type(),
           at::cuda::getCurrentCUDAStream());
    return out;
}

at::Tensor
norm_rope(at::Tensor x, at::Tensor weight, at::Tensor cos, at::Tensor sin, int64_t mode, bool interleaved, double eps)
{
    int b, seq, n, d;
    check_x(x, b, seq, n, d);
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 (per-head) or 1 (cross-head)");
    TORCH_CHECK(weight.numel() == (mode == 0 ? d : n * d), "weight numel mismatch for the chosen mode");
    check_cos_sin(cos, sin, seq, d);
    x                             = x.contiguous();
    at::Tensor                out = at::empty_like(x);
    const at::cuda::CUDAGuard guard(x.device());
    run_qk(x.data_ptr(),
           out.data_ptr(),
           fptr(weight),
           fptr(cos),
           fptr(sin),
           b,
           seq,
           n,
           d,
           true,
           true,
           static_cast<int>(mode),
           interleaved,
           static_cast<float>(eps),
           x.scalar_type(),
           at::cuda::getCurrentCUDAStream());
    return out;
}

}  // namespace ulysses
