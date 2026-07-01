#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace ulysses {

// Fused QK RMSNorm + RoPE building blocks (elementwise, DiT q/k). All replicate Wan's exact math:
// RMSNorm in fp32 (eps inside rsqrt, per-channel weight, no bias), RoPE over the full head_dim.
// x is [b, seq, n, d] (contiguous), fp16/bf16, d a power of two, d<=1024. weight/cos/sin are fp32.
// mode: 0 = per-head (reduce over d, weight [d]); 1 = cross-head (reduce over n*d, weight [n*d]).
// cos/sin: [seq, d/2], indexed by the seq position of each row. interleaved: true=GPT-J adjacent-pair,
// false=NeoX half-split. All return a new tensor (torch::empty_like(x)).

at::Tensor rms_norm(at::Tensor x, at::Tensor weight, int64_t mode, double eps);

at::Tensor rope(at::Tensor x, at::Tensor cos, at::Tensor sin, bool interleaved);

at::Tensor norm_rope(at::Tensor x, at::Tensor weight, at::Tensor cos, at::Tensor sin, int64_t mode,
                     bool interleaved, double eps);

// In-place variant used by the a2a-fused op: apply norm+rope to `x` writing into `out` (may alias x).
// Same semantics as norm_rope; separated so the a2a path can target a caller-provided buffer + stream.
void norm_rope_out(const void* x, void* out, const float* weight, const float* cos, const float* sin, int b,
                   int seq, int n, int d, int mode, bool interleaved, float eps, at::ScalarType dtype,
                   cudaStream_t stream);

}  // namespace ulysses
