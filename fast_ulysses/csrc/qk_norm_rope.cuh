#pragma once
#include "a2a_config.cuh"  // Ulysses4DDims (via ulysses_common.cuh) + A2AConfig
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

at::Tensor
norm_rope(at::Tensor x, at::Tensor weight, at::Tensor cos, at::Tensor sin, int64_t mode, bool interleaved, double eps);

// mode0 all-to-all with QK RMSNorm+RoPE fused into the scatter (all_to_all_qk.cu). inv_rms is the per-token
// cross-head inverse RMS ([b*s_local]) when cross_head, else null (per-head reduces inside the scatter).
// cfg: threads/unroll/blocks from resolve_config_qk (tile_n/tile_s unused).
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
                   cudaStream_t                 stream);

// cross-head pre-pass: fill inv_rms[b*s_local] from source [b,s_local,n_global,d].
void launch_token_inv_rms(
    const void* x, float* inv_rms, const Ulysses4DDims& dims, float eps, at::ScalarType dtype, cudaStream_t stream);

// Fused-scatter autotune sweep (threads x unroll x blocks microbench; mirrors resolve_config_nontma).
// Each timed run is the REAL per-call op: (cross-head pre-pass +) scatter + finish(quiet+barrier).
// inv_rms must already be allocated AND filled when cross_head (probe launches dereference it).
// Result is cached by the caller via UlyssesGroup::resolve_config_cached.
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
                            const std::function<void()>& finish);

}  // namespace ulysses
