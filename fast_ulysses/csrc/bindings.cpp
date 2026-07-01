#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "a2a_config.cuh"
#include "qk_norm_rope.cuh"
#include "ulysses_common.cuh"
#include "ulysses_group.cuh"

namespace ulysses {

// Smoke test: include NVSHMEM headers + reference its types to prove the
// compile/host-link path works; no runtime init required.
int64_t nvshmem_uniqueid_nbytes()
{
    return static_cast<int64_t>(sizeof(nvshmemx_uniqueid_t));
}

at::Tensor all_to_all_single_4d(const c10::intrusive_ptr<UlyssesGroup>& group,
                                at::Tensor                              input,
                                int64_t                                 mode,
                                std::string                             tag,
                                c10::optional<bool>                     use_tma)
{
    TORCH_CHECK(input.is_cuda() && input.dim() == 4, "input must be a 4D CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "dtype must be float16 or bfloat16");
    input        = input.contiguous();
    const int ws = static_cast<int>(group->world_size());
    TORCH_CHECK(ws >= 1 && ws <= 8, "world_size must be in [1, 8] (single-node NVLink), got ", ws);
    const int me   = static_cast<int>(group->rank());
    const int b    = static_cast<int>(input.size(0));
    const int x1   = static_cast<int>(input.size(1));
    const int x2   = static_cast<int>(input.size(2));
    const int d    = static_cast<int>(input.size(3));
    const int elem = static_cast<int>(input.element_size());
    TORCH_CHECK((static_cast<int64_t>(d) * elem) % 16 == 0, "d*elem_size must be 16B-aligned");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");

    const at::cuda::CUDAGuard guard(input.device());
    cudaStream_t              stream = at::cuda::getCurrentCUDAStream();

    // ---- Uniform path (s/n divisible by world_size) ----
    Ulysses4DDims dims;
    dims.b    = b;
    dims.d    = d;
    dims.rank = me;
    std::vector<int64_t> out_shape;
    if (mode == 0) {
        TORCH_CHECK(x2 % ws == 0, "n_global must be divisible by world_size");
        dims.s_local  = x1;
        dims.n_global = x2;
        dims.s_global = x1 * ws;
        dims.n_local  = x2 / ws;
        out_shape     = {b, dims.s_global, dims.n_local, d};
    }
    else {
        TORCH_CHECK(x1 % ws == 0, "s_global must be divisible by world_size");
        dims.s_global = x1;
        dims.n_local  = x2;
        dims.s_local  = x1 / ws;
        dims.n_global = x2 * ws;
        out_shape     = {b, dims.s_local, dims.n_global, d};
    }

    const auto& buf = group->pool().acquire(out_shape, input.scalar_type(), tag);
    // Tri-state use_tma: -1 auto / 0 non-TMA / 1 TMA. resolve_config picks the path (auto -> runtime-faster
    // of the two; explicit -> forced) and its launch config, caching both. All ranks must pass the same
    // use_tma, else they diverge on kernel/barrier + caches and hang.
    int use_tma_i = use_tma.has_value() ? (*use_tma ? 1 : 0) : -1;
    if (use_tma_i > 0)
        TORCH_CHECK(group->sm_major() >= 9, "use_tma=True requires sm90+ (TMA unavailable on this GPU)");
    const auto pc =
        group->resolve_config(dims, static_cast<int>(mode), use_tma_i, input.data_ptr(), buf.peer_ptrs, elem, stream);
    if (pc.tma)
        launch_a2a_tma(input.data_ptr(), buf.peer_ptrs, dims, static_cast<int>(mode), elem, pc.cfg, stream);
    else
        launch_a2a(input.data_ptr(), buf.peer_ptrs, dims, static_cast<int>(mode), elem, pc.cfg, stream);
    ULYSSES_CUDA_CHECK(cudaGetLastError());  // catch an a2a kernel launch failure
    nvshmemx_quiet_on_stream(stream);        // complete this rank's P2P writes (globally visible)
    group->fast_barrier(stream);             // custom NVLink flag barrier (cross-rank sync)
    return buf.view;
}

// mode0 Ulysses input all-to-all with fused source-side QK RMSNorm + RoPE. Input [b, s_local, n_global, d]
// (this rank holds its sequence shard's full head set); applies norm(+rope) per Wan semantics, then scatters
// to [b, s_global, n_local, d]. norm_mode: 0 per-head (weight [d]) / 1 cross-head (weight [n_global*d]).
// cos/sin: [s_local, d/2], already sliced to this rank's GLOBAL positions by the caller. v (no norm/rope)
// should keep using all_to_all_single_4d.
at::Tensor all_to_all_single_4d_qk(const c10::intrusive_ptr<UlyssesGroup>& group,
                                   at::Tensor                              input,
                                   at::Tensor                              weight,
                                   at::Tensor                              cos,
                                   at::Tensor                              sin,
                                   int64_t                                 norm_mode,
                                   bool                                    interleaved,
                                   double                                  eps,
                                   std::string                             tag,
                                   c10::optional<bool>                     use_tma)
{
    TORCH_CHECK(input.is_cuda() && input.dim() == 4, "input must be a 4D CUDA tensor [b,s_local,n_global,d]");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "dtype must be float16 or bfloat16");
    input        = input.contiguous();
    const int ws = static_cast<int>(group->world_size());
    TORCH_CHECK(ws >= 1 && ws <= 8, "world_size must be in [1, 8] (single-node NVLink), got ", ws);
    const int b    = static_cast<int>(input.size(0));
    const int x1   = static_cast<int>(input.size(1));  // s_local
    const int x2   = static_cast<int>(input.size(2));  // n_global
    const int d    = static_cast<int>(input.size(3));
    const int elem = static_cast<int>(input.element_size());
    TORCH_CHECK((static_cast<int64_t>(d) * elem) % 16 == 0, "d*elem_size must be 16B-aligned");
    TORCH_CHECK(d > 0 && d <= 1024 && (d & (d - 1)) == 0, "d must be a power of two in (0,1024]");
    TORCH_CHECK(x2 % ws == 0, "n_global must be divisible by world_size");
    TORCH_CHECK(norm_mode == 0 || norm_mode == 1, "norm_mode must be 0 (per-head) or 1 (cross-head)");
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kFloat && weight.is_contiguous() &&
                    weight.numel() == (norm_mode == 0 ? d : x2 * d),
                "weight must be contiguous fp32 with numel d (per-head) or n_global*d (cross-head)");
    TORCH_CHECK(cos.is_cuda() && cos.scalar_type() == at::kFloat && cos.is_contiguous() && cos.size(-1) == d / 2 &&
                    sin.is_cuda() && sin.scalar_type() == at::kFloat && sin.is_contiguous() && sin.size(-1) == d / 2,
                "cos/sin must be contiguous fp32 with last dim d/2");

    const at::cuda::CUDAGuard guard(input.device());
    cudaStream_t              stream = at::cuda::getCurrentCUDAStream();

    // Source-side fuse: norm+rope on [b, s_local, n_global, d] into a scratch, then the plain mode0 scatter.
    at::Tensor xt = at::empty_like(input);
    norm_rope_out(input.data_ptr(), xt.data_ptr(), weight.data_ptr<float>(), cos.data_ptr<float>(),
                  sin.data_ptr<float>(), b, x1, x2, d, static_cast<int>(norm_mode), interleaved,
                  static_cast<float>(eps), input.scalar_type(), stream);

    Ulysses4DDims dims;
    dims.b        = b;
    dims.d        = d;
    dims.rank     = static_cast<int>(group->rank());
    dims.s_local  = x1;
    dims.n_global = x2;
    dims.s_global = x1 * ws;
    dims.n_local  = x2 / ws;
    const std::vector<int64_t> out_shape = {b, dims.s_global, dims.n_local, d};

    const auto& buf       = group->pool().acquire(out_shape, input.scalar_type(), tag);
    int         use_tma_i = use_tma.has_value() ? (*use_tma ? 1 : 0) : -1;
    if (use_tma_i > 0)
        TORCH_CHECK(group->sm_major() >= 9, "use_tma=True requires sm90+ (TMA unavailable on this GPU)");
    const auto pc = group->resolve_config(dims, 0, use_tma_i, xt.data_ptr(), buf.peer_ptrs, elem, stream);
    if (pc.tma)
        launch_a2a_tma(xt.data_ptr(), buf.peer_ptrs, dims, 0, elem, pc.cfg, stream);
    else
        launch_a2a(xt.data_ptr(), buf.peer_ptrs, dims, 0, elem, pc.cfg, stream);
    ULYSSES_CUDA_CHECK(cudaGetLastError());
    nvshmemx_quiet_on_stream(stream);
    group->fast_barrier(stream);
    return buf.view;
}

}  // namespace ulysses

TORCH_LIBRARY(fast_ulysses, m)
{
    m.def("nvshmem_uniqueid_nbytes() -> int");
    m.impl("nvshmem_uniqueid_nbytes", &ulysses::nvshmem_uniqueid_nbytes);

    m.class_<ulysses::UlyssesGroup>("UlyssesGroup")
        .def(torch::init<std::vector<int64_t>, int64_t, int64_t, int64_t>())
        .def("rank", &ulysses::UlyssesGroup::rank)
        .def("world_size", &ulysses::UlyssesGroup::world_size)
        .def("destroy", &ulysses::UlyssesGroup::destroy)
        .def_static("uniqueid_nints", &ulysses::UlyssesGroup::uniqueid_nints)
        .def_static("get_uniqueid", &ulysses::UlyssesGroup::get_uniqueid)
        .def_static("init_world", &ulysses::UlyssesGroup::init_world);

    m.def("all_to_all_single_4d(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
          "Tensor input, int mode, str tag, bool? use_tma=None) -> Tensor");
    m.impl("all_to_all_single_4d", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_single_4d);

    // Standalone fused QK RMSNorm / RoPE building blocks (no group; single-GPU elementwise).
    m.def("rms_norm(Tensor x, Tensor weight, int mode, float eps) -> Tensor");
    m.impl("rms_norm", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::rms_norm);
    m.def("rope(Tensor x, Tensor cos, Tensor sin, bool interleaved) -> Tensor");
    m.impl("rope", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::rope);
    m.def("norm_rope(Tensor x, Tensor weight, Tensor cos, Tensor sin, int mode, bool interleaved, float eps) -> Tensor");
    m.impl("norm_rope", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::norm_rope);

    m.def("all_to_all_single_4d_qk(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, Tensor input, "
          "Tensor weight, Tensor cos, Tensor sin, int norm_mode, bool interleaved, float eps, str tag, "
          "bool? use_tma=None) -> Tensor");
    m.impl("all_to_all_single_4d_qk", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_single_4d_qk);
}

// Python `import _C` needs PyInit__C; TORCH_LIBRARY already registered at dlopen time.
PYBIND11_MODULE(_C, m) {}
