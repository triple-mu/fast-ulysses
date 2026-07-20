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

namespace {

// Shared validation + dims for the uniform (non-fused) 4D a2a entry points. The input
// must already be contiguous.
void check_uniform_args(const at::Tensor&     input,
                        int64_t               mode,
                        int                   ws,
                        Ulysses4DDims&        dims,
                        std::vector<int64_t>& out_shape)
{
    TORCH_CHECK(input.is_cuda() && input.dim() == 4, "input must be a 4D CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "dtype must be float16 or bfloat16");
    const int b    = static_cast<int>(input.size(0));
    const int x1   = static_cast<int>(input.size(1));
    const int x2   = static_cast<int>(input.size(2));
    const int d    = static_cast<int>(input.size(3));
    const int elem = static_cast<int>(input.element_size());
    TORCH_CHECK((static_cast<int64_t>(d) * elem) % 16 == 0, "d*elem_size must be 16B-aligned");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");
    dims.b = b;
    dims.d = d;
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
}

}  // namespace

at::Tensor all_to_all_single_4d(const c10::intrusive_ptr<UlyssesGroup>& group,
                                at::Tensor                              input,
                                int64_t                                 mode,
                                std::string                             tag,
                                c10::optional<bool>                     use_tma)
{
    input        = input.contiguous();
    const int ws = static_cast<int>(group->world_size());
    TORCH_CHECK(ws >= 1 && ws <= 8, "world_size must be in [1, 8] (single-node NVLink), got ", ws);
    Ulysses4DDims        dims;
    std::vector<int64_t> out_shape;
    check_uniform_args(input, mode, ws, dims, out_shape);
    dims.rank      = static_cast<int>(group->rank());
    const int elem = static_cast<int>(input.element_size());

    const at::cuda::CUDAGuard guard(input.device());
    cudaStream_t              stream = at::cuda::getCurrentCUDAStream();

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

// CE (copy-engine) transfer path: same collective semantics, layouts, tag-scoped output
// buffers and barrier epochs as all_to_all_single_4d, but the data movement is a per-peer
// cudaMemcpy2DAsync fan-out on DMA engines (see all_to_all_ce.cu) -- zero SM usage, no
// launch config, no autotune. This is the third path next to the SM scatter and TMA: pick
// it when the a2a must overlap concurrent compute (the kernel paths cannot get an SM block
// slot while e.g. nvjet GEMMs hold them all).
at::Tensor all_to_all_single_4d_ce(const c10::intrusive_ptr<UlyssesGroup>& group,
                                   at::Tensor                              input,
                                   int64_t                                 mode,
                                   std::string                             tag)
{
    input        = input.contiguous();
    const int ws = static_cast<int>(group->world_size());
    TORCH_CHECK(ws >= 1 && ws <= 8, "world_size must be in [1, 8] (single-node NVLink), got ", ws);
    Ulysses4DDims        dims;
    std::vector<int64_t> out_shape;
    check_uniform_args(input, mode, ws, dims, out_shape);
    dims.rank = static_cast<int>(group->rank());

    const at::cuda::CUDAGuard guard(input.device());
    cudaStream_t              stream = at::cuda::getCurrentCUDAStream();

    const auto& buf = group->pool().acquire(out_shape, input.scalar_type(), tag);
    launch_a2a_ce(input.data_ptr(),
                  buf.peer_ptrs,
                  dims,
                  static_cast<int>(mode),
                  static_cast<int>(input.element_size()),
                  group->ce_resources(),
                  stream);
    // No nvshmemx_quiet: the transfers are CE memcpy operations, not NVSHMEM proxy writes.
    // Their completion is already joined onto `stream` inside launch_a2a_ce (stream-ordered
    // completion of a P2P memcpy implies global visibility -- the same contract NVSHMEM's
    // own host-issued P2P puts rely on), so the flag barrier can publish directly.
    group->fast_barrier(stream);
    return buf.view;
}

// mode0 Ulysses input all-to-all with fused source-side QK RMSNorm + RoPE. Input [b, s_local, n_global, d]
// (this rank holds its sequence shard's full head set); applies norm(+rope) per Wan semantics, then scatters
// to [b, s_global, n_local, d]. norm_mode: 0 per-head (weight [d]) / 1 cross-head (weight [n_global*d]).
// cos/sin: [s_local, d/2], already sliced to this rank's GLOBAL positions by the caller. v (no norm/rope)
// should keep using all_to_all_single_4d.
namespace {

// Shared validation + mode0 dims for the fused QK ops (input must already be contiguous).
void check_qk_args(const at::Tensor& input,
                   const at::Tensor& weight,
                   const at::Tensor& cos,
                   const at::Tensor& sin,
                   int64_t           norm_mode,
                   int               ws,
                   Ulysses4DDims&    dims)
{
    TORCH_CHECK(input.is_cuda() && input.dim() == 4, "input must be a 4D CUDA tensor [b,s_local,n_global,d]");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "dtype must be float16 or bfloat16");
    const int b    = static_cast<int>(input.size(0));
    const int x1   = static_cast<int>(input.size(1));  // s_local
    const int x2   = static_cast<int>(input.size(2));  // n_global
    const int d    = static_cast<int>(input.size(3));
    const int elem = static_cast<int>(input.element_size());
    TORCH_CHECK(d > 0 && d <= 1024 && (d & (d - 1)) == 0, "d must be a power of two in (0,1024]");
    // The fused scatter works on (vec j, vec j+vecs/2) uint4 pairs of a d-row -> needs >= 2 vecs/row.
    TORCH_CHECK(static_cast<int64_t>(d) * elem >= 32, "fused a2a needs d*elem_size >= 32B, got d=", d);
    TORCH_CHECK(x2 % ws == 0, "n_global must be divisible by world_size");
    TORCH_CHECK(norm_mode == 0 || norm_mode == 1, "norm_mode must be 0 (per-head) or 1 (cross-head)");
    // per-head reduces a d-row over its vecs/2 lanes with warp shuffles -> the row must fit in a warp.
    TORCH_CHECK(norm_mode == 1 || static_cast<int64_t>(d) * elem <= 1024,
                "per-head fused a2a needs d*elem_size <= 1024B (row reduction within one warp), got d=",
                d);
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kFloat && weight.is_contiguous()
                    && weight.numel() == (norm_mode == 0 ? d : static_cast<int64_t>(x2) * d),
                "weight must be contiguous fp32 with numel d (per-head) or n_global*d (cross-head)");
    TORCH_CHECK(cos.is_cuda() && cos.scalar_type() == at::kFloat && cos.is_contiguous() && cos.size(-1) == d / 2
                    && sin.is_cuda() && sin.scalar_type() == at::kFloat && sin.is_contiguous() && sin.size(-1) == d / 2,
                "cos/sin must be contiguous fp32 with last dim d/2");
    // Kernels index cos/sin by row up to s_local-1 -- an under-sized table is a GPU OOB read.
    TORCH_CHECK(cos.numel() >= static_cast<int64_t>(x1) * (d / 2) && sin.numel() >= static_cast<int64_t>(x1) * (d / 2),
                "cos/sin must cover s_local rows: numel >= s_local*d/2");
    dims.b        = b;
    dims.d        = d;
    dims.s_local  = x1;
    dims.n_global = x2;
    dims.s_global = x1 * ws;
    dims.n_local  = x2 / ws;
}

// One fused scatter: (cross-head inv-rms pre-pass +) config resolve + launch, WITHOUT the trailing
// quiet+barrier -- the callers append it, which is how qk2 shares a single barrier across q and k.
// The pre-pass MUST run before resolve_config_cached (the autotune probes dereference inv_rms).
at::Tensor qk_scatter_one(const c10::intrusive_ptr<UlyssesGroup>& group,
                          const at::Tensor&                       input,
                          const at::Tensor&                       weight,
                          const at::Tensor&                       cos,
                          const at::Tensor&                       sin,
                          int64_t                                 norm_mode,
                          bool                                    interleaved,
                          double                                  eps,
                          const std::string&                      tag,
                          const Ulysses4DDims&                    dims,
                          cudaStream_t                            stream)
{
    const int                  ws        = static_cast<int>(group->world_size());
    const std::vector<int64_t> out_shape = {dims.b, dims.s_global, dims.n_local, dims.d};
    const auto&                buf       = group->pool().acquire(out_shape, input.scalar_type(), tag);

    const bool   cross = (norm_mode == 1);
    at::Tensor   inv_rms;
    const float* inv_ptr = nullptr;
    if (cross) {
        inv_rms = at::empty({static_cast<int64_t>(dims.b) * dims.s_local}, input.options().dtype(at::kFloat));
        launch_token_inv_rms(
            input.data_ptr(), inv_rms.data_ptr<float>(), dims, static_cast<float>(eps), input.scalar_type(), stream);
        inv_ptr = inv_rms.data_ptr<float>();
    }
    auto finish = [&group, stream] {
        nvshmemx_quiet_on_stream(stream);
        group->fast_barrier(stream);
    };
    // FAST_ULYSSES_QK_TUNE_VERBOSE=1: each rank prints its tuned config (diagnostics only).
    static const bool tune_verbose = [] {
        const char* e = std::getenv("FAST_ULYSSES_QK_TUNE_VERBOSE");
        return e && e[0] == '1';
    }();
    const A2AConfig cfg =
        group->resolve_config_cached(config_key_qk(ws, static_cast<int>(norm_mode), dims), [&]() -> A2AConfig {
            return resolve_config_qk(input.data_ptr(),
                                     inv_ptr,
                                     buf.peer_ptrs,
                                     weight.data_ptr<float>(),
                                     cos.data_ptr<float>(),
                                     sin.data_ptr<float>(),
                                     dims,
                                     cross,
                                     interleaved,
                                     static_cast<float>(eps),
                                     input.scalar_type(),
                                     stream,
                                     tune_verbose,
                                     finish);
        });
    launch_a2a_qk(input.data_ptr(),
                  inv_ptr,
                  buf.peer_ptrs,
                  weight.data_ptr<float>(),
                  cos.data_ptr<float>(),
                  sin.data_ptr<float>(),
                  dims,
                  cross,
                  interleaved,
                  static_cast<float>(eps),
                  input.scalar_type(),
                  cfg,
                  stream);
    return buf.view;
}

}  // namespace

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
    input        = input.contiguous();
    const int ws = static_cast<int>(group->world_size());
    TORCH_CHECK(ws >= 1 && ws <= 8, "world_size must be in [1, 8] (single-node NVLink), got ", ws);
    Ulysses4DDims dims;
    check_qk_args(input, weight, cos, sin, norm_mode, ws, dims);
    dims.rank = static_cast<int>(group->rank());

    (void)use_tma;  // fused path always uses the direct-write scatter kernel; TMA-fused is future work
    const at::cuda::CUDAGuard guard(input.device());
    cudaStream_t              stream = at::cuda::getCurrentCUDAStream();
    at::Tensor out = qk_scatter_one(group, input, weight, cos, sin, norm_mode, interleaved, eps, tag, dims, stream);
    nvshmemx_quiet_on_stream(stream);  // complete this rank's P2P writes (globally visible)
    group->fast_barrier(stream);       // cross-rank sync
    return out;
}

// q and k in ONE collective call: two fused scatters back-to-back, then a SINGLE shared quiet +
// fast_barrier (vs 2x with two all_to_all_single_4d_qk calls -- the barrier is pure latency, so
// sharing it is a straight win). q/k share shape/dtype/cos/sin but have their own norm weights.
// Exposed as one op so every rank issues identical barrier counts (lockstep). Outputs live in two
// distinct tag-scoped buffers (tag::q / tag::k).
std::vector<at::Tensor> all_to_all_single_4d_qk2(const c10::intrusive_ptr<UlyssesGroup>& group,
                                                 at::Tensor                              q,
                                                 at::Tensor                              k,
                                                 at::Tensor                              weight_q,
                                                 at::Tensor                              weight_k,
                                                 at::Tensor                              cos,
                                                 at::Tensor                              sin,
                                                 int64_t                                 norm_mode,
                                                 bool                                    interleaved,
                                                 double                                  eps,
                                                 std::string                             tag,
                                                 c10::optional<bool>                     use_tma)
{
    q            = q.contiguous();
    k            = k.contiguous();
    const int ws = static_cast<int>(group->world_size());
    TORCH_CHECK(ws >= 1 && ws <= 8, "world_size must be in [1, 8] (single-node NVLink), got ", ws);
    TORCH_CHECK(k.is_cuda() && q.sizes() == k.sizes() && q.scalar_type() == k.scalar_type(),
                "k must be a CUDA tensor with the same shape and dtype as q");
    Ulysses4DDims dims;
    check_qk_args(q, weight_q, cos, sin, norm_mode, ws, dims);
    TORCH_CHECK(weight_k.is_cuda() && weight_k.scalar_type() == at::kFloat && weight_k.is_contiguous()
                    && weight_k.numel() == weight_q.numel(),
                "weight_k must be contiguous fp32 with the same numel as weight_q");
    dims.rank = static_cast<int>(group->rank());

    (void)use_tma;  // fused path always uses the direct-write scatter kernel; TMA-fused is future work
    const at::cuda::CUDAGuard guard(q.device());
    cudaStream_t              stream = at::cuda::getCurrentCUDAStream();
    at::Tensor                oq =
        qk_scatter_one(group, q, weight_q, cos, sin, norm_mode, interleaved, eps, tag + "::q", dims, stream);
    at::Tensor ok =
        qk_scatter_one(group, k, weight_k, cos, sin, norm_mode, interleaved, eps, tag + "::k", dims, stream);
    nvshmemx_quiet_on_stream(stream);  // one completion for BOTH scatters
    group->fast_barrier(stream);       // one shared cross-rank sync
    return {oq, ok};
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

    // CE (copy-engine) transfer path: zero-SM DMA fan-out, overlaps concurrent compute.
    m.def("all_to_all_single_4d_ce(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
          "Tensor input, int mode, str tag) -> Tensor");
    m.impl("all_to_all_single_4d_ce", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_single_4d_ce);

    // Standalone fused QK RMSNorm / RoPE building blocks (no group; single-GPU elementwise).
    m.def("rms_norm(Tensor x, Tensor weight, int mode, float eps) -> Tensor");
    m.impl("rms_norm", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::rms_norm);
    m.def("rope(Tensor x, Tensor cos, Tensor sin, bool interleaved) -> Tensor");
    m.impl("rope", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::rope);
    m.def(
        "norm_rope(Tensor x, Tensor weight, Tensor cos, Tensor sin, int mode, bool interleaved, float eps) -> Tensor");
    m.impl("norm_rope", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::norm_rope);

    m.def("all_to_all_single_4d_qk(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, Tensor input, "
          "Tensor weight, Tensor cos, Tensor sin, int norm_mode, bool interleaved, float eps, str tag, "
          "bool? use_tma=None) -> Tensor");
    m.impl("all_to_all_single_4d_qk", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_single_4d_qk);

    // q + k in one collective call (two fused scatters, ONE shared quiet+barrier).
    m.def("all_to_all_single_4d_qk2(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, Tensor q, Tensor k, "
          "Tensor weight_q, Tensor weight_k, Tensor cos, Tensor sin, int norm_mode, bool interleaved, float eps, "
          "str tag, bool? use_tma=None) -> Tensor[]");
    m.impl("all_to_all_single_4d_qk2", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_single_4d_qk2);
}

// Python `import _C` needs PyInit__C; TORCH_LIBRARY already registered at dlopen time.
PYBIND11_MODULE(_C, m) {}
