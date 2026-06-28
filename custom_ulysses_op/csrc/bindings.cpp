#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "a2a_config.cuh"
#include "ulysses_common.cuh"
#include "ulysses_group.cuh"

namespace ulysses {

// Smoke test: include NVSHMEM headers + reference its types to prove the
// compile/host-link path works; no runtime init required.
int64_t nvshmem_uniqueid_nbytes()
{
    return static_cast<int64_t>(sizeof(nvshmemx_uniqueid_t));
}

// Decide whether this call takes the TMA path (thin wrapper over the shared
// helper decide_tma_path; single source of truth in a2a_config.cuh).
// use_tma: None = auto by arch, True/False = explicit on/off. Explicit True
// with sm<9 errors here.
static bool decide_tma(c10::optional<bool> use_tma, int sm_major, bool is_varlen)
{
    int i = use_tma.has_value() ? (*use_tma ? 1 : 0) : -1;
    if (i > 0)
        TORCH_CHECK(sm_major >= 9, "use_tma=True requires sm90+ (TMA unavailable on this GPU)");
    return decide_tma_path(i, sm_major, is_varlen);
}

at::Tensor all_to_all_single_4d(const c10::intrusive_ptr<UlyssesGroup>& group,
                                at::Tensor                              input,
                                int64_t                                 mode,
                                std::string                             tag,
                                c10::optional<std::vector<int64_t>>     seq_lens,
                                c10::optional<std::vector<int64_t>>     head_splits,
                                c10::optional<bool>                     use_tma)
{
    TORCH_CHECK(input.is_cuda() && input.dim() == 4, "input must be a 4D CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "dtype must be float16 or bfloat16");
    input        = input.contiguous();
    const int ws = static_cast<int>(group->world_size());
    TORCH_CHECK(ws == 1 || ws == 2 || ws == 4 || ws == 8, "world_size must be 1, 2, 4, or 8");
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

    // ---- Varlen path: caller supplies splits (no runtime gather); symmetric heap allocated collectively by max ----
    if (seq_lens.has_value() || head_splits.has_value()) {
        TORCH_CHECK(seq_lens.has_value() && head_splits.has_value(), "varlen needs BOTH seq_lens and head_splits");
        const auto& sl = *seq_lens;
        const auto& nl = *head_splits;
        TORCH_CHECK(static_cast<int>(sl.size()) == ws && static_cast<int>(nl.size()) == ws,
                    "seq_lens/head_splits length must equal world_size");
        SplitInfo sp;
        sp.world_size = ws;
        sp.rank       = me;
        sp.b          = b;
        sp.d          = d;
        sp.s_off[0]   = 0;
        sp.n_off[0]   = 0;
        for (int r = 0; r < ws; ++r) {
            sp.s_off[r + 1] = sp.s_off[r] + static_cast<int>(sl[r]);
            sp.n_off[r + 1] = sp.n_off[r] + static_cast<int>(nl[r]);
        }
        const int            S = sp.s_off[ws];
        const int            N = sp.n_off[ws];
        std::vector<int64_t> out_shape;
        if (mode == 0) {
            TORCH_CHECK(x1 == static_cast<int>(sl[me]) && x2 == N,
                        "mode0 varlen input must be [b, seq_lens[rank], sum(head_splits), d]");
            out_shape = {b, S, static_cast<int64_t>(nl[me]), d};
        }
        else {
            TORCH_CHECK(x1 == S && x2 == static_cast<int>(nl[me]),
                        "mode1 varlen input must be [b, sum(seq_lens), head_splits[rank], d]");
            out_shape = {b, static_cast<int64_t>(sl[me]), N, d};
        }
        const auto& buf = group->pool().acquire(out_shape, input.scalar_type(), tag);
        // Varlen defaults to the faster non-TMA path: with uneven seq, TMA is forced to tile_s=1
        // and ends up ~1.5x slower. auto/None always picks non-TMA; only explicit use_tma=True
        // takes TMA-varlen (is_varlen=true inside decide_tma).
        const bool tma = decide_tma(use_tma, group->sm_major(), /*is_varlen=*/true);
        if (tma)
            launch_a2a_tma_varlen(input.data_ptr(), buf.peer_ptrs, sp, static_cast<int>(mode), elem, stream);
        else
            launch_a2a_varlen(input.data_ptr(), buf.peer_ptrs, sp, static_cast<int>(mode), elem, stream);
        nvshmemx_quiet_on_stream(stream);  // complete this rank's P2P writes (globally visible)
        group->fast_barrier(stream);       // custom NVLink flag barrier (cross-rank sync)
        return buf.view;
    }

    // ---- Uniform fast path (s/n divisible by world_size) ----
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
    // Path decision (auto: sm90+ -> TMA). All ranks must pass the same use_tma; otherwise they
    // diverge on kernel/barrier + cfg_cache and hang.
    const bool tma = decide_tma(use_tma, group->sm_major(), /*is_varlen=*/false);
    // Hit in the group cache returns directly; miss runs a micro-benchmark to pick the best and
    // caches it (collective call -- all ranks must issue the same shape sequence).
    A2AConfig cfg =
        group->resolve_config(dims, static_cast<int>(mode), tma, input.data_ptr(), buf.peer_ptrs, elem, stream);
    if (tma)
        launch_a2a_tma(input.data_ptr(), buf.peer_ptrs, dims, static_cast<int>(mode), elem, cfg, stream);
    else
        launch_a2a(input.data_ptr(), buf.peer_ptrs, dims, static_cast<int>(mode), elem, cfg, stream);
    nvshmemx_quiet_on_stream(stream);  // complete this rank's P2P writes (globally visible)
    group->fast_barrier(stream);       // custom NVLink flag barrier (cross-rank sync)
    return buf.view;
}

}  // namespace ulysses

TORCH_LIBRARY(ulysses, m)
{
    m.def("nvshmem_uniqueid_nbytes() -> int");
    m.impl("nvshmem_uniqueid_nbytes", &ulysses::nvshmem_uniqueid_nbytes);

    m.class_<ulysses::UlyssesGroup>("UlyssesGroup")
        .def(torch::init<std::vector<int64_t>, int64_t, int64_t, int64_t>())
        .def("rank", &ulysses::UlyssesGroup::rank)
        .def("world_size", &ulysses::UlyssesGroup::world_size)
        .def("tune", &ulysses::UlyssesGroup::tune)
        .def("destroy", &ulysses::UlyssesGroup::destroy)
        .def_static("uniqueid_nints", &ulysses::UlyssesGroup::uniqueid_nints)
        .def_static("get_uniqueid", &ulysses::UlyssesGroup::get_uniqueid)
        .def_static("init_world", &ulysses::UlyssesGroup::init_world);

    m.def("all_to_all_single_4d(__torch__.torch.classes.ulysses.UlyssesGroup group, "
          "Tensor input, int mode, str tag, int[]? seq_lens=None, int[]? head_splits=None, "
          "bool? use_tma=None) -> Tensor");
    m.impl("all_to_all_single_4d", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_single_4d);
}

// Python `import _C` needs PyInit__C; TORCH_LIBRARY already registered at dlopen time.
PYBIND11_MODULE(_C, m) {}
