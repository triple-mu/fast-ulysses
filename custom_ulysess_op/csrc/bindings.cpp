#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "ulysses_common.cuh"
#include "ulysses_group.cuh"

namespace ulysess {

// 冒烟：包含 NVSHMEM 头 + 引用其类型，证明编译与 host 链接通路；无需 runtime
// init。
int64_t nvshmem_uniqueid_nbytes()
{
    return static_cast<int64_t>(sizeof(nvshmemx_uniqueid_t));
}

at::Tensor all_to_all_single_4d(const c10::intrusive_ptr<UlyssesGroup>& group,
                                at::Tensor                              input,
                                int64_t                                 mode,
                                std::string                             tag,
                                c10::optional<std::vector<int64_t>>     seq_lens,
                                c10::optional<std::vector<int64_t>>     head_splits)
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

    // ---- 变长路径：调用方提供 split（无运行时 gather）；对称堆按 max 集体分配 ----
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
        launch_a2a_varlen(input.data_ptr(), buf.peer_ptrs, sp, static_cast<int>(mode), elem, stream);
        nvshmemx_quiet_on_stream(stream);
        nvshmemx_barrier_on_stream(group->team(), stream);
        return buf.view;
    }

    // ---- 均匀快路径（s/n 被 world_size 整除）----
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
    launch_a2a(input.data_ptr(), buf.peer_ptrs, dims, static_cast<int>(mode), elem, stream);
    nvshmemx_quiet_on_stream(stream);                   // 完成本 rank 的 P2P 写
    nvshmemx_barrier_on_stream(group->team(), stream);  // 跨 rank 同步 + 可见
    return buf.view;
}

}  // namespace ulysess

TORCH_LIBRARY(ulysess, m)
{
    m.def("nvshmem_uniqueid_nbytes() -> int");
    m.impl("nvshmem_uniqueid_nbytes", &ulysess::nvshmem_uniqueid_nbytes);

    m.class_<ulysess::UlyssesGroup>("UlyssesGroup")
        .def(torch::init<std::vector<int64_t>, int64_t, int64_t, int64_t>())
        .def("rank", &ulysess::UlyssesGroup::rank)
        .def("world_size", &ulysess::UlyssesGroup::world_size)
        .def("destroy", &ulysess::UlyssesGroup::destroy)
        .def_static("uniqueid_nints", &ulysess::UlyssesGroup::uniqueid_nints)
        .def_static("get_uniqueid", &ulysess::UlyssesGroup::get_uniqueid)
        .def_static("init_world", &ulysess::UlyssesGroup::init_world);

    m.def("all_to_all_single_4d(__torch__.torch.classes.ulysess.UlyssesGroup group, "
          "Tensor input, int mode, str tag, int[]? seq_lens=None, int[]? head_splits=None) -> Tensor");
    m.impl("all_to_all_single_4d", c10::DispatchKey::CompositeExplicitAutograd, &ulysess::all_to_all_single_4d);
}

// Python `import _C` 需要 PyInit__C；TORCH_LIBRARY 在 dlopen 时已完成注册。
PYBIND11_MODULE(_C, m) {}
