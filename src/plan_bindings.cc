// Exposes the pure-host plan builder to Python so test/test_plan.py can replay it over numpy
// buffers on a machine with no GPU. a2a_plan.cc itself stays free of torch and CUDA; this file is
// the only place the two meet. A separate TU is safe here, unlike the operator schemas in
// bindings.cpp: this schema names no custom class, so static-initialisation order does not matter.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <tuple>

#include <fast_ulysses/a2a_plan.hpp>

namespace ulysses {
namespace {

// peer, src_off, dst_off, src_pitch, dst_pitch, width, rows, depth, src_slice, dst_slice
constexpr int64_t kOpFields = 10;

std::tuple<std::vector<int64_t>, int64_t, at::Tensor> a2a_plan_debug(int64_t              b,
                                                                     int64_t              d,
                                                                     int64_t              rank,
                                                                     int64_t              world_size,
                                                                     std::vector<int64_t> seq_splits,
                                                                     std::vector<int64_t> head_splits,
                                                                     int64_t              mode,
                                                                     int64_t              elem_size)
{
    A2ADims dims;
    dims.b           = b;
    dims.d           = d;
    dims.rank        = static_cast<int>(rank);
    dims.world_size  = static_cast<int>(world_size);
    dims.seq_splits  = std::move(seq_splits);
    dims.head_splits = std::move(head_splits);

    const A2APlan plan = build_plan(dims, static_cast<int>(mode), elem_size);

    at::Tensor ops =
        at::empty({static_cast<int64_t>(plan.ops.size()), kOpFields}, at::TensorOptions().dtype(at::kLong));
    auto a = ops.accessor<int64_t, 2>();
    for (size_t i = 0; i < plan.ops.size(); ++i) {
        a[i][0] = plan.ops[i].peer;
        a[i][1] = plan.ops[i].src_offset;
        a[i][2] = plan.ops[i].dst_offset;
        a[i][3] = plan.ops[i].src_pitch;
        a[i][4] = plan.ops[i].dst_pitch;
        a[i][5] = plan.ops[i].width;
        a[i][6] = plan.ops[i].rows;
        a[i][7] = plan.ops[i].depth;
        a[i][8] = plan.ops[i].src_slice;
        a[i][9] = plan.ops[i].dst_slice;
    }
    return {plan.output_shape, plan.window_numel, ops};
}

}  // namespace
}  // namespace ulysses

TORCH_LIBRARY_FRAGMENT(fast_ulysses, m)
{
    m.def("a2a_plan_debug(int b, int d, int rank, int world_size, int[] seq_splits, "
          "int[] head_splits, int mode, int elem_size) -> (int[], int, Tensor)",
          &ulysses::a2a_plan_debug);
}
