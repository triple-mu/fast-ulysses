// The torch op layer: validate, plan, barrier, transfer, barrier, copy out.
//
// Every address this file touches arrives as an argument. The windows and their flags are torch
// symmetric-memory allocations owned by the Python side (python/fast_ulysses/comm.py), so no
// communication library appears here, and the only state the group object holds is one CUDA stream.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <optional>
#include <torch/extension.h>
#include <torch/library.h>

#include <map>
#include <string>
#include <tuple>

#include <fast_ulysses/a2a_plan.hpp>
#include <fast_ulysses/common.hpp>
#include <fast_ulysses/transfer.hpp>
#include <fast_ulysses/work.hpp>

namespace ulysses {

// One CUDA stream, created on first use: the transfer stream the remote peer copies are serialised
// onto. Rank and world size ride along because every entry point needs them.
class UlyssesGroup: public torch::CustomClassHolder {
public:
    UlyssesGroup(int64_t rank, int64_t world_size):
        rank_(static_cast<int>(rank)),
        world_size_(static_cast<int>(world_size))
    {
        TORCH_CHECK(world_size_ >= 1 && world_size_ <= 8, "world_size must be in [1, 8] (one node), got ", world_size_);
        TORCH_CHECK(rank_ >= 0 && rank_ < world_size_, "rank ", rank_, " out of range for world_size ", world_size_);
    }

    ~UlyssesGroup() override
    {
        destroy();
    }

    int64_t rank() const
    {
        return rank_;
    }

    int64_t world_size() const
    {
        return world_size_;
    }

    cudaStream_t xfer_stream()
    {
        if (xfer_ == nullptr) {
            ULYSSES_CUDA_CHECK(cudaStreamCreateWithFlags(&xfer_, cudaStreamNonBlocking));
        }
        return xfer_;
    }

    // Nothing here is collective -- one rank may run it alone -- so the destructor can call it.
    void destroy()
    {
        if (xfer_ != nullptr) {
            cudaStreamSynchronize(xfer_);
            cudaStreamDestroy(xfer_);
            xfer_ = nullptr;
        }
    }

private:
    int          rank_, world_size_;
    cudaStream_t xfer_ = nullptr;
};

namespace {

// Validation + dims for the 4D a2a entry point. Shape-only, so a window can be sized for a call
// that has no tensor yet. The plan treats uneven as the general case, so the only decision here is
// what the splits ARE: the caller's, or the even ones the shape implies.
A2ADims make_dims_from_shape(at::IntArrayRef                            sizes,
                             c10::ScalarType                            dtype,
                             int64_t                                    mode,
                             int                                        ws,
                             int                                        rank,
                             const std::optional<std::vector<int64_t>>& seq_splits,
                             const std::optional<std::vector<int64_t>>& head_splits)
{
    TORCH_CHECK(sizes.size() == 4, "input must be 4D, got ", sizes.size(), " dims");
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "dtype must be float16 or bfloat16");
    const int64_t x1   = sizes[1];
    const int64_t x2   = sizes[2];
    const int64_t d    = sizes[3];
    const int64_t elem = static_cast<int64_t>(c10::elementSize(dtype));
    TORCH_CHECK((d * elem) % 16 == 0, "d*elem_size must be 16B-aligned");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");
    A2ADims dims;
    dims.b          = sizes[0];
    dims.d          = d;
    dims.rank       = rank;
    dims.world_size = ws;

    if (seq_splits.has_value() || head_splits.has_value()) {
        // One without the other has no meaning: the plan needs the WHOLE group's geometry, and the
        // missing half cannot be inferred from a shape that is itself already sharded.
        TORCH_CHECK(seq_splits.has_value() && head_splits.has_value(),
                    "pass both seq_splits and head_splits, or neither");
        dims.seq_splits  = *seq_splits;
        dims.head_splits = *head_splits;
        dims.validate();  // length/sign checks, before indexing by rank below
        // Cross-check the splits against the tensor handed in, so a caller that mis-shards gets an
        // error here instead of a silently corrupt result.
        const int64_t expect_x1 = (mode == 0) ? dims.seq_splits[rank] : dims.seq_total();
        const int64_t expect_x2 = (mode == 0) ? dims.head_total() : dims.head_splits[rank];
        TORCH_CHECK(x1 == expect_x1 && x2 == expect_x2,
                    "input is [",
                    dims.b,
                    ", ",
                    x1,
                    ", ",
                    x2,
                    ", ",
                    d,
                    "] but the splits imply [",
                    dims.b,
                    ", ",
                    expect_x1,
                    ", ",
                    expect_x2,
                    ", ",
                    d,
                    "]");
        return dims;
    }

    // No splits given: the even special case, which the shape alone determines only if the
    // scattered axis divides. The other axis is already sharded on entry, so it never has to.
    if (mode == 0) {
        // x1 is this rank's sequence shard, x2 the global head count.
        TORCH_CHECK(x2 % ws == 0, "n_global must be divisible by world_size (or pass head_splits)");
        dims.seq_splits.assign(ws, x1);
        dims.head_splits.assign(ws, x2 / ws);
    }
    else {
        // x1 is the global sequence length, x2 this rank's head shard.
        TORCH_CHECK(x1 % ws == 0, "s_global must be divisible by world_size (or pass seq_splits)");
        dims.seq_splits.assign(ws, x1 / ws);
        dims.head_splits.assign(ws, x2);
    }
    return dims;
}

// Validated input, the plan, and the tensor the result lands in -- which may be the window itself.
struct Prepared {
    at::Tensor x;
    at::Tensor output;
    A2APlan    plan;
};

// Byte intervals [a, a+a_bytes) and [b, b+b_bytes) intersect.
bool intervals_overlap(const void* a, int64_t a_bytes, const void* b, int64_t b_bytes)
{
    const auto* pa = static_cast<const char*>(a);
    const auto* pb = static_cast<const char*>(b);
    return pa < pb + b_bytes && pb < pa + a_bytes;
}

// Refuse a call whose input, or whose `out`, shares bytes with the window it is about to fill --
// it would be read while every peer writes it. `out` BEING the window is the zero-copy path and is
// the one legitimate case; the caller reaches it by passing a buffer from group.empty_output().
// Intervals rather than a pointer comparison, because an overlap can start past the window base.
void check_window_aliasing(const Prepared& prepared, const void* window, int64_t window_bytes, bool out_is_window)
{
    TORCH_CHECK(!intervals_overlap(prepared.x.data_ptr(), prepared.x.nbytes(), window, window_bytes),
                "input overlaps the window this call fills: it would be read while every peer "
                "writes it. Pass a separate output buffer, or let the call allocate one.");
    if (prepared.output.defined() && !out_is_window) {
        TORCH_CHECK(!intervals_overlap(prepared.output.data_ptr(), prepared.output.nbytes(), window, window_bytes),
                    "out partially overlaps the window this call fills, which neither writes it "
                    "correctly nor copies out correctly. Pass either the whole window (from "
                    "group.empty_output()) or a buffer outside the symmetric pool.");
    }
}

// Everything this does runs BEFORE the call's first barrier, so a rejected argument leaves no rank
// waiting on peers that did not reject it.
Prepared prepare(const c10::intrusive_ptr<UlyssesGroup>&    group,
                 const at::Tensor&                          input,
                 int64_t                                    mode,
                 const std::optional<std::vector<int64_t>>& seq_splits,
                 const std::optional<std::vector<int64_t>>& head_splits,
                 const std::optional<at::Tensor>&           out)
{
    const int ws = static_cast<int>(group->world_size());

    Prepared prepared;
    prepared.x = input.contiguous();
    TORCH_CHECK(prepared.x.is_cuda(), "input must be a CUDA tensor");
    const A2ADims dims = make_dims_from_shape(prepared.x.sizes(),
                                              prepared.x.scalar_type(),
                                              mode,
                                              ws,
                                              static_cast<int>(group->rank()),
                                              seq_splits,
                                              head_splits);
    prepared.plan      = build_plan(dims, static_cast<int>(mode), prepared.x.element_size());

    if (out.has_value()) {
        prepared.output = *out;
        TORCH_CHECK(prepared.output.is_cuda() && prepared.output.is_contiguous(),
                    "out must be a contiguous CUDA tensor");
        TORCH_CHECK(prepared.output.scalar_type() == prepared.x.scalar_type(),
                    "out has dtype ",
                    prepared.output.scalar_type(),
                    ", expected ",
                    prepared.x.scalar_type());
        TORCH_CHECK(prepared.output.sizes() == at::IntArrayRef(prepared.plan.output_shape),
                    "out has shape ",
                    prepared.output.sizes(),
                    ", expected ",
                    at::IntArrayRef(prepared.plan.output_shape));
    }
    else {
        prepared.output = at::empty(prepared.plan.output_shape, prepared.x.options());
    }
    return prepared;
}

// The window this rank writes its own share into and reads its result from, checked against what
// the plan needs. `window_ptrs[rank]` is our own base; the others are the peers' as we address them.
void* local_window(
    const Prepared& prepared, const std::vector<int64_t>& window_ptrs, int64_t window_numel, int rank, int ws)
{
    TORCH_CHECK(static_cast<int>(window_ptrs.size()) == ws,
                "window_ptrs has ",
                window_ptrs.size(),
                " entries, expected world_size ",
                ws);
    TORCH_CHECK(window_numel >= prepared.plan.window_numel,
                "the window holds ",
                window_numel,
                " elements but this call needs ",
                prepared.plan.window_numel);
    return reinterpret_cast<void*>(window_ptrs[rank]);
}

// Barrier, transfer, barrier: everything the call does to the window, ordered on `stream`.
void transfer_on_stream(const c10::intrusive_ptr<UlyssesGroup>& group,
                        const Prepared&                         prepared,
                        const std::vector<int64_t>&             window_ptrs,
                        const std::vector<int64_t>&             flag_ptrs,
                        cudaStream_t                            stream)
{
    const int                   rank = static_cast<int>(group->rank());
    const std::vector<uint64_t> peers(window_ptrs.begin(), window_ptrs.end());
    const std::vector<uint64_t> flags(flag_ptrs.begin(), flag_ptrs.end());

    // WRITERS WAIT FOR READERS, before writing anything. The window is single-buffered, so this
    // call is about to overwrite what the previous call produced, which a peer may still be
    // reading; the closing barrier below proves everyone's WRITES landed and nothing about their
    // READS. It guards the START of a call rather than the end of the previous one because on the
    // zero-copy path the result is read by the caller at a time the operator never sees.
    fast_barrier(stream, flags, rank);

    launch_a2a_ce(prepared.x.data_ptr(), peers, prepared.plan, group->xfer_stream(), rank, stream);

    // That a completed peer memcpy is VISIBLE at the destination when a later kernel's release
    // store arrives is an ASSUMPTION, not a documented guarantee -- test/distributed's CE ordering
    // worker is the negative control for it.
    fast_barrier(stream, flags, rank);
}

// Window -> the caller's tensor, ordered after the closing barrier on the same stream. Every rank's
// result is dense from the window base, so this is one flat device-to-device copy -- this rank's
// own share included, since it travels through the window like every peer's.
void copy_out(const Prepared& prepared, const void* window, cudaStream_t stream)
{
    ULYSSES_CUDA_CHECK(cudaMemcpyAsync(prepared.output.data_ptr(),
                                       window,
                                       static_cast<size_t>(prepared.output.numel() * prepared.output.element_size()),
                                       cudaMemcpyDeviceToDevice,
                                       stream));
}

}  // namespace

// What the Python side needs before it can allocate: the window capacity and this rank's output
// shape. The capacity is the LARGEST rank's output rather than this rank's -- peer offsets only
// line up while every rank allocates the same size, and each rank can compute the max without
// communicating. Both come from the plan, so neither is re-derived in Python.
std::tuple<int64_t, std::vector<int64_t>> plan_shapes(std::vector<int64_t>                       sizes,
                                                      int64_t                                    mode,
                                                      at::ScalarType                             dtype,
                                                      int64_t                                    world_size,
                                                      int64_t                                    rank,
                                                      const std::optional<std::vector<int64_t>>& seq_splits,
                                                      const std::optional<std::vector<int64_t>>& head_splits)
{
    const A2ADims dims = make_dims_from_shape(at::IntArrayRef(sizes),
                                              dtype,
                                              mode,
                                              static_cast<int>(world_size),
                                              static_cast<int>(rank),
                                              seq_splits,
                                              head_splits);
    const A2APlan plan = build_plan(dims, static_cast<int>(mode), static_cast<int64_t>(c10::elementSize(dtype)));
    return {plan.window_numel, plan.output_shape};
}

// The one collective. The result is always a tensor the caller owns, with no lifetime rules.
//
// Two speeds, decided by where `out` points rather than by a second entry point: when `out` IS the
// window (a buffer from group.empty_output()), the peers write it directly and there is no
// copy-out; otherwise the transfer lands in a window this group keeps and one flat device-to-device
// copy moves it out.
at::Tensor all_to_all_4d(const c10::intrusive_ptr<UlyssesGroup>&    group,
                         const at::Tensor&                          input,
                         int64_t                                    mode,
                         std::vector<int64_t>                       window_ptrs,
                         std::vector<int64_t>                       flag_ptrs,
                         int64_t                                    window_numel,
                         const std::optional<std::vector<int64_t>>& seq_splits,
                         const std::optional<std::vector<int64_t>>& head_splits,
                         const std::optional<at::Tensor>&           out)
{
    const at::cuda::CUDAGuard guard(input.device());
    const Prepared            prepared = prepare(group, input, mode, seq_splits, head_splits, out);
    const int                 ws       = static_cast<int>(group->world_size());
    void*      window        = local_window(prepared, window_ptrs, window_numel, static_cast<int>(group->rank()), ws);
    const bool out_is_window = prepared.output.data_ptr() == window;
    check_window_aliasing(prepared, window, window_numel * prepared.x.element_size(), out_is_window);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    transfer_on_stream(group, prepared, window_ptrs, flag_ptrs, stream);
    if (!out_is_window) {
        copy_out(prepared, window, stream);
    }
    return prepared.output;
}

// Benchmark-only: the copying call with CUDA events between its stages. `transfer` covers the peer
// copies plus this rank's own share, which runs on the caller's stream underneath them and so
// cannot be timed apart. Strictly ordered on one stream, so the stages sum to the whole call.
std::tuple<at::Tensor, std::vector<double>> all_to_all_4d_timed(const c10::intrusive_ptr<UlyssesGroup>&    group,
                                                                const at::Tensor&                          input,
                                                                int64_t                                    mode,
                                                                std::vector<int64_t>                       window_ptrs,
                                                                std::vector<int64_t>                       flag_ptrs,
                                                                int64_t                                    window_numel,
                                                                const std::optional<std::vector<int64_t>>& seq_splits,
                                                                const std::optional<std::vector<int64_t>>& head_splits)
{
    const at::cuda::CUDAGuard guard(input.device());
    const Prepared            prepared = prepare(group, input, mode, seq_splits, head_splits, std::nullopt);
    const int                 rank     = static_cast<int>(group->rank());
    const int                 ws       = static_cast<int>(group->world_size());
    void*                     window   = local_window(prepared, window_ptrs, window_numel, rank, ws);
    check_window_aliasing(prepared, window, window_numel * prepared.x.element_size(), /*out_is_window=*/false);

    const std::vector<uint64_t> peers(window_ptrs.begin(), window_ptrs.end());
    const std::vector<uint64_t> flags(flag_ptrs.begin(), flag_ptrs.end());
    cudaStream_t                stream = at::cuda::getCurrentCUDAStream();

    cudaEvent_t marks[5];
    for (auto& ev : marks) {
        ULYSSES_CUDA_CHECK(cudaEventCreate(&ev));
    }

    ULYSSES_CUDA_CHECK(cudaEventRecord(marks[0], stream));
    fast_barrier(stream, flags, rank);
    ULYSSES_CUDA_CHECK(cudaEventRecord(marks[1], stream));
    launch_a2a_ce(prepared.x.data_ptr(), peers, prepared.plan, group->xfer_stream(), rank, stream);
    ULYSSES_CUDA_CHECK(cudaEventRecord(marks[2], stream));
    fast_barrier(stream, flags, rank);
    ULYSSES_CUDA_CHECK(cudaEventRecord(marks[3], stream));
    copy_out(prepared, window, stream);
    ULYSSES_CUDA_CHECK(cudaEventRecord(marks[4], stream));

    ULYSSES_CUDA_CHECK(cudaEventSynchronize(marks[4]));
    std::vector<double> stages(4, 0.0);
    for (int i = 0; i < 4; ++i) {
        float ms = 0.0F;
        ULYSSES_CUDA_CHECK(cudaEventElapsedTime(&ms, marks[i], marks[i + 1]));
        stages[i] = static_cast<double>(ms);
    }
    for (auto& ev : marks) {
        ULYSSES_CUDA_CHECK(cudaEventDestroy(ev));
    }
    return {prepared.output, stages};
}

}  // namespace ulysses

TORCH_LIBRARY(fast_ulysses, m)
{
    m.class_<ulysses::UlyssesGroup>("UlyssesGroup")
        .def(torch::init<int64_t, int64_t>())
        .def("destroy", &ulysses::UlyssesGroup::destroy);

    m.def("plan_shapes(int[] sizes, int mode, ScalarType dtype, int world_size, int rank, "
          "int[]? seq_splits=None, int[]? head_splits=None) -> (int, int[])");
    m.impl("plan_shapes", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::plan_shapes);

    // `out` is an optional preallocated destination; passing the window itself is the zero-copy
    // path. The Python side is what knows which buffers are windows.
    m.def("all_to_all_4d(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
          "Tensor input, int mode, int[] window_ptrs, int[] flag_ptrs, int window_numel, "
          "int[]? seq_splits=None, int[]? head_splits=None, Tensor? out=None) -> Tensor");
    m.impl("all_to_all_4d", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_4d);

    m.def("all_to_all_4d_timed(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
          "Tensor input, int mode, int[] window_ptrs, int[] flag_ptrs, int window_numel, "
          "int[]? seq_splits=None, int[]? head_splits=None) -> (Tensor, float[])");
    m.impl("all_to_all_4d_timed", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_4d_timed);
}

// Python `import _C` needs PyInit__C; TORCH_LIBRARY already registered at dlopen time.
PYBIND11_MODULE(_C, m)
{
    // The async entry points are Python-side, so this is the one piece that has to be C++:
    // c10d::Work is a C++ interface. Not a torch op -- it takes a raw stream handle and mutates a
    // process-wide registry, neither of which belongs in a schema. See include/fast_ulysses/work.hpp.
    m.def("register_stream_completion", [](const at::Tensor& tensor, int64_t comm_stream) {
        return ulysses::register_stream_completion(tensor, reinterpret_cast<cudaStream_t>(comm_stream));
    });

    // TESTS ONLY. Underscored, and not a torch op, because arming it deliberately breaks the
    // operator: it is the negative control for a2a_ce_flag_ordering.py. See src/transfer.cu.
    m.def("_set_ce_fault", &ulysses::set_ce_fault);

    m.def("build_info", []() {
        std::map<std::string, std::string> out;
        out["version"]        = FAST_ULYSSES_VERSION;
        out["cuda_arch_list"] = FAST_ULYSSES_CUDA_ARCH_LIST;
        return out;
    });
}
