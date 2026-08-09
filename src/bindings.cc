// The torch op layer: validate the tensor, take the plan and the window from the group, then
// barrier, transfer, barrier, copy out.
//
// The group owns everything that survives a call -- windows, plans, staging buffers -- so the
// Python side is a constructor and two forwards.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <optional>
#include <torch/extension.h>
#include <torch/library.h>

#include <map>
#include <string>
#include <tuple>

#include <fast_ulysses/common.hpp>
#include <fast_ulysses/group.hpp>
#include <fast_ulysses/nvlink.hpp>
#include <fast_ulysses/transfer.hpp>
#include <fast_ulysses/work.hpp>

namespace ulysses {

namespace {

// Byte intervals [a, a+a_bytes) and [b, b+b_bytes) intersect.
bool intervals_overlap(const void* a, int64_t a_bytes, const void* b, int64_t b_bytes)
{
    const auto* pa = static_cast<const char*>(a);
    const auto* pb = static_cast<const char*>(b);
    return pa < pb + b_bytes && pb < pa + a_bytes;
}

// The validated input, the destination, the window the transfer fills, and whether those last two
// are the same buffer.
struct Call {
    at::Tensor     x;
    at::Tensor     output;
    const A2APlan* plan          = nullptr;
    const Window*  win           = nullptr;
    bool           out_is_window = false;
};

// Everything here runs BEFORE the call's first barrier, so a rejected argument leaves no rank
// waiting on peers that did not reject it.
Call prepare(const c10::intrusive_ptr<UlyssesGroup>&    group,
             const at::Tensor&                          input,
             int64_t                                    mode,
             const std::optional<std::vector<int64_t>>& seq_splits,
             const std::optional<std::vector<int64_t>>& head_splits,
             const std::optional<at::Tensor>&           out,
             WindowRole                                 role)
{
    Call call;
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor, got one on ", input.device());
    call.x    = input.contiguous();
    call.plan = &group->plan(call.x.sizes(), mode, call.x.scalar_type(), seq_splits, head_splits);

    if (out.has_value()) {
        call.output = *out;
        TORCH_CHECK(call.output.is_cuda() && call.output.is_contiguous(), "out must be a contiguous CUDA tensor");
        TORCH_CHECK(call.output.scalar_type() == call.x.scalar_type(),
                    "out has dtype ",
                    call.output.scalar_type(),
                    ", expected ",
                    call.x.scalar_type());
        TORCH_CHECK(call.output.sizes() == at::IntArrayRef(call.plan->output_shape),
                    "out has shape ",
                    call.output.sizes(),
                    ", expected ",
                    at::IntArrayRef(call.plan->output_shape));
        // A buffer from empty_output() IS a window, and the peers write it directly.
        const Window* owned = group->window_of(call.output);
        if (owned != nullptr && owned->numel >= call.plan->window_numel) {
            call.win           = owned;
            call.out_is_window = true;
        }
    }
    else {
        call.output = at::empty(call.plan->output_shape, call.x.options());
    }

    if (call.win == nullptr) {
        call.win = &group->window(role, call.x.scalar_type(), call.plan->window_numel);
    }

    // Refuse a call whose input, or whose `out`, shares bytes with the window it is about to fill
    // -- it would be read while every peer writes it. `out` BEING the window is the zero-copy path
    // and the one legitimate case. Intervals, not a pointer comparison, because an overlap can
    // start past the window's base.
    const void*   base  = reinterpret_cast<const void*>(call.win->peer_ptrs[group->rank()]);
    const int64_t bytes = call.win->numel * call.x.element_size();
    TORCH_CHECK(!intervals_overlap(call.x.data_ptr(), call.x.nbytes(), base, bytes),
                "input overlaps the window this call fills: it would be read while every peer "
                "writes it. Pass a separate output buffer, or let the call allocate one.");
    if (!call.out_is_window) {
        TORCH_CHECK(!intervals_overlap(call.output.data_ptr(), call.output.nbytes(), base, bytes),
                    "out overlaps the window this call fills without being it, which neither "
                    "writes correctly nor copies out correctly. Pass a whole buffer from "
                    "group.empty_output(), or one outside symmetric memory.");
    }
    return call;
}

// Barrier, transfer, barrier: everything the call does to the window, ordered on `stream`.
void transfer_on_stream(const c10::intrusive_ptr<UlyssesGroup>& group, const Call& call, cudaStream_t stream)
{
    const int rank = static_cast<int>(group->rank());

    // WRITERS WAIT FOR READERS, before writing anything. The window is single-buffered, so this
    // call is about to overwrite what the previous one produced, which a peer may still be
    // reading; the closing barrier below proves everyone's WRITES landed and nothing about their
    // READS. It guards the START of a call rather than the end of the previous one because on the
    // zero-copy path the result is read by the caller at a time the operator never sees.
    fast_barrier(stream, call.win->flag_ptrs, rank);

    launch_a2a_ce(call.x.data_ptr(), call.win->peer_ptrs, *call.plan, group->xfer_stream(), rank, stream);

    // That a completed peer memcpy is VISIBLE at the destination when this kernel's release store
    // arrives is an ASSUMPTION, not a documented guarantee -- test/distributed/ce_ordering.py is
    // the negative control for it.
    fast_barrier(stream, call.win->flag_ptrs, rank);
}

// Window -> the caller's tensor, ordered after the closing barrier on the same stream. Every
// rank's result is dense from the window base, so this is one flat device-to-device copy -- this
// rank's own share included, since it travels through the window like every peer's.
void copy_out(const Call& call, cudaStream_t stream, int rank)
{
    ULYSSES_CUDA_CHECK(cudaMemcpyAsync(call.output.data_ptr(),
                                       reinterpret_cast<const void*>(call.win->peer_ptrs[rank]),
                                       static_cast<size_t>(call.output.nbytes()),
                                       cudaMemcpyDeviceToDevice,
                                       stream));
}

at::Tensor run(const c10::intrusive_ptr<UlyssesGroup>&    group,
               const at::Tensor&                          input,
               int64_t                                    mode,
               const std::optional<std::vector<int64_t>>& seq_splits,
               const std::optional<std::vector<int64_t>>& head_splits,
               const std::optional<at::Tensor>&           out,
               WindowRole                                 role,
               cudaStream_t                               stream)
{
    const Call call = prepare(group, input, mode, seq_splits, head_splits, out, role);
    transfer_on_stream(group, call, stream);
    if (!call.out_is_window) {
        copy_out(call, stream, static_cast<int>(group->rank()));
    }
    return call.output;
}

}  // namespace

// The one collective. The result is always a tensor the caller owns, with no lifetime rules; `out`
// from empty_output() removes the copy-out, because the peers then write it directly.
at::Tensor all_to_all_4d(const c10::intrusive_ptr<UlyssesGroup>&    group,
                         const at::Tensor&                          input,
                         int64_t                                    mode,
                         const std::optional<std::vector<int64_t>>& seq_splits,
                         const std::optional<std::vector<int64_t>>& head_splits,
                         const std::optional<at::Tensor>&           out)
{
    const at::cuda::CUDAGuard guard(input.device());
    return run(group, input, mode, seq_splits, head_splits, out, kSyncWindow, at::cuda::getCurrentCUDAStream());
}

// The async form's device-side half. Python has already made the comm stream current; `caller` is
// the stream the user was on, which is where the input is staged.
at::Tensor all_to_all_4d_staged(const c10::intrusive_ptr<UlyssesGroup>&    group,
                                const at::Tensor&                          input,
                                int64_t                                    mode,
                                const std::optional<std::vector<int64_t>>& seq_splits,
                                const std::optional<std::vector<int64_t>>& head_splits,
                                const std::optional<at::Tensor>&           out,
                                int64_t                                    caller_stream_id,
                                int64_t                                    caller_device_index)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor, got one on ", input.device());
    const at::cuda::CUDAGuard guard(input.device());
    cudaStream_t              comm = at::cuda::getCurrentCUDAStream();
    // Rebuilt from torch's own identifiers rather than wrapped from a raw handle, so the caching
    // allocator sees the same stream object the caller was using.
    const c10::cuda::CUDAStream caller(c10::Stream::unpack3(
        caller_stream_id, static_cast<c10::DeviceIndex>(caller_device_index), c10::DeviceType::CUDA));
    // Stage on the caller's stream, so the caller's tensor is never retained cross-stream --
    // record_stream would instead pin every freed input until the comm stream caught up.
    const at::Tensor& staged = group->stage(input.contiguous(), caller, comm);
    at::Tensor        result = run(group, staged, mode, seq_splits, head_splits, out, kAsyncWindow, comm);
    group->release_staging(comm);
    return result;
}

// A buffer shaped like this call's output, in symmetric memory, for the caller to pass back as
// `out`. COLLECTIVE.
at::Tensor empty_output(const c10::intrusive_ptr<UlyssesGroup>&    group,
                        const at::Tensor&                          input,
                        int64_t                                    mode,
                        const std::optional<std::vector<int64_t>>& seq_splits,
                        const std::optional<std::vector<int64_t>>& head_splits)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor, got one on ", input.device());
    const at::cuda::CUDAGuard guard(input.device());
    const A2APlan&            plan = group->plan(input.sizes(), mode, input.scalar_type(), seq_splits, head_splits);
    return group->make_output(plan.output_shape, input.scalar_type(), plan.window_numel);
}

// Benchmark-only: the copying call with CUDA events between its stages, strictly ordered on one
// stream so they sum to the whole call. `transfer` covers the peer copies plus this rank's own
// share, which runs on the caller's stream underneath them and so cannot be timed apart.
std::tuple<at::Tensor, std::vector<double>> all_to_all_4d_timed(const c10::intrusive_ptr<UlyssesGroup>&    group,
                                                                const at::Tensor&                          input,
                                                                int64_t                                    mode,
                                                                const std::optional<std::vector<int64_t>>& seq_splits,
                                                                const std::optional<std::vector<int64_t>>& head_splits)
{
    const at::cuda::CUDAGuard guard(input.device());
    const Call                call   = prepare(group, input, mode, seq_splits, head_splits, std::nullopt, kSyncWindow);
    const int                 rank   = static_cast<int>(group->rank());
    cudaStream_t              stream = at::cuda::getCurrentCUDAStream();

    cudaEvent_t marks[5];
    for (auto& ev : marks) {
        ULYSSES_CUDA_CHECK(cudaEventCreate(&ev));
    }
    ULYSSES_CUDA_CHECK(cudaEventRecord(marks[0], stream));
    fast_barrier(stream, call.win->flag_ptrs, rank);
    ULYSSES_CUDA_CHECK(cudaEventRecord(marks[1], stream));
    launch_a2a_ce(call.x.data_ptr(), call.win->peer_ptrs, *call.plan, group->xfer_stream(), rank, stream);
    ULYSSES_CUDA_CHECK(cudaEventRecord(marks[2], stream));
    fast_barrier(stream, call.win->flag_ptrs, rank);
    ULYSSES_CUDA_CHECK(cudaEventRecord(marks[3], stream));
    copy_out(call, stream, rank);
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
    return {call.output, stages};
}

}  // namespace ulysses

TORCH_LIBRARY(fast_ulysses, m)
{
    m.class_<ulysses::UlyssesGroup>("UlyssesGroup")
        .def(torch::init<std::string, int64_t, int64_t, int64_t>())
        .def("destroy", &ulysses::UlyssesGroup::destroy);

    m.def("all_to_all_4d(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, Tensor input, "
          "int mode, int[]? seq_splits=None, int[]? head_splits=None, Tensor? out=None) -> Tensor");
    m.impl("all_to_all_4d", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_4d);

    m.def("all_to_all_4d_staged(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
          "Tensor input, int mode, int[]? seq_splits, int[]? head_splits, Tensor? out, "
          "int caller_stream_id, int caller_device_index) -> Tensor");
    m.impl("all_to_all_4d_staged", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_4d_staged);

    m.def("empty_output(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, Tensor input, "
          "int mode, int[]? seq_splits=None, int[]? head_splits=None) -> Tensor");
    m.impl("empty_output", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::empty_output);

    m.def("all_to_all_4d_timed(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
          "Tensor input, int mode, int[]? seq_splits=None, int[]? head_splits=None) "
          "-> (Tensor, float[])");
    m.impl("all_to_all_4d_timed", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_4d_timed);
}

// Python `import _C` needs PyInit__C; TORCH_LIBRARY already registered at dlopen time.
PYBIND11_MODULE(_C, m)
{
    // The async wrapper is Python-side, so this is the one piece that has to be here: c10d::Work
    // is a C++ interface. Not a torch op -- it takes a raw stream handle and mutates a
    // process-wide registry, neither of which belongs in a schema.
    m.def("register_stream_completion", [](const at::Tensor& tensor, int64_t comm_stream) {
        return ulysses::register_stream_completion(tensor, reinterpret_cast<cudaStream_t>(comm_stream));
    });

    // {(i, j): joined by NVLink} over CUDA device indices, or None when NVML cannot say.
    m.def("nvlink_matrix", [](const std::vector<int64_t>& devices) -> py::object {
        const auto matrix = ulysses::nvlink_matrix(devices);
        if (!matrix.has_value()) {
            return py::none();
        }
        py::dict out;
        for (const auto& entry : *matrix) {
            out[py::make_tuple(entry.first.first, entry.first.second)] = py::bool_(entry.second);
        }
        return out;
    });

    // Empty when every pair is NVLink-joined, or when NVML cannot say.
    m.def("check_nvlink", &ulysses::check_nvlink);

    // TESTS ONLY. Underscored, and not a torch op, because arming it deliberately breaks the
    // operator: it is the negative control for test/distributed/ce_ordering.py.
    m.def("_set_ce_fault", &ulysses::set_ce_fault);

    m.def("build_info", []() {
        std::map<std::string, std::string> out;
        out["version"]        = FAST_ULYSSES_VERSION;
        out["cuda_arch_list"] = FAST_ULYSSES_CUDA_ARCH_LIST;
        return out;
    });
}
