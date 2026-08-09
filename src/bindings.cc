// The torch op layer: validate the tensor, take the plan and the window from the group, then
// barrier, transfer, barrier, copy out.
//
// The group owns everything that survives a call -- windows, plans, staging buffers -- so the
// Python side is a constructor and two forwards.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <optional>
#include <torch/csrc/autograd/custom_function.h>
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

// Before any CUDAGuard: constructing one from a CPU tensor aborts inside c10 with a message that
// names neither this library nor the argument. docs/api.md promises this wording.
void require_cuda(const at::Tensor& input)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor, got one on ", input.device());
}

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
    // Here rather than deeper in: window() and make_output() also check, but the zero-copy path
    // reaches neither, and by the time anything else would notice the opening barrier has already
    // been issued -- so the rank that noticed would be waiting on peers instead of throwing.
    group->check_alive();
    // Released buffers go back to the allocator here, before this call takes any Window& out of
    // owned_. make_output() is the only other place that can prune, so a caller who takes one
    // buffer from empty_output() and drops it would otherwise pin it until destroy().
    group->prune_owned();
    require_cuda(input);
    // Per-tensor, so it stays out of the plan cache key. The windows live on one device and the
    // barrier kernel dereferences their flag pointers; a tensor from another device would launch
    // that kernel on the wrong one.
    TORCH_CHECK(input.device().index() == group->device_index(),
                "input is on cuda:",
                input.device().index(),
                " but this UlyssesGroup was built for cuda:",
                group->device_index(),
                "; one group serves exactly one device");
    call.x    = input.contiguous();
    call.plan = &group->plan(call.x.sizes(), mode, call.x.scalar_type(), seq_splits, head_splits);

    if (out.has_value()) {
        call.output = *out;
        TORCH_CHECK(call.output.is_cuda() && call.output.is_contiguous(), "out must be a contiguous CUDA tensor");
        TORCH_CHECK(call.output.device().index() == group->device_index(),
                    "out is on cuda:",
                    call.output.device().index(),
                    " but this UlyssesGroup was built for cuda:",
                    group->device_index());
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

// Records the staging release on scope exit, INCLUDING when the transfer throws. Without it a
// throw after the copies were issued leaves the release event unrecorded, and the next call for
// that shape overwrites the staging buffer while the comm stream is still reading it -- silently,
// because waiting on an unrecorded event succeeds and does nothing.
struct StagingRelease {
    UlyssesGroup* group;
    cudaStream_t  comm;

    ~StagingRelease()
    {
        group->release_staging(comm);  // noexcept; see the header
    }
};

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

// The one collective, in two forms.
//
// They are two ops rather than one with an optional `out` because a single op cannot be both
// honest and differentiable. Declaring the alias `out` really has -- Tensor(a!) -- makes the
// schema mutable, and torch/library.py refuses to register an autograd formula for a mutable
// operator. Splitting gives the functional form autograd and a meta kernel, and gives the
// out-variant an annotation that matches what it does. `-> ()` rather than `-> Tensor(a!)`
// because can_auto_functionalize accepts only that shape, so only that shape can enter a graph.
at::Tensor all_to_all_4d(const c10::intrusive_ptr<UlyssesGroup>&    group,
                         const at::Tensor&                          input,
                         int64_t                                    mode,
                         const std::optional<std::vector<int64_t>>& seq_splits,
                         const std::optional<std::vector<int64_t>>& head_splits)
{
    require_cuda(input);
    const at::cuda::CUDAGuard guard(input.device());
    return run(
        group, input, mode, seq_splits, head_splits, std::nullopt, kSyncWindow, at::cuda::getCurrentCUDAStream());
}

// Shape propagation for FakeTensor and AOTAutograd. CompositeExplicitAutograd already covers the
// Meta key, so without this a fake tensor reaches the real kernel and dies on require_cuda -- a
// message about a CUDA tensor, for what is really an untraceable-op problem. A direct Meta
// registration takes precedence over the alias.
//
// The arithmetic stays in group->plan(): it is pure host code that allocates nothing on the device,
// so it is safe here, and it runs every shape, dtype and splits check on the way. Rewriting the
// uneven-splits rule in Python would put a second copy of it somewhere test_plan.py cannot reach --
// which is the whole reason a2a_plan.cc is a torch-free, CUDA-free translation unit.
//
// The group is a real object on this path: FakeTensorMode's conversion only touches tensors, so the
// ScriptObject arrives intact and world_size is available -- which it is not from the input shape
// alone when the splits are absent.
at::Tensor all_to_all_4d_meta(const c10::intrusive_ptr<UlyssesGroup>&    group,
                              const at::Tensor&                          input,
                              int64_t                                    mode,
                              const std::optional<std::vector<int64_t>>& seq_splits,
                              const std::optional<std::vector<int64_t>>& head_splits)
{
    const A2APlan& plan = group->plan(input.sizes(), mode, input.scalar_type(), seq_splits, head_splits);
    return at::empty(plan.output_shape, input.options());
}

// empty_output is a COLLECTIVE allocation, so it cannot be traced at all: replaying a trace would
// put the rendezvous wherever the graph happens to place it rather than where every rank's own
// program reaches it. Say that, instead of letting it fall through to a CUDA-tensor complaint.
at::Tensor empty_output_meta(const c10::intrusive_ptr<UlyssesGroup>&,
                             const at::Tensor&,
                             int64_t,
                             const std::optional<std::vector<int64_t>>&,
                             const std::optional<std::vector<int64_t>>&)
{
    TORCH_CHECK(false,
                "empty_output is a collective allocation and cannot be traced: every rank has to "
                "reach it at the same point in its own program. Call it eagerly, outside the "
                "region being traced, and pass the buffer in.");
}

// Fills `out`. A buffer from empty_output() IS the window, so the peers write it directly and the
// copy-out disappears; any other contiguous tensor of the output shape is copied into.
void all_to_all_4d_out(const c10::intrusive_ptr<UlyssesGroup>&    group,
                       const at::Tensor&                          input,
                       int64_t                                    mode,
                       const std::optional<std::vector<int64_t>>& seq_splits,
                       const std::optional<std::vector<int64_t>>& head_splits,
                       const at::Tensor&                          out)
{
    require_cuda(input);
    const at::cuda::CUDAGuard guard(input.device());
    run(group, input, mode, seq_splits, head_splits, out, kSyncWindow, at::cuda::getCurrentCUDAStream());
}

// The async form's device-side half. Python has already made the comm stream current; `caller` is
// the stream the user was on, which is where the input is staged.
//
// Everything both staged forms share, so the split above costs one wrapper rather than a
// duplicated body.
namespace {

at::Tensor run_staged(const c10::intrusive_ptr<UlyssesGroup>&    group,
                      const at::Tensor&                          input,
                      int64_t                                    mode,
                      const std::optional<std::vector<int64_t>>& seq_splits,
                      const std::optional<std::vector<int64_t>>& head_splits,
                      const std::optional<at::Tensor>&           out,
                      int64_t                                    caller_stream_id,
                      int64_t                                    caller_device_index)
{
    require_cuda(input);
    const at::cuda::CUDAGuard guard(input.device());
    cudaStream_t              comm = at::cuda::getCurrentCUDAStream();
    // Rebuilt from torch's own identifiers rather than wrapped from a raw handle, so the caching
    // allocator sees the same stream object the caller was using.
    const c10::cuda::CUDAStream caller(c10::Stream::unpack3(
        caller_stream_id, static_cast<c10::DeviceIndex>(caller_device_index), c10::DeviceType::CUDA));
    // Stage on the caller's stream, so the caller's tensor is never read by the comm stream --
    // record_stream would instead pin every freed input until the comm stream caught up. The
    // staging copy also absorbs a strided input, which is why nothing calls contiguous() here:
    // doing so would launch that copy on the comm stream, which is exactly what this avoids.
    const at::Tensor&    staged = group->stage(input, caller, comm);
    const StagingRelease release{group.get(), comm};
    return run(group, staged, mode, seq_splits, head_splits, out, kAsyncWindow, comm);
}

}  // namespace

at::Tensor all_to_all_4d_staged(const c10::intrusive_ptr<UlyssesGroup>&    group,
                                const at::Tensor&                          input,
                                int64_t                                    mode,
                                const std::optional<std::vector<int64_t>>& seq_splits,
                                const std::optional<std::vector<int64_t>>& head_splits,
                                int64_t                                    caller_stream_id,
                                int64_t                                    caller_device_index)
{
    return run_staged(group, input, mode, seq_splits, head_splits, std::nullopt, caller_stream_id, caller_device_index);
}

void all_to_all_4d_staged_out(const c10::intrusive_ptr<UlyssesGroup>&    group,
                              const at::Tensor&                          input,
                              int64_t                                    mode,
                              const std::optional<std::vector<int64_t>>& seq_splits,
                              const std::optional<std::vector<int64_t>>& head_splits,
                              const at::Tensor&                          out,
                              int64_t                                    caller_stream_id,
                              int64_t                                    caller_device_index)
{
    run_staged(group, input, mode, seq_splits, head_splits, out, caller_stream_id, caller_device_index);
}

// Re-enter the DISPATCHER with the autograd keys excluded.
//
// Not a direct call to all_to_all_4d(): that would leave dispatch for good, and every key below
// Autograd -- Python, and through it FakeTensorMode, and Meta -- would never be consulted, so a
// fake tensor would reach the real kernel and die in cudaMemcpy3DAsync. Measured with a
// TorchDispatchMode: with a direct call it sees this op's inner aten::empty and never the op.
//
// The guard is what stops op.call from arriving back at the Autograd kernel. GradMode being off
// is not enough: it governs graph construction, not which kernel the dispatcher picks.
at::Tensor dispatch_below_autograd(const c10::intrusive_ptr<UlyssesGroup>&    group,
                                   const at::Tensor&                          input,
                                   int64_t                                    mode,
                                   const std::optional<std::vector<int64_t>>& seq_splits,
                                   const std::optional<std::vector<int64_t>>& head_splits)
{
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto                          op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("fast_ulysses::all_to_all_4d", "")
                         .typed<at::Tensor(const c10::intrusive_ptr<UlyssesGroup>&,
                                           const at::Tensor&,
                                           int64_t,
                                           const std::optional<std::vector<int64_t>>&,
                                           const std::optional<std::vector<int64_t>>&)>();
    return op.call(group, input, mode, seq_splits, head_splits);
}

// The adjoint of the collective is the collective in the other direction.
//
// mode 0 is a bijection on the global element set -- rank r's (batch, s, n, k) lands at rank
// owner(n) position (batch, seq_offset[r]+s, n-head_offset[p], k) -- and mode 1 with the SAME
// splits is `build_plan`'s scatter branch read right to left, i.e. its inverse. The Jacobian of a
// permutation is a permutation matrix, whose transpose is its inverse, so the vjp is `1 - mode`.
//
// The splits do NOT swap. That is worth stating because all_to_all_single's backward DOES swap its
// input and output split sizes: those describe one call's send and receive counts, so reversing the
// direction reverses their roles. seq_splits[p] / head_splits[p] are a different object -- rank p's
// sequence and head shard, a property of the GROUP that holds whichever way the data moves. `mode`
// selects which axis arrives already sharded, not what the lists mean. test_plan.py's round trip
// builds both plans from one pair of genuinely uneven lists and confirms it numerically.
//
// Absent splits pass through too: mode 0 with none derives seq=[x1]*ws, head=[x2/ws]*ws, and mode 1
// with none on that output shape derives the same pair -- and can never fail a divisibility check
// the forward did not already pass, since the axis it tests is ws times something.
class A2AFunction: public torch::autograd::Function<A2AFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext*          ctx,
                              const c10::intrusive_ptr<UlyssesGroup>&    group,
                              const at::Tensor&                          input,
                              int64_t                                    mode,
                              const std::optional<std::vector<int64_t>>& seq_splits,
                              const std::optional<std::vector<int64_t>>& head_splits)
    {
        // The graph holds the group alive. A graph that outlives destroy() raises from backward,
        // which is the right answer -- there is nothing left to run the collective on.
        ctx->saved_data["group"] = group;
        ctx->saved_data["mode"]  = mode;
        ctx->saved_data["seq"]   = seq_splits.has_value() ? c10::IValue(*seq_splits) : c10::IValue();
        ctx->saved_data["head"]  = head_splits.has_value() ? c10::IValue(*head_splits) : c10::IValue();

        return dispatch_below_autograd(group, input, mode, seq_splits, head_splits);
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list    grads)
    {
        // Five entries, one per non-ctx forward argument; only `input` is differentiable.
        if (!grads[0].defined()) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }
        auto          group = ctx->saved_data["group"].toCustomClass<UlyssesGroup>();
        const int64_t mode  = ctx->saved_data["mode"].toInt();

        std::optional<std::vector<int64_t>> seq, head;
        if (!ctx->saved_data["seq"].isNone()) {
            seq = ctx->saved_data["seq"].toIntVector();
        }
        if (!ctx->saved_data["head"].isNone()) {
            head = ctx->saved_data["head"].toIntVector();
        }

        // The copying form, never `out` from empty_output(): that is a collective allocation, and
        // putting one inside backward would hand the rendezvous position to the autograd engine's
        // node order rather than to the caller's own program. A gradient also belongs to the
        // engine, which may accumulate into it or hold it, while a window is single-buffered.
        //
        // apply(), not the plain function, so grad-of-grad routes back through here.
        return {at::Tensor(),
                A2AFunction::apply(group, grads[0].contiguous(), 1 - mode, seq, head),
                at::Tensor(),
                at::Tensor(),
                at::Tensor()};
    }
};

at::Tensor all_to_all_4d_autograd(const c10::intrusive_ptr<UlyssesGroup>&    group,
                                  const at::Tensor&                          input,
                                  int64_t                                    mode,
                                  const std::optional<std::vector<int64_t>>& seq_splits,
                                  const std::optional<std::vector<int64_t>>& head_splits)
{
    // The node apply() allocates is dead weight when nothing will ever use it, and this operator
    // runs twice per attention layer per step. Measured on 4x B200: routing every call through
    // apply() cost about 1 us of host submit time per call, which is the reason the bookkeeping is
    // in C++ at all. The predicate is the same one apply() computes internally as `is_executable`.
    if (!at::GradMode::is_enabled() || !input.requires_grad()) {
        return dispatch_below_autograd(group, input, mode, seq_splits, head_splits);
    }
    return A2AFunction::apply(group, input, mode, seq_splits, head_splits);
}

// A buffer shaped like this call's output, in symmetric memory, for the caller to pass back as
// `out`. COLLECTIVE.
at::Tensor empty_output(const c10::intrusive_ptr<UlyssesGroup>&    group,
                        const at::Tensor&                          input,
                        int64_t                                    mode,
                        const std::optional<std::vector<int64_t>>& seq_splits,
                        const std::optional<std::vector<int64_t>>& head_splits)
{
    require_cuda(input);
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
    require_cuda(input);
    const at::cuda::CUDAGuard guard(input.device());
    const Call                call   = prepare(group, input, mode, seq_splits, head_splits, std::nullopt, kSyncWindow);
    const int                 rank   = static_cast<int>(group->rank());
    cudaStream_t              stream = at::cuda::getCurrentCUDAStream();

    // Default flags, not cudaEventDisableTiming: cudaEventElapsedTime below needs the timestamps.
    const Event marks[5];
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
    return {call.output, stages};
}

}  // namespace ulysses

TORCH_LIBRARY(fast_ulysses, m)
{
    m.class_<ulysses::UlyssesGroup>("UlyssesGroup")
        .def(torch::init<std::string, int64_t, int64_t, int64_t>())
        .def("destroy", &ulysses::UlyssesGroup::destroy);

    // Functional: no alias, so it can carry an autograd formula and a meta kernel.
    m.def("all_to_all_4d(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, Tensor input, "
          "int mode, int[]? seq_splits=None, int[]? head_splits=None) -> Tensor");
    m.impl("all_to_all_4d", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_4d);
    m.impl("all_to_all_4d", c10::DispatchKey::Autograd, &ulysses::all_to_all_4d_autograd);
    // Explicit Meta, overriding the alias above. Deliberately NOT also torch.library.register_fake:
    // on 2.13 that silently replaces an existing Meta kernel, leaving two shape rules and no error
    // saying which one ran.
    m.impl("all_to_all_4d", c10::DispatchKey::Meta, &ulysses::all_to_all_4d_meta);

    // Mutating, and says so. Not differentiable: the gradient would have to flow back out of a
    // buffer the caller owns and may already have overwritten.
    m.def("all_to_all_4d_out(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, Tensor input, "
          "int mode, int[]? seq_splits, int[]? head_splits, Tensor(a!) out) -> ()");
    m.impl("all_to_all_4d_out", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_4d_out);

    m.def("all_to_all_4d_staged(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
          "Tensor input, int mode, int[]? seq_splits, int[]? head_splits, "
          "int caller_stream_id, int caller_device_index) -> Tensor");
    m.impl("all_to_all_4d_staged", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_4d_staged);

    m.def("all_to_all_4d_staged_out(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
          "Tensor input, int mode, int[]? seq_splits, int[]? head_splits, Tensor(a!) out, "
          "int caller_stream_id, int caller_device_index) -> ()");
    m.impl("all_to_all_4d_staged_out", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_4d_staged_out);

    m.def("empty_output(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, Tensor input, "
          "int mode, int[]? seq_splits=None, int[]? head_splits=None) -> Tensor");
    m.impl("empty_output", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::empty_output);
    m.impl("empty_output", c10::DispatchKey::Meta, &ulysses::empty_output_meta);

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

    // TESTS ONLY, and underscored for the same reason: it synchronises, and it exposes a counter
    // the barrier kernel owns. a2a_cudagraph.py needs it because torn data is a SUFFICIENT signal
    // that the handshake died, not a necessary one -- a replay where no peer happened to be late
    // comes back clean either way, and only the epoch says whether the barrier was alive.
    m.def("_epoch", [](const c10::intrusive_ptr<ulysses::UlyssesGroup>& group, int64_t role, const at::Tensor& like) {
        return group->epoch(static_cast<ulysses::WindowRole>(role), like.scalar_type());
    });

    m.def("build_info", []() {
        std::map<std::string, std::string> out;
        out["version"]        = FAST_ULYSSES_VERSION;
        out["cuda_arch_list"] = FAST_ULYSSES_CUDA_ARCH_LIST;
        return out;
    });
}
