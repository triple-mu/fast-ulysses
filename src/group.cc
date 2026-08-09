// Contracts and layout: include/fast_ulysses/group.hpp.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <fast_ulysses/common.hpp>
#include <fast_ulysses/group.hpp>

namespace ulysses {

namespace {

namespace symm = c10d::symmetric_memory;

// The handshake needs uint64 flags[ws] followed by uint64 epoch. It lives in the LAST slots of the
// allocation's signal pad, because torch's own SymmetricMemory::barrier() -- which window setup
// calls -- uses channels counted from the front, and the two must not overlap.
int64_t flag_slots(int world_size)
{
    return world_size + 1;
}

int64_t flag_offset_slots(int world_size)
{
    const int64_t pad_slots = static_cast<int64_t>(symm::get_signal_pad_size()) / 8;
    TORCH_CHECK(pad_slots >= flag_slots(world_size) + 8,
                "the symmetric-memory signal pad holds ",
                pad_slots,
                " uint64 slots, too few for a ",
                world_size,
                "-rank handshake plus torch's own channels. Raise it with "
                "torch.distributed._symmetric_memory.set_signal_pad_size() before any allocation.");
    return pad_slots - flag_slots(world_size);
}

// Validation and dims for the 4D a2a. Everything checked here is part of the plan cache key, so a
// cache hit cannot skip a check that would have failed: the same values already passed it. Checks
// that depend on the tensor rather than its shape (device, contiguity, aliasing) stay per-call.
//
// The plan treats uneven as the general case, so the only decision here is what the splits ARE:
// the caller's, or the even ones the shape implies.
A2ADims make_dims(at::IntArrayRef                            sizes,
                  at::ScalarType                             dtype,
                  int64_t                                    mode,
                  int                                        ws,
                  int                                        rank,
                  const std::optional<std::vector<int64_t>>& seq_splits,
                  const std::optional<std::vector<int64_t>>& head_splits)
{
    TORCH_CHECK(sizes.size() == 4, "input must be 4D, got ", sizes.size(), " dims");
    // Listed, not opened up. Nothing below this line is dtype-specific -- build_plan takes an
    // elem_size, the transport copies bytes, the window map and the plan key both carry the dtype,
    // and empty_strided_p2p takes a plain ScalarType -- so the set is a decision, not a constraint.
    // Leaving the check off entirely would admit kBool, the complex types and the quantized ones,
    // which nothing here has reasoned about.
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat || dtype == at::kFloat8_e4m3fn
                    || dtype == at::kFloat8_e5m2 || dtype == at::kChar || dtype == at::kByte,
                "dtype must be float16, bfloat16, float32, float8_e4m3fn, float8_e5m2, int8 or uint8, got ",
                dtype);
    const int64_t x1   = sizes[1];
    const int64_t x2   = sizes[2];
    const int64_t d    = sizes[3];
    const int64_t elem = static_cast<int64_t>(c10::elementSize(dtype));
    // One check covers the whole plan: every byte quantity build_plan emits -- src_offset,
    // dst_offset, width, both pitches, the batch strides -- is a multiple of d * elem. Note the
    // rule TIGHTENS as the element shrinks: d % 8 at fp16, d % 4 at fp32, d % 16 at fp8 and int8.
    TORCH_CHECK(
        (d * elem) % 16 == 0, "the head dim must be 16-byte aligned: d=", d, " x ", elem, " B is ", d * elem, " B");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1, got ", mode);

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
        dims.validate();  // length and sign, before indexing by rank below
        // Cross-check against the tensor handed in, so a caller that mis-shards gets an error here
        // instead of a silently corrupt result.
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

    // No splits: the even special case, which the shape alone determines only if the scattered
    // axis divides. The other axis is already sharded on entry, so it never has to.
    if (mode == 0) {
        TORCH_CHECK(x2 % ws == 0,
                    "mode 0 scatters the head axis, so n_global (",
                    x2,
                    ") must divide world_size (",
                    ws,
                    ") -- or pass seq_splits and head_splits");
        dims.seq_splits.assign(ws, x1);
        dims.head_splits.assign(ws, x2 / ws);
    }
    else {
        TORCH_CHECK(x1 % ws == 0,
                    "mode 1 scatters the sequence axis, so s_global (",
                    x1,
                    ") must divide world_size (",
                    ws,
                    ") -- or pass seq_splits and head_splits");
        dims.seq_splits.assign(ws, x1 / ws);
        dims.head_splits.assign(ws, x2);
    }
    return dims;
}

}  // namespace

bool UlyssesGroup::PlanKey::operator<(const PlanKey& o) const
{
    return std::tie(sizes, seq, head, mode, dtype) < std::tie(o.sizes, o.seq, o.head, o.mode, o.dtype);
}

UlyssesGroup::UlyssesGroup(std::string group_name, int64_t rank, int64_t world_size, int64_t device_index):
    group_name_(std::move(group_name)),
    rank_(static_cast<int>(rank)),
    world_size_(static_cast<int>(world_size)),
    device_index_(static_cast<int>(device_index))
{
    TORCH_CHECK(world_size_ >= 1 && world_size_ <= 8,
                "world_size must be in [1, 8] -- fast-ulysses is single-node -- got ",
                world_size_);
    TORCH_CHECK(rank_ >= 0 && rank_ < world_size_, "rank ", rank_, " out of range for world_size ", world_size_);
}

UlyssesGroup::~UlyssesGroup()
{
    destroy();
}

void UlyssesGroup::check_alive() const
{
    TORCH_CHECK(!destroyed_, "this UlyssesGroup has been destroyed");
}

const A2APlan& UlyssesGroup::plan(at::IntArrayRef                            sizes,
                                  int64_t                                    mode,
                                  at::ScalarType                             dtype,
                                  const std::optional<std::vector<int64_t>>& seq_splits,
                                  const std::optional<std::vector<int64_t>>& head_splits)
{
    PlanKey key;
    key.sizes = sizes.vec();
    key.mode  = mode;
    key.dtype = dtype;
    // Kept optional, so "no splits" and "empty splits" are distinct keys. Collapsing them would
    // let an even-split call warm the cache for a later empty-split one, which make_dims rejects.
    key.seq  = seq_splits;
    key.head = head_splits;

    auto it = plans_.find(key);
    if (it != plans_.end()) {
        return it->second;
    }
    // A caller whose shapes are data-dependent would otherwise grow this without bound. Dropping
    // everything costs one rebuild per live shape, which is microseconds.
    if (plans_.size() >= 512) {
        plans_.clear();
    }
    const A2ADims dims = make_dims(sizes, dtype, mode, world_size_, rank_, seq_splits, head_splits);
    return plans_
        .emplace(std::move(key),
                 build_plan(dims, static_cast<int>(mode), static_cast<int64_t>(c10::elementSize(dtype))))
        .first->second;
}

void UlyssesGroup::prune_owned()
{
    // The caller's buffer is a view over the window's storage, so a use count of one means we are
    // the only holder left and the allocation can go back to the symmetric allocator.
    for (auto it = owned_.begin(); it != owned_.end();) {
        it = (it->second.tensor.storage().use_count() == 1) ? owned_.erase(it) : std::next(it);
    }
}

Window UlyssesGroup::allocate(int64_t numel, at::ScalarType dtype)
{
    // Every rank computes the same window size (it is the max over all ranks), so this throws on
    // all of them together and cannot leave anyone waiting. A zero-sized symmetric allocation has
    // a null data_ptr, which rendezvous refuses with a null handle.
    TORCH_CHECK(numel > 0,
                "this call moves no data: every rank's shard is empty. Check the shape and the "
                "splits -- a sequence or head axis of 0 on ALL ranks has nothing to exchange");
    const at::cuda::CUDAGuard guard(device_index_);
    // empty_strided_p2p rather than a MemPool: it is the allocator entry point directly, so no
    // unrelated allocation can interleave with ours and desync the rendezvous order across ranks.
    at::Tensor t = symm::empty_strided_p2p(
        {numel}, {1}, dtype, c10::Device(c10::DeviceType::CUDA, device_index_), group_name_, std::nullopt);
    auto sym = symm::rendezvous(t, group_name_);
    TORCH_CHECK(sym,
                "torch symmetric memory did not establish a rendezvous for a ",
                numel,
                "-element window on group '",
                group_name_,
                "'. The allocation is not a symmetric one, or the group is not registered with "
                "torch's symmetric-memory bootstrap.");

    Window win;
    win.tensor = t;
    win.numel  = numel;
    for (void* p : sym->get_buffer_ptrs()) {
        win.peer_ptrs.push_back(reinterpret_cast<uint64_t>(p));
    }
    const int64_t off = flag_offset_slots(world_size_);
    for (void* p : sym->get_signal_pad_ptrs()) {
        win.flag_ptrs.push_back(reinterpret_cast<uint64_t>(p) + static_cast<uint64_t>(off * 8));
    }
    TORCH_CHECK(static_cast<int>(win.peer_ptrs.size()) == world_size_,
                "rendezvous returned ",
                win.peer_ptrs.size(),
                " peers for a group of ",
                world_size_,
                "; the process group and this group disagree about their membership");

    // This allocation may be reusing memory an earlier window freed, and the epoch has to start at
    // zero. Clear only our own region -- torch's channels are its business -- then hold every rank
    // until all have cleared, or one rank's first publish lands in a pad another has yet to clear,
    // is erased, and that rank waits forever for a write that already happened.
    sym->get_signal_pad(rank_, {flag_slots(world_size_)}, at::kLong, off).zero_();
    sym->barrier(0, 0);
    return win;
}

const Window& UlyssesGroup::window(WindowRole role, at::ScalarType dtype, int64_t numel)
{
    check_alive();
    const auto key = std::make_pair(static_cast<int64_t>(role), dtype);
    auto       it  = windows_.find(key);
    if (it != windows_.end() && it->second.numel >= numel) {
        return it->second;
    }
    if (it != windows_.end()) {
        windows_.erase(it);  // free before allocating, so the allocator can reuse the segment
    }
    return windows_.emplace(key, allocate(numel, dtype)).first->second;
}

at::Tensor UlyssesGroup::make_output(at::IntArrayRef shape, at::ScalarType dtype, int64_t numel)
{
    check_alive();
    prune_owned();  // before allocating, so a released buffer's memory can be reused right here
    Window     win = allocate(numel, dtype);
    at::Tensor out = win.tensor.narrow(0, 0, c10::multiply_integers(shape)).view(shape);
    owned_.emplace(win.tensor.data_ptr(), std::move(win));
    return out;
}

const Window* UlyssesGroup::window_of(const at::Tensor& out) const
{
    auto it = owned_.find(out.data_ptr());
    return it == owned_.end() ? nullptr : &it->second;
}

const at::Tensor& UlyssesGroup::stage(const at::Tensor& x, c10::cuda::CUDAStream caller, cudaStream_t comm)
{
    auto key = std::make_pair(x.sizes().vec(), x.scalar_type());
    // Unlike plans_, every entry here pins a full input's worth of device memory, so the bound
    // matters more. Dropping them all costs one extra copy per live shape on the next call, and
    // release_staging() has already been called for every entry by the time we get here.
    if (staging_.size() >= 16 && staging_.find(key) == staging_.end()) {
        for (auto& entry : staging_) {
            if (entry.second.release != nullptr) {
                ULYSSES_CUDA_CHECK(cudaEventSynchronize(entry.second.release));
                ULYSSES_CUDA_CHECK(cudaEventDestroy(entry.second.release));
            }
        }
        staging_.clear();
        last_staged_ = nullptr;
    }
    auto& s = staging_[key];
    // Everything below happens on the CALLER's stream, so the caller's tensor is only ever read
    // there and the comm stream sees only the staged copy.
    const c10::cuda::CUDAStreamGuard guard(caller);
    if (!s.tensor.defined()) {
        // Both built into locals and committed together, so a throw from either leaves the entry
        // untouched and the next call retries it. A half-built entry -- tensor defined, release
        // still null -- would take the else branch below forever and wait on a null event.
        //
        // at::empty, not empty_like: empty_like would copy a strided input's layout, and the
        // transport reads the staged buffer as dense.
        at::Tensor  staged  = at::empty(x.sizes(), x.options());
        cudaEvent_t release = nullptr;
        ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&release, cudaEventDisableTiming));
        s.tensor  = std::move(staged);
        s.release = release;
    }
    else {
        // Wait GPU-side for the comm stream to have finished reading the previous contents.
        ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(caller, s.release, 0));
    }
    s.tensor.copy_(x);
    // The comm stream must not read the staged copy before that copy has run, and the ready event
    // trails everything already submitted on the caller's stream -- including any earlier consumer
    // of this same buffer.
    const Event ready(cudaEventDisableTiming);
    ULYSSES_CUDA_CHECK(cudaEventRecord(ready, caller));
    ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(comm, ready, 0));

    last_staged_ = &s;
    return s.tensor;
}

void UlyssesGroup::release_staging(cudaStream_t comm) noexcept
{
    if (last_staged_ != nullptr) {
        // See the header: unchecked because this also runs while an exception from the transfer is
        // propagating, and because a checked record that failed would leave the same state.
        cudaEventRecord(last_staged_->release, comm);
        last_staged_ = nullptr;
    }
}

int64_t UlyssesGroup::epoch(WindowRole role, at::ScalarType dtype) const
{
    auto it = windows_.find(std::make_pair(static_cast<int64_t>(role), dtype));
    if (it == windows_.end()) {
        return -1;
    }
    // The epoch sits one slot past the ws flags, which is where barrier_kernel's atomicAdd lands.
    const auto         addr  = it->second.flag_ptrs[rank_] + static_cast<uint64_t>(world_size_) * 8;
    unsigned long long value = 0;
    ULYSSES_CUDA_CHECK(cudaMemcpy(&value, reinterpret_cast<const void*>(addr), sizeof(value), cudaMemcpyDeviceToHost));
    return static_cast<int64_t>(value);
}

cudaStream_t UlyssesGroup::xfer_stream()
{
    if (xfer_ == nullptr) {
        const at::cuda::CUDAGuard guard(device_index_);
        ULYSSES_CUDA_CHECK(cudaStreamCreateWithFlags(&xfer_, cudaStreamNonBlocking));
    }
    return xfer_;
}

void UlyssesGroup::destroy()
{
    if (destroyed_) {
        return;
    }
    destroyed_ = true;
    if (xfer_ != nullptr) {
        // Unchecked, like the rest of teardown: the caller has already quiesced the group, and a
        // throw from here would run during interpreter shutdown.
        cudaStreamSynchronize(xfer_);
        cudaStreamDestroy(xfer_);
        xfer_ = nullptr;
    }
    for (auto& entry : staging_) {
        if (entry.second.release != nullptr) {
            cudaEventDestroy(entry.second.release);
        }
    }
    staging_.clear();
    last_staged_ = nullptr;
    plans_.clear();
    windows_.clear();
    // Buffers the caller still holds stay alive through their own storage reference; dropping our
    // record here only releases the ones nobody kept.
    prune_owned();
}

}  // namespace ulysses
