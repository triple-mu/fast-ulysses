#include "fast_ulysses.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/TypeCast.h>
#include <c10/util/safe_numerics.h>
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <set>

namespace ulysses {
namespace symm = c10d::symmetric_memory;

namespace {

bool supports_p2p(const std::vector<int64_t>& devices)
{
    for (int64_t src : devices) {
        for (int64_t dst : devices) {
            if (src == dst) continue;
            int ok = 0;
            FU_CUDA_CHECK(cudaDeviceCanAccessPeer(&ok, src, dst));
            if (!ok) return false;
        }
    }
    return true;
}

// How many times the caching allocator has handed a segment back to the driver.
//
// That is the one event that can leave an input memory region pointing at pages it no longer
// owns, and the NIC would read them with no completion saying anything was wrong. Torch will
// report it, which is enough: re-registering costs milliseconds on a path nobody times, and
// being silently wrong costs everything.
//
// The callback runs with the allocator's per-device lock held and possibly with the GIL, so it
// does one lock-free increment and nothing else -- no allocation, no lock, no Python. It does
// not record *which* segment on purpose: narrowing that down needs a registry and therefore a
// lock, and this fires on empty_cache() and OOM recovery, not on anything hot.
//
// SEGMENT_UNMAP as well as SEGMENT_FREE. Under `expandable_segments:True` the allocator gives
// memory back by unmapping part of a segment rather than freeing the whole one, and it is
// documented to record only SEGMENT_UNMAP for that. Watching SEGMENT_FREE alone leaves the guard
// silently inert exactly where it is configured on -- which parts of the calling stack do.
std::atomic<uint64_t>& segment_releases()
{
    static std::atomic<uint64_t> counter{0};
    static std::once_flag once;
    std::call_once(once, [] {
        c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
            [](const c10::CachingDeviceAllocator::TraceEntry& entry) {
                using Action = c10::CachingDeviceAllocator::TraceEntry::Action;
                if (entry.action_ == Action::SEGMENT_FREE ||
                    entry.action_ == Action::SEGMENT_UNMAP) {
                    counter.fetch_add(1, std::memory_order_relaxed);
                }
            });
    });
    return counter;
}

int64_t tensor_nbytes(const at::Tensor& tensor)
{
    return c10::checked_convert<int64_t>(tensor.nbytes(), "tensor byte size");
}

int64_t checked_scale(int64_t value, int64_t scale, const char* dimension)
{
    int64_t result = 0;
    TORCH_CHECK(value >= 0 && !c10::mul_overflows(value, scale, &result), dimension,
                " exceeds int64 range");
    return result;
}

void assert_no_byte_overlap(const at::Tensor& input, const at::Tensor& output)
{
    const auto input_begin = reinterpret_cast<uintptr_t>(input.data_ptr());
    const auto output_begin = reinterpret_cast<uintptr_t>(output.data_ptr());
    const auto input_bytes =
        c10::checked_convert<uintptr_t>(input.nbytes(), "input byte size");
    const auto output_bytes =
        c10::checked_convert<uintptr_t>(output.nbytes(), "output byte size");
    const auto maximum = std::numeric_limits<uintptr_t>::max();
    TORCH_CHECK(input_begin <= maximum - input_bytes &&
                    output_begin <= maximum - output_bytes,
                "tensor address range overflows uintptr_t");
    const auto input_end = input_begin + input_bytes;
    const auto output_end = output_begin + output_bytes;
    TORCH_CHECK(input_end <= output_begin || output_end <= input_begin,
                "input and output overlap");
}

}  // namespace

uint64_t segment_releases_seen() { return segment_releases().load(std::memory_order_relaxed); }

UlyssesGroup::UlyssesGroup(std::string name,
                           int64_t rank,
                           int64_t world_size,
                           int64_t device,
                           std::vector<int64_t> devices,
                           bool enable_rdma,
                           std::vector<std::string> nics)
    : name_(std::move(name)),
      rank_(rank),
      world_size_(world_size),
      device_(device)
{
    TORCH_CHECK(supported_world_size(world_size_), "world_size must be 1, 2, 4, or 8");
    TORCH_CHECK(rank_ >= 0 && rank_ < world_size_, "invalid rank");
    TORCH_CHECK(static_cast<int>(devices.size()) == world_size_, "invalid device list");
    TORCH_CHECK(static_cast<int>(std::set<int64_t>(devices.begin(), devices.end()).size()) ==
                    world_size_,
                "one rank per GPU is required");
    TORCH_CHECK(devices[rank_] == device_, "rank is using the wrong GPU");
    TORCH_CHECK(supports_p2p(devices), "CUDA P2P is required between every GPU");
    const at::cuda::CUDAGuard guard(device_);
    // clock64() counts cycles, so the timeout has to be expressed in them. The rate is nominal
    // and the clock drifts under load; a watchdog set a thousand times past any real wait does
    // not care. cudaDeviceProp lost its clockRate field, so ask for the attribute.
    int clock_khz = 0;
    FU_CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device_));
    TORCH_CHECK(clock_khz > 0, "could not read the device clock rate");
    barrier_deadline_cycles_ = static_cast<long long>(clock_khz) * kBarrierTimeoutMs;
    rdma_ = std::make_unique<RdmaTransport>(rank_, world_size_, device_, devices,
                                            enable_rdma, nics);
}

UlyssesGroup::~UlyssesGroup() noexcept
{
    // Destruction can be triggered by Python under an arbitrary CUDA context. If explicit
    // shutdown was skipped, retain the RDMA resources until process exit rather than tearing
    // down a verbs context under a context that may not be the one that built it.
    if (!destroyed_) leak_unsafe_resources();
}

std::string UlyssesGroup::unsupported_reason(std::vector<int64_t> sizes,
                                             at::ScalarType dtype,
                                             int64_t mode) const
{
    if (mode != 0 && mode != 1) return "mode must be 0 or 1";
    if (sizes.size() != 4) return "shape must be 4-D [B, S, H, D]";
    for (int64_t size : sizes) {
        if (size <= 0) return "empty dimensions are unsupported";
    }
    if (!supported_dtype(dtype)) return "dtype must be float16 or bfloat16";
    if (sizes[mode == 0 ? 2 : 1] % world_size_ != 0) {
        return "the dimension this mode splits must be divisible by world_size";
    }
    // Everything past here is the transport's own admissibility, and the transport answers it:
    // batch, the divisibility mlx5 needs on top of the group's, the UINT32 MKey bounds, and the
    // 16-bit stride. Restating any of it here is how the two would drift apart.
    if (rdma_ && rdma_->enabled()) {
        return rdma_->shape_reason(static_cast<int>(mode), sizes[0], sizes[1], sizes[2],
                                   sizes[3], c10::elementSize(dtype));
    }
    return {};
}

void UlyssesGroup::require_supported(std::vector<int64_t> sizes,
                                     at::ScalarType dtype,
                                     int64_t mode) const
{
    const std::string reason = unsupported_reason(std::move(sizes), dtype, mode);
    TORCH_CHECK(reason.empty(), reason);
}

void UlyssesGroup::validate(const at::Tensor& input, int64_t mode) const
{
    TORCH_CHECK(!destroyed_, "group is destroyed");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");
    TORCH_CHECK(input.is_cuda() && input.get_device() == device_, "wrong CUDA device");
    TORCH_CHECK(input.dim() == 4 && input.is_contiguous(),
                "input must be contiguous [B, S, H, D]");
    TORCH_CHECK(!input.requires_grad(), "inference only");
}

std::vector<int64_t> UlyssesGroup::output_shape_for(std::vector<int64_t> sizes,
                                                    int64_t mode) const
{
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");
    TORCH_CHECK(sizes.size() == 4, "shape must be 4-D [B, S, H, D]");
    TORCH_CHECK(std::all_of(sizes.begin(), sizes.end(), [](int64_t size) { return size >= 0; }),
                "negative dimensions are unsupported");
    TORCH_CHECK(sizes[mode == 0 ? 2 : 1] % world_size_ == 0,
                "the dimension this mode splits must be divisible by world_size");
    const auto b = sizes[0], s = sizes[1], h = sizes[2], d = sizes[3];
    if (mode == 0)
        return {b, checked_scale(s, world_size_, "output sequence dimension"),
                h / world_size_, d};
    return {b, s / world_size_, checked_scale(h, world_size_, "output head dimension"), d};
}

bool UlyssesGroup::register_output(at::Tensor output,
                                   std::vector<int64_t> input_sizes,
                                   int64_t mode)
{
    TORCH_CHECK(!destroyed_, "group is destroyed");
    const auto existing = buffers_.find(output.data_ptr());
    if (existing != buffers_.end()) {
        const Buffer& buffer = *existing->second;
        if (buffer.mode == mode && buffer.input_sizes == input_sizes &&
            output.sizes().equals(buffer.output_sizes) &&
            buffer.dtype == output.scalar_type() && buffer.nbytes == tensor_nbytes(output)) {
            return false;
        }
    }

    // Cold path, but it lands inside whichever call first sees a shape, and neither half of it
    // is visible as GPU work: the rendezvous is collective and an RDMA registration is
    // milliseconds.
    FU_NVTX("fu::register_output");
    TORCH_CHECK(input_sizes.size() == 4, "input shape must be 4-D [B, S, H, D]");
    require_supported(input_sizes, output.scalar_type(), mode);
    TORCH_CHECK(output.is_cuda() && output.get_device() == device_ && output.is_contiguous(),
                "output must be contiguous and on the group's device");
    TORCH_CHECK(output.sizes().vec() == output_shape_for(input_sizes, mode),
                "output shape does not match the mode and input shape");

    const at::cuda::CUDAGuard guard(device_);
    auto buffer = std::make_unique<Buffer>();
    buffer->pointer = output.data_ptr();
    buffer->nbytes = tensor_nbytes(output);
    buffer->mode = mode;
    buffer->input_sizes = input_sizes;
    buffer->output_sizes = output.sizes().vec();
    buffer->dtype = output.scalar_type();
    {
        // Collective: every rank has to reach it, with the same allocation, in the same order.
        FU_NVTX("fu::register_output[rendezvous]");
        buffer->memory = symm::rendezvous(output, name_);
        TORCH_CHECK(buffer->memory, "symmetric-memory rendezvous failed");
    }
    const size_t allocation_size = buffer->memory->get_buffer_size();
    const size_t allocation_offset = buffer->memory->get_offset();
    TORCH_CHECK(allocation_offset <= allocation_size &&
                    static_cast<size_t>(buffer->nbytes) <= allocation_size - allocation_offset,
                "output exceeds its symmetric-memory allocation");
    const auto peer_pointers = buffer->memory->get_buffer_ptrs();
    buffer->peers.reserve(peer_pointers.size());
    for (void* pointer : peer_pointers) {
        const auto address = reinterpret_cast<uintptr_t>(pointer);
        TORCH_CHECK(pointer && address <= std::numeric_limits<uintptr_t>::max() - allocation_offset,
                    "invalid symmetric-memory peer pointer");
        buffer->peers.push_back(address + allocation_offset);
    }
    TORCH_CHECK(static_cast<int>(buffer->peers.size()) == world_size_,
                "symmetric-memory group size mismatch");
    TORCH_CHECK(buffer->peers[rank_] == reinterpret_cast<uintptr_t>(output.data_ptr()),
                "symmetric-memory output offset mismatch");
    // The tail of the signal pad. Torch's own primitives number their channels from the front,
    // so taking the last world_size words keeps the two out of each other's way -- and nothing
    // has to zero it, because the epoch only moves forward and whatever a recycled pad still
    // holds is an announcement from an epoch already past.
    const size_t pad_words = buffer->memory->get_signal_pad_size() / sizeof(uint64_t);
    TORCH_CHECK(pad_words >= static_cast<size_t>(world_size_) + 8,
                "symmetric-memory signal pad is too small");
    const size_t pad_offset = pad_words - world_size_;
    const auto pad_pointers = buffer->memory->get_signal_pad_ptrs();
    buffer->pads.reserve(pad_pointers.size());
    for (void* pointer : pad_pointers) {
        TORCH_CHECK(pointer, "invalid symmetric-memory signal pad pointer");
        buffer->pads.push_back(reinterpret_cast<uintptr_t>(pointer) +
                               pad_offset * sizeof(uint64_t));
    }
    TORCH_CHECK(static_cast<int>(buffer->pads.size()) == world_size_,
                "symmetric-memory signal-pad group size mismatch");
    if (rdma_ && rdma_->enabled()) {
        FU_NVTX("fu::register_output[mlx5]");
        buffer->rdma = rdma_->register_buffer(
            buffer->pointer, buffer->nbytes, static_cast<int>(mode), input_sizes[0],
            input_sizes[1], input_sizes[2], input_sizes[3], output.element_size());
    }
    buffers_[buffer->pointer] = std::move(buffer);
    return true;
}

Buffer& UlyssesGroup::lookup_output(const at::Tensor& output) const
{
    TORCH_CHECK(!destroyed_, "group is destroyed");
    const auto it = buffers_.find(output.data_ptr());
    TORCH_CHECK(it != buffers_.end(), "output was not registered with this group");
    return *it->second;
}

const void* UlyssesGroup::bind_input(Buffer& buffer,
                                     const at::Tensor& input,
                                     cudaStream_t stream) const
{
    // Only mlx5 cares where the input lives. It registers a memory region for the NIC and, in
    // mode 0, builds MKeys that gather from that exact address, both cached for the buffer's
    // life. The p2p path takes the pointer as a call argument and registers nothing, so it
    // accepts a different allocation every call.
    if (!(rdma_ && rdma_->enabled())) return input.data_ptr();

    const int64_t bytes = tensor_nbytes(input);

    // Nothing here holds the input alive, and that is the point. Holding it -- the obvious way
    // to keep a registered address meaningful -- stops the caching allocator reusing that block,
    // so the next input is allocated somewhere else and has to be staged, and the one after
    // that, forever. Letting the block go is what lets it come back.
    //
    // What holding it bought was safety against the segment being released; the release counter
    // buys the same thing without keeping the address out of circulation.
    // Once a landing block has been adopted the registration is on memory this buffer holds, so
    // nothing the allocator does can take it away and there is nothing to watch for.
    const uint64_t releases = segment_releases().load(std::memory_order_relaxed);
    if (!buffer.input_landing.defined() && buffer.input_data_ptr &&
        releases != buffer.input_epoch) {
        FU_NVTX("fu::reregister_input");
        rdma_->forget_input(*buffer.rdma);
        buffer.input_data_ptr = nullptr;
    }
    if (!buffer.input_data_ptr) {
        buffer.input_data_ptr = input.data_ptr();
        buffer.input_nbytes = bytes;
        buffer.input_epoch = releases;
        return input.data_ptr();
    }
    TORCH_CHECK(bytes == buffer.input_nbytes,
                "an mlx5 output is registered for one input byte size: expected ",
                buffer.input_nbytes, ", got ", bytes);
    if (input.data_ptr() == buffer.input_data_ptr) return input.data_ptr();

    // The address moved, and the two reasons it might have want opposite responses. A producer
    // that allocated once during warmup and then settled somewhere else wants the registration
    // moved, which costs one ibv_reg_mr. A producer that allocates afresh every call wants a
    // block of its own to be staged into, because moving the registration every time is
    // milliseconds against a copy that is tens of microseconds. Try the first once; if it moves
    // again, commit to the second.
    if (buffer.input_rebinds == 0 && !buffer.input_landing.defined()) {
        FU_NVTX("fu::rebind_input");
        ++buffer.input_rebinds;
        rdma_->forget_input(*buffer.rdma);
        buffer.input_data_ptr = input.data_ptr();
        buffer.input_epoch = releases;
        return input.data_ptr();
    }
    if (!buffer.input_landing.defined()) {
        // Never stage into the registered address as it stands. Nothing holds it -- that is what
        // lets the zero-copy case work at all -- so the allocator may have handed it to a live
        // tensor since, and writing an input's worth of bytes over somebody else's data is
        // rank-local, silent, and reported by nothing: the region is LOCAL_WRITE only and no
        // completion describes it. Take a block this buffer owns and move the registration onto
        // it, once.
        FU_NVTX("fu::adopt_staging_block");
        buffer.input_landing = at::empty(
            {bytes}, at::TensorOptions().dtype(at::kByte).device(input.device()));
        rdma_->forget_input(*buffer.rdma);
        buffer.input_data_ptr = buffer.input_landing.data_ptr();
        buffer.input_epoch = releases;
    }
    FU_NVTX("fu::stage_moving_input");
    // validate() has already established that the input is 4-D, contiguous, on this device and
    // of a supported dtype, so this is a flat byte copy between two allocations of equal size.
    FU_CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(buffer.input_data_ptr), input.data_ptr(),
                                  static_cast<size_t>(bytes), cudaMemcpyDeviceToDevice, stream));
    return buffer.input_data_ptr;
}

void UlyssesGroup::all_to_all_4d(const at::Tensor& input, at::Tensor output, int64_t mode)
{
    validate(input, mode);
    Buffer& buffer = lookup_output(output);
    TORCH_CHECK(output.is_cuda() && output.get_device() == device_ && output.is_contiguous(),
                "invalid output");
    TORCH_CHECK(output.scalar_type() == input.scalar_type() &&
                    output.sizes().equals(buffer.output_sizes),
                "output shape or dtype mismatch");
    TORCH_CHECK(buffer.mode == mode && input.sizes().equals(buffer.input_sizes) &&
                    buffer.dtype == input.scalar_type(),
                "output was registered for another mode or shape");
    // StorageImpl alias checks miss independent DLPack wrappers around the same device address.
    // Both tensors are contiguous here, so checked byte intervals cover every aliasing route.
    assert_no_byte_overlap(input, output);

    const at::cuda::CUDAGuard guard(device_);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    FU_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                "CUDA Graph capture is unsupported");
    const void* source = bind_input(buffer, input, stream);
    FU_NVTX(mode == 0 ? "fu::exchange(mode=0)" : "fu::exchange(mode=1)");
    if (rdma_ && rdma_->enabled()) {
        TORCH_CHECK(buffer.rdma, "RDMA output is not registered");
        // A cross-quad RDMA write lands in the peer's output the moment it is posted -- a receive
        // queue entry holds back the completion, not the data -- so the next call would overwrite
        // a result its peers have not read yet. The opening barrier is what stops that. It covers
        // all eight ranks, not just the quad, because it is the cross-quad half that races.
        {
            FU_NVTX("fu::open_barrier");
            barrier(stream, buffer.pads, rank_, ++epoch_, barrier_deadline_cycles_, true);
        }
        {
            // The NIC reads the input on its own, outside the stream. This one wait covers both
            // that and the barrier above, since the posting below is host-side and would
            // otherwise run ahead of each. It is a host wait, so it shows on no CUDA timeline.
            FU_NVTX("fu::drain_before_post");
            FU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        rdma_->start_exchange(source, tensor_nbytes(input),
                              *buffer.rdma, static_cast<int>(mode), input.size(0), input.size(1),
                              input.size(2), input.size(3), input.element_size());
        launch_all_to_all(source, buffer.peers, mode, input.size(0), input.size(1),
                          input.size(2), input.size(3), input.element_size(), rank_, stream,
                          true);
        {
            FU_NVTX("fu::close_barrier");
            barrier(stream, buffer.pads, rank_, ++epoch_, barrier_deadline_cycles_, false);
        }
        rdma_->finish_exchange();
        rdma_->flush();
        return;
    }
    {
        FU_NVTX("fu::open_barrier");
        barrier(stream, buffer.pads, rank_, ++epoch_, barrier_deadline_cycles_, true);
    }
    launch_all_to_all(source, buffer.peers, mode, input.size(0), input.size(1),
                      input.size(2), input.size(3), input.element_size(), rank_, stream, false);
    {
        FU_NVTX("fu::close_barrier");
        barrier(stream, buffer.pads, rank_, ++epoch_, barrier_deadline_cycles_, false);
    }
}

std::string UlyssesGroup::backend() const
{
    if (rdma_ && rdma_->enabled()) return "mlx5";
    return "p2p";
}

std::vector<int64_t> UlyssesGroup::connection_info() const
{
    TORCH_CHECK(!destroyed_, "group is destroyed");
    TORCH_CHECK(rdma_ && rdma_->enabled(), "mlx5 backend is disabled");
    return rdma_->connection_info();
}

void UlyssesGroup::connect(const std::vector<std::vector<int64_t>>& peers)
{
    TORCH_CHECK(!destroyed_, "group is destroyed");
    TORCH_CHECK(rdma_ && rdma_->enabled(), "mlx5 backend is disabled");
    rdma_->connect(peers);
}

std::vector<int64_t> UlyssesGroup::buffer_info(at::Tensor output) const
{
    Buffer& buffer = lookup_output(output);
    TORCH_CHECK(buffer.rdma, "output is not an mlx5 registration");
    return rdma_->buffer_info(*buffer.rdma);
}

void UlyssesGroup::connect_buffer(
    at::Tensor output,
    const std::vector<std::vector<int64_t>>& peers)
{
    Buffer& buffer = lookup_output(output);
    TORCH_CHECK(buffer.rdma, "output is not an mlx5 registration");
    rdma_->connect_buffer(*buffer.rdma, peers);
}

void UlyssesGroup::leak_unsafe_resources() noexcept
{
    for (auto& entry : buffers_) static_cast<void>(entry.second.release());
    static_cast<void>(rdma_.release());
}

void UlyssesGroup::destroy()
{
    if (destroyed_) return;
    // Peer mappings belong to torch's symmetric memory now, so there is nothing here that has to
    // be closed in a particular order relative to the other ranks. What is left is local: memory
    // regions, MKeys, queue pairs.
    const at::cuda::CUDAGuard guard(device_);
    destroyed_ = true;
    buffers_.clear();
    rdma_.reset();
}

}  // namespace ulysses

TORCH_LIBRARY(fast_ulysses, m)
{
    m.class_<ulysses::UlyssesGroup>("UlyssesGroup")
        .def(torch::init<std::string, int64_t, int64_t, int64_t, std::vector<int64_t>,
                         bool, std::vector<std::string>>())
        .def("register_output", &ulysses::UlyssesGroup::register_output)
        .def("all_to_all_4d", &ulysses::UlyssesGroup::all_to_all_4d)
        .def("unsupported_reason", &ulysses::UlyssesGroup::unsupported_reason)
        .def("output_shape_for", &ulysses::UlyssesGroup::output_shape_for)
        .def("backend", &ulysses::UlyssesGroup::backend)
        .def("connection_info", &ulysses::UlyssesGroup::connection_info)
        .def("connect", &ulysses::UlyssesGroup::connect)
        .def("buffer_info", &ulysses::UlyssesGroup::buffer_info)
        .def("connect_buffer", &ulysses::UlyssesGroup::connect_buffer)
        .def("destroy", &ulysses::UlyssesGroup::destroy);
}

// The two rules a caller has to consult before a group exists, so they cannot be methods. Both
// are derived from the predicates the constructor itself uses, rather than restated, because a
// caller that has to transcribe a limit will eventually transcribe it wrongly -- and will not
// find out, since the transcription lives where the library cannot see it.
PYBIND11_MODULE(_C, m)
{
    std::vector<int64_t> world_sizes;
    for (int64_t size = 1; size <= ulysses::kMaxWorldSize; ++size) {
        if (ulysses::supported_world_size(size)) world_sizes.push_back(size);
    }
    m.attr("SUPPORTED_WORLD_SIZES") = pybind11::cast(world_sizes);
    m.def("supports_world_size", &ulysses::supported_world_size);
    m.def("supports_dtype", &ulysses::supported_dtype);
    m.def("segment_releases_seen", &ulysses::segment_releases_seen);
}
