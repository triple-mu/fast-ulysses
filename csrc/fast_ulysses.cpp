#include "fast_ulysses.h"

#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/extension.h>

#include <limits>
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

int64_t tensor_nbytes(const at::Tensor& tensor)
{
    const int64_t elements = tensor.numel();
    const int64_t element_size = tensor.element_size();
    TORCH_CHECK(element_size > 0 &&
                    elements <= std::numeric_limits<int64_t>::max() / element_size,
                "tensor byte size overflows int64_t");
    return elements * element_size;
}

}  // namespace

UlyssesGroup::UlyssesGroup(std::string name,
                           int64_t rank,
                           int64_t world_size,
                           int64_t device,
                           std::vector<int64_t> devices)
    : name_(std::move(name)),
      rank_(rank),
      world_size_(world_size),
      device_(device)
{
    TORCH_CHECK(world_size_ >= 1 && world_size_ <= 8 &&
                    (world_size_ & (world_size_ - 1)) == 0,
                "world_size must be 1, 2, 4, or 8");
    TORCH_CHECK(rank_ >= 0 && rank_ < world_size_, "invalid rank");
    TORCH_CHECK(static_cast<int>(devices.size()) == world_size_, "invalid device list");
    TORCH_CHECK(static_cast<int>(std::set<int64_t>(devices.begin(), devices.end()).size()) ==
                    world_size_,
                "one rank per GPU is required");
    TORCH_CHECK(devices[rank_] == device_, "rank is using the wrong GPU");
    TORCH_CHECK(supports_p2p(devices), "CUDA P2P is required between every GPU");
}

UlyssesGroup::~UlyssesGroup() noexcept
{
    // Destruction can be triggered by Python under an arbitrary CUDA context and cannot perform
    // the rank-coordinated IPC shutdown protocol. If explicit shutdown was skipped, retain all
    // CUDA resources until process exit rather than freeing them under an unsafe context
    // or closing peer mappings out of order.
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
        return "the dimension this mode splits must divide world_size";
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
    TORCH_CHECK(input.is_cuda() && input.get_device() == device_, "wrong CUDA device");
    TORCH_CHECK(input.dim() == 4 && input.is_contiguous(),
                "input must be contiguous [B, S, H, D]");
    TORCH_CHECK(!input.requires_grad(), "inference only");
    // The shape and dtype half is the part a caller can ask about in advance; keeping it in one
    // function is what makes the answer to unsupported_reason() and the reason this throws the
    // same sentence.
    require_supported(input.sizes().vec(), input.scalar_type(), mode);
}

std::vector<int64_t> UlyssesGroup::output_shape_for(std::vector<int64_t> sizes,
                                                    int64_t mode) const
{
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");
    TORCH_CHECK(sizes.size() == 4, "shape must be 4-D [B, S, H, D]");
    TORCH_CHECK(sizes[mode == 0 ? 2 : 1] % world_size_ == 0,
                "the dimension this mode splits must divide world_size");
    const auto b = sizes[0], s = sizes[1], h = sizes[2], d = sizes[3];
    if (mode == 0) return {b, s * world_size_, h / world_size_, d};
    return {b, s / world_size_, h * world_size_, d};
}

std::vector<int64_t> UlyssesGroup::output_shape(const at::Tensor& input, int64_t mode) const
{
    validate(input, mode);
    return output_shape_for(input.sizes().vec(), mode);
}

at::Tensor UlyssesGroup::allocate_output(const at::Tensor& input, int64_t mode)
{
    // Cold path, but it lands inside whichever request first sees a shape, so it is worth
    // seeing: the symmetric-memory rendezvous is collective and is not visible as GPU work.
    FU_NVTX("fu::allocate_output");
    const auto shape = output_shape(input, mode);
    const at::cuda::CUDAGuard guard(device_);
    auto buffer = std::make_unique<Buffer>();
    at::Tensor tensor;
    {
        FU_NVTX("fu::allocate_output[symm_mem]");
        tensor = symm::empty_strided_p2p(
            {input.numel()}, {1}, input.scalar_type(),
            c10::Device(c10::DeviceType::CUDA, device_), name_, std::nullopt);
        // Collective, and the only one on the p2p allocation path: every rank has to reach it.
        FU_NVTX("fu::allocate_output[rendezvous]");
        auto memory = symm::rendezvous(tensor, name_);
        TORCH_CHECK(memory, "symmetric-memory rendezvous failed");
        for (void* ptr : memory->get_buffer_ptrs())
            buffer->peers.push_back(reinterpret_cast<uint64_t>(ptr));

        const int64_t offset = symm::get_signal_pad_size() / 8 - world_size_;
        TORCH_CHECK(offset >= 8, "symmetric-memory signal pad is too small");
        for (void* ptr : memory->get_signal_pad_ptrs())
            buffer->flags.push_back(reinterpret_cast<uint64_t>(ptr) + offset * 8);
        TORCH_CHECK(static_cast<int>(buffer->peers.size()) == world_size_,
                    "symmetric-memory group size mismatch");
        memory->get_signal_pad(rank_, {world_size_}, at::kLong, offset).zero_();
    }
    buffer->shape = shape;
    buffer->dtype = input.scalar_type();

    auto output = tensor.view(shape);
    // Keep this exact TensorImpl alive so registry identity cannot be reused.
    buffer->tensor = output;
    buffer->output_storage = output.storage();
    buffer->output_storage_impl = buffer->output_storage.unsafeGetStorageImpl();
    buffer->output_data_ptr = output.data_ptr();
    buffer->output_storage_offset = output.storage_offset();
    buffer->output_nbytes = tensor_nbytes(output);
    Buffer* raw = buffer.get();
    buffers_.push_back(std::move(buffer));
    outputs_[output.unsafeGetTensorImpl()] = raw;
    return output;
}

Buffer& UlyssesGroup::lookup_output(const at::Tensor& output) const
{
    TORCH_CHECK(!destroyed_, "group is destroyed");
    const auto it = outputs_.find(output.unsafeGetTensorImpl());
    TORCH_CHECK(it != outputs_.end(), "output must come from allocate_output()");
    Buffer& buffer = *it->second;

    // Check StorageImpl before dereferencing the current data pointer. In particular, set_() keeps
    // TensorImpl identity unchanged, so the registry lookup alone is not an ownership check.
    TORCH_CHECK(output.has_storage() &&
                    output.storage().unsafeGetStorageImpl() == buffer.output_storage_impl &&
                    buffer.output_storage.unsafeGetStorageImpl() == buffer.output_storage_impl,
                "output storage was rebound after allocate_output()");
    TORCH_CHECK(output.data_ptr() == buffer.output_data_ptr,
                "output data pointer changed after allocate_output()");
    TORCH_CHECK(output.storage_offset() == buffer.output_storage_offset &&
                    tensor_nbytes(output) == buffer.output_nbytes,
                "output storage offset or byte size changed after allocate_output()");
    return buffer;
}

void UlyssesGroup::bind_or_validate_input(Buffer& buffer, const at::Tensor& input) const
{
    TORCH_CHECK(input.has_storage(), "input must have storage");
    const int64_t bytes = tensor_nbytes(input);
    const auto* storage_impl = input.storage().unsafeGetStorageImpl();
    if (!buffer.input_storage) {
        buffer.input_storage = input.storage();
        buffer.input_storage_impl = storage_impl;
        buffer.input_data_ptr = input.data_ptr();
        buffer.input_storage_offset = input.storage_offset();
        buffer.input_nbytes = bytes;
        return;
    }
    TORCH_CHECK(storage_impl == buffer.input_storage_impl &&
                    buffer.input_storage.unsafeGetStorageImpl() == buffer.input_storage_impl &&
                    input.data_ptr() == buffer.input_data_ptr &&
                    input.storage_offset() == buffer.input_storage_offset &&
                    bytes == buffer.input_nbytes,
                "each output workspace requires one fixed input storage, pointer, offset, and "
                "byte size");
}

void UlyssesGroup::all_to_all_4d(const at::Tensor& input,
                                 at::Tensor output,
                                 int64_t mode,
                                 int64_t stream_ptr)
{
    const auto shape = output_shape(input, mode);
    Buffer& buffer = lookup_output(output);
    TORCH_CHECK(output.is_cuda() && output.get_device() == device_ && output.is_contiguous(),
                "invalid output");
    TORCH_CHECK(output.scalar_type() == input.scalar_type() && output.sizes().vec() == shape,
                "output shape or dtype mismatch");
    TORCH_CHECK(buffer.shape == shape && buffer.dtype == input.scalar_type(),
                "output was allocated for another mode or shape");
    TORCH_CHECK(input.data_ptr() != output.data_ptr(), "input and output alias");
    TORCH_CHECK(buffer.peers.size() == static_cast<size_t>(world_size_),
                "output has not been connected to its peers");

    const at::cuda::CUDAGuard guard(device_);
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    int stream_device = -1;
    FU_CUDA_CHECK(cudaStreamGetDevice(stream, &stream_device));
    TORCH_CHECK(stream_device == device_, "stream is on the wrong CUDA device");
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    FU_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                "CUDA Graph capture is unsupported");
    bind_or_validate_input(buffer, input);
    FU_NVTX(mode == 0 ? "fu::exchange(mode=0)" : "fu::exchange(mode=1)");
    {
        FU_NVTX("fu::open_barrier");
        barrier(stream, buffer.flags, rank_, ++buffer.epoch);
    }
    {
        FU_NVTX("fu::copies_enqueue");
        launch_all_to_all(input.data_ptr(), buffer.peers, mode, input.size(0), input.size(1),
                          input.size(2), input.size(3), input.element_size(), rank_,
                          stream);
    }
    {
        FU_NVTX("fu::close_barrier");
        barrier(stream, buffer.flags, rank_, ++buffer.epoch);
    }
}

std::string UlyssesGroup::backend() const { return "p2p"; }

void UlyssesGroup::leak_unsafe_resources() noexcept
{
    for (auto& buffer : buffers_) static_cast<void>(buffer.release());
}

void UlyssesGroup::destroy()
{
    if (destroyed_) return;
    // Symmetric-memory teardown acts on the current device's context, while explicit shutdown
    // may be initiated under any device.
    const at::cuda::CUDAGuard guard(device_);
    destroyed_ = true;
    outputs_.clear();
    buffers_.clear();
}

}  // namespace ulysses

TORCH_LIBRARY(fast_ulysses, m)
{
    m.class_<ulysses::UlyssesGroup>("UlyssesGroup")
        .def(torch::init<std::string, int64_t, int64_t, int64_t, std::vector<int64_t>>())
        .def("allocate_output", &ulysses::UlyssesGroup::allocate_output)
        .def("all_to_all_4d", &ulysses::UlyssesGroup::all_to_all_4d)
        .def("unsupported_reason", &ulysses::UlyssesGroup::unsupported_reason)
        .def("output_shape_for", &ulysses::UlyssesGroup::output_shape_for)
        .def("backend", &ulysses::UlyssesGroup::backend)
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
}
