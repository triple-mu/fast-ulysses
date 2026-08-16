#include "fast_ulysses.h"

#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/extension.h>

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

}  // namespace

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
    const at::cuda::CUDAGuard guard(device_);
    rdma_ = std::make_unique<RdmaTransport>(rank_, world_size_, device_, devices,
                                            enable_rdma, nics);
}

UlyssesGroup::~UlyssesGroup()
{
    destroy();
}

void UlyssesGroup::validate(const at::Tensor& input, int64_t mode) const
{
    TORCH_CHECK(!destroyed_, "group is destroyed");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");
    TORCH_CHECK(input.is_cuda() && input.get_device() == device_, "wrong CUDA device");
    TORCH_CHECK(input.dim() == 4 && input.is_contiguous(),
                "input must be contiguous [B, S, H, D]");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "input must be float16 or bfloat16");
    TORCH_CHECK(!input.requires_grad(), "inference only");
    for (int64_t size : input.sizes()) TORCH_CHECK(size > 0, "empty dimensions are unsupported");
    if (rdma_ && rdma_->enabled()) {
        TORCH_CHECK(input.size(0) == 1, "mlx5 RDMA backend currently supports batch=1");
        // Both modes program the same stride: one row of the global head dimension.
        const int64_t heads = mode == 0 ? input.size(2) : input.size(2) * world_size_;
        const int64_t stride = heads * input.size(3) * input.element_size();
        TORCH_CHECK(stride <= kMaxInterleavedStride,
                    "heads*dim*itemsize over all ranks is ", stride,
                    " bytes, above the ", kMaxInterleavedStride,
                    "-byte mlx5 MKey stride limit; set FAST_ULYSSES_DISABLE_RDMA=1 to use the "
                    "CUDA P2P backend for this shape");
    }
    TORCH_CHECK(input.size(mode == 0 ? 2 : 1) % world_size_ == 0,
                "split dimension must divide world_size");
}

std::vector<int64_t> UlyssesGroup::output_shape(const at::Tensor& input, int64_t mode) const
{
    validate(input, mode);
    const auto b = input.size(0), s = input.size(1);
    const auto h = input.size(2), d = input.size(3);
    if (mode == 0) return {b, s * world_size_, h / world_size_, d};
    return {b, s / world_size_, h * world_size_, d};
}

at::Tensor UlyssesGroup::allocate_output(const at::Tensor& input, int64_t mode)
{
    const auto shape = output_shape(input, mode);
    const at::cuda::CUDAGuard guard(device_);
    auto buffer = std::make_unique<Buffer>();
    at::Tensor tensor;
    if (rdma_ && rdma_->enabled()) {
        void* pointer = nullptr;
        FU_CUDA_CHECK(cudaMalloc(&pointer, input.numel() * input.element_size()));
        tensor = at::from_blob(
            pointer, {input.numel()},
            [device = device_](void* allocation) {
                int previous = 0;
                cudaGetDevice(&previous);
                cudaSetDevice(device);
                cudaFree(allocation);
                cudaSetDevice(previous);
            },
            input.options());
    } else {
        tensor = symm::empty_strided_p2p(
            {input.numel()}, {1}, input.scalar_type(),
            c10::Device(c10::DeviceType::CUDA, device_), name_, std::nullopt);
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
    if (rdma_ && rdma_->enabled()) {
        buffer->rdma = rdma_->register_buffer(
            output.data_ptr(), output.numel() * output.element_size(), mode,
            input.size(0), input.size(1), input.size(2), input.size(3),
            input.element_size());
    }
    Buffer* raw = buffer.get();
    buffers_.push_back(std::move(buffer));
    outputs_[output.unsafeGetTensorImpl()] = raw;
    return output;
}

void UlyssesGroup::all_to_all_4d(const at::Tensor& input,
                                 at::Tensor output,
                                 int64_t mode,
                                 int64_t stream_ptr)
{
    const auto shape = output_shape(input, mode);
    const auto it = outputs_.find(output.unsafeGetTensorImpl());
    TORCH_CHECK(it != outputs_.end(), "output must come from allocate_output()");
    Buffer& buffer = *it->second;
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
    if (rdma_ && rdma_->enabled()) {
        TORCH_CHECK(buffer.rdma, "RDMA output is not registered");
        // The input MR is cached on the bare pointer, so the tensor behind it has to stay alive:
        // once it is freed the caching allocator can hand the same address to something else and
        // the registration would silently point at other data. Dropping the previous owner only
        // after the new one is held keeps the address from being recycled in between.
        at::Tensor previous_input;
        if (!buffer.input_owner.defined() ||
            buffer.input_owner.data_ptr() != input.data_ptr()) {
            previous_input = std::move(buffer.input_owner);
            buffer.input_owner = input;
        }
        // A cross-quad RDMA write lands in the peer's output the moment it is posted -- a receive
        // queue entry holds back the completion, not the data -- so the next call would overwrite
        // a result its peers have not read yet. The opening barrier is what stops that. It covers
        // all eight ranks, not just the quad, because it is the cross-quad half that races.
        barrier(stream, buffer.flags, rank_, ++buffer.epoch);
        // The NIC reads the input on its own, outside the stream. This one wait covers both that
        // and the barrier above, since the posting below is host-side and would otherwise run
        // ahead of each.
        FU_CUDA_CHECK(cudaStreamSynchronize(stream));
        rdma_->start_exchange(input.data_ptr(), input.numel() * input.element_size(),
                              *buffer.rdma, mode, input.size(0), input.size(1),
                              input.size(2), input.size(3), input.element_size());
        launch_all_to_all(input.data_ptr(), buffer.peers, mode, input.size(0),
                          input.size(1), input.size(2), input.size(3),
                          input.element_size(), rank_, stream, true);
        barrier(stream, buffer.flags, rank_, ++buffer.epoch);
        rdma_->finish_exchange();
        rdma_->flush();
        return;
    }
    barrier(stream, buffer.flags, rank_, ++buffer.epoch);
    launch_all_to_all(input.data_ptr(), buffer.peers, mode, input.size(0), input.size(1),
                      input.size(2), input.size(3), input.element_size(), rank_, stream,
                      false);
    barrier(stream, buffer.flags, rank_, ++buffer.epoch);
}

std::string UlyssesGroup::backend() const
{
    if (rdma_ && rdma_->enabled()) return "mlx5";
    return "p2p";
}

std::vector<int64_t> UlyssesGroup::connection_info() const
{
    TORCH_CHECK(rdma_ && rdma_->enabled(), "mlx5 backend is disabled");
    return rdma_->connection_info();
}

void UlyssesGroup::connect(const std::vector<std::vector<int64_t>>& peers)
{
    TORCH_CHECK(rdma_ && rdma_->enabled(), "mlx5 backend is disabled");
    rdma_->connect(peers);
}

std::vector<int64_t> UlyssesGroup::buffer_info(at::Tensor output) const
{
    const auto it = outputs_.find(output.unsafeGetTensorImpl());
    TORCH_CHECK(it != outputs_.end() && it->second->rdma,
                "output must be an RDMA allocate_output() result");
    return rdma_->buffer_info(*it->second->rdma);
}

void UlyssesGroup::connect_buffer(
    at::Tensor output,
    const std::vector<std::vector<int64_t>>& peers)
{
    const auto it = outputs_.find(output.unsafeGetTensorImpl());
    TORCH_CHECK(it != outputs_.end() && it->second->rdma,
                "output must be an RDMA allocate_output() result");
    rdma_->connect_buffer(*it->second->rdma, peers);
    it->second->peers = rdma_->peer_pointers(*it->second->rdma);
    it->second->flags = rdma_->peer_flags(*it->second->rdma);
}

void UlyssesGroup::flush() const
{
    if (rdma_) rdma_->flush();
}

void UlyssesGroup::destroy()
{
    if (destroyed_) return;
    destroyed_ = true;
    // cudaIpcCloseMemHandle acts on the current device's context, and this runs from a destructor
    // that Python can trigger under any device.
    const at::cuda::CUDAGuard guard(device_);
    outputs_.clear();
    buffers_.clear();
    rdma_.reset();
}

}  // namespace ulysses

TORCH_LIBRARY(fast_ulysses, m)
{
    m.class_<ulysses::UlyssesGroup>("UlyssesGroup")
        .def(torch::init<std::string, int64_t, int64_t, int64_t, std::vector<int64_t>,
                         bool, std::vector<std::string>>())
        .def("allocate_output", &ulysses::UlyssesGroup::allocate_output)
        .def("all_to_all_4d", &ulysses::UlyssesGroup::all_to_all_4d)
        .def("backend", &ulysses::UlyssesGroup::backend)
        .def("connection_info", &ulysses::UlyssesGroup::connection_info)
        .def("connect", &ulysses::UlyssesGroup::connect)
        .def("buffer_info", &ulysses::UlyssesGroup::buffer_info)
        .def("connect_buffer", &ulysses::UlyssesGroup::connect_buffer)
        .def("flush", &ulysses::UlyssesGroup::flush)
        .def("destroy", &ulysses::UlyssesGroup::destroy);
}

PYBIND11_MODULE(_C, m) {}
