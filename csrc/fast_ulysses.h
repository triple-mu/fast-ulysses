#pragma once

#include "nvtx.h"
#include "rdma.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <cuda_runtime.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/custom_class.h>
#include <torch/version.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

static_assert(TORCH_VERSION_MAJOR > 2 ||
                  (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 10),
              "fast-ulysses requires torch 2.10 or newer");

#define FU_CUDA_CHECK(expr)                                                    \
    do {                                                                       \
        const cudaError_t err = (expr);                                        \
        TORCH_CHECK(err == cudaSuccess, #expr, ": ", cudaGetErrorString(err)); \
    } while (0)

namespace ulysses {

// mlx5's entry test is exactly eight ranks with device i on rank i, and that lives in rdma.cpp.
// This is the separate question of how wide a p2p group may be, which nothing but the world-size
// predicate below depends on now that the barrier comes from torch.
constexpr int64_t kMaxWorldSize = 8;

// The window a barrier waits before deciding a peer is gone. It only has to exceed the largest
// legitimate arrival skew, which is one layer of compute; a minute is three orders of magnitude
// past that, and the alternative is a job nobody notices is dead.
constexpr int64_t kBarrierTimeoutMs = 60'000;

// The two admissibility rules that a caller has to answer before a group exists, so they are
// free functions rather than members. Everything else a shape can be refused for depends on
// the group's transport and lives in UlyssesGroup::unsupported_reason.
constexpr bool supported_world_size(int64_t world_size)
{
    return world_size >= 1 && world_size <= kMaxWorldSize &&
           (world_size & (world_size - 1)) == 0;
}

inline bool supported_dtype(at::ScalarType dtype)
{
    return dtype == at::kHalf || dtype == at::kBFloat16;
}

// What one output allocation needs to be exchanged into. Outputs come from a torch symmetric
// memory pool now, so the caller owns the tensor and the caching allocator owns its lifetime;
// this holds only what the transports cannot rediscover per call.
struct Buffer {
    void* pointer = nullptr;
    int64_t nbytes = 0;
    int64_t mode = 0;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_sizes;
    at::ScalarType dtype = at::kBFloat16;

    // The rendezvous result. Torch caches it per allocation, so this is a reference rather than
    // a second copy of anything; it supplies the peer pointers the copies write into and the
    // signal pads the barrier announces through.
    c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> memory;
    std::vector<uint64_t> peers;
    std::vector<uint64_t> pads;

    std::unique_ptr<RdmaBuffer> rdma;

    // mlx5 only: the address the NIC's registration was made against, and the count of segment
    // releases that had happened when it was made. Deliberately NOT a reference to the caller's
    // input -- see bind_input.
    const void* input_data_ptr = nullptr;
    int64_t input_nbytes = 0;
    uint64_t input_epoch = 0;
    int input_rebinds = 0;
    // Where a moving input is staged to. Empty until one actually moves, and owned from then
    // on, because staging into an address nothing holds writes over whatever the allocator has
    // since put there.
    at::Tensor input_landing;
};

class UlyssesGroup final : public torch::CustomClassHolder {
public:
    UlyssesGroup(std::string name,
                 int64_t rank,
                 int64_t world_size,
                 int64_t device,
                 std::vector<int64_t> devices,
                 bool enable_rdma,
                 std::vector<std::string> nics);
    ~UlyssesGroup() noexcept override;

    // Once per distinct output allocation and geometry. Python drives this because the two
    // steps that are collective -- the symmetric-memory rendezvous, and mlx5's remote key
    // exchange -- have to be reached by every rank in the same order, and every other collective
    // in this library is issued from there.
    bool register_output(at::Tensor output, std::vector<int64_t> input_sizes, int64_t mode);
    // Runs on the caller's stream, like any other torch operation.
    void all_to_all_4d(const at::Tensor& input, at::Tensor output, int64_t mode);
    // Why this shape cannot be exchanged, or "" if it can. Pure, local, and collective-free,
    // so a caller can decide before entering the collective registration path -- which hangs the
    // other ranks if only some of them reach it. The answer is a function of
    // (mode, sizes, dtype, world_size, transport) alone: every term is identical on every rank,
    // so acting on it cannot make the ranks disagree. Nothing per-tensor or per-host may enter
    // it for that reason.
    std::string unsupported_reason(std::vector<int64_t> sizes,
                                   at::ScalarType dtype,
                                   int64_t mode) const;
    std::vector<int64_t> output_shape_for(std::vector<int64_t> sizes, int64_t mode) const;
    std::string backend() const;
    std::vector<int64_t> connection_info() const;
    void connect(const std::vector<std::vector<int64_t>>& peers);
    std::vector<int64_t> buffer_info(at::Tensor output) const;
    void connect_buffer(at::Tensor output,
                        const std::vector<std::vector<int64_t>>& peers);
    void destroy();

private:
    void validate(const at::Tensor& input, int64_t mode) const;
    void require_supported(std::vector<int64_t> sizes, at::ScalarType dtype, int64_t mode) const;
    Buffer& lookup_output(const at::Tensor& output) const;
    const void* bind_input(Buffer& buffer, const at::Tensor& input,
                           cudaStream_t stream) const;
    void leak_unsafe_resources() noexcept;

    std::string name_;
    int rank_;
    int world_size_;
    int device_;
    // One counter for the whole group, not one per buffer. A buffer's pad outlives the buffer --
    // the block is recycled, and the announcements written into it are still there -- so a
    // per-buffer counter restarting at zero would find a stale pad already past it. A global one
    // only ever moves forward, so a stale value is always behind.
    uint64_t epoch_ = 0;
    long long barrier_deadline_cycles_ = 0;
    bool destroyed_ = false;
    std::unique_ptr<RdmaTransport> rdma_;
    // Keyed on the output's device address. A symmetric-memory pool block is a whole segment
    // (the pool is created with no_split), so an address names one allocation for as long as
    // anything holds it, and a recycled address is the same peer allocation on every rank.
    std::unordered_map<void*, std::unique_ptr<Buffer>> buffers_;
};

// How many times the caching allocator has taken memory back out of a registration's reach.
// Exposed only so a test can prove the counter is not inert: a guard nobody has watched fire is
// a guard nobody has tested, and this one is silent when it fails to fire.
uint64_t segment_releases_seen();

// quad_only restricts the copies to this rank's quad; the NIC carries the rest.
void launch_all_to_all(const void* input,
                       const std::vector<uint64_t>& peers,
                       int mode,
                       int64_t batch,
                       int64_t seq,
                       int64_t heads,
                       int64_t dim,
                       int64_t element_size,
                       int rank,
                       cudaStream_t stream,
                       bool quad_only);

// `opening` picks which of the two identically-bodied kernels to launch, so the two barriers of
// an exchange are told apart on the kernel timeline.
void barrier(cudaStream_t stream,
             const std::vector<uint64_t>& pads,
             int rank,
             uint64_t epoch,
             long long deadline,
             bool opening);

}  // namespace ulysses
