#pragma once

#include "nvtx.h"

#include <ATen/ATen.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
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

// The barrier addresses peers through PeerFlags::ptr[8] in transfer.cu, which is what bounds
// a group.
constexpr int64_t kMaxWorldSize = 8;

// Peers this far apart in rank sit on opposite sides of the host's socket boundary on the
// machine this targets, and a copy across it runs at roughly half the rate of one that stays
// on the near side under the concurrent all-to-all pattern. Only instrumentation reads this
// today; it is here because it is the axis any scheduling decision would be made along, and
// deriving it from cudaDevP2PAttrPerformanceRank rather than asserting it is the obvious
// next step.
constexpr int kNearPeers = 4;

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

struct Buffer {
    // Keep the allocation itself alive independently of tensor: set_() mutates the shared
    // TensorImpl behind every at::Tensor handle, but cannot replace this Storage owner.
    at::Tensor tensor;
    c10::Storage output_storage;
    const c10::StorageImpl* output_storage_impl = nullptr;
    void* output_data_ptr = nullptr;
    int64_t output_storage_offset = 0;
    int64_t output_nbytes = 0;

    // A workspace is permanently paired with the first input allocation used with it. Holding
    // Storage rather than Tensor also makes a later input.set_() observable instead of silently
    // following the rebound TensorImpl.
    c10::Storage input_storage;
    const c10::StorageImpl* input_storage_impl = nullptr;
    const void* input_data_ptr = nullptr;
    int64_t input_storage_offset = 0;
    int64_t input_nbytes = 0;
    std::vector<uint64_t> peers;
    std::vector<uint64_t> flags;
    std::vector<int64_t> shape;
    at::ScalarType dtype = at::kBFloat16;
    uint64_t epoch = 0;
};

class UlyssesGroup final : public torch::CustomClassHolder {
public:
    UlyssesGroup(std::string name,
                 int64_t rank,
                 int64_t world_size,
                 int64_t device,
                 std::vector<int64_t> devices);
    ~UlyssesGroup() noexcept override;

    at::Tensor allocate_output(const at::Tensor& input, int64_t mode);
    void all_to_all_4d(const at::Tensor& input,
                       at::Tensor output,
                       int64_t mode,
                       int64_t stream);
    // Why this shape cannot be exchanged, or "" if it can. Pure, local, and collective-free,
    // so a caller can decide before entering allocate_output -- which is collective, and which
    // hangs the other ranks if only some of them reach it. The answer is a function of
    // (mode, sizes, dtype, world_size, transport) alone: every term is identical on every rank,
    // so acting on it cannot make the ranks disagree. Nothing per-tensor or per-host may enter
    // it for that reason.
    std::string unsupported_reason(std::vector<int64_t> sizes,
                                   at::ScalarType dtype,
                                   int64_t mode) const;
    std::vector<int64_t> output_shape_for(std::vector<int64_t> sizes, int64_t mode) const;
    std::string backend() const;
    void destroy();

private:
    void validate(const at::Tensor& input, int64_t mode) const;
    std::vector<int64_t> output_shape(const at::Tensor& input, int64_t mode) const;
    void require_supported(std::vector<int64_t> sizes, at::ScalarType dtype, int64_t mode) const;
    Buffer& lookup_output(const at::Tensor& output) const;
    void bind_or_validate_input(Buffer& buffer, const at::Tensor& input) const;
    void leak_unsafe_resources() noexcept;

    std::string name_;
    int rank_;
    int world_size_;
    int device_;
    bool destroyed_ = false;
    std::vector<std::unique_ptr<Buffer>> buffers_;
    std::unordered_map<const c10::TensorImpl*, Buffer*> outputs_;
};

void launch_all_to_all(const void* input,
                       const std::vector<uint64_t>& peers,
                       int mode,
                       int64_t batch,
                       int64_t seq,
                       int64_t heads,
                       int64_t dim,
                       int64_t element_size,
                       int rank,
                       cudaStream_t stream);

void barrier(cudaStream_t stream,
             const std::vector<uint64_t>& flags,
             int rank,
             uint64_t epoch);

}  // namespace ulysses
