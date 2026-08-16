#pragma once

#include "rdma.h"

#include <ATen/ATen.h>
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

struct Buffer {
    at::Tensor tensor;
    at::Tensor input_owner;
    std::vector<uint64_t> peers;
    std::vector<uint64_t> flags;
    std::vector<int64_t> shape;
    at::ScalarType dtype = at::kBFloat16;
    uint64_t epoch = 0;
    std::unique_ptr<RdmaBuffer> rdma;
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
    ~UlyssesGroup() override;

    at::Tensor allocate_output(const at::Tensor& input, int64_t mode);
    void all_to_all_4d(const at::Tensor& input,
                       at::Tensor output,
                       int64_t mode,
                       int64_t stream);
    std::string backend() const;
    std::vector<int64_t> connection_info() const;
    void connect(const std::vector<std::vector<int64_t>>& peers);
    std::vector<int64_t> buffer_info(at::Tensor output) const;
    void connect_buffer(at::Tensor output,
                        const std::vector<std::vector<int64_t>>& peers);
    void flush() const;
    void destroy();

private:
    void validate(const at::Tensor& input, int64_t mode) const;
    std::vector<int64_t> output_shape(const at::Tensor& input, int64_t mode) const;

    std::string name_;
    int rank_;
    int world_size_;
    int device_;
    bool destroyed_ = false;
    std::unique_ptr<RdmaTransport> rdma_;
    std::vector<std::unique_ptr<Buffer>> buffers_;
    std::unordered_map<const c10::TensorImpl*, Buffer*> outputs_;
};

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

void barrier(cudaStream_t stream,
             const std::vector<uint64_t>& flags,
             int rank,
             uint64_t epoch);

}  // namespace ulysses
