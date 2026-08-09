#pragma once
/// @file
/// The group: symmetric windows, the cached plans, and the transfer stream.
///
/// Everything a call needs on the host lives here, so the Python side is a constructor and two
/// one-line forwards. Windows are torch symmetric-memory allocations
/// (`c10d::symmetric_memory::empty_strided_p2p` + `rendezvous`), which is the same mechanism
/// torch's own `symm_mem.empty()` uses and is available unchanged from torch 2.10 on.
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <map>
#include <optional>
#include <string>
#include <torch/custom_class.h>
#include <vector>

#include <fast_ulysses/a2a_plan.hpp>

namespace ulysses {

/// @brief One symmetric allocation plus the addresses the transport needs.
struct Window {
    at::Tensor            tensor;     ///< holds the allocation alive
    std::vector<uint64_t> peer_ptrs;  ///< peer p's copy of it, as addressed from here
    std::vector<uint64_t> flag_ptrs;  ///< peer p's handshake region inside its signal pad
    int64_t               numel = 0;
};

/// @brief Which stream a window belongs to. A window is single-buffered, so two calls may share
/// one only when a stream orders them -- and the sync and async calls run on different streams.
enum WindowRole : int64_t {
    kSyncWindow  = 0,
    kAsyncWindow = 1,
};

class UlyssesGroup: public torch::CustomClassHolder {
public:
    /// @param group_name  the process group's name, which is what rendezvous keys on
    UlyssesGroup(std::string group_name, int64_t rank, int64_t world_size, int64_t device_index);
    ~UlyssesGroup() override;

    int64_t rank() const
    {
        return rank_;
    }

    int64_t world_size() const
    {
        return world_size_;
    }

    /// @brief The plan for this call, built once per distinct (shape, mode, dtype, splits).
    ///
    /// build_plan allocates several vectors, which is a few microseconds of host time on every
    /// call; the answer only depends on the key, so it is kept.
    const A2APlan& plan(at::IntArrayRef                            sizes,
                        int64_t                                    mode,
                        at::ScalarType                             dtype,
                        const std::optional<std::vector<int64_t>>& seq_splits,
                        const std::optional<std::vector<int64_t>>& head_splits);

    /// @brief This group's window for `role`, allocated on first use and grown when a call needs
    /// more than the last one did. COLLECTIVE when it allocates.
    const Window& window(WindowRole role, at::ScalarType dtype, int64_t numel);

    /// @brief A window the caller owns, for the zero-copy path. COLLECTIVE.
    at::Tensor make_output(at::IntArrayRef shape, at::ScalarType dtype, int64_t numel);

    /// @brief The window `out` is, or nullptr when it is an ordinary tensor.
    const Window* window_of(const at::Tensor& out) const;

    /// @brief Staging buffer for the async path, so the caller's tensor is never retained
    /// cross-stream. Copies `x` into it on `caller`, then makes `comm` wait for that copy.
    /// @return the staging tensor, whose release event is recorded by release_staging().
    const at::Tensor& stage(const at::Tensor& x, c10::cuda::CUDAStream caller, cudaStream_t comm);

    /// @brief Record, on `comm`, that the last staged buffer may be overwritten again.
    void release_staging(cudaStream_t comm);

    cudaStream_t xfer_stream();

    /// @brief Release the windows and the transfer stream. Buffers handed out by make_output are
    /// the caller's and are unaffected.
    void destroy();

private:
    struct PlanKey {
        std::vector<int64_t> sizes, seq, head;
        int64_t              mode;
        at::ScalarType       dtype;
        bool                 operator<(const PlanKey& o) const;
    };

    /// Allocate `numel` elements of symmetric memory, rendezvous, and clear the handshake region.
    Window allocate(int64_t numel, at::ScalarType dtype);

    std::string  group_name_;
    int          rank_, world_size_, device_index_;
    cudaStream_t xfer_ = nullptr;
    bool         destroyed_ = false;

    std::map<PlanKey, A2APlan>                          plans_;
    std::map<std::pair<int64_t, at::ScalarType>, Window> windows_;
    std::map<const void*, Window>                       owned_;  // handed to callers, by address

    // (shape, dtype) -> staging buffer + the event saying the comm stream is done reading it.
    struct Staging {
        at::Tensor  tensor;
        cudaEvent_t release = nullptr;
    };
    std::map<std::pair<std::vector<int64_t>, at::ScalarType>, Staging> staging_;
    Staging*                                                           last_staged_ = nullptr;
};

}  // namespace ulysses
