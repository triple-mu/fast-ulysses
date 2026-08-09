#pragma once
/// @file
/// Error checking shared by every translation unit here.
#include <c10/util/Exception.h>
#include <cuda_runtime.h>

namespace ulysses {

/// @brief Check a CUDA runtime call and throw (TORCH_CHECK) on failure, naming the call text and
/// the driver's error string. For kernel launches, pass cudaGetLastError().
#define ULYSSES_CUDA_CHECK(expr)                                                                                       \
    do {                                                                                                               \
        cudaError_t err_ = (expr);                                                                                     \
        TORCH_CHECK(err_ == cudaSuccess, "CUDA error (" #expr "): ", cudaGetErrorString(err_));                        \
    } while (0)

/// @brief A cudaEvent_t destroyed on scope exit.
///
/// Every create/destroy pair here has ULYSSES_CUDA_CHECK calls between it, so the hand-paired form
/// leaked one event per call that threw -- and a call throws exactly when something is already
/// going wrong and the process is likely to keep running.
class Event {
public:
    /// @param flags cudaEventDefault keeps timing, which is what the timed op needs.
    explicit Event(unsigned flags = cudaEventDefault)
    {
        ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }

    ~Event()
    {
        cudaEventDestroy(event_);  // unchecked: a destructor must not throw
    }

    Event(const Event&)            = delete;
    Event& operator=(const Event&) = delete;

    /// Implicit, so the call sites read as they did with the raw handle.
    operator cudaEvent_t() const
    {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

}  // namespace ulysses
