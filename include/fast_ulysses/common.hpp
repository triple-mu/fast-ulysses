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

}  // namespace ulysses
