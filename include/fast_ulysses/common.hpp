#pragma once
#include <c10/util/Exception.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace ulysses {

// Check a CUDA runtime call's status and throw (TORCH_CHECK) on failure, naming the call text and
// the driver error string. For kernel launches pass cudaGetLastError().
#define ULYSSES_CUDA_CHECK(expr)                                                                                       \
    do {                                                                                                               \
        cudaError_t err_ = (expr);                                                                                     \
        TORCH_CHECK(err_ == cudaSuccess, "CUDA error (" #expr "): ", cudaGetErrorString(err_));                        \
    } while (0)

// System-scope release publish / acquire load, used by fast_barrier. Requires sm_70 or newer.

// The publish is a MAX, not a store, so a flag can never move backwards. With a tag's calls ordered
// this cannot arise and the MAX is NOT load-bearing; what it buys is that a caller who breaks that
// contract cannot pin a peer's flag BELOW the epoch that peer is waiting for.
__device__ __forceinline__ void red_release_sys_max_u64(uint64_t* addr, uint64_t v)
{
    asm volatile("red.release.sys.global.max.u64 [%0], %1;" ::"l"(addr), "l"(v) : "memory");
}

__device__ __forceinline__ uint64_t ld_acquire_sys_u64(const uint64_t* addr)
{
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(addr) : "memory");
    return v;
}

}  // namespace ulysses
