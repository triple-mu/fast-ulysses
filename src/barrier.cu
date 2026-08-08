// Flag barrier over P2P-accessible memory. Contract and layout: include/fast_ulysses/transfer.hpp.
#include <fast_ulysses/common.hpp>
#include <fast_ulysses/transfer.hpp>

namespace ulysses {

namespace {

// System-scope release publish / acquire load. Requires sm_70 or newer.
//
// The publish is a MAX, not a store, so a flag can never move backwards. With a window's calls
// ordered this cannot arise and the MAX is not load-bearing; what it buys is that a caller who
// breaks that ordering cannot pin a peer's flag BELOW the epoch that peer is waiting for.
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

// world_size is bounded by the [1, 8] single-node limit the op layer enforces.
struct BarPeers {
    uint64_t p[8];
};

__global__ void barrier_kernel(uint64_t* local, BarPeers peers, int ws, int rank)
{
    // atomicAdd rather than a plain increment, so a caller who breaks the per-window ordering
    // contract cannot lose an epoch outright.
    __shared__ uint64_t epoch;
    if (threadIdx.x == 0)
        epoch = atomicAdd(reinterpret_cast<unsigned long long*>(local + ws), 1ULL) + 1;
    __syncthreads();  // must precede the early-out: every launched thread has to reach it

    int t = threadIdx.x;
    if (t >= ws)
        return;
    red_release_sys_max_u64(reinterpret_cast<uint64_t*>(peers.p[t]) + rank, epoch);  // peer t's flags[rank]
    uint64_t        v;
    const uint64_t* mine = local + t;  // own flags[t], written by peer t
    do {
        v = ld_acquire_sys_u64(mine);
    } while (v < epoch);
}

}  // namespace

void fast_barrier(cudaStream_t stream, const std::vector<uint64_t>& flag_ptrs, int rank)
{
    const int ws = static_cast<int>(flag_ptrs.size());
    if (ws == 1)
        return;
    BarPeers peers;
    for (int i = 0; i < ws; ++i)
        peers.p[i] = flag_ptrs[i];
    // A spin kernel, not stream memops (cuStreamWriteValue64/WaitValue64): the waiting memop needs
    // a remote-write-flush device attribute that is not always present, and measured worse under
    // concurrent compute. This kernel's inline PTX only requires sm_70, and it needs an SM slot
    // only at a kernel boundary.
    barrier_kernel<<<1, 32, 0, stream>>>(reinterpret_cast<uint64_t*>(flag_ptrs[rank]), peers, ws, rank);
    ULYSSES_CUDA_CHECK(cudaGetLastError());
}

}  // namespace ulysses
