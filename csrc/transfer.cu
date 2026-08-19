#include "fast_ulysses.h"

#include <cstdio>

namespace ulysses {
namespace {

__device__ __forceinline__ void publish(uint64_t* address, uint64_t value)
{
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::
                 "l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ uint64_t acquire(const uint64_t* address)
{
    uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) :
                 "l"(address) : "memory");
    return value;
}

struct PeerPads { uint64_t ptr[8]; };

// One thread per peer: announce this epoch to it, then wait for its announcement.
//
// The epoch is the whole design. It counts every barrier the group has ever issued, never
// resets, and the wait is >= rather than ==, so an announcement that arrives early -- from a
// peer already inside the next exchange -- satisfies this one instead of being lost. That is
// what makes the two barriers of an exchange, and consecutive exchanges on a recycled buffer,
// safe without any channel bookkeeping.
__device__ __forceinline__ void barrier_body(uint64_t* local,
                                             PeerPads peers,
                                             int world_size,
                                             int rank,
                                             uint64_t epoch,
                                             long long deadline)
{
    const int peer = threadIdx.x;
    if (peer >= world_size) return;
    publish(reinterpret_cast<uint64_t*>(peers.ptr[peer]) + rank, epoch);
    const long long begin = clock64();
    while (acquire(local + peer) < epoch) {
        // A barrier that waits forever turns one dead rank into a job a human has to notice.
        if (clock64() - begin > deadline) {
            std::printf("fast-ulysses: rank %d gave up waiting for rank %d at epoch %llu\n",
                        rank, peer, static_cast<unsigned long long>(epoch));
            __trap();
        }
    }
}

// Two entry points rather than one, so the opening and closing barrier are distinguishable on
// the kernel timeline. That axis is the only one where the phases of an exchange appear in the
// order the GPU ran them, and it costs nothing: same body, same launch, different name.
__global__ void fu_barrier_open(uint64_t* local, PeerPads peers, int world_size, int rank,
                                uint64_t epoch, long long deadline)
{
    barrier_body(local, peers, world_size, rank, epoch, deadline);
}

__global__ void fu_barrier_close(uint64_t* local, PeerPads peers, int world_size, int rank,
                                 uint64_t epoch, long long deadline)
{
    barrier_body(local, peers, world_size, rank, epoch, deadline);
}

}  // namespace

void barrier(cudaStream_t stream,
             const std::vector<uint64_t>& pads,
             int rank,
             uint64_t epoch,
             long long deadline,
             bool opening)
{
    if (pads.size() <= 1) return;
    PeerPads peers{};
    for (size_t i = 0; i < pads.size(); ++i) peers.ptr[i] = pads[i];
    auto* local = reinterpret_cast<uint64_t*>(pads[rank]);
    const int world_size = static_cast<int>(pads.size());
    if (opening) {
        fu_barrier_open<<<1, 32, 0, stream>>>(local, peers, world_size, rank, epoch, deadline);
    } else {
        fu_barrier_close<<<1, 32, 0, stream>>>(local, peers, world_size, rank, epoch, deadline);
    }
    FU_CUDA_CHECK(cudaGetLastError());
}

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
                       bool quad_only)
{
    FU_NVTX(quad_only ? "fu::copies[near-quad only]" : "fu::copies[all peers]");
    const int world_size = peers.size();
    const auto* source = static_cast<const uint8_t*>(input);
    const int64_t width = (mode == 0 ? heads / world_size : heads) * dim * element_size;

    auto copy = [&](int peer) {
        // One range per destination, named by the only distinction the transfer itself makes:
        // whether the peer is in this rank's quad. The two classes do not run at the same rate,
        // and nsys attributes each device-side copy to the API call inside this range, so
        // without the name a profile shows one undifferentiated band of peer memcpys.
        char name[56];
        std::snprintf(name, sizeof(name), "fu::copy[peer=%d %s]", peer,
                      peer == rank ? "self"
                                   : (peer / kQuad == rank / kQuad ? "near-quad" : "cross-quad"));
        FU_NVTX_DETAIL(name);
        for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
            int64_t src_offset, dst_offset, src_pitch, dst_pitch, rows;
            if (mode == 0) {
                const int64_t local_heads = heads / world_size;
                src_offset = (batch_index * seq * heads + peer * local_heads) * dim * element_size;
                dst_offset = (batch_index * seq * world_size + rank * seq) * local_heads * dim *
                             element_size;
                src_pitch = heads * dim * element_size;
                dst_pitch = local_heads * dim * element_size;
                rows = seq;
            } else {
                const int64_t local_seq = seq / world_size;
                const int64_t global_heads = heads * world_size;
                src_offset = (batch_index * seq + peer * local_seq) * heads * dim * element_size;
                dst_offset = (batch_index * local_seq * global_heads + rank * heads) * dim *
                             element_size;
                src_pitch = heads * dim * element_size;
                dst_pitch = global_heads * dim * element_size;
                rows = local_seq;
            }
            auto* destination = reinterpret_cast<uint8_t*>(peers[peer]);
            FU_CUDA_CHECK(cudaMemcpy2DAsync(
                destination + dst_offset, dst_pitch, source + src_offset, src_pitch, width, rows,
                cudaMemcpyDefault, stream));
        }
    };

    // XOR-shift peer order. Stopping at kQuad keeps the copies inside this rank's quad, which is
    // exactly the set reachable through peer pointers when the NIC carries the other quad.
    const int end = quad_only ? kQuad : world_size;
    for (int step = 1; step < end; ++step) copy(rank ^ step);
    copy(rank);
}

}  // namespace ulysses
