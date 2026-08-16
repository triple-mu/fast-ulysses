#include "fast_ulysses.h"

namespace ulysses {
namespace {

struct PeerFlags { uint64_t ptr[8]; };

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

__global__ void barrier_kernel(uint64_t* local,
                               PeerFlags peers,
                               int world_size,
                               int rank,
                               uint64_t epoch)
{
    const int peer = threadIdx.x;
    if (peer >= world_size) return;
    publish(reinterpret_cast<uint64_t*>(peers.ptr[peer]) + rank, epoch);
    while (acquire(local + peer) < epoch) { }
}

}  // namespace

void barrier(cudaStream_t stream,
             const std::vector<uint64_t>& flags,
             int rank,
             uint64_t epoch)
{
    if (flags.size() <= 1) return;
    PeerFlags peers{};
    for (size_t i = 0; i < flags.size(); ++i) peers.ptr[i] = flags[i];
    barrier_kernel<<<1, 32, 0, stream>>>(reinterpret_cast<uint64_t*>(flags[rank]), peers,
                                         flags.size(), rank, epoch);
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
    const int world_size = peers.size();
    const auto* source = static_cast<const uint8_t*>(input);
    const int64_t width = (mode == 0 ? heads / world_size : heads) * dim * element_size;

    auto copy = [&](int peer) {
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
    // exactly the set reachable through IPC pointers when the NIC carries the other quad.
    const int end = quad_only ? kQuad : world_size;
    for (int step = 1; step < end; ++step) copy(rank ^ step);
    copy(rank);
}

}  // namespace ulysses
