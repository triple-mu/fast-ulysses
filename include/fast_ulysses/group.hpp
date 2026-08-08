#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <fast_ulysses/a2a_plan.hpp>
#include <fast_ulysses/common.hpp>
#include <fast_ulysses/symmetric_pool.hpp>
#include <map>
#include <memory>
#include <nvshmem.h>
#include <string>
#include <torch/custom_class.h>
#include <vector>

namespace ulysses {

// Per-group CE (copy-engine) transfer resources: ONE stream, which the remote peer copies are
// serialised onto because concurrent copies contend for the same egress. Created lazily by
// ce_resources(), released in destroy(). Join events are deliberately NOT pooled here -- see the
// fresh-event note in launch_a2a_ce.
struct CEResources {
    cudaStream_t xfer = nullptr;
};

// CE transfer path: issues `plan.ops` as pitched cudaMemcpy2D/3DAsync copies -- remote peers
// serialised on `ce.xfer`, this rank's own share on `stream` -- joined back to `stream` with
// events. `peer_ptrs[p]` is the base of peer p's window, which the op offsets are relative to.
void launch_a2a_ce(const void*                  src,
                   const std::vector<uint64_t>& peer_ptrs,
                   const A2APlan&               plan,
                   const CEResources&           ce,
                   int                          rank,
                   cudaStream_t                 stream);

// TESTS ONLY: publish the flag before the payload has landed, on purpose. See transfer.cu.
void set_ce_fault(int64_t delay_us);

class UlyssesGroup: public torch::CustomClassHolder {
public:
    static int64_t              uniqueid_nints();  // ceil(sizeof(nvshmemx_uniqueid_t)/8)
    static std::vector<int64_t> get_uniqueid();    // rank0 only
    static void init_world(std::vector<int64_t> uid_ints, int64_t global_rank, int64_t global_nranks);  // idempotent

    UlyssesGroup(std::vector<int64_t> peer_global_pes, int64_t my_rank, int64_t device_id, int64_t reserved_bytes);
    ~UlyssesGroup() override;

    int64_t rank() const
    {
        return my_rank_;
    }
    int64_t world_size() const
    {
        return world_size_;
    }
    void destroy();

    SymmetricHeapPool& pool()
    {
        return *pool_;
    }

    // Custom single-node flag barrier over P2P-mapped memory, in place of NVSHMEM's own sync. No
    // nvshmem quiet is needed (or would help): the transport issues raw cudaMemcpy2DAsync into
    // nvshmem_ptr addresses, which are not NVSHMEM operations for quiet to order. No-op at
    // world_size == 1. `tag` picks the barrier state, of which there is ONE SET PER TAG.
    void fast_barrier(cudaStream_t stream, const std::string& tag);

    // TESTS: the tag's device-side epoch counter, read back to the host; 0 if the tag has no
    // barrier state yet. Lets a test assert that a handshake ADVANCED.
    int64_t barrier_epoch(const std::string& tag);

    // Create a tag's barrier state now, so a later fast_barrier on it allocates nothing. Its flags
    // live in the same pool as the data windows, so sealing without this makes the tag's FIRST
    // handshake fail rather than its first undeclared window.
    void reserve_barrier(const std::string& tag, cudaStream_t stream)
    {
        barrier_state_(tag, stream);
    }

    const CEResources& ce_resources();  // creates the transfer stream on first use

private:
    int                                my_rank_, world_size_, device_id_;
    std::vector<int>                   peer_global_pes_;
    nvshmem_team_t                     team_;
    bool                               owns_team_ = false;
    bool                               destroyed_ = false;
    std::unique_ptr<SymmetricHeapPool> pool_;

    CEResources ce_;
    bool        ce_ready_ = false;

    // fast_barrier state: symmetric flag buffer (uint64[ws]) + monotonic epoch. ONE SET PER TAG.
    //
    // The epoch protocol needs every rank to assign epochs to the same handshakes in the same
    // order, and program order gives that only WITHIN a tag. With one counter for the whole group,
    // two collectives on unordered streams interleave differently on different ranks, so a rank
    // waits on an epoch its peer published for the OTHER collective and reads the window before it
    // is written; making the counter atomic fixes the lost increment, not the ordering. Per tag
    // there is nothing to interleave (a2a_overlapping_barriers.py). The flags are a symmetric-heap
    // buffer like the data; the epoch is this rank's own device memory, device-side so a captured
    // graph advances it on replay (ulysses_barrier_kernel).
    struct BarrierState {
        void*                 my_flags = nullptr;  // this rank's flag base
        std::vector<uint64_t> peer_flags;          // per-peer flag base (including self)
        uint64_t*             epoch = nullptr;     // device counter, incremented by the kernel
    };
    std::map<std::string, BarrierState> barriers_;

    // This tag's barrier state, created on first use (collective pool alloc + zeroed flags).
    const BarrierState& barrier_state_(const std::string& tag, cudaStream_t stream);
};

}  // namespace ulysses
