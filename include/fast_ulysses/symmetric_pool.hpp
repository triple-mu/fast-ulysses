#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ulysses {

// One symmetric allocation for the whole group, carved up locally: the constructor takes the entire
// pool with a single nvshmem_align, and acquire() only ever returns an offset into it. Nothing on
// the call path may allocate -- nvshmem_align is collective and synchronizes the CUDA stream, while
// the flag barrier is a spin kernel that barrier=False deliberately leaves in flight, so allocating
// mid-run parks every rank's host inside nvshmem_align where it can no longer issue the publish its
// peers are spinning for. The offsets line up because every rank hands them out in the SAME ORDER.
class SymmetricHeapPool {
public:
    // reserved_bytes: the whole pool, allocated here (must be <= NVSHMEM_SYMMETRIC_SIZE).
    // Collective. Every peer must be P2P-mappable; an unreachable pair is refused by name.
    SymmetricHeapPool(int64_t reserved_bytes, int world_size, std::vector<int> peer_global_pes);

    struct Buffer {
        void*                 sym_base;
        int64_t               numel;      // capacity, in elements of the dtype it was made for
        std::vector<uint64_t> peer_ptrs;  // this window's address in each peer's slab
    };

    // Reuse a tag's window when it is big enough; carve a new one when it is not. `numel` is a
    // CAPACITY, not a shape: the key is (tag, dtype) and the match is capacity >= requested, so a
    // tag costs one window at its high-water mark and the key stays off the output shape, which
    // under uneven splits differs per rank and would fork the hit/miss pattern.
    const Buffer& acquire(int64_t numel, c10::ScalarType dtype, const std::string& tag);

    // After this, a miss in acquire() is an error instead of a new window. Catches a shape that
    // drifts upward (growth does not reclaim the offset it outgrew) and ranks that stop agreeing on
    // what they allocate, which is what the local offsets rest on.
    void seal()
    {
        sealed_ = true;
    }

    // Terminal collective: release every view built over an acquire()'d buffer and quiesce all
    // collectives first, since this nvshmem_free's the slab those views alias.
    void destroy();

private:
    using Key = std::pair<std::string, c10::ScalarType>;
    int64_t               reserved_, used_ = 0;
    int                   world_size_;
    std::vector<int>      peer_global_pes_;
    void*                 slab_ = nullptr;  // the one allocation
    std::vector<uint64_t> slab_peer_;       // nvshmem_ptr(slab_, peer_global_pe)
    std::map<Key, Buffer> registry_;
    bool                  destroyed_ = false;
    bool                  sealed_    = false;
};

}  // namespace ulysses
