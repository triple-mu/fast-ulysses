#include <c10/cuda/CUDAGuard.h>
#include <fast_ulysses/symmetric_pool.hpp>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/torch.h>

namespace ulysses {

SymmetricHeapPool::SymmetricHeapPool(int64_t reserved_bytes, int world_size, std::vector<int> peer_global_pes):
    reserved_(reserved_bytes),
    world_size_(world_size),
    peer_global_pes_(std::move(peer_global_pes))
{
    // THE one symmetric allocation. See the class comment for why it may not happen anywhere else.
    slab_ = nvshmem_align(256, static_cast<size_t>(reserved_));
    TORCH_CHECK(slab_ != nullptr,
                "nvshmem_align failed for a ",
                reserved_,
                " B symmetric pool. Lower initial_pool_bytes, or raise NVSHMEM_SYMMETRIC_SIZE.");

    // Every peer must be load/store reachable, because the transport writes peer windows with plain
    // cudaMemcpy*Async: a null here would become a copy to address 0. nvshmem_ptr is non-null
    // exactly when the pair is P2P-mappable, and when it is not the cause is outside this process,
    // so refuse by name and let `fast-ulysses doctor` print the pairwise matrix.
    slab_peer_.resize(world_size_);
    for (int i = 0; i < world_size_; ++i) {
        slab_peer_[i] = reinterpret_cast<uint64_t>(nvshmem_ptr(slab_, peer_global_pes_[i]));
        TORCH_CHECK(slab_peer_[i] != 0,
                    "nvshmem_ptr returned NULL for group peer ",
                    i,
                    " (global PE ",
                    peer_global_pes_[i],
                    "): that pair is not P2P-mappable, so this group cannot be formed. Run "
                    "`fast-ulysses doctor` for the pairwise reachability matrix.");
    }
}

const SymmetricHeapPool::Buffer&
SymmetricHeapPool::acquire(int64_t numel, c10::ScalarType dtype, const std::string& tag)
{
    TORCH_CHECK(!destroyed_, "SymmetricHeapPool::acquire called after destroy()");
    Key  key{tag, dtype};
    auto it = registry_.find(key);
    if (it != registry_.end() && it->second.numel >= numel)
        return it->second;  // big enough

    TORCH_CHECK(!sealed_,
                "the symmetric pool is sealed and tag '",
                tag,
                "' (",
                c10::toString(dtype),
                ") needs capacity ",
                numel,
                " elements, but has ",
                it != registry_.end() ? it->second.numel : 0,
                ". Declare this call in reserve(), or pass allow_growth=True.");

    // 256-byte granularity keeps every window aligned for the pitched copies and for the uint4
    // stores the transport issues.
    int64_t nbytes = numel * c10::elementSize(dtype);
    nbytes         = (nbytes + 255) / 256 * 256;
    TORCH_CHECK(used_ + nbytes <= reserved_,
                "SymmetricHeapPool OOM: tag '",
                tag,
                "' needs ",
                nbytes,
                " B, used ",
                used_,
                " / reserved ",
                reserved_,
                " B. Increase initial_pool_bytes.");

    Buffer buf;
    buf.sym_base = static_cast<char*>(slab_) + used_;
    buf.numel    = numel;
    // The peer's copy of this window sits at the same offset in the peer's slab, because every
    // rank hands out offsets in the same order (see the class comment).
    buf.peer_ptrs.resize(world_size_);
    for (int i = 0; i < world_size_; ++i)
        buf.peer_ptrs[i] = slab_peer_[i] + static_cast<uint64_t>(used_);
    used_ += nbytes;

    // Growing replaces the entry; the outgrown offset is not reclaimed.
    return registry_.insert_or_assign(std::move(key), std::move(buf)).first->second;
}

void SymmetricHeapPool::destroy()
{
    if (destroyed_)
        return;
    registry_.clear();
    if (slab_ != nullptr) {
        nvshmem_free(slab_);
        slab_ = nullptr;
    }
    destroyed_ = true;
}

}  // namespace ulysses
