#include "symmetric_pool.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/torch.h>

namespace ulysses {

SymmetricHeapPool::SymmetricHeapPool(int64_t          reserved_bytes,
                                     int              world_size,
                                     std::vector<int> peer_global_pes,
                                     nvshmem_team_t   team):
    reserved_(reserved_bytes),
    world_size_(world_size),
    peer_global_pes_(std::move(peer_global_pes)),
    team_(team)
{
    // Collectively allocate a small symmetric scratch for nbytes consistency checks in acquire.
    scratch_ = static_cast<int64_t*>(nvshmem_align(256, 2 * sizeof(int64_t)));
    TORCH_CHECK(scratch_ != nullptr, "nvshmem_align scratch failed");
    segments_.push_back(scratch_);
}

int64_t SymmetricHeapPool::collective_max_nbytes(int64_t nbytes)
{
    // Max-reduce nbytes across all PEs (varlen: per-rank output sizes differ, but the symmetric
    // heap requires a uniform size, so allocate by the global max).
    // scratch_ is DEVICE symmetric memory from nvshmem_align and cannot be dereferenced on host,
    // so we cudaMemcpy host<->device and run the reduce on device (host-blocking).
    int64_t gmax = 0;
    cudaMemcpy(scratch_, &nbytes, sizeof(int64_t), cudaMemcpyHostToDevice);
    nvshmem_int64_max_reduce(team_, scratch_ + 1, scratch_, 1);
    cudaMemcpy(&gmax, scratch_ + 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
    return gmax;
}

const SymmetricHeapPool::Buffer&
SymmetricHeapPool::acquire(const std::vector<int64_t>& shape, c10::ScalarType dtype, const std::string& tag)
{
    TORCH_CHECK(!destroyed_, "SymmetricHeapPool::acquire called after destroy()");
    Key  key{tag, shape, dtype};
    auto it = registry_.find(key);
    if (it != registry_.end())
        return it->second;  // reuse

    int64_t numel = 1;
    for (auto s : shape)
        numel *= s;
    const int64_t elem   = c10::elementSize(dtype);
    int64_t       nbytes = numel * elem;
    nbytes               = (nbytes + 15) / 16 * 16;  // uint4 alignment

    // Varlen: per-rank output sizes differ, but the symmetric heap requires a uniform size, so all
    // ranks allocate the global max and each uses only its local portion.
    const int64_t alloc_bytes = collective_max_nbytes(nbytes);
    TORCH_CHECK(used_ + alloc_bytes <= reserved_,
                "SymmetricHeapPool OOM: need ",
                alloc_bytes,
                " B, used ",
                used_,
                " / reserved ",
                reserved_,
                " B. Increase initial_pool_bytes.");

    void* p = nvshmem_align(256, alloc_bytes);  // collective alloc (uniform size); address never moves
    TORCH_CHECK(p != nullptr, "nvshmem_align failed for ", alloc_bytes, " B");
    segments_.push_back(p);
    used_ += alloc_bytes;

    Buffer buf;
    buf.sym_base = p;
    buf.nbytes   = alloc_bytes;
    buf.peer_ptrs.resize(world_size_);
    for (int i = 0; i < world_size_; ++i)
        buf.peer_ptrs[i] = reinterpret_cast<uint64_t>(nvshmem_ptr(p, peer_global_pes_[i]));
    // Single-node P2P: all peer pointers must be non-null.
    for (int i = 0; i < world_size_; ++i)
        TORCH_CHECK(buf.peer_ptrs[i] != 0,
                    "nvshmem_ptr returned NULL for peer ",
                    i,
                    " (non-P2P-reachable; phase-1 requires single-node NVLink).");

    auto opts = at::TensorOptions().dtype(dtype).device(at::kCUDA, at::cuda::current_device());
    buf.view  = at::from_blob(
        p, shape, [](void*) {}, opts);  // no-op deleter

    auto res = registry_.emplace(std::move(key), std::move(buf));
    return res.first->second;
}

void SymmetricHeapPool::destroy()
{
    if (destroyed_)
        return;
    registry_.clear();  // drop from_blob views (does not free underlying memory)
    for (void* p : segments_)
        nvshmem_free(p);
    segments_.clear();
    scratch_   = nullptr;  // segments freed; null it out to avoid dangling-pointer misuse
    destroyed_ = true;
}

}  // namespace ulysses
