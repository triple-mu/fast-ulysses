#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <map>
#include <nvshmem.h>
#include <string>
#include <tuple>
#include <vector>

namespace ulysses {

class SymmetricHeapPool {
public:
    // reserved_bytes: per-group cap (must be <= NVSHMEM_SYMMETRIC_SIZE reserved at init).
    SymmetricHeapPool(int64_t reserved_bytes, int world_size, std::vector<int> peer_global_pes, nvshmem_team_t team);

    struct Buffer {
        void*                 sym_base;
        int64_t               nbytes;
        std::vector<uint64_t> peer_ptrs;  // nvshmem_ptr(sym_base, peer_global_pe)
        at::Tensor            view;       // from_blob with no-op deleter (pool owns lifetime)
    };

    // Reuse on (tag,shape,dtype) hit; otherwise collectively allocate a new segment and register it.
    const Buffer& acquire(const std::vector<int64_t>& shape, c10::ScalarType dtype, const std::string& tag);

    // Terminal collective op: before calling, release all from_blob views returned by acquire() and
    // ensure no A2A/collective is in flight, since this nvshmem_free's the segments those views alias.
    void destroy();  // nvshmem_free all segments + clear registry

private:
    using Key = std::tuple<std::string, std::vector<int64_t>, c10::ScalarType>;
    int64_t               reserved_, used_ = 0;
    int                   world_size_;
    std::vector<int>      peer_global_pes_;
    nvshmem_team_t        team_;
    int64_t*              scratch_ = nullptr;  // 2 symmetric int64s for nbytes consistency check
    std::vector<void*>    segments_;
    std::map<Key, Buffer> registry_;
    bool                  destroyed_ = false;

    // Collectively reduce nbytes to global max and return it (in varlen mode ranks have different
    // output sizes, so the symmetric heap allocates the same max size on every rank and each rank
    // uses only its own slice); in uniform mode gmax==nbytes, so behavior is unchanged.
    int64_t collective_max_nbytes(int64_t nbytes);
};

}  // namespace ulysses
