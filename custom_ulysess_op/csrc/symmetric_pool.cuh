#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <map>
#include <nvshmem.h>
#include <string>
#include <tuple>
#include <vector>

namespace ulysess {

class SymmetricHeapPool {
public:
    // reserved_bytes：本 group 自设上限（须 ≤ init 时的 NVSHMEM_SYMMETRIC_SIZE 预留）。
    SymmetricHeapPool(int64_t reserved_bytes, int world_size, std::vector<int> peer_global_pes, nvshmem_team_t team);

    struct Buffer {
        void*                 sym_base;
        int64_t               nbytes;
        std::vector<uint64_t> peer_ptrs;  // nvshmem_ptr(sym_base, peer_global_pe)
        at::Tensor            view;       // from_blob，no-op deleter（pool 持生命周期）
    };

    // 命中 (tag,shape,dtype) 直接复用；否则集体新分配一段并登记。
    const Buffer& acquire(const std::vector<int64_t>& shape, c10::ScalarType dtype, const std::string& tag);

    // 终态且为集体操作：调用前须释放所有 acquire() 返回的 from_blob 视图，且不得有进行中的 A2A/集体操作，因为它会
    // nvshmem_free 这些视图所别名的段。
    void destroy();  // 释放全部段（nvshmem_free）+ 清注册表

private:
    using Key = std::tuple<std::string, std::vector<int64_t>, c10::ScalarType>;
    int64_t               reserved_, used_ = 0;
    int                   world_size_;
    std::vector<int>      peer_global_pes_;
    nvshmem_team_t        team_;
    int64_t*              scratch_ = nullptr;  // 2 个对称 int64，做 nbytes 一致性校验
    std::vector<void*>    segments_;
    std::map<Key, Buffer> registry_;
    bool                  destroyed_ = false;

    // 集体对 nbytes 取全局 max 并返回（变长时各 rank 输出大小不同，对称堆按 max 同尺寸分配，
    // 每 rank 只用自己那块）；uniform 时 gmax==nbytes，行为不变。
    int64_t collective_max_nbytes(int64_t nbytes);
};

}  // namespace ulysess
