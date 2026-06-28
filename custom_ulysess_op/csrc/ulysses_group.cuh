#pragma once
#include "symmetric_pool.cuh"
#include <cstdint>
#include <memory>
#include <nvshmem.h>
#include <torch/custom_class.h>
#include <vector>

namespace ulysess {

class UlyssesGroup: public torch::CustomClassHolder {
public:
    static int64_t              uniqueid_nints();  // ceil(sizeof(nvshmemx_uniqueid_t)/8)
    static std::vector<int64_t> get_uniqueid();    // 仅 rank0 调用
    static void init_world(std::vector<int64_t> uid_ints, int64_t global_rank, int64_t global_nranks);  // 幂等

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
    nvshmem_team_t team() const
    {
        return team_;
    }

private:
    int                                my_rank_, world_size_, device_id_;
    std::vector<int>                   peer_global_pes_;
    nvshmem_team_t                     team_;
    bool                               owns_team_ = false;
    bool                               destroyed_ = false;
    std::unique_ptr<SymmetricHeapPool> pool_;
};

}  // namespace ulysess
