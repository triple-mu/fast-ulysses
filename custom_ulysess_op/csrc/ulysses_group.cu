#include "ulysses_group.cuh"
#include <cstring>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/torch.h>

namespace ulysess {

static bool g_world_inited = false;
static int  g_live_groups  = 0;

int64_t UlyssesGroup::uniqueid_nints()
{
    return static_cast<int64_t>((sizeof(nvshmemx_uniqueid_t) + 7) / 8);
}

std::vector<int64_t> UlyssesGroup::get_uniqueid()
{
    nvshmemx_uniqueid_t uid;
    std::memset(&uid, 0, sizeof(uid));
    TORCH_CHECK(nvshmemx_get_uniqueid(&uid) == 0, "nvshmemx_get_uniqueid failed");
    std::vector<int64_t> out(uniqueid_nints(), 0);
    std::memcpy(out.data(), &uid, sizeof(uid));
    return out;
}

void UlyssesGroup::init_world(std::vector<int64_t> uid_ints, int64_t global_rank, int64_t global_nranks)
{
    if (g_world_inited)
        return;
    TORCH_CHECK(static_cast<int64_t>(uid_ints.size()) >= uniqueid_nints(), "uid_ints too short");
    nvshmemx_uniqueid_t uid;
    std::memcpy(&uid, uid_ints.data(), sizeof(uid));
    // 用 INITIALIZER 而非 memset(0)：stamps attr/args/uid_args 的 version 字段。
    // nvshmemx_set_attr_uniqueid_args 不会写 version，而 hostlib_init_attr 据
    // attr.args.version 分派 V2 路径，所以版本必须先 stamp（inline nvshmemx_init_attr
    // 会在 version 非法时自动 stamp，此处显式 substitute 了那一步）。
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    TORCH_CHECK(
        nvshmemx_set_attr_uniqueid_args(static_cast<int>(global_rank), static_cast<int>(global_nranks), &uid, &attr)
            == 0,
        "nvshmemx_set_attr_uniqueid_args failed");
    // DEVIATION（详见 task-5-report）：用 host-lib 直接入口 nvshmemx_hostlib_init_attr
    // 代替 inline nvshmemx_init_attr。后者内联调 nvshmemi_init_thread，该符号仅在静态
    // libnvshmem_device.a 中，链接它会与 torch 自带 libtorch_nvshmem.so 的 NVSHMEM 版本
    // 节点冲突（undefined symbol nvshmem_selected_device_transport）。hostlib_init_attr
    // 是 host 共享库直接导出的等价入口（NVSHMEM 自身 python UID 路径即用此）。
    TORCH_CHECK(nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) == 0,
                "nvshmemx_hostlib_init_attr failed");
    g_world_inited = true;
}

UlyssesGroup::UlyssesGroup(std::vector<int64_t> peer_global_pes,
                           int64_t              my_rank,
                           int64_t              device_id,
                           int64_t              reserved_bytes):
    my_rank_(static_cast<int>(my_rank)),
    world_size_(static_cast<int>(peer_global_pes.size())),
    device_id_(static_cast<int>(device_id))
{
    TORCH_CHECK(g_world_inited, "init_world must be called before constructing UlyssesGroup");
    TORCH_CHECK(world_size_ == 1 || world_size_ == 2 || world_size_ == 4 || world_size_ == 8,
                "world_size must be in {1,2,4,8}, got ",
                world_size_);
    cudaSetDevice(device_id_);
    peer_global_pes_.reserve(world_size_);
    for (auto pe : peer_global_pes)
        peer_global_pes_.push_back(static_cast<int>(pe));

    // team：组覆盖整个 world 且连续 → TEAM_WORLD；否则连续步长 1 子组用 split_strided。
    const int gpes          = nvshmem_n_pes();
    bool      is_full_world = (world_size_ == gpes);
    for (int i = 0; i < world_size_ && is_full_world; ++i)
        if (peer_global_pes_[i] != i)
            is_full_world = false;
    if (is_full_world) {
        team_      = NVSHMEM_TEAM_WORLD;
        owns_team_ = false;
    }
    else {
        int start = peer_global_pes_[0];
        for (int i = 1; i < world_size_; ++i)
            TORCH_CHECK(peer_global_pes_[i] == start + i, "phase-1 only supports a contiguous PE subgroup");
        nvshmem_team_config_t cfg;
        std::memset(&cfg, 0, sizeof(cfg));
        TORCH_CHECK(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, start, 1, world_size_, &cfg, 0, &team_) == 0,
                    "nvshmem_team_split_strided failed");
        owns_team_ = true;
    }

    pool_ = std::make_unique<SymmetricHeapPool>(reserved_bytes, world_size_, peer_global_pes_, team_);
    ++g_live_groups;
}

void UlyssesGroup::destroy()
{
    if (destroyed_)
        return;
    if (pool_)
        pool_->destroy();
    if (owns_team_)
        nvshmem_team_destroy(team_);
    destroyed_ = true;
    if (--g_live_groups == 0 && g_world_inited) {
        // DEVIATION：nvshmem_finalize() 是 inline，内联调 nvshmemi_finalize()（同样仅在
        // 静态 device.a 中）。用 host 共享库导出的 nvshmemx_hostlib_finalize() 代替。
        nvshmemx_hostlib_finalize();
        g_world_inited = false;
    }
}

UlyssesGroup::~UlyssesGroup()
{
    destroy();
}

}  // namespace ulysess
