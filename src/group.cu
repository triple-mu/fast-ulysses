#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <cuda_runtime.h>
#include <fast_ulysses/group.hpp>
#include <iostream>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/torch.h>

namespace ulysses {

static bool g_world_inited = false;
static int  g_live_groups  = 0;

// Flag barrier. Each rank holds uint64 flags[ws] PER TAG. On arrival, rank r publishes epoch into
// every peer p's flags[r] (release/sys MAX), then spins until its own flags[0..ws-1] are all >=
// epoch (acquire/sys). A monotonically increasing epoch means no reset and no ABA.
struct BarPeers {
    uint64_t p[8];
};

__global__ void ulysses_barrier_kernel(uint64_t* local, BarPeers peers, int ws, int rank, uint64_t* epoch_counter)
{
    // The epoch is advanced ON THE DEVICE, which is what makes a captured call replayable: a
    // host-computed epoch is a constant baked into the graph, so every replay announces a value the
    // peer flags already hold and the wait is satisfied by stale state. atomicAdd, so that a caller
    // who breaks the same-tag ordering contract cannot lose an epoch outright.
    __shared__ uint64_t epoch;
    if (threadIdx.x == 0)
        epoch = atomicAdd(reinterpret_cast<unsigned long long*>(epoch_counter), 1ULL) + 1;
    __syncthreads();  // must precede the early-out: every launched thread has to reach it

    int t = threadIdx.x;
    if (t >= ws)
        return;
    uint64_t* remote = reinterpret_cast<uint64_t*>(peers.p[t]) + rank;  // peer t's flags[rank]
    red_release_sys_max_u64(remote, epoch);
    uint64_t  v;
    uint64_t* mine = local + t;  // own flags[t] (written by peer t)
    do {
        v = ld_acquire_sys_u64(mine);
    } while (v < epoch);
}

const UlyssesGroup::BarrierState& UlyssesGroup::barrier_state_(const std::string& tag, cudaStream_t stream)
{
    auto it = barriers_.find(tag);
    if (it != barriers_.end())
        return it->second;

    BarrierState st;
    // The tag goes in the pool name, so each tag's handshake gets its own flags -- see BarrierState.
    const auto& buf = pool_->acquire(static_cast<int64_t>(world_size_), at::kLong, "__ulysses_sync__" + tag);
    st.my_flags     = buf.sym_base;
    st.peer_flags   = buf.peer_ptrs;
    // Epoch counter: never addressed by a peer, but taken from the pool rather than cudaMalloc'd,
    // which is not stream-ordered and blocks the host until the device drains -- with a spin
    // barrier in flight, the deadlock the pool's one-slab design exists to remove.
    const auto& ep = pool_->acquire(1, at::kLong, "__ulysses_epoch__" + tag);
    st.epoch       = reinterpret_cast<uint64_t*>(ep.sym_base);
    ULYSSES_CUDA_CHECK(cudaMemsetAsync(st.epoch, 0, sizeof(uint64_t), stream));  // first kernel publishes 1
    ULYSSES_CUDA_CHECK(cudaMemsetAsync(st.my_flags, 0, world_size_ * sizeof(uint64_t), stream));
    // All ranks must finish clearing before anyone writes a flag, or a clear overwrites an epoch.
    nvshmemx_barrier_on_stream(team_, stream);
    return barriers_.emplace(tag, std::move(st)).first->second;
}

int64_t UlyssesGroup::barrier_epoch(const std::string& tag)
{
    auto it = barriers_.find(tag);
    if (it == barriers_.end())
        return 0;
    uint64_t value = 0;
    ULYSSES_CUDA_CHECK(cudaMemcpy(&value, it->second.epoch, sizeof(value), cudaMemcpyDeviceToHost));
    return static_cast<int64_t>(value);
}

void UlyssesGroup::fast_barrier(cudaStream_t stream, const std::string& tag)
{
    if (world_size_ == 1)
        return;
    const BarrierState& st = barrier_state_(tag, stream);
    BarPeers            peers;
    for (int i = 0; i < world_size_; ++i)
        peers.p[i] = st.peer_flags[i];
    // A spin kernel, not stream memops (cuStreamWriteValue64/WaitValue64): the waiting memop needs
    // a remote-write-flush device attribute that is not always present, while this kernel's inline
    // PTX only requires sm_70. It needs an SM slot only at a kernel boundary.
    ulysses_barrier_kernel<<<1, 32, 0, stream>>>(
        reinterpret_cast<uint64_t*>(st.my_flags), peers, world_size_, my_rank_, st.epoch);
    ULYSSES_CUDA_CHECK(cudaGetLastError());  // catch a barrier-kernel launch failure
}

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
    // Use INITIALIZER, not memset(0): it stamps the version field, which
    // nvshmemx_set_attr_uniqueid_args does not write and hostlib_init_attr dispatches on. Inline
    // nvshmemx_init_attr would have stamped it for us.
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    TORCH_CHECK(
        nvshmemx_set_attr_uniqueid_args(static_cast<int>(global_rank), static_cast<int>(global_nranks), &uid, &attr)
            == 0,
        "nvshmemx_set_attr_uniqueid_args failed");
    // The host-lib direct entry, not inline nvshmemx_init_attr, which calls nvshmemi_init_thread:
    // that lives only in static libnvshmem_device.a, and linking it clashes with the version node
    // of torch's bundled libtorch_nvshmem.so.
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
    TORCH_CHECK(world_size_ >= 1 && world_size_ <= 8, "world_size must be in [1, 8] (one node), got ", world_size_);
    ULYSSES_CUDA_CHECK(cudaSetDevice(device_id_));
    peer_global_pes_.reserve(world_size_);
    for (auto pe : peer_global_pes)
        peer_global_pes_.push_back(static_cast<int>(pe));

    // team: the whole world -> TEAM_WORLD; else the strided slice of it this PE set describes. The
    // stride comes from the PE list, so a 2-D layout works: tp=2 x ulysses-sp=4 on 8 GPUs gives sp
    // groups {0,2,4,6} and {1,3,5,7}, both live at once.
    //
    // Each PE passes only its OWN triplet, which is not the shape the API is built for
    // (nvshmem_team_split_2d loops split_strided with EVERY PE in every call). It works because the
    // split wraps its body in PARENT-team collectives while the body that picks the team index
    // reduces over the CHILD triplet alone. Hence the contract comm.py states: the live groups must
    // PARTITION the job and construct together, so those parent-team collectives still pair up.
    // Overlapping groups hang right here -- members block on the parent collectives, non-members
    // walk past them. NOT ENFORCED: such a check has to run on every rank, and only group members
    // reach this constructor.
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
        const int   start  = peer_global_pes_[0];
        const int   stride = world_size_ > 1 ? peer_global_pes_[1] - start : 1;
        std::string pes;  // the offending list, named by both checks below (<= 8 entries)
        for (int pe : peer_global_pes_)
            pes += std::to_string(pe) + " ";
        // A PE set that is not an arithmetic progression is not a strided team and has no other
        // representation here -- refuse it instead of splitting some other PE set.
        TORCH_CHECK(stride >= 1, "process-group ranks must be increasing; got [", pes, "]");
        for (int i = 1; i < world_size_; ++i)
            TORCH_CHECK(peer_global_pes_[i] == start + i * stride,
                        "process-group ranks must be evenly strided (they become an NVSHMEM strided team); got [",
                        pes,
                        "], which breaks stride ",
                        stride,
                        " at index ",
                        i,
                        " (rank ",
                        peer_global_pes_[i],
                        ", expected ",
                        start + i * stride,
                        ")");
        nvshmem_team_config_t cfg;
        std::memset(&cfg, 0, sizeof(cfg));
        TORCH_CHECK(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, start, stride, world_size_, &cfg, 0, &team_) == 0,
                    "nvshmem_team_split_strided(start=",
                    start,
                    ", stride=",
                    stride,
                    ", size=",
                    world_size_,
                    ") failed");
        owns_team_ = true;
    }

    pool_ = std::make_unique<SymmetricHeapPool>(reserved_bytes, world_size_, peer_global_pes_);
    ++g_live_groups;
}

const CEResources& UlyssesGroup::ce_resources()
{
    if (!ce_ready_) {
        ULYSSES_CUDA_CHECK(cudaStreamCreateWithFlags(&ce_.xfer, cudaStreamNonBlocking));
        ce_ready_ = true;
    }
    return ce_;
}

void UlyssesGroup::destroy()
{
    if (destroyed_)
        return;
    if (ce_ready_) {
        // Unchecked, like the rest of destroy(): the caller already guarantees quiescence.
        cudaStreamSynchronize(ce_.xfer);
        cudaStreamDestroy(ce_.xfer);
        ce_       = {};
        ce_ready_ = false;
    }
    // Both the flags and the epoch counters are pool windows, so pool_->destroy() reclaims them.
    barriers_.clear();
    if (pool_)
        pool_->destroy();
    if (owns_team_)
        nvshmem_team_destroy(team_);
    destroyed_ = true;
    if (--g_live_groups == 0 && g_world_inited) {
        // nvshmem_finalize() is inline and calls nvshmemi_finalize(), again only in the static
        // device library; this is the host shared library's equivalent.
        nvshmemx_hostlib_finalize();
        g_world_inited = false;
    }
}

UlyssesGroup::~UlyssesGroup()
{
    // No collective teardown from the destructor: destroy() is collective throughout, while GC and
    // interpreter-exit timing differ across ranks, so a rank destructing alone would hang the
    // group. Leak and warn instead.
    if (!destroyed_)
        std::cerr << "[fast_ulysses] UlyssesGroup dropped without destroy(); leaking symmetric-heap "
                     "resources (call group.destroy() on all ranks)"
                  << std::endl;
}

}  // namespace ulysses
