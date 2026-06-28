#include "tma_ptx.cuh"
#include "ulysses_group.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/torch.h>

namespace ulysses {

static bool g_world_inited = false;
static int  g_live_groups  = 0;

// ---- Custom NVLink flag barrier ----
// Flag layout: each rank holds uint64 flags[ws]. On arrival, rank r writes epoch into every peer p's
// flags[r] (P2P, release/sys), then spins until its own flags[0..ws-1] all == epoch (acquire/sys).
// Monotonically increasing epoch means no reset and no ABA. Strict lockstep (SPMD collective) keeps
// epoch identical across ranks throughout.
struct BarPeers {
    uint64_t p[8];
};

__global__ void ulysses_barrier_kernel(uint64_t* local, BarPeers peers, int ws, int rank, uint64_t epoch)
{
    int t = threadIdx.x;
    if (t >= ws)
        return;
    uint64_t* remote = reinterpret_cast<uint64_t*>(peers.p[t]) + rank;  // peer t's flags[rank]
    st_release_sys_u64(remote, epoch);
    uint64_t  v;
    uint64_t* mine = local + t;  // own flags[t] (written by peer t)
    do {
        v = ld_acquire_sys_u64(mine);
    } while (v < epoch);
}

void UlyssesGroup::fast_barrier(cudaStream_t stream)
{
    if (world_size_ == 1)
        return;
    if (!bar_ready_) {
        const auto& buf = pool_->acquire({static_cast<int64_t>(world_size_)}, at::kLong, "__ulysses_sync__");
        bar_local_      = buf.sym_base;
        bar_peers_      = buf.peer_ptrs;
        cudaMemsetAsync(bar_local_, 0, world_size_ * sizeof(uint64_t), stream);  // init 0; epoch starts at 1
        // One slow sync: ensure all ranks finish clearing before anyone writes a flag (otherwise the
        // clear could overwrite an already-written epoch).
        nvshmemx_barrier_on_stream(team_, stream);
        bar_ready_ = true;
    }
    ++bar_epoch_;
    BarPeers peers;
    for (int i = 0; i < world_size_; ++i)
        peers.p[i] = bar_peers_[i];
    ulysses_barrier_kernel<<<1, 32, 0, stream>>>(
        reinterpret_cast<uint64_t*>(bar_local_), peers, world_size_, my_rank_, bar_epoch_);
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
    // Use INITIALIZER, not memset(0): it stamps the version field of attr/args/uid_args.
    // nvshmemx_set_attr_uniqueid_args does not write version, and hostlib_init_attr dispatches the V2
    // path based on attr.args.version, so the version must be stamped first (inline nvshmemx_init_attr
    // auto-stamps when version is invalid; here we explicitly substitute that step).
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    TORCH_CHECK(
        nvshmemx_set_attr_uniqueid_args(static_cast<int>(global_rank), static_cast<int>(global_nranks), &uid, &attr)
            == 0,
        "nvshmemx_set_attr_uniqueid_args failed");
    // DEVIATION (see task-5-report): use the host-lib direct entry nvshmemx_hostlib_init_attr instead of
    // inline nvshmemx_init_attr. The inline version calls nvshmemi_init_thread, a symbol that lives only
    // in static libnvshmem_device.a; linking it clashes with the NVSHMEM version node of torch's bundled
    // libtorch_nvshmem.so (undefined symbol nvshmem_selected_device_transport). hostlib_init_attr is the
    // equivalent entry exported directly by the host shared library (NVSHMEM's own python UID path uses it).
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
    // Cache compute capability major (TMA cp.async.bulk.tensor is Hopper(sm90)+ only); whether a call
    // actually takes the TMA path is decided per call (all_to_all/tune) via decide_tma_path on use_tma.
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id_);
    sm_major_ = major;
    TORCH_CHECK(world_size_ == 1 || world_size_ == 2 || world_size_ == 4 || world_size_ == 8,
                "world_size must be in {1,2,4,8}, got ",
                world_size_);
    cudaSetDevice(device_id_);
    peer_global_pes_.reserve(world_size_);
    for (auto pe : peer_global_pes)
        peer_global_pes_.push_back(static_cast<int>(pe));

    // team: if the group covers the whole world contiguously -> TEAM_WORLD; else a contiguous stride-1
    // subgroup via split_strided.
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

// See the "collective-call hard invariant" and thread-safety notes in ulysses_group.cuh: on a cache hit
// return immediately (no collective); only on a miss do we enter the collective.
A2AConfig UlyssesGroup::resolve_config(const Ulysses4DDims&         dims,
                                       int                          mode,
                                       bool                         use_tma,
                                       const void*                  src,
                                       const std::vector<uint64_t>& peers,
                                       int                          elem,
                                       cudaStream_t                 stream)
{
    const ConfigKey key = config_key(world_size_, mode, use_tma, dims);
    auto            it  = cfg_cache_.find(key);
    if (it != cfg_cache_.end())
        return it->second;  // hit: no collective primitive
    A2AConfig cfg = use_tma ? resolve_config_tma(src, peers, dims, mode, elem, static_cast<int>(team_), false, stream) :
                              resolve_config_nontma(src, peers, dims, mode, elem, stream, /*verbose=*/false);
    cfg_cache_[key] = cfg;
    return cfg;
}

// See the "collective-call hard invariant" note in ulysses_group.cuh. Params are the per-rank LOCAL
// logical dims (b, s_local, n_local, d), identical for mode0/mode1 (s_local=S_global/ws, n_local=H/ws):
// passing local on both axes removes the old ambiguity (one axis was global, and which one flipped by
// mode). The mode-dependent physical input shape is reconstructed here (global = local*ws), so the
// resulting dims/cache-key match the bindings.cpp uniform path and the lazy all_to_all hits this entry.
// use_tma: -1 auto / 0 / 1, resolved via decide_tma_path (uniform, is_varlen=false); only that path warms.
void UlyssesGroup::tune(
    int64_t b, int64_t s_local, int64_t n_local, int64_t d, int64_t mode, int64_t use_tma, bool verbose)
{
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");
    if (use_tma > 0)
        TORCH_CHECK(sm_major_ >= 9, "use_tma=True requires sm90+ (TMA unavailable on this GPU)");
    const bool    tma = decide_tma_path(static_cast<int>(use_tma), sm_major_, /*is_varlen=*/false);
    const int     ws  = world_size_;
    Ulysses4DDims dims;
    dims.b        = static_cast<int>(b);
    dims.d        = static_cast<int>(d);
    dims.rank     = my_rank_;
    dims.s_local  = static_cast<int>(s_local);
    dims.n_local  = static_cast<int>(n_local);
    dims.s_global = static_cast<int>(s_local) * ws;
    dims.n_global = static_cast<int>(n_local) * ws;
    // Physical all_to_all input/output by mode: mode0 input (b,s_local,n_global,d) -> output
    // (b,s_global,n_local,d); mode1 is the reverse.
    std::vector<int64_t> in_shape, out_shape;
    if (mode == 0) {
        in_shape  = {b, dims.s_local, dims.n_global, d};
        out_shape = {b, dims.s_global, dims.n_local, d};
    }
    else {
        in_shape  = {b, dims.s_global, dims.n_local, d};
        out_shape = {b, dims.s_local, dims.n_global, d};
    }

    const at::cuda::CUDAGuard guard(at::Device(at::kCUDA, device_id_));
    cudaStream_t              stream = at::cuda::getCurrentCUDAStream();
    // The temp src must live past the microbenchmark's last cudaEventSynchronize (resolve syncs
    // internally); src is destroyed only at end of this scope, which satisfies that naturally -- do not
    // free it early. __tune__ uses a single tag (segments accumulate per shape; the number of pre-tuned
    // shapes is bounded by initial_pool_bytes).
    at::Tensor  src = at::empty(in_shape, at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_id_));
    const auto& buf = pool().acquire(out_shape, at::kBFloat16, "__tune__");
    A2AConfig   cfg =
        tma ? resolve_config_tma(
            src.data_ptr(), buf.peer_ptrs, dims, static_cast<int>(mode), 2, static_cast<int>(team_), verbose, stream) :
                resolve_config_nontma(src.data_ptr(), buf.peer_ptrs, dims, static_cast<int>(mode), 2, stream, verbose);
    cfg_cache_[config_key(ws, static_cast<int>(mode), tma, dims)] = cfg;
}

// Varlen tune: build SplitInfo from the per-rank splits, allocate a temp physical src + symmetric scratch
// output, and warm the non-TMA varlen config (resolve_config_varlen's own process-static cache). See tune()
// for the collective-call invariant. Shapes match the bindings.cpp varlen section.
void UlyssesGroup::tune_varlen(int64_t              b,
                               std::vector<int64_t> seq_lens,
                               std::vector<int64_t> head_splits,
                               int64_t              d,
                               int64_t              mode,
                               int64_t              use_tma,
                               bool                 verbose)
{
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");
    const int ws = world_size_;
    TORCH_CHECK(static_cast<int>(seq_lens.size()) == ws && static_cast<int>(head_splits.size()) == ws,
                "seq_lens/head_splits length must equal world_size");
    if (use_tma > 0)
        TORCH_CHECK(sm_major_ >= 9, "use_tma=True requires sm90+ (TMA unavailable on this GPU)");
    const int me = my_rank_;
    SplitInfo sp;
    sp.world_size = ws;
    sp.rank       = me;
    sp.b          = static_cast<int>(b);
    sp.d          = static_cast<int>(d);
    sp.s_off[0]   = 0;
    sp.n_off[0]   = 0;
    for (int r = 0; r < ws; ++r) {
        sp.s_off[r + 1] = sp.s_off[r] + static_cast<int>(seq_lens[r]);
        sp.n_off[r + 1] = sp.n_off[r] + static_cast<int>(head_splits[r]);
    }
    const int S = sp.s_off[ws];
    const int N = sp.n_off[ws];
    // Physical input/output for this rank (matches bindings varlen): mode0 input [b, seq_lens[me], N, d] ->
    // output [b, S, head_splits[me], d]; mode1 is the reverse.
    std::vector<int64_t> in_shape, out_shape;
    if (mode == 0) {
        in_shape  = {b, seq_lens[me], N, d};
        out_shape = {b, S, head_splits[me], d};
    }
    else {
        in_shape  = {b, S, head_splits[me], d};
        out_shape = {b, seq_lens[me], N, d};
    }

    const at::cuda::CUDAGuard guard(at::Device(at::kCUDA, device_id_));
    cudaStream_t              stream = at::cuda::getCurrentCUDAStream();
    at::Tensor  src = at::empty(in_shape, at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_id_));
    const auto& buf = pool().acquire(out_shape, at::kBFloat16, "__tune_varlen__");
    // Varlen path: auto/false -> non-TMA (the tuned path); explicit True -> TMA-varlen (fixed launch, nothing to
    // tune). pool().acquire above is still the collective both branches must issue in lockstep.
    const bool tma = decide_tma_path(static_cast<int>(use_tma), sm_major_, /*is_varlen=*/true);
    if (tma) {
        if (verbose && me == 0)
            printf("[ulysses varlen tune] use_tma=True -> TMA-varlen has no tunable launch config (skip)\n");
        return;
    }
    resolve_config_varlen(src.data_ptr(), buf.peer_ptrs, sp, static_cast<int>(mode), 2, stream, verbose);
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
        // DEVIATION: nvshmem_finalize() is inline and calls nvshmemi_finalize() (again only in static
        // device.a). Use nvshmemx_hostlib_finalize() exported by the host shared library instead.
        nvshmemx_hostlib_finalize();
        g_world_inited = false;
    }
}

UlyssesGroup::~UlyssesGroup()
{
    destroy();
}

}  // namespace ulysses
