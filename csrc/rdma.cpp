#include "rdma.h"

#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

namespace ulysses {
namespace {

constexpr int kWorld = 8;
constexpr int kPort = 1;

struct GroupWire {
    uint32_t qpn[kWorld]{};
    uint32_t psn[kWorld]{};
    uint32_t mtu = 0;
    uint8_t gid[16]{};
};

struct BufferWire {
    uint64_t address = 0;
    uint32_t rkey = 0;
    uint32_t destination_rkey[kWorld]{};
    cudaIpcMemHandle_t ipc{};
    cudaIpcMemHandle_t flag_ipc{};
};

template <typename T>
std::vector<int64_t> encode(const T& value)
{
    const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
    std::vector<int64_t> result(sizeof(T));
    for (size_t i = 0; i < sizeof(T); ++i) result[i] = bytes[i];
    return result;
}

template <typename T>
T decode(const std::vector<int64_t>& bytes)
{
    TORCH_CHECK(bytes.size() == sizeof(T), "invalid RDMA metadata size");
    T result{};
    auto* output = reinterpret_cast<uint8_t*>(&result);
    for (size_t i = 0; i < sizeof(T); ++i) {
        TORCH_CHECK(bytes[i] >= 0 && bytes[i] <= 255, "invalid RDMA metadata byte");
        output[i] = static_cast<uint8_t>(bytes[i]);
    }
    return result;
}

void cuda_check(cudaError_t result, const char* operation)
{
    TORCH_CHECK(result == cudaSuccess, operation, ": ", cudaGetErrorString(result));
}

void verbs_check(int result, const char* operation)
{
    TORCH_CHECK(result == 0, operation, ": ", std::strerror(result > 0 ? result : errno));
}

int64_t checked_mul(int64_t left, int64_t right, const char* value)
{
    TORCH_CHECK(left > 0 && right > 0, value, " requires positive factors");
    TORCH_CHECK(left <= std::numeric_limits<int64_t>::max() / right,
                value, " exceeds int64 range");
    return left * right;
}

uint64_t checked_address_add(uint64_t address, uint64_t offset, const char* value)
{
    TORCH_CHECK(address <= std::numeric_limits<uint64_t>::max() - offset,
                value, " address overflows uint64");
    return address + offset;
}

uint64_t checked_offset_mul(uint64_t left, uint64_t right, const char* value)
{
    TORCH_CHECK(right == 0 || left <= std::numeric_limits<uint64_t>::max() / right,
                value, " offset exceeds uint64 range");
    return left * right;
}

size_t checked_registration_bytes(int64_t bytes)
{
    TORCH_CHECK(bytes > 0, "RDMA buffer size must be positive");
    TORCH_CHECK(static_cast<uint64_t>(bytes) <= std::numeric_limits<size_t>::max(),
                "RDMA buffer size exceeds size_t range");
    return static_cast<size_t>(bytes);
}

int64_t checked_tensor_bytes(int mode,
                             int64_t bytes,
                             int64_t batch,
                             int64_t seq,
                             int64_t heads,
                             int64_t dim,
                             int64_t element_size)
{
    TORCH_CHECK(mode == 0 || mode == 1, "RDMA mode must be 0 or 1");
    TORCH_CHECK(batch > 0 && seq > 0 && heads > 0 && dim > 0 && element_size > 0,
                "RDMA dimensions and element size must be positive");
    if (mode == 0) {
        TORCH_CHECK(heads % kWorld == 0,
                    "mode=0 head dimension must divide world_size");
    } else {
        TORCH_CHECK(seq % kWorld == 0,
                    "mode=1 sequence dimension must divide world_size");
    }

    const int64_t rows = checked_mul(batch, seq, "RDMA element count");
    const int64_t columns = checked_mul(heads, dim, "RDMA element count");
    const int64_t elements = checked_mul(rows, columns, "RDMA element count");
    const int64_t expected = checked_mul(elements, element_size, "RDMA tensor size");
    TORCH_CHECK(bytes == expected, "RDMA buffer size does not match its shape: expected ",
                expected, " bytes but received ", bytes);
    checked_registration_bytes(bytes);
    return expected;
}

struct MkeyGeometry {
    uint32_t rows = 0;
    uint32_t width = 0;
    uint32_t skip = 0;
    uint64_t pitch = 0;
};

MkeyGeometry mkey_geometry(int mode,
                           int64_t batch,
                           int64_t seq,
                           int64_t heads,
                           int64_t dim,
                           int64_t element_size)
{
    int64_t rows = 0;
    int64_t width = 0;
    int64_t pitch = 0;
    if (mode == 0) {
        rows = checked_mul(batch, seq, "mode=0 MKey rows");
        const int64_t head_width = checked_mul(heads / kWorld, dim,
                                               "mode=0 MKey width");
        width = checked_mul(head_width, element_size, "mode=0 MKey width");
        pitch = checked_mul(checked_mul(heads, dim, "mode=0 MKey pitch"),
                            element_size, "mode=0 MKey pitch");
    } else {
        rows = checked_mul(batch, seq / kWorld, "mode=1 MKey rows");
        width = checked_mul(checked_mul(heads, dim, "mode=1 MKey width"),
                            element_size, "mode=1 MKey width");
        pitch = checked_mul(checked_mul(heads, kWorld, "mode=1 MKey pitch"),
                            checked_mul(dim, element_size, "mode=1 MKey pitch"),
                            "mode=1 MKey pitch");
    }
    TORCH_CHECK(rows <= UINT32_MAX, "mode=", mode,
                " MKey row count exceeds UINT32_MAX");
    TORCH_CHECK(width <= UINT32_MAX, "mode=", mode,
                " MKey width exceeds UINT32_MAX");
    TORCH_CHECK(width <= pitch, "mode=", mode, " MKey width exceeds pitch");
    TORCH_CHECK(pitch <= kMaxInterleavedStride, "mode=", mode,
                " MKey pitch exceeds the device interleaved-stride limit of ",
                kMaxInterleavedStride, " bytes");
    const int64_t skip = pitch - width;
    TORCH_CHECK(skip <= UINT32_MAX, "mode=", mode,
                " MKey skip exceeds UINT32_MAX");
    return {static_cast<uint32_t>(rows), static_cast<uint32_t>(width),
            static_cast<uint32_t>(skip), static_cast<uint64_t>(pitch)};
}

uint64_t mkey_peer_address(const void* pointer,
                           int peer,
                           int64_t bytes,
                           const MkeyGeometry& geometry,
                           const char* value)
{
    const uint64_t peer_offset = checked_offset_mul(peer, geometry.width, value);
    const uint64_t row_offset = checked_offset_mul(geometry.rows - 1, geometry.pitch, value);
    const uint64_t end = checked_address_add(
        checked_address_add(peer_offset, row_offset, value), geometry.width, value);
    TORCH_CHECK(end <= static_cast<uint64_t>(bytes), value,
                " interleaved layout exceeds registered memory");
    const uint64_t base = reinterpret_cast<uint64_t>(pointer);
    checked_address_add(base, end, value);
    return checked_address_add(base, peer_offset, value);
}

uint32_t checked_payload(int64_t input_bytes)
{
    checked_registration_bytes(input_bytes);
    TORCH_CHECK(input_bytes % kWorld == 0,
                "RDMA input bytes must divide world_size");
    const int64_t payload = input_bytes / kWorld;
    TORCH_CHECK(payload > 0 && payload <= UINT32_MAX,
                "per-peer payload exceeds mlx5 WR limits");
    return static_cast<uint32_t>(payload);
}

ibv_mr* register_gpu_mr(ibv_pd* pd, void* pointer, size_t bytes, int access)
{
    TORCH_CHECK(pointer, "cannot register a null GPU pointer");
    auto* mr = ibv_reg_mr(pd, pointer, bytes, access);
    TORCH_CHECK(mr, "ibv_reg_mr failed: ", std::strerror(errno));
    return mr;
}

struct MrDeleter {
    void operator()(ibv_mr* mr) const noexcept
    {
        if (mr) ibv_dereg_mr(mr);
    }
};

struct MkeyDeleter {
    void operator()(mlx5dv_mkey* mkey) const noexcept
    {
        if (mkey) mlx5dv_destroy_mkey(mkey);
    }
};

using MrHandle = std::unique_ptr<ibv_mr, MrDeleter>;
using MkeyHandle = std::unique_ptr<mlx5dv_mkey, MkeyDeleter>;

int nic_number(const std::string& name)
{
    return std::stoi(name.substr(name.find('_') + 1));
}

int path_distance(const std::filesystem::path& left,
                  const std::filesystem::path& right)
{
    std::vector<std::string> a, b;
    for (const auto& part : left) a.push_back(part.string());
    for (const auto& part : right) b.push_back(part.string());
    size_t common = 0;
    while (common < a.size() && common < b.size() && a[common] == b[common]) ++common;
    return static_cast<int>(a.size() + b.size() - 2 * common);
}

std::string select_nic(int rank, int device, const std::vector<std::string>& configured)
{
    if (!configured.empty()) {
        TORCH_CHECK(static_cast<int>(configured.size()) == kWorld,
                    "an explicit NIC list must name one mlx5 device per rank");
        return configured[rank];
    }

    char pci_id[32]{};
    cuda_check(cudaDeviceGetPCIBusId(pci_id, sizeof(pci_id), device),
               "cudaDeviceGetPCIBusId");
    std::string pci_name(pci_id);
    std::transform(pci_name.begin(), pci_name.end(), pci_name.begin(),
                   [](unsigned char value) { return std::tolower(value); });
    std::error_code error;
    const auto gpu_path = std::filesystem::canonical(
        std::filesystem::path("/sys/bus/pci/devices") / pci_name, error);
    TORCH_CHECK(!error, "cannot resolve GPU PCI path ", pci_name, ": ", error.message());

    struct Candidate {
        std::string name;
        int distance;
    };
    std::vector<Candidate> candidates;
    for (const auto& entry : std::filesystem::directory_iterator("/sys/class/infiniband")) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("mlx5_", 0) != 0) continue;
        const std::string suffix = name.substr(5);
        if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(), ::isdigit)) continue;
        error.clear();
        const auto nic_path = std::filesystem::canonical(entry.path() / "device", error);
        if (!error) candidates.push_back({name, path_distance(gpu_path, nic_path)});
    }
    TORCH_CHECK(!candidates.empty(), "no mlx5 RDMA devices found");
    const int minimum = std::min_element(
        candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.distance < b.distance; })
                            ->distance;
    std::vector<std::string> closest;
    for (const auto& candidate : candidates)
        if (candidate.distance == minimum) closest.push_back(candidate.name);
    std::sort(closest.begin(), closest.end(),
              [](const std::string& a, const std::string& b) {
                  return nic_number(a) < nic_number(b);
              });
    return closest[device % closest.size()];
}

int select_gid_index(const std::string& nic, ibv_context* context)
{
    for (int index = 0; index < 128; ++index) {
        ibv_gid gid{};
        // An unpopulated index fails without meaning the table ends there, so keep scanning.
        if (ibv_query_gid(context, kPort, index, &gid)) continue;
        const auto* bytes = reinterpret_cast<const uint8_t*>(&gid);
        const bool ipv4 = bytes[10] == 0xff && bytes[11] == 0xff;
        std::ifstream type("/sys/class/infiniband/" + nic +
                           "/ports/1/gid_attrs/types/" + std::to_string(index));
        std::string value;
        std::getline(type, value);
        if (ipv4 && value.find("RoCE v2") != std::string::npos) return index;
    }
    TORCH_CHECK(false, "no IPv4 RoCE v2 GID found for ", nic);
}

}  // namespace

struct RdmaTransport::Impl {
    int rank = 0;
    int world_size = 0;
    int device = 0;
    bool active = false;
    bool connected = false;
    int write_ordering = cudaGPUDirectRDMAWritesOrderingNone;
    std::string nic_name;
    int gid_index = -1;
    ibv_context* context = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* cq = nullptr;
    std::array<ibv_qp*, kWorld> qps{};
    std::array<ibv_qp_ex*, kWorld> qpxs{};
    std::array<mlx5dv_qp_ex*, kWorld> mlx5_qpxs{};
    GroupWire local{};
    std::array<GroupWire, kWorld> peers{};
    uint64_t next_wr_id = 1;
    int pending_completions = 0;
    bool failed = false;

    bool cross(int peer) const { return peer / kQuad != rank / kQuad; }

    // A failed or timed-out exchange leaves work requests outstanding, and the NIC may still
    // write into memory this rank is about to reuse or release. Moving every QP to ERR flushes
    // them; draining the CQ afterwards retires the flush completions so a later poll cannot
    // miscount them as its own. Both are best-effort -- this runs while unwinding -- and the
    // transport is marked dead either way, so every entry point reports the original failure
    // instead of "previous RDMA exchange is unfinished" forever after.
    void quiesce() noexcept
    {
        failed = true;
        pending_completions = 0;
        for (auto* qp : qps) {
            if (!qp) continue;
            ibv_qp_attr attr{};
            attr.qp_state = IBV_QPS_ERR;
            ibv_modify_qp(qp, &attr, IBV_QP_STATE);
        }
        if (!cq) return;
        // The flush completions are not all queued the instant the QP reaches ERR, so one empty
        // poll does not mean the queue is drained. Two in a row does, near enough, and the
        // deadline bounds the case where it never settles.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        int consecutive_empty = 0;
        while (consecutive_empty < 2 && std::chrono::steady_clock::now() < deadline) {
            ibv_wc entries[kWorld]{};
            consecutive_empty = ibv_poll_cq(cq, kWorld, entries) > 0 ? 0 : consecutive_empty + 1;
        }
    }

    // Wraps anything that posts work requests or waits for them. Once a request is on a send
    // queue, giving up without retiring it leaves completions that the next poll would count as
    // its own, so every such region ends either complete or quiesced.
    template <typename Body>
    void retire_on_failure(Body&& body)
    {
        try {
            body();
        } catch (...) {
            quiesce();
            throw;
        }
    }

    void poll(int expected)
    {
        int completed = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (completed < expected) {
            ibv_wc entries[kWorld]{};
            const int count = ibv_poll_cq(cq, std::min(kWorld, expected - completed), entries);
            TORCH_CHECK(count >= 0, "ibv_poll_cq failed");
            for (int i = 0; i < count; ++i) {
                TORCH_CHECK(entries[i].status == IBV_WC_SUCCESS,
                            "mlx5 completion failed: ", ibv_wc_status_str(entries[i].status),
                            " vendor_err=", entries[i].vendor_err);
            }
            completed += count;
            // Only an empty poll can time out: a poll that completed the set has succeeded
            // whatever the clock says.
            TORCH_CHECK(count > 0 || std::chrono::steady_clock::now() < deadline,
                        "timed out waiting for mlx5 completion");
        }
    }

    mlx5dv_mkey* create_mkey()
    {
        mlx5dv_mkey_init_attr attr{};
        attr.pd = pd;
        attr.create_flags = MLX5DV_MKEY_INIT_ATTR_FLAGS_INDIRECT;
        attr.max_entries = 2;
        auto* result = mlx5dv_create_mkey(&attr);
        TORCH_CHECK(result, "mlx5dv_create_mkey failed: ", std::strerror(errno));
        return result;
    }

    void configure_mkey(int peer,
                        mlx5dv_mkey* mkey,
                        uint32_t access,
                        uint64_t address,
                        uint32_t width,
                        uint32_t skip,
                        uint32_t rows,
                        uint32_t lkey)
    {
        mlx5dv_mkey_conf_attr config{};
        mlx5dv_mr_interleaved layout{};
        layout.addr = address;
        layout.bytes_count = width;
        layout.bytes_skip = skip;
        layout.lkey = lkey;
        ibv_wr_start(qpxs[peer]);
        qpxs[peer]->wr_id = next_wr_id++;
        qpxs[peer]->wr_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
        mlx5dv_wr_mkey_configure(mlx5_qpxs[peer], mkey, 2, &config);
        mlx5dv_wr_set_mkey_access_flags(mlx5_qpxs[peer], access);
        mlx5dv_wr_set_mkey_layout_interleaved(mlx5_qpxs[peer], rows, 1, &layout);
        const int result = ibv_wr_complete(qpxs[peer]);
        TORCH_CHECK(result == 0, "configure interleaved MKey failed: ",
                    std::strerror(result));
    }

    void post_write(int peer,
                    uint32_t local_key,
                    uint64_t local_address,
                    uint32_t bytes,
                    uint32_t remote_key,
                    uint64_t remote_address)
    {
        auto* qp = qpxs[peer];
        ibv_wr_start(qp);
        qp->wr_id = next_wr_id++;
        qp->wr_flags = IBV_SEND_SIGNALED;
        ibv_wr_rdma_write_imm(qp, remote_key, remote_address, 0);
        ibv_wr_set_sge(qp, local_key, local_address, bytes);
        const int result = ibv_wr_complete(qp);
        TORCH_CHECK(result == 0, "post RDMA write failed: ", std::strerror(result));
    }

    void post_receive(int peer)
    {
        ibv_recv_wr request{};
        request.wr_id = next_wr_id++;
        ibv_recv_wr* bad = nullptr;
        const int result = ibv_post_recv(qps[peer], &request, &bad);
        TORCH_CHECK(result == 0, "post RDMA receive failed: ", std::strerror(result));
    }

    ~Impl()
    {
        for (auto* qp : qps)
            if (qp) ibv_destroy_qp(qp);
        if (cq) ibv_destroy_cq(cq);
        if (pd) ibv_dealloc_pd(pd);
        if (context) ibv_close_device(context);
    }
};

struct RdmaBuffer::Impl {
    ibv_mr* output_mr = nullptr;
    void* output_pointer = nullptr;
    int64_t output_bytes = 0;
    std::array<mlx5dv_mkey*, kWorld> destination_mkeys{};
    ibv_mr* input_mr = nullptr;
    std::array<mlx5dv_mkey*, kWorld> source_mkeys{};
    const void* input_pointer = nullptr;
    int64_t input_bytes = 0;
    int mode = 0;
    int64_t batch = 0;
    int64_t seq = 0;
    int64_t heads = 0;
    int64_t dim = 0;
    int64_t element_size = 0;
    std::array<BufferWire, kWorld> peers{};
    std::array<void*, kWorld> peer_pointers{};
    void* flags = nullptr;
    std::array<void*, kWorld> peer_flags{};
    bool connected = false;
    bool imports_closed = false;

    cudaError_t close_imports() noexcept
    {
        // Closing is terminal even if one mapping reports an error. Successfully closed handles
        // stay null so a coordinated retry only visits the remainder.
        connected = false;
        cudaError_t first_error = cudaSuccess;
        auto close = [&first_error](void*& pointer, const void* local) {
            if (!pointer || pointer == local) return;
            const cudaError_t result = cudaIpcCloseMemHandle(pointer);
            if (result == cudaSuccess)
                pointer = nullptr;
            else if (first_error == cudaSuccess)
                first_error = result;
        };
        for (auto*& pointer : peer_pointers) close(pointer, output_pointer);
        for (auto*& pointer : peer_flags) close(pointer, flags);
        if (first_error == cudaSuccess) imports_closed = true;
        return first_error;
    }

    void release_local() noexcept
    {
        for (auto*& mkey : source_mkeys) {
            if (mkey) mlx5dv_destroy_mkey(mkey);
            mkey = nullptr;
        }
        if (input_mr) ibv_dereg_mr(input_mr);
        input_mr = nullptr;
        input_pointer = nullptr;
        input_bytes = 0;
        for (auto*& mkey : destination_mkeys) {
            if (mkey) mlx5dv_destroy_mkey(mkey);
            mkey = nullptr;
        }
        if (output_mr) ibv_dereg_mr(output_mr);
        output_mr = nullptr;
        for (auto*& pointer : peer_pointers)
            if (pointer == output_pointer) pointer = nullptr;
        for (auto*& pointer : peer_flags)
            if (pointer == flags) pointer = nullptr;
        if (flags) cudaFree(flags);
        flags = nullptr;
        output_pointer = nullptr;
        output_bytes = 0;
    }

    ~Impl()
    {
        // The coordinated path closes imports before peers release their exports. This fallback
        // is deliberately noexcept and makes a successful explicit close safe to repeat.
        close_imports();
        release_local();
    }
};

RdmaBuffer::RdmaBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
RdmaBuffer::~RdmaBuffer() = default;

RdmaTransport::RdmaTransport(int rank,
                             int world_size,
                             int device,
                             const std::vector<int64_t>& devices,
                             bool enable,
                             const std::vector<std::string>& nics)
    : impl_(std::make_unique<Impl>())
{
    impl_->rank = rank;
    impl_->world_size = world_size;
    impl_->device = device;
    if (world_size != kWorld || !enable) return;
    if (devices.size() != kWorld) return;
    for (int i = 0; i < kWorld; ++i)
        if (devices[i] != i) return;
    TORCH_CHECK(rank >= 0 && rank < kWorld && device == rank,
                "mlx5 RDMA requires rank i to own CUDA device i");

    cuda_check(cudaDeviceGetAttribute(&impl_->write_ordering,
                                      cudaDevAttrGPUDirectRDMAWritesOrdering, device),
               "cudaDeviceGetAttribute(GPUDirectRDMAWritesOrdering)");
    if (impl_->write_ordering < cudaGPUDirectRDMAWritesOrderingOwner) {
        int flush_options = 0;
        cuda_check(cudaDeviceGetAttribute(
                       &flush_options, cudaDevAttrGPUDirectRDMAFlushWritesOptions, device),
                   "cudaDeviceGetAttribute(GPUDirectRDMAFlushWritesOptions)");
        TORCH_CHECK(
            flush_options & cudaFlushGPUDirectRDMAWritesOptionHost,
            "mlx5 RDMA requires cudaDeviceFlushGPUDirectRDMAWrites(), but device ", device,
            " does not advertise cudaFlushGPUDirectRDMAWritesOptionHost");
    }

    impl_->nic_name = select_nic(rank, device, nics);
    int count = 0;
    ibv_device** list = ibv_get_device_list(&count);
    TORCH_CHECK(list, "ibv_get_device_list failed: ", std::strerror(errno));
    mlx5dv_context_attr context_attr{};
    context_attr.flags = MLX5DV_CONTEXT_FLAGS_DEVX;
    for (int i = 0; i < count; ++i) {
        if (impl_->nic_name == ibv_get_device_name(list[i])) {
            impl_->context = mlx5dv_open_device(list[i], &context_attr);
            break;
        }
    }
    ibv_free_device_list(list);
    TORCH_CHECK(impl_->context, "cannot open ", impl_->nic_name,
                " with DEVX: ", std::strerror(errno));
    impl_->pd = ibv_alloc_pd(impl_->context);
    TORCH_CHECK(impl_->pd, "ibv_alloc_pd failed: ", std::strerror(errno));
    impl_->cq = ibv_create_cq(impl_->context, 256, nullptr, nullptr, 0);
    TORCH_CHECK(impl_->cq, "ibv_create_cq failed: ", std::strerror(errno));
    impl_->gid_index = select_gid_index(impl_->nic_name, impl_->context);

    for (int peer = 0; peer < kWorld; ++peer) {
        if (!impl_->cross(peer)) continue;
        ibv_qp_init_attr_ex qp_attr{};
        qp_attr.send_cq = impl_->cq;
        qp_attr.recv_cq = impl_->cq;
        qp_attr.cap.max_send_wr = 128;
        qp_attr.cap.max_recv_wr = 1;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp_attr.cap.max_inline_data = 128;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
        qp_attr.pd = impl_->pd;
        qp_attr.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM;
        mlx5dv_qp_init_attr dv_attr{};
        dv_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS;
        dv_attr.send_ops_flags = MLX5DV_QP_EX_WITH_MKEY_CONFIGURE;
        auto* qp = mlx5dv_create_qp(impl_->context, &qp_attr, &dv_attr);
        TORCH_CHECK(qp, "mlx5dv_create_qp failed: ", std::strerror(errno));
        impl_->qps[peer] = qp;
        impl_->qpxs[peer] = ibv_qp_to_qp_ex(qp);
        impl_->mlx5_qpxs[peer] = mlx5dv_qp_ex_from_ibv_qp_ex(impl_->qpxs[peer]);
        TORCH_CHECK(impl_->qpxs[peer] && impl_->mlx5_qpxs[peer],
                    "cannot create extended mlx5 QP");
        impl_->local.qpn[peer] = qp->qp_num;
        impl_->local.psn[peer] = 0x120000 + rank * 0x1000 + peer * 0x10;
    }
    ibv_port_attr port{};
    verbs_check(ibv_query_port(impl_->context, kPort, &port), "ibv_query_port");
    impl_->local.mtu = port.active_mtu;
    ibv_gid gid{};
    verbs_check(ibv_query_gid(impl_->context, kPort, impl_->gid_index, &gid),
                "ibv_query_gid");
    std::memcpy(impl_->local.gid, &gid, sizeof(gid));
    impl_->active = true;
}

RdmaTransport::~RdmaTransport() = default;

bool RdmaTransport::enabled() const { return impl_->active; }
const std::string& RdmaTransport::nic() const { return impl_->nic_name; }

std::vector<int64_t> RdmaTransport::connection_info() const
{
    TORCH_CHECK(enabled(), "RDMA transport is disabled");
    return encode(impl_->local);
}

void RdmaTransport::connect(const std::vector<std::vector<int64_t>>& encoded_peers)
{
    TORCH_CHECK(enabled(), "RDMA transport is disabled");
    TORCH_CHECK(encoded_peers.size() == kWorld, "expected eight RDMA peers");
    for (int peer = 0; peer < kWorld; ++peer)
        impl_->peers[peer] = decode<GroupWire>(encoded_peers[peer]);

    for (int peer = 0; peer < kWorld; ++peer) {
        auto* qp = impl_->qps[peer];
        if (!qp) continue;
        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = kPort;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        verbs_check(ibv_modify_qp(qp, &attr,
                                  IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                      IBV_QP_ACCESS_FLAGS),
                    "QP RESET->INIT");

        attr = {};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = static_cast<ibv_mtu>(
            std::min(impl_->local.mtu, impl_->peers[peer].mtu));
        attr.dest_qp_num = impl_->peers[peer].qpn[impl_->rank];
        attr.rq_psn = impl_->peers[peer].psn[impl_->rank];
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.port_num = kPort;
        std::memcpy(&attr.ah_attr.grh.dgid, impl_->peers[peer].gid, 16);
        attr.ah_attr.grh.sgid_index = impl_->gid_index;
        attr.ah_attr.grh.hop_limit = 64;
        verbs_check(ibv_modify_qp(qp, &attr,
                                  IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                      IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                      IBV_QP_MAX_DEST_RD_ATOMIC |
                                      IBV_QP_MIN_RNR_TIMER),
                    "QP INIT->RTR");

        attr = {};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 18;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = impl_->local.psn[peer];
        attr.max_rd_atomic = 1;
        verbs_check(ibv_modify_qp(qp, &attr,
                                  IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                      IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                      IBV_QP_MAX_QP_RD_ATOMIC),
                    "QP RTR->RTS");
    }
    impl_->connected = true;
}

std::unique_ptr<RdmaBuffer> RdmaTransport::register_buffer(void* pointer,
                                                           int64_t bytes,
                                                           int mode,
                                                           int64_t batch,
                                                           int64_t seq,
                                                           int64_t heads,
                                                           int64_t dim,
                                                           int64_t element_size)
{
    TORCH_CHECK(impl_->connected, "RDMA transport is not connected");
    TORCH_CHECK(!impl_->failed, "the mlx5 transport failed and is permanently unusable");
    TORCH_CHECK(pointer, "cannot register a null RDMA output pointer");
    checked_tensor_bytes(mode, bytes, batch, seq, heads, dim, element_size);
    const uint32_t payload = checked_payload(bytes);

    MkeyGeometry geometry{};
    std::array<uint64_t, kWorld> destination_addresses{};
    if (mode == 1) {
        geometry = mkey_geometry(mode, batch, seq, heads, dim, element_size);
        TORCH_CHECK(checked_offset_mul(geometry.rows, geometry.width,
                                       "mode=1 MKey payload") == payload,
                    "mode=1 MKey geometry does not match the per-peer payload");
        for (int peer = 0; peer < kWorld; ++peer) {
            if (!impl_->cross(peer)) continue;
            destination_addresses[peer] = mkey_peer_address(
                pointer, peer, bytes, geometry, "mode=1 destination MKey");
        }
    }

    auto state = std::make_unique<RdmaBuffer::Impl>();
    state->mode = mode;
    state->output_pointer = pointer;
    state->output_bytes = bytes;
    state->batch = batch;
    state->seq = seq;
    state->heads = heads;
    state->dim = dim;
    state->element_size = element_size;
    state->peer_pointers[impl_->rank] = pointer;
    cuda_check(cudaMalloc(&state->flags, kWorld * sizeof(uint64_t)), "cudaMalloc RDMA flags");
    cuda_check(cudaMemset(state->flags, 0, kWorld * sizeof(uint64_t)),
               "cudaMemset RDMA flags");
    state->peer_flags[impl_->rank] = state->flags;
    MrHandle output_mr(register_gpu_mr(
        impl_->pd, pointer, checked_registration_bytes(bytes),
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));

    if (mode == 1) {
        std::array<MkeyHandle, kWorld> destination_mkeys{};
        for (int peer = 0; peer < kWorld; ++peer) {
            if (!impl_->cross(peer)) continue;
            destination_mkeys[peer].reset(impl_->create_mkey());
        }
        impl_->retire_on_failure([&] {
            int configured = 0;
            for (int peer = 0; peer < kWorld; ++peer) {
                if (!impl_->cross(peer)) continue;
                impl_->configure_mkey(
                    peer, destination_mkeys[peer].get(),
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_REMOTE_READ,
                    destination_addresses[peer], geometry.width, geometry.skip,
                    geometry.rows, output_mr->lkey);
                ++configured;
            }
            impl_->poll(configured);
        });
        for (int peer = 0; peer < kWorld; ++peer)
            state->destination_mkeys[peer] = destination_mkeys[peer].release();
    }
    state->output_mr = output_mr.release();
    return std::unique_ptr<RdmaBuffer>(new RdmaBuffer(std::move(state)));
}

std::vector<int64_t> RdmaTransport::buffer_info(const RdmaBuffer& buffer) const
{
    BufferWire wire{};
    wire.address = reinterpret_cast<uint64_t>(buffer.impl_->output_pointer);
    wire.rkey = buffer.impl_->output_mr->rkey;
    cuda_check(cudaIpcGetMemHandle(&wire.ipc, buffer.impl_->output_pointer),
               "cudaIpcGetMemHandle");
    cuda_check(cudaIpcGetMemHandle(&wire.flag_ipc, buffer.impl_->flags),
               "cudaIpcGetMemHandle flags");
    for (int peer = 0; peer < kWorld; ++peer)
        if (buffer.impl_->destination_mkeys[peer])
            wire.destination_rkey[peer] = buffer.impl_->destination_mkeys[peer]->rkey;
    return encode(wire);
}

void RdmaTransport::connect_buffer(
    RdmaBuffer& buffer,
    const std::vector<std::vector<int64_t>>& encoded_peers) const
{
    TORCH_CHECK(encoded_peers.size() == kWorld, "expected eight RDMA buffer peers");
    TORCH_CHECK(!buffer.impl_->connected, "RDMA buffer is already connected");
    TORCH_CHECK(!buffer.impl_->imports_closed,
                "RDMA buffer imports were permanently closed");
    std::array<BufferWire, kWorld> peers{};
    for (int peer = 0; peer < kWorld; ++peer)
        peers[peer] = decode<BufferWire>(encoded_peers[peer]);

    struct PendingImports {
        std::array<void*, kWorld> payloads{};
        std::array<void*, kWorld> flags{};

        ~PendingImports()
        {
            for (auto* pointer : payloads)
                if (pointer) cudaIpcCloseMemHandle(pointer);
            for (auto* pointer : flags)
                if (pointer) cudaIpcCloseMemHandle(pointer);
        }
    } pending;
    for (int peer = 0; peer < kWorld; ++peer) {
        if (peer == impl_->rank) continue;
        if (!impl_->cross(peer))
            cuda_check(cudaIpcOpenMemHandle(&pending.payloads[peer],
                                            peers[peer].ipc,
                                            cudaIpcMemLazyEnablePeerAccess),
                       "cudaIpcOpenMemHandle");
        // Flags are mapped from every rank, not just the quad: the payload crosses the quad
        // boundary through the NIC, but the handshake is a single word and P2P reaches it.
        cuda_check(cudaIpcOpenMemHandle(&pending.flags[peer],
                                        peers[peer].flag_ipc,
                                        cudaIpcMemLazyEnablePeerAccess),
                   "cudaIpcOpenMemHandle flags");
    }
    buffer.impl_->peers = peers;
    for (int peer = 0; peer < kWorld; ++peer) {
        if (peer == impl_->rank) continue;
        buffer.impl_->peer_pointers[peer] = pending.payloads[peer];
        pending.payloads[peer] = nullptr;
        buffer.impl_->peer_flags[peer] = pending.flags[peer];
        pending.flags[peer] = nullptr;
    }
    buffer.impl_->connected = true;
}

std::vector<uint64_t> RdmaTransport::peer_pointers(const RdmaBuffer& buffer) const
{
    std::vector<uint64_t> result(kWorld);
    for (int peer = 0; peer < kWorld; ++peer)
        result[peer] = reinterpret_cast<uint64_t>(buffer.impl_->peer_pointers[peer]);
    return result;
}

std::vector<uint64_t> RdmaTransport::peer_flags(const RdmaBuffer& buffer) const
{
    std::vector<uint64_t> result(kWorld);
    for (int peer = 0; peer < kWorld; ++peer)
        result[peer] = reinterpret_cast<uint64_t>(buffer.impl_->peer_flags[peer]);
    return result;
}

void RdmaTransport::close_buffer_imports(RdmaBuffer& buffer) const
{
    if (buffer.impl_->imports_closed) return;
    TORCH_CHECK(impl_->pending_completions == 0,
                "cannot close RDMA buffer imports while an exchange is pending");
    const cudaError_t result = buffer.impl_->close_imports();
    cuda_check(result, "cudaIpcCloseMemHandle RDMA buffer imports");
}

void RdmaTransport::start_exchange(const void* input,
                                   int64_t input_bytes,
                                   RdmaBuffer& output,
                                   int mode,
                                   int64_t batch,
                                   int64_t seq,
                                   int64_t heads,
                                   int64_t dim,
                                   int64_t element_size)
{
    TORCH_CHECK(!impl_->failed, "the mlx5 transport failed and is permanently unusable");
    TORCH_CHECK(impl_->pending_completions == 0,
                "previous RDMA exchange is unfinished");
    TORCH_CHECK(output.impl_->connected, "RDMA output metadata is not connected");
    TORCH_CHECK(output.impl_->mode == mode, "RDMA output mode mismatch");
    auto& state = *output.impl_;
    TORCH_CHECK(input, "RDMA input pointer must not be null");
    if (state.input_mr) {
        TORCH_CHECK(state.input_pointer == input && state.input_bytes == input_bytes,
                    "RDMA input storage and byte size are fixed after first use; allocate a "
                    "separate output workspace for a different input storage");
    }
    TORCH_CHECK(state.batch == batch && state.seq == seq && state.heads == heads &&
                    state.dim == dim && state.element_size == element_size,
                "RDMA output workspace was registered for a different input shape or dtype");
    TORCH_CHECK(state.output_bytes == input_bytes,
                "RDMA input and output byte sizes must match");
    checked_tensor_bytes(mode, input_bytes, batch, seq, heads, dim, element_size);
    const MkeyGeometry geometry =
        mkey_geometry(mode, batch, seq, heads, dim, element_size);
    const uint32_t payload = checked_payload(input_bytes);
    TORCH_CHECK(checked_offset_mul(geometry.rows, geometry.width,
                                   "RDMA MKey payload") == payload,
                "RDMA MKey geometry does not match the per-peer payload");
    const uint64_t payload64 = payload;

    std::array<uint64_t, kWorld> source_mkey_addresses{};
    std::array<uint64_t, kWorld> local_addresses{};
    std::array<uint64_t, kWorld> remote_addresses{};
    std::array<uint32_t, kWorld> remote_keys{};
    const uint64_t input_address = reinterpret_cast<uint64_t>(input);
    for (int step = kQuad; step < kWorld; ++step) {
        const int peer = impl_->rank ^ step;
        TORCH_CHECK(impl_->qps[peer] && impl_->qpxs[peer] && impl_->mlx5_qpxs[peer],
                    "missing mlx5 QP for peer ", peer);
        if (mode == 0) {
            source_mkey_addresses[peer] = mkey_peer_address(
                input, peer, input_bytes, geometry, "mode=0 source MKey");
            TORCH_CHECK(state.peers[peer].address && state.peers[peer].rkey,
                        "missing mode=0 output registration for peer ", peer);
            const uint64_t remote_offset =
                checked_offset_mul(impl_->rank, payload64, "mode=0 remote payload");
            TORCH_CHECK(checked_address_add(remote_offset, payload64,
                                            "mode=0 remote payload") <=
                            static_cast<uint64_t>(state.output_bytes),
                        "mode=0 remote payload exceeds output buffer");
            checked_address_add(
                state.peers[peer].address,
                checked_address_add(remote_offset, payload64,
                                    "mode=0 remote payload"),
                "mode=0 remote payload");
            remote_addresses[peer] = checked_address_add(
                state.peers[peer].address, remote_offset, "mode=0 remote payload");
            remote_keys[peer] = state.peers[peer].rkey;
        } else {
            remote_keys[peer] = state.peers[peer].destination_rkey[impl_->rank];
            TORCH_CHECK(remote_keys[peer],
                        "missing mode=1 destination MKey for peer ", peer);
            const uint64_t local_offset =
                checked_offset_mul(peer, payload64, "mode=1 local payload");
            TORCH_CHECK(checked_address_add(local_offset, payload64,
                                            "mode=1 local payload") <=
                            static_cast<uint64_t>(input_bytes),
                        "mode=1 local payload exceeds input buffer");
            checked_address_add(
                input_address,
                checked_address_add(local_offset, payload64,
                                    "mode=1 local payload"),
                "mode=1 local payload");
            local_addresses[peer] = checked_address_add(
                input_address, local_offset, "mode=1 local payload");
        }
    }

    if (!state.input_mr) {
        MrHandle input_mr(register_gpu_mr(
            impl_->pd, const_cast<void*>(input), checked_registration_bytes(input_bytes),
            IBV_ACCESS_LOCAL_WRITE));
        std::array<MkeyHandle, kWorld> source_mkeys{};
        if (mode == 0) {
            for (int peer = 0; peer < kWorld; ++peer) {
                if (!impl_->cross(peer)) continue;
                source_mkeys[peer].reset(impl_->create_mkey());
            }
            impl_->retire_on_failure([&] {
                int configured = 0;
                for (int peer = 0; peer < kWorld; ++peer) {
                    if (!impl_->cross(peer)) continue;
                    impl_->configure_mkey(peer, source_mkeys[peer].get(), 0,
                                          source_mkey_addresses[peer], geometry.width,
                                          geometry.skip, geometry.rows, input_mr->lkey);
                    ++configured;
                }
                impl_->poll(configured);
            });
            for (int peer = 0; peer < kWorld; ++peer)
                state.source_mkeys[peer] = source_mkeys[peer].release();
        }
        state.input_pointer = input;
        state.input_bytes = input_bytes;
        state.input_mr = input_mr.release();
    }

    std::array<uint32_t, kWorld> local_keys{};
    for (int step = kQuad; step < kWorld; ++step) {
        const int peer = impl_->rank ^ step;
        if (mode == 0) {
            TORCH_CHECK(state.source_mkeys[peer],
                        "missing mode=0 source MKey for peer ", peer);
            local_keys[peer] = state.source_mkeys[peer]->lkey;
        } else {
            TORCH_CHECK(state.input_mr, "missing RDMA input registration");
            local_keys[peer] = state.input_mr->lkey;
        }
    }

    // Every shape conversion, address calculation, and key lookup above is complete before the
    // first receive is posted. A validation failure therefore cannot strand half an exchange.
    // A posting failure still can, so it retires whatever was posted before it gives up.
    impl_->retire_on_failure([&] {
        for (int step = kQuad; step < kWorld; ++step) {
            impl_->post_receive(impl_->rank ^ step);
            ++impl_->pending_completions;
        }
        for (int step = kQuad; step < kWorld; ++step) {
            const int peer = impl_->rank ^ step;
            if (mode == 0) {
                impl_->post_write(peer, local_keys[peer], 0, payload,
                                  remote_keys[peer], remote_addresses[peer]);
            } else {
                impl_->post_write(peer, local_keys[peer], local_addresses[peer], payload,
                                  remote_keys[peer], 0);
            }
            ++impl_->pending_completions;
        }
    });
}

void RdmaTransport::finish_exchange()
{
    TORCH_CHECK(!impl_->failed, "the mlx5 transport failed and is permanently unusable");
    const int pending = impl_->pending_completions;
    TORCH_CHECK(pending > 0, "no RDMA exchange is pending");
    impl_->retire_on_failure([&] { impl_->poll(pending); });
    impl_->pending_completions = 0;
}

void RdmaTransport::flush() const
{
    if (!enabled()) return;
    if (impl_->write_ordering >= cudaGPUDirectRDMAWritesOrderingOwner) return;
    cuda_check(cudaDeviceFlushGPUDirectRDMAWrites(
                   cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                   cudaFlushGPUDirectRDMAWritesToOwner),
               "cudaDeviceFlushGPUDirectRDMAWrites");
}

}  // namespace ulysses
