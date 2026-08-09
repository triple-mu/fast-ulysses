// Contract: include/fast_ulysses/nvlink.hpp.
//
// NVML is loaded with dlopen rather than linked, so a machine without it -- or with one too old
// for these entry points -- reports "cannot say" instead of failing to load the extension.
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <mutex>
#include <set>
#include <tuple>

#include <fast_ulysses/common.hpp>
#include <fast_ulysses/nvlink.hpp>

namespace ulysses {

namespace {

constexpr unsigned kNvmlSuccess = 0;
// A device with no NVLink at all answers NOT_SUPPORTED for link 0, and one with NVLink answers
// INVALID_ARGUMENT past its last link. Both are the device ANSWERING -- "no more links" -- and
// must not be confused with NVML being unusable, which is what nullopt means to the caller.
constexpr unsigned kNvmlInvalidArgument = 2;
constexpr unsigned kNvmlNotSupported    = 3;
constexpr int      kMaxLinks            = 32;  // past NVML_NVLINK_MAX_LINKS (18 today); the sweep stops on error
constexpr unsigned kRemoteGpu           = 0;   // NVML_NVLINK_DEVICE_TYPE_GPU
constexpr unsigned kRemoteSwitch        = 2;   // NVML_NVLINK_DEVICE_TYPE_SWITCH

// nvmlPciInfo_t. Only domain/bus/device are read, and only after the link's remote end has already
// said it is a GPU; an unmatched result is reported as "cannot determine" rather than as a missing
// link, so a layout change cannot turn into a wrong refusal.
struct NvmlPciInfo {
    char     bus_id_legacy[16];
    unsigned domain;
    unsigned bus;
    unsigned device;
    unsigned pci_device_id;
    unsigned pci_subsystem_id;
    char     bus_id[32];
};

struct Nvml {
    void* handle                                                 = nullptr;
    unsigned (*init)()                                           = nullptr;
    unsigned (*handle_by_pci_bus_id)(const char*, void**)        = nullptr;
    unsigned (*nvlink_state)(void*, unsigned, unsigned*)         = nullptr;
    unsigned (*nvlink_remote_type)(void*, unsigned, unsigned*)   = nullptr;
    unsigned (*nvlink_remote_pci)(void*, unsigned, NvmlPciInfo*) = nullptr;

    bool ok() const
    {
        return handle != nullptr && handle_by_pci_bus_id != nullptr && nvlink_state != nullptr
               && nvlink_remote_type != nullptr && nvlink_remote_pci != nullptr;
    }
};

template<typename Fn>
void resolve(void* handle, const char* name, Fn& out)
{
    out = reinterpret_cast<Fn>(dlsym(handle, name));
}

// Loaded and initialised once. NVML init is idempotent but not free, and this is only ever called
// at group construction and from `doctor`.
const Nvml& nvml()
{
    static Nvml           lib;
    static std::once_flag once;
    std::call_once(once, [] {
        lib.handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (lib.handle == nullptr) {
            return;
        }
        resolve(lib.handle, "nvmlInit_v2", lib.init);
        resolve(lib.handle, "nvmlDeviceGetHandleByPciBusId_v2", lib.handle_by_pci_bus_id);
        resolve(lib.handle, "nvmlDeviceGetNvLinkState", lib.nvlink_state);
        resolve(lib.handle, "nvmlDeviceGetNvLinkRemoteDeviceType", lib.nvlink_remote_type);
        resolve(lib.handle, "nvmlDeviceGetNvLinkRemotePciInfo_v2", lib.nvlink_remote_pci);
        if (lib.init == nullptr || lib.init() != kNvmlSuccess) {
            lib.handle = nullptr;  // treat a failed init as "no NVML"
        }
    });
    return lib;
}

// The PCI triple CUDA reports for a device, which is what an NVML link's remote end is matched against.
struct Bdf {
    unsigned domain, bus, device;
    bool     operator<(const Bdf& o) const
    {
        return std::tie(domain, bus, device) < std::tie(o.domain, o.bus, o.device);
    }
};

bool device_bdf(int64_t index, Bdf& out)
{
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, static_cast<int>(index)) != cudaSuccess) {
        return false;
    }
    out = {static_cast<unsigned>(prop.pciDomainID),
           static_cast<unsigned>(prop.pciBusID),
           static_cast<unsigned>(prop.pciDeviceID)};
    return true;
}

}  // namespace

std::optional<std::map<std::pair<int64_t, int64_t>, bool>> nvlink_matrix(const std::vector<int64_t>& devices)
{
    const Nvml& lib = nvml();
    if (!lib.ok()) {
        return std::nullopt;
    }

    std::map<Bdf, int64_t>   by_bdf;
    std::map<int64_t, void*> handles;
    for (int64_t i : devices) {
        Bdf bdf{};
        if (!device_bdf(i, bdf)) {
            return std::nullopt;
        }
        by_bdf[bdf] = i;
        char bus_id[32];
        std::snprintf(bus_id, sizeof(bus_id), "%08x:%02x:%02x.0", bdf.domain, bdf.bus, bdf.device);
        void* h = nullptr;
        if (lib.handle_by_pci_bus_id(bus_id, &h) != kNvmlSuccess) {
            return std::nullopt;
        }
        handles[i] = h;
    }

    std::map<std::pair<int64_t, int64_t>, bool> linked;
    for (int64_t i : devices) {
        for (int64_t j : devices) {
            linked[{i, j}] = (i == j);
        }
    }

    std::set<int64_t> on_switch;
    for (const auto& entry : handles) {
        // Per DEVICE, not once for the whole sweep. Shared, the first device to answer would let
        // every later one that NVML refused keep the `false` it was initialised with above -- and
        // check_nvlink would then refuse the group claiming a link is missing, when the truth is
        // that nobody asked successfully. The header promises the opposite.
        bool answered = false;
        for (int link = 0; link < kMaxLinks; ++link) {
            unsigned       active = 0;
            const unsigned rc     = lib.nvlink_state(entry.second, static_cast<unsigned>(link), &active);
            if (rc == kNvmlNotSupported || rc == kNvmlInvalidArgument) {
                answered = true;  // the device has no more links, which is an answer
                break;
            }
            if (rc != kNvmlSuccess) {
                break;  // anything else means we genuinely could not ask
            }
            answered = true;
            if (active == 0) {
                continue;
            }
            unsigned kind = 0;
            if (lib.nvlink_remote_type(entry.second, static_cast<unsigned>(link), &kind) != kNvmlSuccess) {
                continue;
            }
            if (kind == kRemoteSwitch) {
                on_switch.insert(entry.first);
            }
            else if (kind == kRemoteGpu) {
                NvmlPciInfo info{};
                if (lib.nvlink_remote_pci(entry.second, static_cast<unsigned>(link), &info) != kNvmlSuccess) {
                    continue;
                }
                auto it = by_bdf.find(Bdf{info.domain, info.bus, info.device});
                if (it != by_bdf.end()) {
                    linked[{entry.first, it->second}] = true;
                    linked[{it->second, entry.first}] = true;
                }
            }
        }
        // One device we could not probe makes the whole matrix unusable: its row and column would
        // read as "not linked" and be indistinguishable from a real absence of links.
        if (!answered) {
            return std::nullopt;
        }
    }

    // One fabric per node, so every GPU with a switch link can reach every other one.
    for (int64_t i : on_switch) {
        for (int64_t j : on_switch) {
            linked[{i, j}] = true;
        }
    }
    return linked;
}

std::string check_nvlink(const std::vector<int64_t>& devices)
{
    const auto matrix = nvlink_matrix(devices);
    if (!matrix.has_value()) {
        return {};
    }
    for (int64_t i : devices) {
        for (int64_t j : devices) {
            if (i != j && !matrix->at({i, j})) {
                return "cuda:" + std::to_string(i) + " and cuda:" + std::to_string(j)
                       + " are not joined by NVLink. fast-ulysses writes peer memory directly, which over "
                         "PCIe -- and especially across a CPU socket -- is slower than torch.distributed, "
                         "because that routes around the link instead. Use torch.distributed for this "
                         "group, or pass require_nvlink=False to measure it anyway. `fast-ulysses doctor` "
                         "prints the full matrix.";
            }
        }
    }
    return {};
}

}  // namespace ulysses
