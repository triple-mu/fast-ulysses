#pragma once

#include <nvtx3/nvToolsExt.h>

namespace ulysses {

// The mlx5 exchange spends most of its time in two host waits -- a stream drain before the
// posting, and a completion-queue spin after it -- around a transfer the NIC performs off
// stream. None of that appears on a CUDA timeline: the GPU simply has nothing queued, so the
// profile shows a gap with no owner. These ranges give the gap a name.
//
// They stay compiled in unconditionally. An NVTX range with no collector attached is a call
// into a stub, and this op issues single-digit ranges per call against milliseconds of
// transfer, so gating them would cost more in dead branches than it saves.
//
// Two rules the layout follows, both learned from reading a profile that did not obey them:
//
//   - **A range holds one kind of work.** A range around both a host wait and a set of kernel
//     launches has a wall clock that measures neither: the launches are asynchronous so the
//     range is far too short for them, and the wait absorbs whatever the caller had already
//     queued so it is far too long for the transfer. Neither number can be quoted.
//   - **A range with one child and the same span is noise.** Nesting is how a profile shows
//     containment; a level that contains exactly one thing shows nothing and costs a row.
class NvtxRange {
public:
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

// Per-peer detail, on its own domain so it can be turned off in a viewer without touching the
// phases. It is one row per peer per exchange -- the deepest and most numerous ranges here --
// and it answers a question ("is this peer near or across the socket") that only matters when
// somebody is already looking at one exchange.
class NvtxDetailRange {
public:
    explicit NvtxDetailRange(const char* name)
    {
        nvtxEventAttributes_t attributes{};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attributes.message.ascii = name;
        nvtxDomainRangePushEx(domain(), &attributes);
    }
    ~NvtxDetailRange() { nvtxDomainRangePop(domain()); }

    NvtxDetailRange(const NvtxDetailRange&) = delete;
    NvtxDetailRange& operator=(const NvtxDetailRange&) = delete;

private:
    static nvtxDomainHandle_t domain()
    {
        static nvtxDomainHandle_t handle = nvtxDomainCreateA("fast-ulysses.detail");
        return handle;
    }
};

}  // namespace ulysses

#define FU_NVTX_JOIN_(a, b) a##b
#define FU_NVTX_JOIN(a, b) FU_NVTX_JOIN_(a, b)
#define FU_NVTX(name) const ::ulysses::NvtxRange FU_NVTX_JOIN(fu_nvtx_, __LINE__)(name)
#define FU_NVTX_DETAIL(name) \
    const ::ulysses::NvtxDetailRange FU_NVTX_JOIN(fu_nvtx_detail_, __LINE__)(name)
