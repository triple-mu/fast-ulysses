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
class NvtxRange {
public:
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace ulysses

#define FU_NVTX_JOIN_(a, b) a##b
#define FU_NVTX_JOIN(a, b) FU_NVTX_JOIN_(a, b)
#define FU_NVTX(name) const ::ulysses::NvtxRange FU_NVTX_JOIN(fu_nvtx_, __LINE__)(name)
