// CE (copy-engine) transfer path for the 4D all-to-all: pitched cudaMemcpy2D/3DAsync straight into
// the peers' window addresses. The copies that cross a link run on the DMA engines and take no
// SMs, so they proceed while compute holds every SM rather than waiting for a block slot. This
// rank's own share does not cross a link and does compete; see transfer.hpp.
// It computes no offsets -- the addressing comes from build_plan in a2a_plan.cc.
#include <fast_ulysses/common.hpp>
#include <fast_ulysses/transfer.hpp>

#include <chrono>
#include <thread>

namespace ulysses {

namespace {

// Debug-only, armed from Python. 0 = off, which is the only state a normal build ever sees.
int64_t g_fault_delay_us = 0;

void CUDART_CB delay_payload(void* arg)
{
    std::this_thread::sleep_for(std::chrono::microseconds(reinterpret_cast<int64_t>(arg)));
}

// One CopyOp -> one CUDA call. The slice counts below are derived, not passed, because
// cudaMemcpy3DParms takes no slice stride (see push_batched).
void issue_copy(void* dst, const void* src, const CopyOp& op, cudaStream_t stream)
{
    if (op.depth <= 1) {
        ULYSSES_CUDA_CHECK(cudaMemcpy2DAsync(dst,
                                             static_cast<size_t>(op.dst_pitch),
                                             src,
                                             static_cast<size_t>(op.src_pitch),
                                             static_cast<size_t>(op.width),
                                             static_cast<size_t>(op.rows),
                                             cudaMemcpyDefault,
                                             stream));
        return;
    }
    cudaMemcpy3DParms parms = {};
    parms.srcPtr            = make_cudaPitchedPtr(const_cast<void*>(src),
                                       static_cast<size_t>(op.src_pitch),
                                       static_cast<size_t>(op.width),
                                       static_cast<size_t>(op.src_slice / op.src_pitch));
    parms.dstPtr            = make_cudaPitchedPtr(dst,
                                       static_cast<size_t>(op.dst_pitch),
                                       static_cast<size_t>(op.width),
                                       static_cast<size_t>(op.dst_slice / op.dst_pitch));
    parms.extent =
        make_cudaExtent(static_cast<size_t>(op.width), static_cast<size_t>(op.rows), static_cast<size_t>(op.depth));
    parms.kind = cudaMemcpyDefault;
    ULYSSES_CUDA_CHECK(cudaMemcpy3DAsync(&parms, stream));
}

}  // namespace

// Arm (delay_us > 0) or disarm (0) the "signal before payload" fault. TESTS ONLY.
//
// The operator depends on copy-engine writes being visible at the destination by the time the
// barrier kernel's release store announcing them arrives, which is undocumented, so the test for it
// is only worth as much as its negative control. Armed, launch_a2a_ce holds the payload back and
// skips the join onto the caller's stream, so the closing barrier publishes while the bytes are in
// flight. See test/distributed/ce_ordering.py.
void set_ce_fault(int64_t delay_us)
{
    g_fault_delay_us = delay_us;
}

void launch_a2a_ce(const void*                  src,
                   const std::vector<uint64_t>& peer_ptrs,
                   const A2APlan&               plan,
                   cudaStream_t                 xfer,
                   int                          rank,
                   cudaStream_t                 stream)
{
    const int      ws        = static_cast<int>(peer_ptrs.size());
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    // Fresh events every call -- do not hoist them onto the group. A shared event re-recorded
    // while earlier waits are still pending lets one of those waits resolve against a later record,
    // which makes the wait depend on the very stream it is blocking: a circular wait. Creating them
    // here is depth-safe, since a wait captures the dependency at call time and destroy defers.
    //
    // ONE STREAM for the remote copies, since they all leave through the same egress and separate
    // streams would only make them contend. This rank's OWN share crosses no link, so it can run
    // alongside them on the caller's stream -- hence the separate emit below. Peers are visited in
    // XOR-shift order, which pairs ranks up without coordination.
    cudaEvent_t ready;
    ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    ULYSSES_CUDA_CHECK(cudaEventRecord(ready, stream));

    auto emit = [&](int p, cudaStream_t on) {
        for (const CopyOp& op : plan.ops) {
            if (op.peer == p) {
                issue_copy(reinterpret_cast<uint8_t*>(peer_ptrs[p]) + op.dst_offset, src_bytes + op.src_offset, op, on);
            }
        }
    };

    ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(xfer, ready, 0));
    // Fault injection (see set_ce_fault): hold the payload back so the flag cannot be behind it.
    if (g_fault_delay_us > 0) {
        ULYSSES_CUDA_CHECK(cudaLaunchHostFunc(xfer, delay_payload, reinterpret_cast<void*>(g_fault_delay_us)));
    }
    for (int k = 1; k < ws; ++k) {
        const int peer = rank ^ k;
        if (peer < ws) {
            emit(peer, xfer);
        }
    }
    // XOR only enumerates every peer when ws is a power of two; sweep for any it missed.
    if ((ws & (ws - 1)) != 0) {
        for (int p = 0; p < ws; ++p) {
            if (p != rank && (p ^ rank) >= ws) {
                emit(p, xfer);
            }
        }
    }
    emit(rank, stream);

    cudaEvent_t done;
    ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    ULYSSES_CUDA_CHECK(cudaEventRecord(done, xfer));
    // Skipping this join is the fault: the caller's stream then reaches the closing barrier, and
    // publishes, without waiting for the remote copies. Nothing else about the call changes.
    if (g_fault_delay_us == 0) {
        ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(stream, done, 0));
    }
    ULYSSES_CUDA_CHECK(cudaEventDestroy(done));
    ULYSSES_CUDA_CHECK(cudaEventDestroy(ready));
}

}  // namespace ulysses
