// CE (copy-engine) transfer path for the uniform 4D all-to-all: per-peer pitched
// cudaMemcpy2DAsync fan-out on dedicated streams, joined back to the launching stream
// with events. The data movement uses DMA engines and zero SMs, so -- unlike the
// SM-resident scatter or the (1-block) TMA kernel, which cannot get a block slot while
// e.g. cuBLAS nvjet GEMMs hold every SM -- it runs at full NVLink bandwidth concurrently
// with compute. Measured (exclusive 4xH100/4xH200, Wan ws=4): standalone ~0.69ms
// (209 GB/s, vs 385 GB/s for a bare pitched peer memcpy), but 93-94% of it overlaps a
// concurrent GEMM chain where the kernel paths overlap ~25-38% -- net exposed time
// ~0.05ms/call vs ~0.36-0.38ms.
//
// Per-peer transfer shape (mode0): destination peer p receives, for every (b, s) row,
// the contiguous n_local*d block at head-column p: s_local rows of n_local*d*elem bytes,
// source pitch n_global*d*elem, contiguous destination rows -> one pitched 2D copy per
// (peer, b). mode1 is the inverse (contiguous source rows, pitched destination).
// Addressing mirrors a2a_copy_generic (all_to_all.cu) byte for byte.
//
// NOTE (measured on CUDA 13.3, exclusive GPUs; alternatives tried and REVERTED):
// - single-stream serial submission: local copy loses its overlap with the remote
//   copies -- 0.82ms standalone, no upside elsewhere;
// - cudaMemcpy3DBatchAsync: the driver's pitched batch path is itself slow (0.82ms,
//   and 1.35ms with cudaMemcpyFlagPreferOverlapWithCompute); it also rejects the
//   LEGACY default stream with "invalid argument" (explicit streams required).
// The per-peer stream pool below beat both on standalone time at equal hiding.
#include "ulysses_group.cuh"

namespace ulysses {

namespace {

// One pitched 2D copy: (peer p, batch i) -> pointers and pitches in bytes.
struct CopyOp {
    const uint8_t* src;
    uint8_t*       dst;
    int64_t        spitch, dpitch;
};

CopyOp make_op(const void* src, uint64_t peer_ptr, const Ulysses4DDims& dims, int mode, int elem_size, int p, int i)
{
    const int64_t row_w  = static_cast<int64_t>(dims.n_local) * dims.d * elem_size;
    const int64_t pitchg = static_cast<int64_t>(dims.n_global) * dims.d * elem_size;
    CopyOp        op;
    if (mode == 0) {
        op.src = static_cast<const uint8_t*>(src) + static_cast<int64_t>(i) * dims.s_local * pitchg
                 + static_cast<int64_t>(p) * row_w;
        op.spitch = pitchg;
        op.dst    = reinterpret_cast<uint8_t*>(peer_ptr)
                 + (static_cast<int64_t>(i) * dims.s_global + static_cast<int64_t>(dims.rank) * dims.s_local) * row_w;
        op.dpitch = row_w;
    }
    else {
        op.src = static_cast<const uint8_t*>(src)
                 + (static_cast<int64_t>(i) * dims.s_global + static_cast<int64_t>(p) * dims.s_local) * row_w;
        op.spitch = row_w;
        op.dst    = reinterpret_cast<uint8_t*>(peer_ptr) + static_cast<int64_t>(i) * dims.s_local * pitchg
                 + static_cast<int64_t>(dims.rank) * row_w;
        op.dpitch = pitchg;
    }
    return op;
}

}  // namespace

void launch_a2a_ce(const void*                  src,
                   const std::vector<uint64_t>& peer_ptrs,
                   const Ulysses4DDims&         dims,
                   int                          mode,
                   int                          elem_size,
                   const CEResources&           ce,
                   cudaStream_t                 stream)
{
    const int     ws    = static_cast<int>(peer_ptrs.size());
    const int64_t row_w = static_cast<int64_t>(dims.n_local) * dims.d * elem_size;
    // Fan out: every CE stream waits for the launching stream (inputs ready), copies its
    // peer's slice, and the launching stream joins all copies before the caller's barrier.
    // The shared source-egress NVLink port caps aggregate bandwidth regardless of stream
    // count; the pool's value is keeping the LOCAL copy concurrent with the remote ones.
    //
    // FRESH events every call -- do not hoist them into CEResources. Re-recording a shared
    // event that still has in-flight stream waits (deep enqueue-ahead: many deferred
    // barrier=False groups queued behind the device) lets a pending wait resolve against a
    // LATER record whose completion depends on this very stream progressing -- a circular
    // wait that deadlocks the group (reproduced at ws=2 with a few undrained groups).
    // Create/destroy is a few us per call and depth-safe: the waits capture the dependency
    // at call time, and destroy defers until the event retires.
    cudaEvent_t ready;
    ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    ULYSSES_CUDA_CHECK(cudaEventRecord(ready, stream));
    for (int p = 0; p < ws; ++p) {
        cudaStream_t cs = ce.streams[p];
        ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(cs, ready, 0));
        for (int i = 0; i < dims.b; ++i) {
            const CopyOp op = make_op(src, peer_ptrs[p], dims, mode, elem_size, p, i);
            ULYSSES_CUDA_CHECK(
                cudaMemcpy2DAsync(op.dst, op.dpitch, op.src, op.spitch, row_w, dims.s_local, cudaMemcpyDefault, cs));
        }
        cudaEvent_t done;
        ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        ULYSSES_CUDA_CHECK(cudaEventRecord(done, cs));
        ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(stream, done, 0));
        ULYSSES_CUDA_CHECK(cudaEventDestroy(done));
    }
    ULYSSES_CUDA_CHECK(cudaEventDestroy(ready));
}

}  // namespace ulysses
