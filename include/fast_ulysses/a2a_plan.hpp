#pragma once
// Addressing for the 4D all-to-all, expressed as pitched copies. Pure host arithmetic with no CUDA
// and no NVSHMEM in it, so the layout contract can be tested on its own (test/test_plan.py replays
// a plan over numpy buffers). UNEVEN splits are the general case -- even splits are just
// seq_splits = [s/P]*P and head_splits = [n/P]*P -- so there is a single code path to get right.
#include <cstdint>
#include <vector>

namespace ulysses {

enum A2AMode : int {
    // [b, seq_splits[me], head_total, d] -> [b, seq_total, head_splits[me], d]
    kScatterHead = 0,
    // the inverse: [b, seq_total, head_splits[me], d] -> [b, seq_splits[me], head_total, d]
    kGatherHead = 1,
};

struct A2ADims {
    int64_t              b          = 0;
    int64_t              d          = 0;
    int                  rank       = 0;
    int                  world_size = 0;
    std::vector<int64_t> seq_splits;   // per-rank local sequence length
    std::vector<int64_t> head_splits;  // per-rank head count

    int64_t              seq_total() const;
    int64_t              head_total() const;
    std::vector<int64_t> seq_offsets() const;   // exclusive prefix sum of seq_splits
    std::vector<int64_t> head_offsets() const;  // exclusive prefix sum of head_splits

    // Throws std::invalid_argument naming the offending field. Call before anything allocates.
    void validate() const;
};

// One pitched copy, offsets in bytes from the base of their buffer (this rank's input tensor, the
// destination peer's symmetric window), so the same struct describes a cudaMemcpy2D/3DAsync and a
// host-side replay in a test. `depth` folds the batch dimension in: b repetitions of the same
// rows x width copy at a fixed stride go as ONE cudaMemcpy3DAsync, when expressible (push_batched).
struct CopyOp {
    int     peer       = 0;
    int64_t src_offset = 0;
    int64_t dst_offset = 0;
    int64_t src_pitch  = 0;
    int64_t dst_pitch  = 0;
    int64_t width      = 0;  // bytes per row
    int64_t rows       = 0;
    int64_t depth      = 1;  // batch elements folded into this op
    int64_t src_slice  = 0;  // bytes between batch elements, source
    int64_t dst_slice  = 0;  // bytes between batch elements, destination
};

struct A2APlan {
    std::vector<int64_t> output_shape;  // shape THIS rank receives, dense from the window base
    // Elements the symmetric window must hold: the LARGEST rank's output, not this rank's. The peer
    // offsets only line up while every rank allocates the same size, and each rank can compute the
    // max without communicating, because the splits describe the whole group.
    int64_t             window_numel = 0;
    std::vector<CopyOp> ops;  // what THIS rank sends; ops[i].peer says where
};

// One plan serves both entry points: `ops` covers all world_size destinations, so this rank's own
// share travels through the window like every peer's -- which is what the borrowed form needs, its
// result BEING the window, and what makes the copying form's copy-out one flat copy.
A2APlan build_plan(const A2ADims& dims, int mode, int64_t elem_size);

}  // namespace ulysses
