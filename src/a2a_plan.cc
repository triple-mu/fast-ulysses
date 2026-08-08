#include <fast_ulysses/a2a_plan.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ulysses {

namespace {

[[noreturn]] void fail(const std::string& message)
{
    throw std::invalid_argument("fast_ulysses: " + message);
}

// A peer's entry in seq_splits/head_splits is bounded by nothing, and a product that wraps yields a
// negative window_numel, which passes acquire's capacity check and is then carved out of the pool
// as an enormous unsigned size.
int64_t checked_mul(int64_t a, int64_t b, const std::string& what)
{
    if (a != 0 && b > std::numeric_limits<int64_t>::max() / a) {
        fail(what + " overflows int64 (" + std::to_string(a) + " * " + std::to_string(b) + ")");
    }
    return a * b;
}

int64_t checked_sum(const std::vector<int64_t>& values, const std::string& what)
{
    int64_t total = 0;
    for (int64_t v : values) {
        if (v > std::numeric_limits<int64_t>::max() - total) {
            fail(what + " overflows int64");
        }
        total += v;
    }
    return total;
}

std::vector<int64_t> exclusive_prefix_sum(const std::vector<int64_t>& values)
{
    std::vector<int64_t> offsets(values.size(), 0);
    int64_t              running = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        offsets[i] = running;
        running += values[i];
    }
    return offsets;
}

// Emit `depth` batch repetitions of `op`, `src_stride` / `dst_stride` bytes apart, as one 3D copy
// when that is both expressible and worth it, and as `depth` 2D copies otherwise.
//
// Only MULTI-ROW copies are fused: a single-row copy is a flat memcpy, and folding b of them into
// one 3D copy would move them onto the strided path, while a pitched copy is already strided and
// fusing removes (b-1) launches. cudaMemcpy3DParms also takes no slice stride -- it derives one as
// pitch * ysize -- so a stride is expressible only when it divides the pitch and fits the rows.
void push_batched(std::vector<CopyOp>& out, CopyOp op, int64_t depth, int64_t src_stride, int64_t dst_stride)
{
    const bool worth_fusing = depth > 1 && op.rows > 1;
    const bool expressible  = worth_fusing && op.src_pitch > 0 && op.dst_pitch > 0 && src_stride % op.src_pitch == 0
                             && dst_stride % op.dst_pitch == 0 && src_stride / op.src_pitch >= op.rows
                             && dst_stride / op.dst_pitch >= op.rows;
    if (expressible) {
        op.depth     = depth;
        op.src_slice = src_stride;
        op.dst_slice = dst_stride;
        out.push_back(op);
        return;
    }
    for (int64_t i = 0; i < depth; ++i) {
        CopyOp one = op;
        one.src_offset += i * src_stride;
        one.dst_offset += i * dst_stride;
        out.push_back(one);
    }
}

}  // namespace

int64_t A2ADims::seq_total() const
{
    return checked_sum(seq_splits, "seq_splits total");
}

int64_t A2ADims::head_total() const
{
    return checked_sum(head_splits, "head_splits total");
}

std::vector<int64_t> A2ADims::seq_offsets() const
{
    return exclusive_prefix_sum(seq_splits);
}

std::vector<int64_t> A2ADims::head_offsets() const
{
    return exclusive_prefix_sum(head_splits);
}

void A2ADims::validate() const
{
    if (world_size <= 0) {
        fail("world_size must be positive, got " + std::to_string(world_size));
    }
    if (rank < 0 || rank >= world_size) {
        fail("rank " + std::to_string(rank) + " out of range for world_size " + std::to_string(world_size));
    }
    if (b <= 0 || d <= 0) {
        fail("b and d must be positive, got b=" + std::to_string(b) + " d=" + std::to_string(d));
    }
    if (static_cast<int>(seq_splits.size()) != world_size) {
        fail("seq_splits has " + std::to_string(seq_splits.size()) + " entries, expected "
             + std::to_string(world_size));
    }
    if (static_cast<int>(head_splits.size()) != world_size) {
        fail("head_splits has " + std::to_string(head_splits.size()) + " entries, expected "
             + std::to_string(world_size));
    }
    // Zero-length shards are allowed (a rank can end up with no sequence), negatives are not.
    for (int p = 0; p < world_size; ++p) {
        if (seq_splits[p] < 0) {
            fail("seq_splits[" + std::to_string(p) + "] is negative");
        }
        if (head_splits[p] < 0) {
            fail("head_splits[" + std::to_string(p) + "] is negative");
        }
    }
}

A2APlan build_plan(const A2ADims& dims, int mode, int64_t elem_size)
{
    dims.validate();
    if (elem_size <= 0) {
        fail("elem_size must be positive, got " + std::to_string(elem_size));
    }
    if (mode != kScatterHead && mode != kGatherHead) {
        fail("mode must be 0 (scatter head) or 1 (gather head), got " + std::to_string(mode));
    }

    const int64_t              seq_total   = dims.seq_total();
    const int64_t              head_total  = dims.head_total();
    const std::vector<int64_t> seq_offset  = dims.seq_offsets();
    const std::vector<int64_t> head_offset = dims.head_offsets();
    const int64_t              s_me        = dims.seq_splits[dims.rank];
    const int64_t              n_me        = dims.head_splits[dims.rank];
    const int64_t              d_bytes     = dims.d * elem_size;

    A2APlan plan;
    plan.output_shape = (mode == kScatterHead) ? std::vector<int64_t>{dims.b, seq_total, n_me, dims.d} :
                                                 std::vector<int64_t>{dims.b, s_me, head_total, dims.d};

    // Window size: the largest receive over all ranks (see A2APlan::window_numel).
    for (int p = 0; p < dims.world_size; ++p) {
        const int64_t rows       = (mode == kScatterHead) ? seq_total : dims.seq_splits[p];
        const int64_t heads      = (mode == kScatterHead) ? dims.head_splits[p] : head_total;
        const int64_t peer_numel = checked_mul(
            checked_mul(checked_mul(dims.b, rows, "window size"), heads, "window size"), dims.d, "window size");
        plan.window_numel = std::max(plan.window_numel, peer_numel);
    }

    for (int p = 0; p < dims.world_size; ++p) {
        if (mode == kScatterHead) {
            // Send peer p the head columns it owns, for every local sequence position; they land in
            // the slice of p's window that this rank's sequence shard occupies. The window rows are
            // contiguous, so this reads strided and writes contiguous.
            const int64_t row = dims.head_splits[p] * d_bytes;
            if (row == 0 || s_me == 0) {
                continue;
            }
            CopyOp op;
            op.peer       = p;
            op.src_offset = head_offset[p] * d_bytes;
            op.src_pitch  = head_total * d_bytes;
            op.dst_offset = seq_offset[dims.rank] * row;
            op.dst_pitch  = row;
            op.width      = row;
            op.rows       = s_me;
            push_batched(plan.ops, op, dims.b, s_me * head_total * d_bytes, seq_total * row);
        }
        else {
            // The inverse: send peer p the sequence rows it owns, carrying only this rank's head
            // columns, into p's window at this rank's head offset -- a strided write. Landing them
            // contiguously in a per-sender segment is not available here: the window IS the
            // returned tensor, so there is no local pass to interleave them afterwards.
            const int64_t row = n_me * d_bytes;
            const int64_t s_p = dims.seq_splits[p];
            if (row == 0 || s_p == 0) {
                continue;
            }
            CopyOp op;
            op.peer       = p;
            op.src_offset = seq_offset[p] * row;
            op.src_pitch  = row;
            op.dst_offset = head_offset[dims.rank] * d_bytes;
            op.dst_pitch  = head_total * d_bytes;
            op.width      = row;
            op.rows       = s_p;
            push_batched(plan.ops, op, dims.b, seq_total * row, s_p * head_total * d_bytes);
        }
    }
    return plan;
}

}  // namespace ulysses
