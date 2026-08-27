#pragma once
/// @file
/// The two device-side pieces of one collective: the copy-engine transfer and the flag barrier.
/// Neither knows where the peer addresses came from -- the caller passes them in, which is what
/// keeps every communication library out of this translation unit.
#include <cstdint>
#include <cuda_runtime.h>
#include <fast_ulysses/a2a_plan.hpp>
#include <vector>

namespace ulysses {

/// @brief Issue `plan.ops` as pitched cudaMemcpy2D/3DAsync copies into the peers' windows.
///
/// Remote peers are serialised on `xfer` because they all leave through the same egress and
/// separate streams would only make them contend; this rank's own share crosses no link and runs
/// on `stream` alongside them. The transfer is joined back onto `stream` before returning.
///
/// The zero-SM property is a property of the REMOTE copies: a peer copy overlaps concurrent compute,
/// while a same-device copy competes with it for the same memory system. This rank's own share is a
/// same-device copy, and so is the copy-out on the copying path.
///
/// @param src        base of this rank's input tensor
/// @param peer_ptrs  peer p's window base, as addressed from this rank; `plan.ops` offsets are
///                   relative to it. Its size is the world size.
/// @param plan       what this rank sends, from build_plan()
/// @param xfer       the group's dedicated transfer stream
/// @param rank       this rank's index within `peer_ptrs`
/// @param stream     the caller's stream
void launch_a2a_ce(const void*                  src,
                   const std::vector<uint64_t>& peer_ptrs,
                   const A2APlan&               plan,
                   cudaStream_t                 xfer,
                   int                          rank,
                   cudaStream_t                 stream);

/// @brief Issue one flat copy per peer from a destination-packed source.
///
/// Source chunk p is sent to peer p. Every source rank writes its chunk at offset
/// `rank * chunk_bytes` in the peer window, so the receiver sees sender-major staging. As in the
/// pitched path, remote copies use `xfer`, this rank's own chunk uses `stream`, and both are joined
/// before returning.
void launch_a2a_flat(const void*                  src,
                     const std::vector<uint64_t>& peer_ptrs,
                     int64_t                      chunk_bytes,
                     cudaStream_t                 xfer,
                     int                          rank,
                     cudaStream_t                 stream);

/// @brief Barrier across the group, over P2P-accessible flags.
///
/// A one-block spin kernel: rank r publishes its epoch into every peer's `flags[r]` with a release
/// store, then waits until its own `flags[0..ws-1]` have all reached that epoch. The epoch lives on
/// the device and is advanced by the kernel, so a captured graph advances it on replay too -- a
/// host-computed epoch would bake a constant into the graph and every replay would be satisfied by
/// stale state.
///
/// The layout inside each rank's flag region is `uint64 flags[ws]` followed by `uint64 epoch`, so
/// the caller must provide at least `(ws + 1) * 8` bytes and zero them before the first call.
///
/// @param stream     the stream to order the handshake on
/// @param flag_ptrs  peer p's flag region, as addressed from this rank. Its size is the world size.
/// @param rank       this rank's index within `flag_ptrs`
void fast_barrier(cudaStream_t stream, const std::vector<uint64_t>& flag_ptrs, int rank);

/// @brief TESTS ONLY: publish the closing flag before the payload has landed, on purpose.
///
/// The operator assumes a completed copy-engine write is visible at the destination by the time a
/// later kernel's release store announcing it arrives, which no vendor document guarantees. Arming
/// this (`delay_us > 0`) makes launch_a2a_ce hold the payload back and skip the join onto the
/// caller's stream, so the test for that assumption has a negative control. 0 disarms.
void set_ce_fault(int64_t delay_us);

}  // namespace ulysses
