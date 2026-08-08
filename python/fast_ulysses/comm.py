"""Python wrapper: build a C++ UlyssesGroup from a torch ProcessGroup (bootstrap is pure C++)."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import AsyncCollectiveTensor

from . import _C

# NVSHMEM reads NVSHMEM_SYMMETRIC_SIZE only when it (re)initializes, i.e. when the first LIVE group
# is constructed, and keeps that size while any group is alive. Track it to warn, not fail later.
_live_groups = 0
_heap_bytes = 0


class CompletedHandle:
    """What the async calls return on a build whose libtorch has no ``c10d::register_work``. With no
    registry to bind the completion event to, the C++ side has already made the caller's stream wait
    on it, so ``wait()`` only unwraps -- a distinct type so that is visible. See csrc/work.h."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def wait(self) -> torch.Tensor:
        return self._tensor

    def __repr__(self) -> str:
        return f"CompletedHandle({tuple(self._tensor.shape)}, {self._tensor.dtype})"


class UlyssesGroup:
    """Ulysses all-to-all group over the NVSHMEM symmetric heap (single-node GPU-to-GPU P2P).

    Construction broadcasts the NVSHMEM unique id, initializes NVSHMEM, and reserves the pool every
    collective takes its output window from (reused per tag+capacity+dtype). ``process_group`` may
    be an evenly strided subgroup, but construction is COLLECTIVE OVER THE WHOLE JOB, because the
    NVSHMEM bootstrap and ``nvshmem_team_split_strided`` are. Every pair of ranks must be
    P2P-mappable, else it raises. Full contract: docs/API.md.
    """

    # False here and True on TorchUlyssesGroup, so a caller holding either can tell them apart.
    fallback = False

    def __init__(
        self,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
        initial_pool_bytes: int = 2 << 30,
    ) -> None:
        """
        Args:
            process_group: bootstrap process group; ``None`` uses ``dist.group.WORLD``. Its ranks
                must be evenly strided (any stride) -- the ulysses subgroup under 2-D parallelism.
            device: this rank's CUDA device; ``None`` uses the current device.
            initial_pool_bytes: the pool, taken in full by one symmetric allocation here (default
                2 GiB); every collective's window is an offset into it. COMMITTED, not a cap.
        """
        pg = process_group if process_group is not None else dist.group.WORLD
        self.pg = pg
        self.rank = dist.get_rank(pg)
        self.world_size = dist.get_world_size(pg)
        self.peer_global_ranks = list(dist.get_process_group_ranks(pg))
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        torch.cuda.set_device(device)

        # Set via env before NVSHMEM init. The heap is sized ABOVE the pool because the pool is one
        # nvshmem_align of all of `initial_pool_bytes` and NVSHMEM's bookkeeping shares the heap.
        global _live_groups, _heap_bytes
        if _live_groups == 0:
            os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(int(initial_pool_bytes) + (64 << 20))
            _heap_bytes = int(initial_pool_bytes)
        elif int(initial_pool_bytes) > _heap_bytes:
            warnings.warn(
                f"initial_pool_bytes={int(initial_pool_bytes)} exceeds the NVSHMEM heap sized by "
                f"the first live UlyssesGroup ({_heap_bytes} B); the extra bytes may not be backed "
                "(size the first group's pool for all concurrently-live groups)",
                stacklevel=2,
            )
        # P2P direct writes do not need the multicast path, whose heap mapping fails and segfaults
        # on some nodes, so disable it by default (overridable via env).
        os.environ.setdefault("NVSHMEM_DISABLE_NVLS", "1")
        # Single-node P2P only: with an IB NIC present NVSHMEM would bring up a remote transport
        # during init that this operator never uses and does not survive.
        os.environ.setdefault("NVSHMEM_REMOTE_TRANSPORT", "none")

        cls = torch.classes.fast_ulysses.UlyssesGroup
        if dist.get_rank() == 0:
            uid = cls.get_uniqueid()
        else:
            uid = [0] * cls.uniqueid_nints()
        uid_t = torch.tensor(uid, dtype=torch.int64, device=device)
        # Generated on GLOBAL rank 0 and broadcast on WORLD, NOT on ``pg``: init_world bootstraps
        # ONE NVSHMEM job of dist.get_world_size() PEs and every PE must join with the SAME id, so
        # narrowing this to ``pg`` gives each subgroup its own id and never completes.
        dist.broadcast(uid_t, src=0, group=dist.group.WORLD)
        cls.init_world(uid_t.tolist(), dist.get_rank(), dist.get_world_size())

        dist.barrier(group=pg)
        self._group = cls(
            [int(r) for r in self.peer_global_ranks],
            int(self.rank),
            int(device.index),
            int(initial_pool_bytes),
        )
        dist.barrier(group=pg)
        _live_groups += 1
        self._destroyed = False

        # High-priority stream for the ASYNC collectives only; sync calls stay on the caller's
        # stream, since routing them through here costs two event hops. The two streams are NOT
        # ordered against each other, which is safe because the fast_barrier state is per TAG.
        _, greatest = torch.cuda.Stream.priority_range()
        self._comm_stream = torch.cuda.Stream(device=device, priority=greatest)
        # (tag, shape, dtype) -> (staging buffer, release_event). See _stage_input.
        self._staging: dict[tuple, tuple[torch.Tensor, torch.cuda.Event]] = {}

    def _stage_input(self, x: torch.Tensor, tag: str) -> tuple[torch.Tensor, torch.cuda.Event]:
        """Copy x into the persistent per-key staging buffer on the CALLER's stream and return
        (staging, release_event). The comm stream reads only the staging copy, so the caller's tensor
        is never retained cross-stream -- record_stream would instead pin every freed input until the
        comm stream caught up. Reuse waits GPU-side for the previous read."""
        key = (tag, tuple(x.shape), x.dtype)
        entry = self._staging.get(key)
        if entry is None:
            entry = (torch.empty_like(x), torch.cuda.Event())
            self._staging[key] = entry
        else:
            torch.cuda.current_stream().wait_event(entry[1])
        entry[0].copy_(x)
        return entry

    def _launch_on_comm_stream(self, releases: list[torch.cuda.Event], fn: Callable):
        """Run a collective on the group's comm stream: wait for the caller's current stream (staged
        inputs ready, and -- since the ready-event trails everything already submitted -- any earlier
        consumer of the same tag's buffer), run fn, record the staging releases, then bind a
        completion event to the result. The binding sits OUTSIDE the stream context deliberately:
        its no-registry fallback wait has to land on the caller's stream."""
        cur = torch.cuda.current_stream()
        ev_ready = torch.cuda.Event()
        ev_ready.record(cur)
        self._comm_stream.wait_event(ev_ready)
        with torch.cuda.stream(self._comm_stream):
            out = fn()
        for ev in releases:
            ev.record(self._comm_stream)
        registered = _C.register_stream_completion(out, self._comm_stream.cuda_stream)
        return out, registered

    def reserve(
        self,
        calls: Sequence[Mapping[str, object]],
        *,
        allow_growth: bool = False,
    ) -> None:
        """Pre-size the symmetric windows for the calls this group will make, then seal the pool.

        Each entry describes one intended call: ``tag``, ``shape`` (the 4D input shape), and
        optionally ``mode`` (default 0), ``dtype`` (default bfloat16), ``seq_splits`` and
        ``head_splits``. Give each tag the LARGEST shape it will ever see; windows are matched by
        capacity. ``allow_growth=True`` leaves a later undeclared call allocating, not raising.
        COLLECTIVE OVER THE WHOLE JOB, same entries in the same order. Contract: docs/API.md."""
        for call in calls:
            torch.ops.fast_ulysses.reserve(
                self._group,
                str(call["tag"]),
                list(call["shape"]),  # type: ignore[arg-type]
                int(call.get("mode", 0)),  # type: ignore[arg-type]
                call.get("dtype", torch.bfloat16),
                call.get("seq_splits"),
                call.get("head_splits"),
            )
        if not allow_growth:
            self._group.seal_pool()

    def barrier_epoch(self, tag: str) -> int:
        """TESTS: the tag's device-side handshake counter, 0 before the tag's first call. Reading it
        synchronises the device, so it is not for a hot path."""
        return self._group.barrier_epoch(tag)

    def all_to_all_single_4d(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
        out: torch.Tensor | None = None,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> torch.Tensor:
        """4D all-to-all: mode 0 scatters heads / gathers sequence, mode 1 inverts it. Returns a
        tensor the CALLER OWNS, with no lifetime rules attached, copied out of the tag's symmetric
        window into ``out`` (CUDA, contiguous, matching dtype and shape) or into a fresh tensor.
        ``seq_splits[p]`` / ``head_splits[p]`` are rank p's sequence and head shard: pass BOTH or
        NEITHER, identical on every rank, matching the shape handed in, else it raises; neither
        means even shards. Collective -- every rank MUST issue the SAME (shape, mode) call sequence,
        sync, async, copying and borrowed alike -- and a tag's calls must stay ORDERED, since a tag
        names one window and one barrier state. Contract: docs/API.md."""
        return torch.ops.fast_ulysses.all_to_all_single_4d(
            self._group, x.contiguous(), mode, tag, seq_splits, head_splits, out
        )

    def all_to_all_single_4d_borrowed(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> torch.Tensor:
        """The same collective, except the result IS the tag's symmetric window: no copy-out.

        Shapes, splits and the collective call sequence are ``all_to_all_single_4d``'s. What is extra
        is a lifetime contract NOTHING IN THIS LIBRARY ENFORCES: the result is valid only until the
        next call carrying this tag, whose transfer writes the same bytes; consume it on the stream
        that produced it before then, synchronising yourself to read it on another; do not read it
        after ``destroy()``; ``.clone()`` is how you keep it. Cross-rank safety IS handled -- a peer
        cannot overwrite this window until every rank has reached the next call's opening barrier,
        and your reads are ordered ahead of that barrier on your stream. Contract: docs/API.md."""
        return torch.ops.fast_ulysses.all_to_all_single_4d_borrowed(
            self._group, x.contiguous(), mode, tag, True, seq_splits, head_splits
        )

    def all_to_all_single_4d_async(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
        out: torch.Tensor | None = None,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> torch.Tensor | CompletedHandle:
        """Async form of ``all_to_all_single_4d``, on the group's comm stream. Returns an
        ``AsyncCollectiveTensor`` wrapping the tensor the caller owns; same collective contract and
        arguments as the sync call. ``result.wait()`` returns the plain tensor, and so does the first
        use of the result by any aten op -- either way the caller's current stream waits on the comm
        stream's completion event, GPU-side; a view op re-wraps without waiting. Wait on, or use,
        every result, or its registry entry outlives it. Ordering is per TAG, so an outstanding
        result must be waited before the next call with THAT tag. Contract: docs/API.md."""
        x, ev_free = self._stage_input(x.contiguous(), tag)
        y, registered = self._launch_on_comm_stream(
            [ev_free],
            lambda: torch.ops.fast_ulysses.all_to_all_single_4d(
                self._group, x, mode, tag, seq_splits, head_splits, out
            ),
        )
        # Two streams touch the output: one allocated it, the other wrote it, and the caching
        # allocator must know about the cross-stream use before it may recycle the block.
        # Registering both covers either origin, since it ignores a block's own stream.
        y.record_stream(self._comm_stream)
        y.record_stream(torch.cuda.current_stream())
        return AsyncCollectiveTensor(y) if registered else CompletedHandle(y)

    def all_to_all_single_4d_borrowed_async(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
        barrier: bool = True,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> torch.Tensor | CompletedHandle:
        """Async form of ``all_to_all_single_4d_borrowed``: an ``AsyncCollectiveTensor`` over the
        WINDOW VIEW, under the same unenforced rules, with "the stream that produced it" being the
        caller's stream from the wait onwards. The wait binds to that CALL, not to the window --
        each borrowed result is a fresh view with its own storage, which is what torch's registry
        keys on. ``barrier=False`` defers the CLOSING handshake to a later ``barrier=True`` call on
        the same stream, so several share one; until then the deferred call's view is not safe to
        read, and all ranks must use the identical pattern. Contract: docs/API.md."""
        x, ev_free = self._stage_input(x.contiguous(), tag)
        out, registered = self._launch_on_comm_stream(
            [ev_free],
            lambda: torch.ops.fast_ulysses.all_to_all_single_4d_borrowed(
                self._group, x, mode, tag, barrier, seq_splits, head_splits
            ),
        )
        return AsyncCollectiveTensor(out) if registered else CompletedHandle(out)

    def all_to_all_single_4d_timed(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run the COPYING collective and return ``(output, {stage: ms})``. Stages are ``barrier_in``
        (writers wait for readers), ``transfer`` (the peer copies, with this rank's own share running
        underneath them on the caller's stream), ``barrier_out`` and ``copy_out``, strictly ordered
        on one stream so they sum to the whole call. BENCHMARK ONLY: reading the events syncs."""
        out, stages = torch.ops.fast_ulysses.all_to_all_single_4d_timed(
            self._group, x.contiguous(), mode, tag, seq_splits, head_splits
        )
        return out, {
            "barrier_in": float(stages[0]),
            "transfer": float(stages[1]),
            "barrier_out": float(stages[2]),
            "copy_out": float(stages[3]),
        }

    def destroy(self) -> None:
        """Release the symmetric-heap resources (collective: ALL ranks must call together)."""
        if self._destroyed:
            return
        # Drain the comm stream first: dist.barrier only syncs the caller's current stream, so an
        # unwaited async a2a could still be writing the buffers nvshmem_free is about to release.
        self._comm_stream.synchronize()
        self._staging.clear()
        dist.barrier(group=self.pg)
        self._group.destroy()
        self._destroyed = True
        global _live_groups
        _live_groups -= 1
