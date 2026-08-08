"""The group object: window allocation, the handshake state, and the async path.

Windows are torch symmetric-memory tensors. They come from a MemPool this group owns, so the
caching allocator hands the memory back when a window is dropped, and ``rendezvous`` gives the peer
addresses the C++ side writes into. Nothing about the transport lives here, and nothing about the
memory lives in C++.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._functional_collectives import AsyncCollectiveTensor

from . import _C

# Each window's flag region is uint64 flags[ws] followed by uint64 epoch; see fast_barrier in
# include/fast_ulysses/transfer.hpp. Signal pads are far larger than this (9 KiB by default), but
# check rather than assume.
_FLAG_SLOTS = 9  # max world_size (8) + 1 epoch


class CompletedHandle:
    """What the async calls return on a build whose libtorch has no ``c10d::register_work``. With no
    registry to bind the completion event to, the C++ side has already made the caller's stream wait
    on it, so ``wait()`` only unwraps -- a distinct type so that is visible."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def wait(self) -> torch.Tensor:
        return self._tensor

    def __repr__(self) -> str:
        return f"CompletedHandle({tuple(self._tensor.shape)}, {self._tensor.dtype})"


class _Window:
    """One symmetric-memory allocation plus the addresses the transport needs.

    ``peer_ptrs[p]`` is peer p's copy of this window as addressed from here, and ``flag_ptrs[p]`` is
    peer p's signal pad. The tensor is held so the allocation outlives them.
    """

    __slots__ = ("tensor", "peer_ptrs", "flag_ptrs", "numel")

    def __init__(self, tensor: torch.Tensor, handle, pg) -> None:
        self.tensor = tensor
        self.numel = tensor.numel()
        self.peer_ptrs = list(handle.buffer_ptrs)
        self.flag_ptrs = list(handle.signal_pad_ptrs)
        if handle.signal_pad_size < _FLAG_SLOTS * 8:
            raise RuntimeError(
                f"the symmetric-memory signal pad is {handle.signal_pad_size} B, but the handshake "
                f"needs {_FLAG_SLOTS * 8} B. Raise it with "
                "torch.distributed._symmetric_memory.set_signal_pad_size() before any allocation."
            )
        # A window taken from the pool may be reusing memory a previous one freed, and the flags
        # carry an epoch that must start at zero. Clear, then hold every rank until all have
        # cleared -- otherwise one rank's first publish lands in a peer's not-yet-cleared pad and is
        # erased, and that peer waits forever for a write that already happened.
        handle.get_signal_pad(handle.rank, (_FLAG_SLOTS,), dtype=torch.int64).zero_()
        dist.barrier(group=pg)


class UlyssesGroup:
    """Ulysses all-to-all over torch symmetric memory (single-node GPU-to-GPU P2P).

    Construction is collective over ``process_group`` only. Each window is a separate collective
    allocation, made on the first call that needs it and cached afterwards, so every rank must issue
    the same ``(shape, mode, tag)`` sequence. Full contract: docs/API.md.
    """

    # False here and True on TorchUlyssesGroup, so a caller holding either can tell them apart.
    fallback = False

    def __init__(
        self,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
        initial_pool_bytes: int = 0,  # noqa: ARG002 -- accepted for source compatibility; unused
    ) -> None:
        """
        Args:
            process_group: the group this collective runs over; ``None`` uses ``dist.group.WORLD``.
            device: this rank's CUDA device; ``None`` uses the current device.
            initial_pool_bytes: ignored. Windows are allocated on demand from a MemPool and freed
                back to it, so there is no pool to size up front.
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

        # OUR OWN pool, not torch's implicit one. rendezvous is collective, so every rank has to
        # reach it for the same allocation at the same point in its own program; a pool only this
        # group allocates from is what keeps the pointer sequence identical across ranks. The
        # options match torch's own symmetric pool: no_split so a window never shares a segment
        # (and therefore never shares a signal pad) with another, use_on_oom off so an unrelated
        # allocation can never be served from here and desync that sequence.
        self._pool = torch.cuda.MemPool(
            symm_mem.get_mempool_allocator(device), use_on_oom=False, no_split=True
        )
        self._group = torch.classes.fast_ulysses.UlyssesGroup(int(self.rank), int(self.world_size))
        self._windows: dict[tuple, _Window] = {}
        self._numel_cache: dict[tuple, int] = {}
        self._destroyed = False

        # High-priority stream for the ASYNC collectives only; sync calls stay on the caller's
        # stream, since routing them through here costs two event hops. The two streams are NOT
        # ordered against each other, which is safe because the handshake state is per WINDOW.
        _, greatest = torch.cuda.Stream.priority_range()
        self._comm_stream = torch.cuda.Stream(device=device, priority=greatest)
        # (tag, shape, dtype) -> (staging buffer, release_event). See _stage_input.
        self._staging: dict[tuple, tuple[torch.Tensor, torch.cuda.Event]] = {}

    # ---- windows -------------------------------------------------------------------------

    def _window_numel(self, shape, mode, dtype, seq_splits, head_splits) -> int:
        """Elements the window for this call must hold, from the plan. Cached: it is pure host
        arithmetic, but it runs on every call and the answer only depends on the key."""
        key = (
            tuple(shape),
            mode,
            dtype,
            tuple(seq_splits) if seq_splits else None,
            tuple(head_splits) if head_splits else None,
        )
        n = self._numel_cache.get(key)
        if n is None:
            n = int(
                torch.ops.fast_ulysses.window_numel(
                    list(shape), mode, dtype, self.world_size, self.rank, seq_splits, head_splits
                )
            )
            self._numel_cache[key] = n
        return n

    def _window(self, tag: str, dtype: torch.dtype, numel: int) -> _Window:
        """The window for ``tag``, grown if this call needs more than the last one did.

        Keyed by (tag, dtype) and matched by capacity, so a tag costs one window at its high-water
        mark. Growing drops the old one, which returns its memory to the pool. COLLECTIVE on a miss.
        """
        key = (tag, dtype)
        win = self._windows.get(key)
        if win is not None and win.numel >= numel:
            return win
        self._windows.pop(key, None)  # free before allocating, so the pool can reuse the segment
        with torch.cuda.use_mem_pool(self._pool):
            t = torch.empty(numel, dtype=dtype, device=self.device)
        win = _Window(t, symm_mem.rendezvous(t, self.pg), self.pg)
        self._windows[key] = win
        return win

    # ---- async plumbing ------------------------------------------------------------------

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

    # ---- entry points --------------------------------------------------------------------

    def reserve(
        self,
        calls: Sequence[Mapping[str, object]],
        *,
        allow_growth: bool = True,  # noqa: ARG002 -- accepted for source compatibility; unused
    ) -> None:
        """Allocate the windows these calls will use, so the collective allocation happens here
        rather than inside the first call that needs it.

        Each entry describes one intended call: ``tag``, ``shape`` (the 4D input shape), and
        optionally ``mode`` (default 0), ``dtype`` (default bfloat16), ``seq_splits`` and
        ``head_splits``. Give each tag the LARGEST shape it will ever see. COLLECTIVE, same entries
        in the same order on every rank. Optional -- a later undeclared call simply allocates."""
        for call in calls:
            dtype = call.get("dtype", torch.bfloat16)
            mode = int(call.get("mode", 0))  # type: ignore[arg-type]
            seq_splits = call.get("seq_splits")
            head_splits = call.get("head_splits")
            numel = self._window_numel(call["shape"], mode, dtype, seq_splits, head_splits)
            self._window(str(call["tag"]), dtype, numel)  # type: ignore[arg-type]

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
        tensor the CALLER OWNS, with no lifetime rules attached, copied out of the window into
        ``out`` (CUDA, contiguous, matching dtype and shape) or into a fresh tensor.
        ``seq_splits[p]`` / ``head_splits[p]`` are rank p's sequence and head shard: pass BOTH or
        NEITHER, identical on every rank, matching the shape handed in, else it raises; neither
        means even shards. Collective -- every rank MUST issue the SAME (shape, mode, tag) call
        sequence, sync, async, copying and borrowed alike. Contract: docs/API.md."""
        x = x.contiguous()
        win = self._acquire(x, mode, tag, seq_splits, head_splits)
        return torch.ops.fast_ulysses.all_to_all_single_4d(
            self._group,
            x,
            mode,
            win.peer_ptrs,
            win.flag_ptrs,
            win.numel,
            seq_splits,
            head_splits,
            out,
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
        """The same collective, except the result IS the tag's window: no copy-out.

        Shapes, splits and the collective call sequence are ``all_to_all_single_4d``'s. What is extra
        is a lifetime contract NOTHING IN THIS LIBRARY ENFORCES: the result is valid only until the
        next call carrying this tag, whose transfer writes the same bytes; consume it on the stream
        that produced it before then, synchronising yourself to read it on another; do not read it
        after ``destroy()``; ``.clone()`` is how you keep it. Cross-rank safety IS handled -- a peer
        cannot overwrite this window until every rank has reached the next call's opening barrier,
        and your reads are ordered ahead of that barrier on your stream. Contract: docs/API.md."""
        x = x.contiguous()
        win = self._acquire(x, mode, tag, seq_splits, head_splits)
        return torch.ops.fast_ulysses.all_to_all_single_4d_borrowed(
            self._group,
            x,
            mode,
            win.peer_ptrs,
            win.flag_ptrs,
            win.numel,
            True,
            seq_splits,
            head_splits,
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
        x = x.contiguous()
        win = self._acquire(x, mode, tag, seq_splits, head_splits)
        x, ev_free = self._stage_input(x, tag)
        y, registered = self._launch_on_comm_stream(
            [ev_free],
            lambda: torch.ops.fast_ulysses.all_to_all_single_4d(
                self._group,
                x,
                mode,
                win.peer_ptrs,
                win.flag_ptrs,
                win.numel,
                seq_splits,
                head_splits,
                out,
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
        x = x.contiguous()
        win = self._acquire(x, mode, tag, seq_splits, head_splits)
        x, ev_free = self._stage_input(x, tag)
        out, registered = self._launch_on_comm_stream(
            [ev_free],
            lambda: torch.ops.fast_ulysses.all_to_all_single_4d_borrowed(
                self._group,
                x,
                mode,
                win.peer_ptrs,
                win.flag_ptrs,
                win.numel,
                barrier,
                seq_splits,
                head_splits,
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
        x = x.contiguous()
        win = self._acquire(x, mode, tag, seq_splits, head_splits)
        out, stages = torch.ops.fast_ulysses.all_to_all_single_4d_timed(
            self._group, x, mode, win.peer_ptrs, win.flag_ptrs, win.numel, seq_splits, head_splits
        )
        return out, {
            "barrier_in": float(stages[0]),
            "transfer": float(stages[1]),
            "barrier_out": float(stages[2]),
            "copy_out": float(stages[3]),
        }

    def _acquire(self, x, mode, tag, seq_splits, head_splits) -> _Window:
        return self._window(
            tag, x.dtype, self._window_numel(x.shape, mode, x.dtype, seq_splits, head_splits)
        )

    def destroy(self) -> None:
        """Release the windows and the transfer stream. Collective, because dropping a window is."""
        if self._destroyed:
            return
        # Drain the comm stream first: dist.barrier only syncs the caller's current stream, so an
        # unwaited async a2a could still be writing a window that is about to be freed.
        self._comm_stream.synchronize()
        self._staging.clear()
        dist.barrier(group=self.pg)
        self._windows.clear()
        self._group.destroy()
        self._destroyed = True
