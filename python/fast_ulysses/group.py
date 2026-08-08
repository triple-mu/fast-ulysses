"""The group object: window allocation, the handshake state, and the async path.

Windows are torch symmetric-memory tensors from a MemPool this group owns, so the caching
allocator takes the memory back when a window is dropped, and ``rendezvous`` supplies the peer
addresses the C++ side writes into. Nothing about the transport lives here, and nothing about the
memory lives in C++.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._functional_collectives import AsyncCollectiveTensor

from . import _C
from .nvlink import check_nvlink

# Each window's flag region is uint64 flags[ws] followed by uint64 epoch; see fast_barrier in
# include/fast_ulysses/transfer.hpp. Signal pads are far larger (9 KiB by default), but check.
_FLAG_SLOTS = 9  # max world_size (8) + 1 epoch


class CompletedHandle:
    """What the async call returns on a build whose libtorch has no ``c10d::register_work``. With
    no registry to bind the completion event to, the C++ side has already made the caller's stream
    wait on it, so ``wait()`` only unwraps -- a distinct type so that is visible."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def wait(self) -> torch.Tensor:
        return self._tensor

    def __repr__(self) -> str:
        return f"CompletedHandle({tuple(self._tensor.shape)}, {self._tensor.dtype})"


class _Window:
    """One symmetric-memory allocation plus the addresses the transport needs.

    ``peer_ptrs[p]`` is peer p's copy of this allocation as addressed from here, and
    ``flag_ptrs[p]`` is peer p's signal pad. The tensor is held so the allocation outlives them.
    """

    __slots__ = ("tensor", "peer_ptrs", "flag_ptrs", "numel", "pool")

    def __init__(self, tensor: torch.Tensor, handle, pg, pool) -> None:
        # The allocation is only valid while the pool it came from is alive, and a buffer handed to
        # a caller can outlive the group. Hold the pool here, and hang this object off that buffer,
        # so the pool dies after the last allocation from it rather than before.
        self.pool = pool
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
        # This allocation may be reusing memory an earlier window freed, and the flags carry an
        # epoch that has to start at zero. Clear, then hold every rank until all have cleared --
        # otherwise one rank's first publish lands in a pad another rank has yet to clear, is
        # erased, and that rank waits forever for a write that already happened.
        handle.get_signal_pad(handle.rank, (_FLAG_SLOTS,), dtype=torch.int64).zero_()
        dist.barrier(group=pg)


class UlyssesGroup:
    """Ulysses all-to-all over NVLink, moved by the copy engines.

    Construction is collective over ``process_group``. Windows are allocated on the first call that
    needs one and cached, so every rank must issue the same sequence of shapes. Contract:
    docs/api.md.
    """

    def __init__(
        self,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
        *,
        require_nvlink: bool = True,
    ) -> None:
        """
        Args:
            process_group: the group this collective runs over; ``None`` uses ``dist.group.WORLD``.
            device: this rank's CUDA device; ``None`` uses the current device.
            require_nvlink: refuse to build a group whose GPUs are not all NVLink-joined. Off is
                for measuring that case, not for running in it.
        """
        pg = process_group if process_group is not None else dist.group.WORLD
        self.pg = pg
        self.rank = dist.get_rank(pg)
        self.world_size = dist.get_world_size(pg)
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        torch.cuda.set_device(device)

        if require_nvlink:
            self._require_nvlink()

        # OUR OWN pool, not torch's implicit one. rendezvous is collective, so every rank has to
        # reach it for the same allocation at the same point in its own program; a pool only this
        # group allocates from is what keeps the allocation sequence identical across ranks. The
        # options match torch's own symmetric pool: no_split so a window never shares a segment
        # (and therefore never shares a signal pad) with another, use_on_oom off so an unrelated
        # allocation can never be served from here and desync that sequence.
        self._pool = torch.cuda.MemPool(
            symm_mem.get_mempool_allocator(device), use_on_oom=False, no_split=True
        )
        self._handle = torch.classes.fast_ulysses.UlyssesGroup(int(self.rank), int(self.world_size))
        # Two internal windows per dtype, one for each stream the collectives run on. A window is
        # single-buffered, so two calls may only share one when the stream orders them.
        self._windows: dict[tuple, _Window] = {}
        # Every symmetric buffer this group has made, by address: what tells all_to_all_4d that a
        # caller's `out` is a window it can write into directly.
        self._by_ptr: dict[int, _Window] = {}
        self._plan_cache: dict[tuple, tuple[int, list[int]]] = {}
        self._destroyed = False

        # High-priority stream for the async call only; the sync call stays on the caller's stream,
        # since routing it through here costs two event hops. The two streams are NOT ordered
        # against each other, which is why they do not share a window.
        _, greatest = torch.cuda.Stream.priority_range()
        self._comm_stream = torch.cuda.Stream(device=device, priority=greatest)
        self._staging: dict[tuple, tuple[torch.Tensor, torch.cuda.Event]] = {}

    def _require_nvlink(self) -> None:
        """Every rank checks its own device against every other one's, so the answer is the same
        on all of them and the refusal is collective in practice."""
        local = torch.tensor([self.device.index], dtype=torch.int64, device=self.device)
        gathered = [torch.zeros_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local, group=self.pg)
        devices = [int(t.item()) for t in gathered]
        if len(set(devices)) != len(devices):
            return  # more than one rank per GPU: not a topology this check is about
        problem = check_nvlink(devices)
        if problem is not None:
            raise RuntimeError(problem)

    # ---- windows -------------------------------------------------------------------------

    def _plan_shapes(self, shape, mode, dtype, seq_splits, head_splits) -> tuple[int, list[int]]:
        """(window capacity, this rank's output shape) from the plan. Cached: pure host arithmetic,
        but it runs on every call and the answer depends only on the key."""
        key = (
            tuple(shape),
            mode,
            dtype,
            tuple(seq_splits) if seq_splits else None,
            tuple(head_splits) if head_splits else None,
        )
        hit = self._plan_cache.get(key)
        if hit is None:
            numel, out_shape = torch.ops.fast_ulysses.plan_shapes(
                list(shape), mode, dtype, self.world_size, self.rank, seq_splits, head_splits
            )
            hit = (int(numel), [int(v) for v in out_shape])
            self._plan_cache[key] = hit
        return hit

    def _allocate(self, numel: int, dtype: torch.dtype) -> _Window:
        """A fresh symmetric window. COLLECTIVE."""
        with torch.cuda.use_mem_pool(self._pool):
            t = torch.empty(numel, dtype=dtype, device=self.device)
        win = _Window(t, symm_mem.rendezvous(t, self.pg), self.pg, self._pool)
        self._by_ptr[t.data_ptr()] = win
        return win

    def _internal_window(self, role: str, dtype: torch.dtype, numel: int) -> _Window:
        """This group's own window for ``role``, grown if this call needs more than the last did.

        Matched by capacity, so a role costs one window at its high-water mark. Growing drops the
        old one, which returns its memory to the pool. COLLECTIVE on a miss.
        """
        key = (role, dtype)
        win = self._windows.get(key)
        if win is not None and win.numel >= numel:
            return win
        if win is not None:
            self._by_ptr.pop(win.tensor.data_ptr(), None)
            del self._windows[key]  # free before allocating, so the pool can reuse the segment
        win = self._allocate(numel, dtype)
        self._windows[key] = win
        return win

    def empty_output(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> torch.Tensor:
        """A buffer shaped like the output of ``all_to_all_4d(x, mode=...)``, in symmetric memory.

        Passing it back as ``out=`` makes the peers write it directly, which removes the copy-out.
        You own it: keep it across steps, free it when you like. COLLECTIVE, so allocate outside
        the loop and reuse -- and use one buffer per concurrent call, since a call overwrites it.
        """
        numel, shape = self._plan_shapes(x.shape, mode, x.dtype, seq_splits, head_splits)
        win = self._allocate(numel, x.dtype)
        buf = win.tensor[: math.prod(shape)].view(shape)
        buf._fast_ulysses_window = win  # keeps the allocation and its pool alive; see _Window
        return buf

    def _window_for(self, x, mode, out, seq_splits, head_splits, role: str) -> _Window:
        numel, _ = self._plan_shapes(x.shape, mode, x.dtype, seq_splits, head_splits)
        if out is not None:
            win = self._by_ptr.get(out.data_ptr())
            if win is not None and win.numel >= numel:
                return win  # zero-copy: the peers write `out` itself
        return self._internal_window(role, x.dtype, numel)

    # ---- entry points --------------------------------------------------------------------

    def all_to_all_4d(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        out: torch.Tensor | None = None,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> torch.Tensor:
        """mode 0 scatters heads and gathers sequence; mode 1 inverts it.

        Returns a tensor the caller owns, with no lifetime rules. ``out`` from ``empty_output()``
        skips the copy-out; any other contiguous CUDA tensor of the output shape is copied into.

        ``seq_splits[p]`` / ``head_splits[p]`` are rank p's sequence and head shard: pass both or
        neither, identical on every rank and matching the shape handed in. Neither means even
        shards. Collective: every rank must issue the same sequence of shapes.
        """
        x = x.contiguous()
        win = self._window_for(x, mode, out, seq_splits, head_splits, "sync")
        return torch.ops.fast_ulysses.all_to_all_4d(
            self._handle,
            x,
            mode,
            win.peer_ptrs,
            win.flag_ptrs,
            win.numel,
            seq_splits,
            head_splits,
            out,
        )

    def all_to_all_4d_async(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        out: torch.Tensor | None = None,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> torch.Tensor | CompletedHandle:
        """``all_to_all_4d`` on this group's comm stream, returning immediately.

        The result is an ``AsyncCollectiveTensor``: ``.wait()`` returns the plain tensor, and so
        does the first use by any aten op -- either way the caller's stream waits on the comm
        stream's completion event GPU-side, and the host does not block. A view op re-wraps without
        waiting.

        Wait on, or use, every result: a dropped one leaves its entry in torch's work registry, and
        ``out=`` is the hole, since reading your own ``out`` never touches the registry. On a
        libtorch with no ``c10d::register_work`` this returns a ``CompletedHandle`` instead --
        correct, no overlap, and a distinct type so it is visible.
        """
        x = x.contiguous()
        win = self._window_for(x, mode, out, seq_splits, head_splits, "async")
        staged, ev_free = self._stage_input(x)
        y, registered = self._launch_on_comm_stream(
            ev_free,
            lambda: torch.ops.fast_ulysses.all_to_all_4d(
                self._handle,
                staged,
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
        y.record_stream(self._comm_stream)
        y.record_stream(torch.cuda.current_stream())
        return AsyncCollectiveTensor(y) if registered else CompletedHandle(y)

    def _timed(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """BENCHMARKS ONLY: the copying call with CUDA events between its stages, which sum to the
        whole call because they are strictly ordered on one stream. Reading the events syncs."""
        x = x.contiguous()
        win = self._window_for(x, mode, None, seq_splits, head_splits, "sync")
        out, stages = torch.ops.fast_ulysses.all_to_all_4d_timed(
            self._handle, x, mode, win.peer_ptrs, win.flag_ptrs, win.numel, seq_splits, head_splits
        )
        names = ("barrier_in", "transfer", "barrier_out", "copy_out")
        return out, dict(zip(names, (float(v) for v in stages), strict=True))

    # ---- async plumbing ------------------------------------------------------------------

    def _stage_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event]:
        """Copy x into a persistent per-(shape, dtype) staging buffer on the CALLER's stream.

        The comm stream reads only the staging copy, so the caller's tensor is never retained
        cross-stream -- record_stream would instead pin every freed input until the comm stream
        caught up. Reuse waits GPU-side for the previous read.
        """
        key = (tuple(x.shape), x.dtype)
        entry = self._staging.get(key)
        if entry is None:
            entry = (torch.empty_like(x), torch.cuda.Event())
            self._staging[key] = entry
        else:
            torch.cuda.current_stream().wait_event(entry[1])
        entry[0].copy_(x)
        return entry

    def _launch_on_comm_stream(self, release: torch.cuda.Event, fn):
        """Run a collective on the comm stream: wait for the caller's stream (the staged input is
        ready, and since the ready-event trails everything already submitted, so is any earlier
        consumer of the same buffer), run fn, release the staging buffer, bind a completion event
        to the result. The binding sits OUTSIDE the stream context deliberately: its no-registry
        fallback wait has to land on the caller's stream."""
        ev_ready = torch.cuda.Event()
        ev_ready.record(torch.cuda.current_stream())
        self._comm_stream.wait_event(ev_ready)
        with torch.cuda.stream(self._comm_stream):
            out = fn()
        release.record(self._comm_stream)
        return out, _C.register_stream_completion(out, self._comm_stream.cuda_stream)

    def destroy(self) -> None:
        """Release the windows and the transfer stream. Collective: dropping a window is."""
        if self._destroyed:
            return
        # Drain the comm stream first: dist.barrier only syncs the caller's current stream, so an
        # unwaited async call could still be writing a window that is about to be freed.
        self._comm_stream.synchronize()
        self._staging.clear()
        dist.barrier(group=self.pg)
        self._windows.clear()
        self._by_ptr.clear()
        self._handle.destroy()
        self._destroyed = True
