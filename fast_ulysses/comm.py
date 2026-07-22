"""Python wrapper: build a C++ UlyssesGroup from a torch ProcessGroup (bootstrap is pure C++)."""

from __future__ import annotations

import os
import warnings
from typing import Callable

import torch
import torch.distributed as dist

# NVSHMEM reads NVSHMEM_SYMMETRIC_SIZE when it (re)initializes -- i.e. when the first LIVE group is
# constructed (destroying the last group finalizes NVSHMEM, so the next group re-initializes with a
# fresh size). While any group is alive the heap keeps its size; track that to warn instead of
# failing at some later nvshmem_align. destroy() keeps the count in step with the C++ group count.
_live_groups = 0
_heap_bytes = 0


class AsyncA2AHandle:
    """Result of an async a2a: the collective runs on the group's comm stream; wait() makes the
    CALLER's current stream wait for it (GPU-side event wait, host does not block) and returns the
    output view. The output lives in the tag-scoped symmetric buffer -- do not issue another call
    with the same tag until this result has been consumed."""

    def __init__(self, out, ev_done: torch.cuda.Event):
        self._out = out
        self._ev_done = ev_done

    def wait(self):
        torch.cuda.current_stream().wait_event(self._ev_done)
        return self._out


class UlyssesGroup:
    """Ulysses all-to-all group over the NVSHMEM symmetric heap (single-node NVLink P2P).

    Wraps the C++ ``UlyssesGroup`` custom class: construction broadcasts the NVSHMEM unique id
    via ``torch.distributed``, initializes NVSHMEM, and reserves a symmetric-heap pool that all
    collectives allocate their outputs from (buffers reused per tag+shape+dtype). Construction
    is itself collective: ALL ranks must build the group together.
    """

    def __init__(
        self,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
        initial_pool_bytes: int = 2 << 30,
    ) -> None:
        """
        Args:
            process_group: bootstrap process group; ``None`` uses ``dist.group.WORLD``.
            device: this rank's CUDA device; ``None`` uses the current device.
            initial_pool_bytes: NVSHMEM symmetric-heap reservation (default 2 GiB); every
                collective's output buffer is carved from this pool.
        """
        pg = process_group if process_group is not None else dist.group.WORLD
        self.pg = pg
        self.rank = dist.get_rank(pg)
        self.world_size = dist.get_world_size(pg)
        self.peer_global_ranks = list(dist.get_process_group_ranks(pg))
        if self.world_size != dist.get_world_size():
            # The uid broadcast below runs on WORLD and nvshmem_team_split_strided is a
            # world-collective -- a subgroup-only construction would hang, so reject it up front.
            raise NotImplementedError(
                "process_group must span all ranks (NVSHMEM bootstrap is world-collective); "
                f"got a subgroup of {self.world_size}/{dist.get_world_size()} ranks"
            )
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        torch.cuda.set_device(device)

        # Reservation must be set via env before NVSHMEM init; it takes effect only when NVSHMEM
        # (re)initializes, which happens while no other group is alive.
        global _live_groups, _heap_bytes
        if _live_groups == 0:
            os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(int(initial_pool_bytes))
            _heap_bytes = int(initial_pool_bytes)
        elif int(initial_pool_bytes) > _heap_bytes:
            warnings.warn(
                f"initial_pool_bytes={int(initial_pool_bytes)} exceeds the NVSHMEM heap sized by "
                f"the first live UlyssesGroup ({_heap_bytes} B); the extra bytes may not be backed "
                "(size the first group's pool for all concurrently-live groups)",
                stacklevel=2,
            )
        # P2P direct writes do not need NVLS (NVLink SHARP multicast); on some nodes its
        # multicast heap mapping fails and segfaults, so disable by default for cross-node
        # robustness (overridable via env).
        os.environ.setdefault("NVSHMEM_DISABLE_NVLS", "1")
        # This op is single-node NVLink P2P only; on nodes with IB NICs, NVSHMEM tries to init
        # the IB remote transport and segfaults, so disable remote transport by default
        # (verified on H200+IB nodes: init SIGSEGVs otherwise).
        os.environ.setdefault("NVSHMEM_REMOTE_TRANSPORT", "none")

        cls = torch.classes.fast_ulysses.UlyssesGroup
        if dist.get_rank() == 0:
            uid = cls.get_uniqueid()
        else:
            uid = [0] * cls.uniqueid_nints()
        uid_t = torch.tensor(uid, dtype=torch.int64, device=device)
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

        # Dedicated high-priority stream for the ASYNC collectives (sync calls run directly on the
        # caller's stream -- routing them through here costs two event hops per call, ~0.27 ms
        # measured, comparable to the a2a itself). The fast_barrier epoch is one per-group monotonic
        # counter, so barrier kernels must execute in submission order across streams: wait() every
        # async handle before issuing the next sync collective (see all_to_all_single_4d_async).
        # High priority lets the comm kernels get SM slots under concurrent compute.
        _, greatest = torch.cuda.Stream.priority_range()
        self._comm_stream = torch.cuda.Stream(device=device, priority=greatest)

    def _launch_on_comm_stream(self, inputs: list[torch.Tensor], fn: Callable):
        """Run a collective on the group's comm stream: comm stream waits for the caller's current
        stream (inputs ready -- and, since the ready-event trails everything already submitted, any
        earlier consumer of the same tag's buffer), runs fn, and returns (result, done_event)."""
        cur = torch.cuda.current_stream()
        ev_ready = torch.cuda.Event()
        ev_ready.record(cur)
        self._comm_stream.wait_event(ev_ready)
        with torch.cuda.stream(self._comm_stream):
            out = fn()
        for t in inputs:
            t.record_stream(self._comm_stream)  # keep the allocator from reusing x too early
        ev_done = torch.cuda.Event()
        ev_done.record(self._comm_stream)
        return out, ev_done

    def all_to_all_single_4d(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
        use_tma: bool | None = None,
    ) -> torch.Tensor:
        """Uniform 4D all-to-all: mode0 scatters heads / gathers sequence; mode1 is its inverse.

        Collective -- every rank MUST issue the SAME (shape, mode, use_tma) call sequence
        (sync and async count together), or the whole group hangs. First call per shape
        micro-benchmarks and caches the launch config; use_tma None/True/False picks
        auto/TMA/non-TMA. Concurrently-live results (e.g. q/k/v) MUST use distinct tags,
        else they alias one symmetric-heap buffer. Full contract: docs/API.md.
        """
        return torch.ops.fast_ulysses.all_to_all_single_4d(
            self._group, x.contiguous(), mode, tag, use_tma
        )

    def all_to_all_single_4d_async(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
        use_tma: bool | None = None,
        barrier: bool = True,
    ) -> AsyncA2AHandle:
        """Async variant on the group's comm stream; handle.wait() makes the caller's stream
        wait (GPU-side) and returns the output view. Same collective contract as the sync call.

        Barrier kernels must execute in submission order: wait() every outstanding handle
        BEFORE the next sync collective. barrier=False defers the handshake so several calls
        share one -- only the barrier-carrying handle's wait() implies peers' writes arrived,
        and all ranks must use the identical barrier pattern. Full contract: docs/API.md.
        """
        x = x.contiguous()
        out, ev_done = self._launch_on_comm_stream(
            [x],
            lambda: torch.ops.fast_ulysses.all_to_all_single_4d(
                self._group, x, mode, tag, use_tma, barrier
            ),
        )
        return AsyncA2AHandle(out, ev_done)

    def all_to_all_single_4d_ce(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
    ) -> torch.Tensor:
        """CE (copy-engine) variant: same collective contract as all_to_all_single_4d, but
        the transfer rides the DMA engines (zero SM) and so overlaps compute that starves
        the kernel paths. Explicit choice -- the use_tma auto-tune never picks CE; prefer
        the kernel paths for tiny shapes (~world_size memcpy launches per call).
        Deliberately no ``barrier`` parameter: a deferred sync result would be an
        unreadable view with nothing left to publish it. Full contract: docs/API.md.
        """
        return torch.ops.fast_ulysses.all_to_all_single_4d_ce(
            self._group, x.contiguous(), mode, tag
        )

    def all_to_all_single_4d_ce_async(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
        barrier: bool = True,
    ) -> AsyncA2AHandle:
        """Async CE variant; the in-flight window genuinely overlaps concurrent
        GEMMs/attention. Ordering and barrier=False grouping exactly as in
        all_to_all_single_4d_async. Full contract: docs/API.md.
        """
        x = x.contiguous()
        out, ev_done = self._launch_on_comm_stream(
            [x],
            lambda: torch.ops.fast_ulysses.all_to_all_single_4d_ce(
                self._group, x, mode, tag, barrier
            ),
        )
        return AsyncA2AHandle(out, ev_done)

    def destroy(self) -> None:
        """Release the symmetric-heap resources (collective: ALL ranks must call together)."""
        if self._destroyed:
            return
        # Drain the comm stream first: dist.barrier only syncs the caller's current stream, so an
        # unwaited async a2a could still be writing the buffers nvshmem_free is about to release.
        self._comm_stream.synchronize()
        dist.barrier(group=self.pg)
        self._group.destroy()
        self._destroyed = True
        global _live_groups
        _live_groups -= 1
