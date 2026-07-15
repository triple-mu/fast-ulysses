"""Python wrapper: build a C++ UlyssesGroup from a torch ProcessGroup (bootstrap is pure C++)."""

from __future__ import annotations

import os
from typing import Callable

import torch
import torch.distributed as dist


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
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        torch.cuda.set_device(device)

        # Reservation must be set via env before NVSHMEM init.
        os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(int(initial_pool_bytes))
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

        COLLECTIVE SEMANTICS: s/n must divide world_size (uniform). The first (shape, mode, use_tma)
        seen runs a local micro-benchmark and caches the launch config; every rank MUST issue the
        SAME (shape, mode, use_tma) call sequence (the nvshmem symmetric alloc + cross-rank barrier
        are collective; all ranks miss the same entry on the first call together). Sync AND async
        calls count in that sequence (both advance the same per-group barrier epoch).

        use_tma (None=auto / True / False): None=auto -> sm<9 uses non-TMA; sm90+ micro-benchmarks
        BOTH paths on the first call for this shape and caches the faster (runtime path selection,
        replacing the old static table). True forces TMA (requires sm90+, else TORCH_CHECK fails);
        False forces non-TMA. Every rank MUST pass the SAME use_tma (a mismatch diverges
        kernel/barrier + cache key -> hang).

        tag scopes the symmetric-heap output buffer (reused on same tag+shape+dtype). Results that
        must stay live together (e.g. q/k/v) MUST use distinct tags, else they alias one buffer.
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
    ) -> AsyncA2AHandle:
        """Async variant: launches on the group's comm stream and returns immediately; kernels
        submitted to the caller's stream afterwards overlap with the a2a until handle.wait().
        Collective constraints are identical to the sync call (same rank-uniform call sequence,
        sync and async counted together).

        ORDERING CONSTRAINT when mixing with sync calls: the fast_barrier epoch is one per-group
        monotonic counter, so barrier kernels must EXECUTE in submission order. wait() every async
        handle of this group before issuing the next sync collective on the main stream -- that
        data dependency forces the comm-stream barriers to complete first. (Sync calls run directly
        on the caller's stream: routing them through the comm stream costs two event hops per call,
        measured ~0.27 ms/call on H200 -- comparable to the a2a itself.)
        """
        x = x.contiguous()
        out, ev_done = self._launch_on_comm_stream(
            [x],
            lambda: torch.ops.fast_ulysses.all_to_all_single_4d(self._group, x, mode, tag, use_tma),
        )
        return AsyncA2AHandle(out, ev_done)

    def all_to_all_single_4d_qk(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        mode: str = "cross_head",
        interleaved: bool = True,
        eps: float = 1e-6,
        tag: str = "",
        use_tma: bool | None = None,
    ) -> torch.Tensor:
        """Mode0 Ulysses input a2a with fused source-side QK RMSNorm + RoPE (for q/k; v uses the
        plain op).

        x: [b, s_local, n_global, d]; weight fp32 ([d] per_head / [n_global*d] cross_head);
        cos/sin fp32 [s_local, d/2] at this rank's GLOBAL positions; interleaved=GPT-J vs NeoX.
        Out [b, s_global, n_local, d].
        """
        m = {"per_head": 0, "cross_head": 1}[mode]
        return torch.ops.fast_ulysses.all_to_all_single_4d_qk(
            self._group, x.contiguous(), weight, cos, sin, m, interleaved, eps, tag, use_tma
        )

    def all_to_all_single_4d_qk2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weight_q: torch.Tensor,
        weight_k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        mode: str = "cross_head",
        interleaved: bool = True,
        eps: float = 1e-6,
        tag: str = "",
        use_tma: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q + k in ONE collective call: two fused scatters back-to-back, then a SINGLE shared
        quiet+fast_barrier (half the sync latency of two all_to_all_single_4d_qk calls). q/k share
        shape/dtype/cos/sin; weight_q/weight_k are their respective norm weights. Outputs land in
        distinct buffers (tag::q / tag::k). Collective constraints are the same as the single op.
        """
        m = {"per_head": 0, "cross_head": 1}[mode]
        oq, ok = torch.ops.fast_ulysses.all_to_all_single_4d_qk2(
            self._group,
            q.contiguous(),
            k.contiguous(),
            weight_q,
            weight_k,
            cos,
            sin,
            m,
            interleaved,
            eps,
            tag,
            use_tma,
        )
        return oq, ok

    def destroy(self) -> None:
        """Release the symmetric-heap resources (collective: ALL ranks must call together)."""
        dist.barrier(group=self.pg)
        self._group.destroy()
