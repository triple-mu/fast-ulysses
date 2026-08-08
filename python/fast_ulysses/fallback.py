"""The same collective, on ``torch.distributed``, for the topologies where that is faster.

A group whose GPUs span more than one CPU socket is the one case where this operator loses:
``all_to_all_single`` does not use direct GPU P2P across that boundary and this transport always
does. Nothing here is a workaround for that -- it is the baseline, offered under the same four
entry points so a caller can keep one code path. See docs/BENCHMARK.md.

``make_group`` picks between this and ``UlyssesGroup``; the object a caller ends up holding is
what says which one it got.
"""

from __future__ import annotations

import pathlib
from itertools import accumulate

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import AsyncCollectiveTensor


def numa_nodes(devices: list[int]) -> dict[int, int]:
    """{device index: NUMA node}, omitting devices whose node the kernel does not report.

    torch gives the PCI address as three integers, so the sysfs directory name has to be rebuilt
    from them (function is always 0 for a GPU). sysfs writes -1 both for "unknown" and for
    "attached to no node", and a single-socket machine reports -1 for everything -- either way
    there is no partition to report, so those devices are left out.
    """
    out = {}
    for i in devices:
        p = torch.cuda.get_device_properties(i)
        bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        try:
            node = int((pathlib.Path("/sys/bus/pci/devices") / bdf / "numa_node").read_text())
        except (OSError, ValueError):
            continue
        if node >= 0:
            out[i] = node
    return out


def spans_sockets(process_group=None) -> bool | None:
    """Does this group's set of devices cross a NUMA boundary? None when it cannot be determined.

    Every rank contributes its own device index, since a rank can only read its own properties.
    Collective on ``process_group``.
    """
    pg = process_group if process_group is not None else dist.group.WORLD
    ws = dist.get_world_size(pg)
    local = torch.cuda.current_device()
    node = numa_nodes([local]).get(local, -1)
    gathered = torch.full((ws,), -1, dtype=torch.int64, device=f"cuda:{local}")
    gathered[dist.get_rank(pg)] = node
    dist.all_reduce(gathered, op=dist.ReduceOp.MAX, group=pg)
    nodes = gathered.tolist()
    if any(n < 0 for n in nodes):
        return None
    return len(set(nodes)) > 1


def _offsets(splits: list[int]) -> list[int]:
    return [0] + list(accumulate(splits))


def _even(x: torch.Tensor, mode: int, ws: int, pg) -> torch.Tensor:
    """permute -> all_to_all_single -> permute, the path sglang's usp.py takes."""
    b, d = x.shape[0], x.shape[-1]
    if mode == 0:
        s_local, n_global = x.shape[1], x.shape[2]
        n_local = n_global // ws
        xt = x.view(b, s_local, ws, n_local, d).permute(2, 0, 1, 3, 4).contiguous()
        out = torch.empty_like(xt)
        dist.all_to_all_single(out, xt, group=pg)
        return out.permute(1, 0, 2, 3, 4).contiguous().view(b, ws * s_local, n_local, d)
    s_global, n_local = x.shape[1], x.shape[2]
    s_local = s_global // ws
    xt = x.view(b, ws, s_local, n_local, d).permute(1, 0, 2, 3, 4).contiguous()
    out = torch.empty_like(xt)
    dist.all_to_all_single(out, xt, group=pg)
    return out.permute(1, 2, 0, 3, 4).contiguous().view(b, s_local, ws * n_local, d)


def _uneven(x, mode: int, seq_splits: list[int], head_splits: list[int], rank: int, pg):
    """dist.all_to_all over a list, the only shape torch takes when the shards differ."""
    b, d = x.shape[0], x.shape[-1]
    ws = len(seq_splits)
    if mode == 0:
        off = _offsets(head_splits)
        send = [x[:, :, off[p] : off[p + 1], :].contiguous() for p in range(ws)]
        recv = [
            torch.empty(b, seq_splits[r], head_splits[rank], d, dtype=x.dtype, device=x.device)
            for r in range(ws)
        ]
        dist.all_to_all(recv, send, group=pg)
        return torch.cat(recv, dim=1)
    off = _offsets(seq_splits)
    send = [x[:, off[p] : off[p + 1], :, :].contiguous() for p in range(ws)]
    recv = [
        torch.empty(b, seq_splits[rank], head_splits[r], d, dtype=x.dtype, device=x.device)
        for r in range(ws)
    ]
    dist.all_to_all(recv, send, group=pg)
    return torch.cat(recv, dim=2)


class TorchUlyssesGroup:
    """``UlyssesGroup``'s four entry points, backed by ``torch.distributed``.

    Differences a caller can observe, all of them the baseline being less restrictive:
    the result is always owned (so the borrowed forms have no lifetime rule), ``tag`` is ignored
    because nothing is reused between calls, and the async forms complete before they return, so
    there is no overlap to gain. ``reserve`` and ``destroy`` are no-ops.
    """

    fallback = True

    def __init__(
        self,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
        initial_pool_bytes: int = 2 << 30,  # noqa: ARG002 -- accepted so the two are swappable
    ) -> None:
        pg = process_group if process_group is not None else dist.group.WORLD
        self.pg = pg
        self.rank = dist.get_rank(pg)
        self.world_size = dist.get_world_size(pg)
        self.peer_global_ranks = list(dist.get_process_group_ranks(pg))
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device

    def _run(self, x, mode, seq_splits, head_splits, out):
        x = x.contiguous()
        if seq_splits is None and head_splits is None:
            y = _even(x, mode, self.world_size, self.pg)
        elif seq_splits is None or head_splits is None:
            raise ValueError("pass both seq_splits and head_splits, or neither")
        else:
            y = _uneven(x, mode, list(seq_splits), list(head_splits), self.rank, self.pg)
        if out is None:
            return y
        out.copy_(y)
        return out

    def reserve(self, calls, *, allow_growth: bool = False) -> None:
        """No-op: there is no symmetric window to size."""

    def all_to_all_single_4d(
        self, x, *, mode=0, tag="", out=None, seq_splits=None, head_splits=None
    ):
        return self._run(x, mode, seq_splits, head_splits, out)

    def all_to_all_single_4d_borrowed(
        self, x, *, mode=0, tag="", seq_splits=None, head_splits=None
    ):
        return self._run(x, mode, seq_splits, head_splits, None)

    def all_to_all_single_4d_async(
        self, x, *, mode=0, tag="", out=None, seq_splits=None, head_splits=None
    ):
        return AsyncCollectiveTensor(self._run(x, mode, seq_splits, head_splits, out))

    def all_to_all_single_4d_borrowed_async(
        self, x, *, mode=0, tag="", barrier=True, seq_splits=None, head_splits=None
    ):
        return AsyncCollectiveTensor(self._run(x, mode, seq_splits, head_splits, None))

    def destroy(self) -> None:
        """No-op: nothing was allocated."""


def make_group(
    process_group: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
    initial_pool_bytes: int = 2 << 30,
    prefer: str = "auto",
):
    """Return whichever of the two is faster on this machine.

    ``prefer``:
      ``"auto"``    ``TorchUlyssesGroup`` when the group's GPUs span more than one CPU socket,
                    ``UlyssesGroup`` otherwise -- including when the socket layout cannot be
                    determined, since that is the common single-socket case.
      ``"fast"``    always ``UlyssesGroup``.
      ``"torch"``   always ``TorchUlyssesGroup``.

    Collective, like either constructor. ``result.fallback`` says which one came back.
    """
    if prefer not in ("auto", "fast", "torch"):
        raise ValueError(f"prefer must be auto, fast or torch, got {prefer!r}")
    from .comm import UlyssesGroup

    if prefer == "auto":
        prefer = "torch" if spans_sockets(process_group) else "fast"
    cls = TorchUlyssesGroup if prefer == "torch" else UlyssesGroup
    return cls(process_group=process_group, device=device, initial_pool_bytes=initial_pool_bytes)
