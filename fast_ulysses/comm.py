"""Python wrapper: build a C++ UlyssesGroup from a torch ProcessGroup (bootstrap is pure C++)."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


class UlyssesGroup:
    def __init__(
            self,
            process_group: Optional[dist.ProcessGroup] = None,
            device: Optional[torch.device] = None,
            initial_pool_bytes: int = 2 << 30,
    ) -> None:
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

    def all_to_all_single_4d(
            self,
            x: torch.Tensor,
            *,
            mode: int = 0,
            tag: str = "",
            use_tma: bool | None = None,
    ) -> torch.Tensor:
        # COLLECTIVE SEMANTICS: s/n must divide world_size (uniform). The first (shape, mode, use_tma)
        # seen runs a local micro-benchmark and caches the launch config; every rank MUST issue the SAME
        # (shape, mode, use_tma) call sequence (the nvshmem symmetric alloc + cross-rank barrier are
        # collective; all ranks miss the same entry on the first call together).
        #
        # use_tma (None=auto / True / False): None=auto -> sm<9 uses non-TMA; sm90+ micro-benchmarks BOTH
        # paths on the first call for this shape and caches the faster (runtime path selection, replacing the
        # old static table). True forces TMA (requires sm90+, else TORCH_CHECK fails); False forces non-TMA.
        # Every rank MUST pass the SAME use_tma (a mismatch diverges kernel/barrier + cache key -> hang).
        #
        # tag scopes the symmetric-heap output buffer (reused on same tag+shape+dtype). Results that
        # must stay live together (e.g. q/k/v) MUST use distinct tags, else they alias one buffer.
        return torch.ops.fast_ulysses.all_to_all_single_4d(
            self._group, x.contiguous(), mode, tag, use_tma
        )

    def destroy(self) -> None:
        dist.barrier(group=self.pg)
        self._group.destroy()
