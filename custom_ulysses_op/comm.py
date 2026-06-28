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

        cls = torch.classes.ulysses.UlyssesGroup
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
        seq_lens: list[int] | None = None,
        head_splits: list[int] | None = None,
        use_tma: bool | None = None,
    ) -> torch.Tensor:
        # seq_lens/head_splits are the per-rank sequence lengths / head counts (varlen, supplied by
        # the caller, no runtime gather); both None -> uniform path (s/n must divide world_size).
        # COLLECTIVE SEMANTICS: the uniform path runs a collective micro-benchmark on the first
        # (shape, mode) seen and caches it; every rank MUST issue the SAME (shape, mode) call
        # sequence, and it is mutually exclusive with tune() (either all tune or all lazy, else the
        # whole group hangs).
        #
        # use_tma (None=auto / True / False): selects the kernel path. None auto-picks per arch
        # (uniform on sm90+ -> TMA; varlen always non-TMA). True forces TMA (uniform and varlen;
        # requires sm90+, else TORCH_CHECK fails); False forces non-TMA.
        # HARD COLLECTIVE CONSTRAINT: every rank MUST pass the SAME use_tma value (same severity as
        # shape/mode). A mismatch makes ranks take different kernels/barriers and diverge in the
        # resolve_config cache key (tma bit) -> the whole group hangs.
        return torch.ops.ulysses.all_to_all_single_4d(
            self._group, x.contiguous(), mode, tag, seq_lens, head_splits, use_tma
        )

    def tune(
        self,
        shape: tuple[int, int, int, int],
        *,
        mode: int = 0,
        use_tma: bool | None = None,
        verbose: bool = False,
    ) -> None:
        """Warm up the launch config for a shape (collective; warm-up only, result unchanged).

        COLLECTIVE SEMANTICS: every rank MUST call this with the SAME (shape, mode, use_tma)
        sequence; tune and the lazy fallback are mutually exclusive -- either all ranks pre-tune the
        same shape set, or all ranks rely on the lazy fallback in the first all_to_all. Mixing
        tune/lazy across ranks hangs the whole group.

        shape is the all_to_all input physical shape (b, x1, x2, d):
          mode0 -> (b, s_local, n_global, d); mode1 -> (b, s_global, n_local, d).
        The number of pre-tuned shapes is bounded by initial_pool_bytes (each shape accumulates a
        scratch output in the symmetric heap).

        use_tma (None=auto / True / False) is resolved to a single path (uniform: sm90+ -> TMA on
        auto) and ONLY that path is warmed (the tma bit is part of the cache key). If a later
        all_to_all uses the opposite use_tma for the same shape, it misses the cache and triggers
        one more lazy collective micro-benchmark (which must again be consistent across all ranks).
        Every rank MUST pass the SAME use_tma to tune (same hard constraint as all_to_all).
        """
        assert len(shape) == 4, "shape must be (b,x1,x2,d)"
        ut = -1 if use_tma is None else int(bool(use_tma))
        self._group.tune(
            int(shape[0]),
            int(shape[1]),
            int(shape[2]),
            int(shape[3]),
            int(mode),
            ut,
            bool(verbose),
        )

    def destroy(self) -> None:
        dist.barrier(group=self.pg)
        self._group.destroy()
