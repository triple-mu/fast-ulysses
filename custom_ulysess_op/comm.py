"""Python 封装：从 torch ProcessGroup 构造 C++ UlyssesGroup（bootstrap 纯 C++）。"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


class UlyssesGroup:
    def __init__(
            self, process_group=None, device=None, initial_pool_bytes: int = 2 << 30
    ):
        pg = process_group if process_group is not None else dist.group.WORLD
        self.pg = pg
        self.rank = dist.get_rank(pg)
        self.world_size = dist.get_world_size(pg)
        self.peer_global_ranks = list(dist.get_process_group_ranks(pg))
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        torch.cuda.set_device(device)

        # 预留必须在 NVSHMEM init 前经 env 生效。
        os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(int(initial_pool_bytes))

        cls = torch.classes.ulysess.UlyssesGroup
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
            x: "torch.Tensor",
            *,
            mode: int = 0,
            tag: str = "",
            seq_lens: "list[int] | None" = None,
            head_splits: "list[int] | None" = None,
    ):
        # seq_lens/head_splits 为各 rank 的序列长/头数（变长，调用方提供，无运行时 gather）；
        # 均为 None 时走均匀路径（s/n 须被 world_size 整除）。
        return torch.ops.ulysess.all_to_all_single_4d(
            self._group, x.contiguous(), mode, tag, seq_lens, head_splits
        )

    def destroy(self) -> None:
        dist.barrier(group=self.pg)
        self._group.destroy()
