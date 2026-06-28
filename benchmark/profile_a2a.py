"""最小 profiling 驱动（配合 nsys/ncu）。单一配置、少量迭代，便于抓 timeline。

    PROF_MODE=0 nsys profile --trace=cuda -o /tmp/a2a \
        torchrun --nproc_per_node=8 custom_ulysess_op/benchmark/profile_a2a.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from custom_ulysess_op import UlyssesGroup


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)

    g = UlyssesGroup(process_group=dist.group.WORLD, initial_pool_bytes=4 << 30)
    mode = int(os.environ.get("PROF_MODE", "0"))
    iters = int(os.environ.get("PROF_ITERS", "30"))
    b, d = 2, 128
    shape = (b, 2048, 8 * ws, d) if mode == 0 else (b, 2048 * ws, 8, d)
    x = torch.randn(shape, dtype=torch.bfloat16, device=dev)

    for _ in range(5):
        g.all_to_all_single_4d(x, mode=mode, tag="prof")
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("a2a_timed")
    for _ in range(iters):
        g.all_to_all_single_4d(x, mode=mode, tag="prof")
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    g.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
