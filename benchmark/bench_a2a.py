"""Ulysses all-to-all 延迟/带宽 benchmark。

运行（目标多卡机，ws ∈ {2,4,8}）：
    torchrun --nproc_per_node=8 custom_ulysess_op/benchmark/bench_a2a.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from custom_ulysess_op import UlyssesGroup


def bench_case(group, shape, dtype, mode, dev, warmup=20, iters=100):
    inp = torch.randn(shape, dtype=dtype, device=dev)
    tag = f"bench_m{mode}_d{shape[-1]}"
    for _ in range(warmup):
        group.all_to_all_single_4d(inp, mode=mode, tag=tag)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        group.all_to_all_single_4d(inp, mode=mode, tag=tag)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    nbytes = inp.numel() * inp.element_size()
    # 算子读一次本地输入、写一次 peer 输出：等效访存 2*nbytes。
    gbps = (nbytes * 2) / (ms * 1e-3) / 1e9
    return ms * 1e3, gbps, nbytes


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    group = UlyssesGroup(process_group=dist.group.WORLD, initial_pool_bytes=4 << 30)

    b, d = 2, 128
    # mode0 输入 [b, s_local, n_global, d]；mode1 输入 [b, s_global, n_local, d]
    cfgs = [
        (0, (b, 2048, 8 * ws, d), torch.bfloat16),
        (1, (b, 2048 * ws, 8, d), torch.bfloat16),
    ]
    if rank == 0:
        print(
            f"=== Ulysses A2A bench: ws={ws} GPU={torch.cuda.get_device_name(0)} ===",
            flush=True,
        )
    for mode, shape, dtype in cfgs:
        us, gbps, nbytes = bench_case(group, shape, dtype, mode, dev)
        if rank == 0:
            print(
                f"mode={mode} shape={tuple(shape)} {str(dtype).split('.')[-1]} | "
                f"{us:8.1f} us/iter | {nbytes / 1e6:7.1f} MB in/rank | {gbps:7.1f} GB/s (2x)",
                flush=True,
            )

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
