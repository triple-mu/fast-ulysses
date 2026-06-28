"""对比测试：custom_ulysess_op vs torch 的 permute + all_to_all_single + permute。

同时校验精度（纯搬运 → 逐位相等）与性能（us/iter + 加速比）。
运行（目标多卡机，ws ∈ {2,4,8}）：
    torchrun --nproc_per_node=8 custom_ulysess_op/benchmark/compare_torch.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from custom_ulysess_op import UlyssesGroup


def torch_a2a(x: torch.Tensor, mode: int, ws: int, group) -> torch.Tensor:
    """标准 Ulysses all-to-all：permute → all_to_all_single → permute。

    mode 0: [b, s_local, n_global, d] -> [b, s_global, n_local, d]
    mode 1: [b, s_global, n_local, d] -> [b, s_local, n_global, d]
    """
    b = x.shape[0]
    d = x.shape[-1]
    if mode == 0:
        s_local, n_global = x.shape[1], x.shape[2]
        n_local = n_global // ws
        # 把目标 rank(=head 块) 提到 dim0 供 all_to_all_single 切分
        xt = x.view(b, s_local, ws, n_local, d).permute(2, 0, 1, 3, 4).contiguous()
        out = torch.empty_like(xt)
        dist.all_to_all_single(out, xt, group=group)
        # out[r] = rank r 的 s_local 段（本 rank 的 head 块）→ 沿 s 拼成 s_global
        return out.permute(1, 0, 2, 3, 4).contiguous().view(b, ws * s_local, n_local, d)
    else:
        s_global, n_local = x.shape[1], x.shape[2]
        s_local = s_global // ws
        xt = x.view(b, ws, s_local, n_local, d).permute(1, 0, 2, 3, 4).contiguous()
        out = torch.empty_like(xt)
        dist.all_to_all_single(out, xt, group=group)
        # out[r] = rank r 为本 rank 序列块贡献的 head 块 → 拼成 n_global
        return out.permute(1, 2, 0, 3, 4).contiguous().view(b, s_local, ws * n_local, d)


def _time(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us/iter


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    pg = dist.group.WORLD

    group = UlyssesGroup(process_group=pg, initial_pool_bytes=4 << 30)

    b, d = 2, 128
    cfgs = [
        (0, (b, 2048, 8 * ws, d), torch.bfloat16),
        (1, (b, 2048 * ws, 8, d), torch.bfloat16),
    ]
    if rank == 0:
        print(
            f"=== ours vs torch(permute+a2a+permute): ws={ws} {torch.cuda.get_device_name(0)} ===",
            flush=True,
        )

    all_exact = True
    for mode, shape, dtype in cfgs:
        inp = torch.randn(shape, dtype=dtype, device=dev)
        tag = f"cmp_m{mode}_d{shape[-1]}"

        ours = group.all_to_all_single_4d(inp, mode=mode, tag=tag).clone()
        ref = torch_a2a(inp, mode, ws, pg)
        exact = torch.equal(ours, ref)
        all_exact = all_exact and exact

        t_ours = _time(lambda: group.all_to_all_single_4d(inp, mode=mode, tag=tag))
        t_ref = _time(lambda: torch_a2a(inp, mode, ws, pg))

        if rank == 0:
            speed = t_ref / t_ours if t_ours > 0 else 0.0
            print(
                f"mode={mode} shape={tuple(shape)} | exact={exact} | "
                f"ours={t_ours:8.1f}us  torch={t_ref:8.1f}us  speedup={speed:5.2f}x",
                flush=True,
            )

    if rank == 0:
        print(f"ACCURACY: {'ALL EXACT' if all_exact else 'MISMATCH!'}", flush=True)

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
