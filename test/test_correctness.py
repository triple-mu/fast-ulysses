"""torchrun 正确性对拍：custom_ulysess_op vs torch permute + all_to_all_single + permute。

纯数据搬运 → 逐位相等。运行（目标多卡机，ws ∈ {2,4,8}）：
    torchrun --nproc_per_node=8 custom_ulysess_op/test/test_correctness.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from custom_ulysess_op import UlyssesGroup


def torch_a2a(x: torch.Tensor, mode: int, ws: int, group) -> torch.Tensor:
    """标准 Ulysses all-to-all 参考实现（permute → all_to_all_single → permute）。"""
    b, d = x.shape[0], x.shape[-1]
    if mode == 0:
        s_local, n_global = x.shape[1], x.shape[2]
        n_local = n_global // ws
        xt = x.view(b, s_local, ws, n_local, d).permute(2, 0, 1, 3, 4).contiguous()
        out = torch.empty_like(xt)
        dist.all_to_all_single(out, xt, group=group)
        return out.permute(1, 0, 2, 3, 4).contiguous().view(b, ws * s_local, n_local, d)
    else:
        s_global, n_local = x.shape[1], x.shape[2]
        s_local = s_global // ws
        xt = x.view(b, ws, s_local, n_local, d).permute(1, 0, 2, 3, 4).contiguous()
        out = torch.empty_like(xt)
        dist.all_to_all_single(out, xt, group=group)
        return out.permute(1, 2, 0, 3, 4).contiguous().view(b, s_local, ws * n_local, d)


def torch_a2a_varlen(x, mode, me, ws, s_off, n_off, group):
    """变长参考：用 torch.distributed.all_to_all（list 形式）做 uneven A2A。"""
    b, d = x.shape[0], x.shape[-1]
    if mode == 0:
        # 发给 peer r：me 的序列 × r 的头块；从 r 收：r 的序列 × me 的头块 → 沿序列拼
        send = [x[:, :, n_off[r] : n_off[r + 1], :].contiguous() for r in range(ws)]
        recv = [
            torch.empty(
                b,
                s_off[r + 1] - s_off[r],
                n_off[me + 1] - n_off[me],
                d,
                dtype=x.dtype,
                device=x.device,
            )
            for r in range(ws)
        ]
        dist.all_to_all(recv, send, group=group)
        return torch.cat(recv, dim=1)
    else:
        send = [x[:, s_off[r] : s_off[r + 1], :, :].contiguous() for r in range(ws)]
        recv = [
            torch.empty(
                b,
                s_off[me + 1] - s_off[me],
                n_off[r + 1] - n_off[r],
                d,
                dtype=x.dtype,
                device=x.device,
            )
            for r in range(ws)
        ]
        dist.all_to_all(recv, send, group=group)
        return torch.cat(recv, dim=2)


def _prefix(xs):
    off = [0]
    for v in xs:
        off.append(off[-1] + v)
    return off


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    pg = dist.group.WORLD

    group = UlyssesGroup(process_group=pg, initial_pool_bytes=1 << 30)

    b = 2
    for dtype in (torch.float16, torch.bfloat16):
        for d in (64, 128, 256):
            for mode in (0, 1):
                shape = (b, 16, 4 * ws, d) if mode == 0 else (b, 16 * ws, 4, d)
                x = torch.randn(shape, dtype=dtype, device=dev)
                tag = f"t_{str(dtype).split('.')[-1]}_d{d}_m{mode}"

                ours = group.all_to_all_single_4d(x, mode=mode, tag=tag)
                ref = torch_a2a(x, mode, ws, pg)
                if not torch.equal(ours, ref):
                    raise AssertionError(
                        f"MISMATCH rank={rank} ws={ws} dtype={dtype} d={d} mode={mode}"
                    )
                if rank == 0:
                    print(
                        f"OK ws={ws} {str(dtype).split('.')[-1]} d={d} mode={mode} shape={tuple(ours.shape)}",
                        flush=True,
                    )
                dist.barrier()

    # 变长（uneven s/n，不能被 world_size 整除）：split 由调用方提供，无运行时 gather。
    seq_lens = [4, 6, 3, 7, 5, 2, 8, 1][:ws]
    head_splits = [2, 3, 1, 4, 2, 5, 1, 3][:ws]
    s_off = _prefix(seq_lens)
    n_off = _prefix(head_splits)
    S, N = s_off[-1], n_off[-1]
    for dtype in (torch.float16, torch.bfloat16):
        for d in (64, 128, 256):
            for mode in (0, 1):
                shape = (
                    (b, seq_lens[rank], N, d)
                    if mode == 0
                    else (b, S, head_splits[rank], d)
                )
                x = torch.randn(shape, dtype=dtype, device=dev)
                tag = f"v_{str(dtype).split('.')[-1]}_d{d}_m{mode}"

                ours = group.all_to_all_single_4d(
                    x, mode=mode, tag=tag, seq_lens=seq_lens, head_splits=head_splits
                )
                ref = torch_a2a_varlen(x, mode, rank, ws, s_off, n_off, pg)
                if not torch.equal(ours, ref):
                    raise AssertionError(
                        f"VARLEN MISMATCH rank={rank} ws={ws} dtype={dtype} d={d} mode={mode}"
                    )
                if rank == 0:
                    print(
                        f"OK[varlen] ws={ws} {str(dtype).split('.')[-1]} d={d} mode={mode} shape={tuple(ours.shape)}",
                        flush=True,
                    )
                dist.barrier()

    if rank == 0:
        print("ALL PASS", flush=True)

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
