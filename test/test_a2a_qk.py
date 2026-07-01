"""torchrun correctness check for the fused mode0 a2a + QK RMSNorm/RoPE op.

    torchrun --nproc_per_node=8 fast-ulysses/test/test_a2a_qk.py

Compares group.all_to_all_single_4d_qk(x, w, cos, sin, ...) against a reference that applies the same
fp32 RMSNorm + RoPE to the source [b, s_local, n_global, d] and then the reference permute+all_to_all
(mode0). The a2a is an exact permutation, so the only diff is the fused norm+rope's fp32 result rounded to
bf16/fp16 (~1 ULP). Covers per-head/cross-head x interleaved/non x fp16/bf16.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


def rms_ref_f32(x, w, mode, eps):
    b, s, n, d = x.shape
    xf = x.float()
    if mode == "per_head":
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return xf * torch.rsqrt(var + eps) * w.view(1, 1, 1, d)
    xf2 = xf.reshape(b, s, n * d)
    var = xf2.pow(2).mean(dim=-1, keepdim=True)
    return (xf2 * torch.rsqrt(var + eps) * w.view(1, 1, n * d)).reshape(b, s, n, d)


def rope_ref_f32(xf, cos, sin, interleaved):
    b, s, n, d = xf.shape
    c = cos.view(1, s, 1, d // 2)
    sn = sin.view(1, s, 1, d // 2)
    if interleaved:
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        return torch.stack([x1 * c - x2 * sn, x1 * sn + x2 * c], dim=-1).flatten(-2)
    half = d // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    return torch.cat([x1 * c - x2 * sn, x2 * c + x1 * sn], dim=-1)


def torch_a2a_mode0(x, ws, group):  # x [b, s_local, n_global, d] -> [b, s_global, n_local, d]
    b, d = x.shape[0], x.shape[-1]
    s_local, n_global = x.shape[1], x.shape[2]
    n_local = n_global // ws
    xt = x.view(b, s_local, ws, n_local, d).permute(2, 0, 1, 3, 4).contiguous()
    out = torch.empty_like(xt)
    dist.all_to_all_single(out, xt, group=group)
    return out.permute(1, 0, 2, 3, 4).contiguous().view(b, ws * s_local, n_local, d)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    pg = dist.group.WORLD
    group = UlyssesGroup(process_group=pg, initial_pool_bytes=1 << 30)

    b, s_local, n_global, d, eps = 2, 16, 8, 128, 1e-6
    n_local = n_global // ws
    atol = {torch.float16: 2e-3, torch.bfloat16: 1e-2}
    rtol = {torch.float16: 2e-3, torch.bfloat16: 1.6e-2}
    torch.manual_seed(100 + rank)  # per-rank distinct data
    fails = 0
    for dtype in (torch.float16, torch.bfloat16):
        x = torch.randn(b, s_local, n_global, d, device=dev, dtype=dtype)
        theta = torch.randn(s_local, d // 2, device=dev, dtype=torch.float32)
        cos, sin = theta.cos().contiguous(), theta.sin().contiguous()
        w_ph = torch.randn(d, device=dev, dtype=torch.float32)
        w_ch = torch.randn(n_global * d, device=dev, dtype=torch.float32)
        for mode, w in (("per_head", w_ph), ("cross_head", w_ch)):
            for il in (True, False):
                ref_src = rope_ref_f32(rms_ref_f32(x, w, mode, eps), cos, sin, il).to(dtype)
                ref = torch_a2a_mode0(ref_src, ws, pg)
                got = group.all_to_all_single_4d_qk(
                    x, w, cos, sin, mode=mode, interleaved=il, eps=eps,
                    tag=f"{str(dtype).split('.')[-1]}_{mode}_{il}",
                )
                ok = torch.allclose(got.float(), ref.float(), atol=atol[dtype], rtol=rtol[dtype])
                md = (got.float() - ref.float()).abs().max().item()
                fails += not ok
                if rank == 0:
                    print(f"{'OK ' if ok else 'FAIL'} ws={ws} {str(dtype).split('.')[-1]:>8} {mode:<10} il={il!s:<5} out={tuple(got.shape)} maxdiff={md:.4e}", flush=True)
                dist.barrier()

    nfail = torch.tensor([fails], device=dev)
    dist.all_reduce(nfail)
    if rank == 0:
        print("ALL PASS" if nfail.item() == 0 else f"FAILED {int(nfail.item())} (summed over ranks)", flush=True)
    group.destroy()
    dist.destroy_process_group()
    if nfail.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
