"""Single-GPU correctness check for the fused QK RMSNorm / RoPE ops vs a torch reference.

    python fast-ulysses/test/test_qk_norm_rope.py

Reference replicates Wan's math in fp32: RMSNorm (eps inside rsqrt, per-channel weight), then RoPE over
the full head_dim, interleaved (GPT-J) or non-interleaved (NeoX). The fused norm_rope kernel keeps fp32
throughout (no intermediate rounding — that is the point of fusing), so the reference composes in fp32 and
rounds only once at the end; agreement is then to ~1 dtype ULP. cos/sin are cos/sin of random angles
(|.|<=1, rotation is norm-preserving) so RoPE does not amplify magnitudes.
"""

from __future__ import annotations

import torch

import fast_ulysses


def rms_ref_f32(x, w, mode, eps):  # returns fp32
    b, s, n, d = x.shape
    xf = x.float()
    if mode == "per_head":
        var = xf.pow(2).mean(dim=-1, keepdim=True)  # over d
        return xf * torch.rsqrt(var + eps) * w.view(1, 1, 1, d)
    xf2 = xf.reshape(b, s, n * d)  # cross_head: reduce over n*d
    var = xf2.pow(2).mean(dim=-1, keepdim=True)
    return (xf2 * torch.rsqrt(var + eps) * w.view(1, 1, n * d)).reshape(b, s, n, d)


def rope_ref_f32(xf, cos, sin, interleaved):  # xf fp32 in, fp32 out
    b, s, n, d = xf.shape
    c = cos.view(1, s, 1, d // 2)
    sn = sin.view(1, s, 1, d // 2)
    if interleaved:
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        o1, o2 = x1 * c - x2 * sn, x1 * sn + x2 * c
        return torch.stack([o1, o2], dim=-1).flatten(-2)
    half = d // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    o1, o2 = x1 * c - x2 * sn, x2 * c + x1 * sn
    return torch.cat([o1, o2], dim=-1)


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    eps = 1e-6
    # kernel and reference do identical fp32 math; disagreement is <=1 dtype ULP, so a relative rtol
    # (not a flat atol) is the right gate — the largest randn elements otherwise trip a tight abs bound.
    atol = {torch.float16: 2e-3, torch.bfloat16: 1e-2}
    rtol = {torch.float16: 2e-3, torch.bfloat16: 1.6e-2}
    fails = 0
    for dtype in (torch.float16, torch.bfloat16):
        for n in (8, 40):
            b, s, d = 2, 16, 128
            x = torch.randn(b, s, n, d, device=dev, dtype=dtype)
            theta = torch.randn(s, d // 2, device=dev, dtype=torch.float32)
            cos, sin = theta.cos().contiguous(), theta.sin().contiguous()
            w_ph = torch.randn(d, device=dev, dtype=torch.float32)
            w_ch = torch.randn(n * d, device=dev, dtype=torch.float32)

            cases = []
            # rms_norm
            cases.append(("rms per_head", fast_ulysses.rms_norm(x, w_ph, mode="per_head", eps=eps), rms_ref_f32(x, w_ph, "per_head", eps).to(dtype)))
            cases.append(("rms cross_head", fast_ulysses.rms_norm(x, w_ch, mode="cross_head", eps=eps), rms_ref_f32(x, w_ch, "cross_head", eps).to(dtype)))
            # rope
            for il in (True, False):
                cases.append((f"rope il={il}", fast_ulysses.rope(x, cos, sin, interleaved=il), rope_ref_f32(x.float(), cos, sin, il).to(dtype)))
            # norm_rope (fused; fp32 throughout, round once)
            for mode, w in (("per_head", w_ph), ("cross_head", w_ch)):
                for il in (True, False):
                    got = fast_ulysses.norm_rope(x, w, cos, sin, mode=mode, interleaved=il, eps=eps)
                    ref = rope_ref_f32(rms_ref_f32(x, w, mode, eps), cos, sin, il).to(dtype)
                    cases.append((f"norm_rope {mode} il={il}", got, ref))

            for name, got, ref in cases:
                md = maxdiff(got, ref)
                ok = torch.allclose(got.float(), ref.float(), atol=atol[dtype], rtol=rtol[dtype])
                fails += not ok
                print(f"{'OK ' if ok else 'FAIL'} {str(dtype).split('.')[-1]:>8} n={n:>2} {name:<24} maxdiff={md:.4e}")
    print("ALL PASS" if fails == 0 else f"FAILED {fails} cases")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
