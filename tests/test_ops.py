"""Single-GPU correctness tests for the standalone QK RMSNorm / RoPE ops vs a torch reference.

Reference replicates Wan's math in fp32: RMSNorm (eps inside rsqrt, per-channel weight), then RoPE
over the full head_dim, interleaved (GPT-J) or non-interleaved (NeoX). The fused norm_rope kernel
keeps fp32 throughout (no intermediate rounding — that is the point of fusing), so the reference
composes in fp32 and rounds only once at the end; agreement is then to ~1 dtype ULP. cos/sin are
cos/sin of random angles (|.|<=1, rotation is norm-preserving) so RoPE does not amplify magnitudes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
fast_ulysses = pytest.importorskip("fast_ulysses", reason="fast_ulysses._C extension not built")

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

EPS = 1e-6
# Kernel and reference do identical fp32 math; disagreement is <=1 dtype ULP, so a relative rtol
# (not a flat atol) is the right gate — the largest randn elements otherwise trip a tight abs bound.
ATOL = {torch.float16: 2e-3, torch.bfloat16: 1e-2}
RTOL = {torch.float16: 2e-3, torch.bfloat16: 1.6e-2}

DTYPES = [torch.float16, torch.bfloat16]
N_HEADS = [8, 40]
# d sweeps the supported power-of-two range up to the documented max 1024 (warp-reduction and
# shared-memory paths have size-dependent branches, so the boundaries matter).
D_HEAD = [64, 128, 256, 512, 1024]


def _skip_if_cross_head_rope_exceeds_smem(mode, n, d):
    """cross_head + RoPE stages n*d fp32 in dynamic smem; skip combos over this device's cap."""
    if mode != "cross_head":
        return
    props = torch.cuda.get_device_properties(0)
    cap = getattr(props, "shared_memory_per_block_optin", 48 * 1024) - 128  # 128B static smem
    if n * d * 4 > cap:
        pytest.skip(f"cross_head rope needs {n * d * 4} B smem > device cap {cap} B")


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


def _make_inputs(dtype, n, d=128):
    torch.manual_seed(0)
    b, s = 2, 16
    dev = torch.device("cuda")
    x = torch.randn(b, s, n, d, device=dev, dtype=dtype)
    theta = torch.randn(s, d // 2, device=dev, dtype=torch.float32)
    cos, sin = theta.cos().contiguous(), theta.sin().contiguous()
    weights = {
        "per_head": torch.randn(d, device=dev, dtype=torch.float32),
        "cross_head": torch.randn(n * d, device=dev, dtype=torch.float32),
    }
    return x, cos, sin, weights


def _assert_close(got, ref, dtype):
    md = (got.float() - ref.float()).abs().max().item()
    assert torch.allclose(got.float(), ref.float(), atol=ATOL[dtype], rtol=RTOL[dtype]), (
        f"maxdiff={md:.4e}"
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("n", N_HEADS)
@pytest.mark.parametrize("d", D_HEAD)
@pytest.mark.parametrize("mode", ["per_head", "cross_head"])
def test_rms_norm(dtype, n, d, mode):
    x, _, _, weights = _make_inputs(dtype, n, d)
    got = fast_ulysses.rms_norm(x, weights[mode], mode=mode, eps=EPS)
    ref = rms_ref_f32(x, weights[mode], mode, EPS).to(dtype)
    _assert_close(got, ref, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("n", N_HEADS)
@pytest.mark.parametrize("d", D_HEAD)
@pytest.mark.parametrize("interleaved", [True, False])
def test_rope(dtype, n, d, interleaved):
    x, cos, sin, _ = _make_inputs(dtype, n, d)
    got = fast_ulysses.rope(x, cos, sin, interleaved=interleaved)
    ref = rope_ref_f32(x.float(), cos, sin, interleaved).to(dtype)
    _assert_close(got, ref, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("n", N_HEADS)
@pytest.mark.parametrize("d", D_HEAD)
@pytest.mark.parametrize("mode", ["per_head", "cross_head"])
@pytest.mark.parametrize("interleaved", [True, False])
def test_norm_rope(dtype, n, d, mode, interleaved):
    _skip_if_cross_head_rope_exceeds_smem(mode, n, d)
    x, cos, sin, weights = _make_inputs(dtype, n, d)
    got = fast_ulysses.norm_rope(
        x, weights[mode], cos, sin, mode=mode, interleaved=interleaved, eps=EPS
    )
    ref = rope_ref_f32(rms_ref_f32(x, weights[mode], mode, EPS), cos, sin, interleaved).to(dtype)
    _assert_close(got, ref, dtype)
