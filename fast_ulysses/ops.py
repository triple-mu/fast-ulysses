"""Standalone fused QK RMSNorm / RoPE ops (module-level; no UlyssesGroup, single-GPU elementwise).

All operate on x = [b, seq, n, d] (fp16/bf16, d a power of two). weight/cos/sin are fp32 CUDA tensors.
mode: "per_head" reduces RMS over d (weight [d]); "cross_head" reduces over n*d (weight [n*d]).
cos/sin: [seq, d/2], indexed by each row's seq position. interleaved: True=GPT-J adjacent-pair,
False=NeoX half-split. Semantics match Wan (RMSNorm in fp32, eps inside rsqrt; RoPE over full d).
"""

from __future__ import annotations

import torch

_MODE = {"per_head": 0, "cross_head": 1}


def _mode_int(mode: str) -> int:
    if mode not in _MODE:
        raise ValueError(f"mode must be 'per_head' or 'cross_head', got {mode!r}")
    return _MODE[mode]


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, *, mode: str = "per_head", eps: float = 1e-6
) -> torch.Tensor:
    return torch.ops.fast_ulysses.rms_norm(x, weight, _mode_int(mode), eps)


def rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, interleaved: bool = True
) -> torch.Tensor:
    return torch.ops.fast_ulysses.rope(x, cos, sin, interleaved)


def norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    mode: str = "per_head",
    interleaved: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    return torch.ops.fast_ulysses.norm_rope(x, weight, cos, sin, _mode_int(mode), interleaved, eps)
