"""fast_ulysses — Ulysses all-to-all custom op over the NVSHMEM symmetric heap."""

import torch  # noqa: F401  load libtorch before dlopen of _C

from . import _C  # noqa: F401,E402  trigger TORCH_LIBRARY registration
from .comm import AsyncA2AHandle, UlyssesGroup  # noqa: E402
from .ops import norm_rope, rms_norm, rope  # noqa: E402

__all__ = ["UlyssesGroup", "AsyncA2AHandle", "_C", "rms_norm", "rope", "norm_rope"]
