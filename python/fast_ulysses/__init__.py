"""fast_ulysses — Ulysses sequence-parallel all-to-all, moved by the GPU copy engines."""

try:
    import torch  # noqa: F401  load libtorch before dlopen of _C

    from . import _C  # noqa: F401  trigger TORCH_LIBRARY registration
except ImportError as exc:  # ld.so names a symbol; _diagnose names the cause
    from ._diagnose import explain  # noqa: E402  imported only on the failure path

    raise ImportError(explain(exc)) from exc

# Written by setup.py from ./VERSION; present whenever _C is, since the same build emits both.
from ._build_meta import VERSION as __version__  # noqa: F401,E402
from .group import CompletedHandle, UlyssesGroup  # noqa: E402

nvlink_matrix = _C.nvlink_matrix

__all__ = ["UlyssesGroup", "CompletedHandle", "nvlink_matrix"]
