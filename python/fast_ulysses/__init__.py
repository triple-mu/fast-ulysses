"""fast_ulysses — Ulysses all-to-all custom op over the NVSHMEM symmetric heap."""

import torch  # noqa: F401  load libtorch before dlopen of _C

try:
    from . import _C  # noqa: F401,E402  trigger TORCH_LIBRARY registration
except ImportError as exc:  # ld.so names a symbol; _diagnose names the cause
    from ._diagnose import explain  # noqa: E402  imported only on the failure path

    raise ImportError(explain(exc)) from exc

# Written by setup.py from ./VERSION; present whenever _C is, since the same build emits both.
from ._build_meta import VERSION as __version__  # noqa: F401,E402
from .comm import CompletedHandle, UlyssesGroup  # noqa: E402
from .fallback import TorchUlyssesGroup, make_group, spans_sockets  # noqa: E402

__all__ = [
    "UlyssesGroup",
    "TorchUlyssesGroup",
    "CompletedHandle",
    "make_group",
    "spans_sockets",
]
