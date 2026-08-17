import torch  # noqa: F401

from . import _C as _C
from ._C import SUPPORTED_WORLD_SIZES, supports_dtype, supports_world_size
from .group import UlyssesGroup

__all__ = [
    "SUPPORTED_WORLD_SIZES",
    "UlyssesGroup",
    "supports_dtype",
    "supports_world_size",
]
__version__ = "0.3.0.dev0"
