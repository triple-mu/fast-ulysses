"""custom_ulysess_op —— NVSHMEM 对称堆 Ulysses all-to-all 自定义算子。"""

import torch  # noqa: F401  在 _C dlopen 前先加载 libtorch

from . import _C  # noqa: F401,E402  触发 TORCH_LIBRARY 注册
from .comm import UlyssesGroup  # noqa: E402

__all__ = ["UlyssesGroup", "_C"]
