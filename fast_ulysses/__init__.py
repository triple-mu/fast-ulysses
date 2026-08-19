"""Equal-split, inference-only Ulysses 4-D all-to-all.

The whole surface is three functions that take a tensor and return one::

    import fast_ulysses as fu

    out = fu.all_to_all_4d(x, mode=0)   # [B, S_local, H, D] -> [B, S_global, H/ws, D]
    out = fu.scatter_heads(x)           # the same thing, named
    out = fu.gather_heads(x)            # and its inverse

There is no group object to thread through, no workspace to allocate, and no stream to bind:
an exchange runs on whichever stream the caller is on, and the result is an ordinary tensor the
caller owns. Which transport carries it -- CUDA P2P over torch symmetric memory, mlx5 RDMA
across the socket boundary, or the process group's own all-to-all -- is decided here, from
facts that are identical on every rank, so the ranks never take different paths.
"""

from __future__ import annotations

import logging
import threading

import torch
import torch.distributed as dist

from . import _C as _C
from . import _fallback
from ._C import SUPPORTED_WORLD_SIZES, supports_dtype, supports_world_size
from .group import UlyssesGroup

__version__ = "0.3.0.dev0"

logger = logging.getLogger(__name__)

# One transport per process group, built on first use. None records "asked, cannot" so the
# construction collectives are not retried on every call.
_TRANSPORTS: dict[str, UlyssesGroup | None] = {}
_DECLINED: set[tuple] = set()
_BUILD_LOCKS: dict[str, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()

__all__ = [
    "SUPPORTED_WORLD_SIZES",
    "__version__",
    "UlyssesGroup",
    "all_to_all_4d",
    "backend",
    "gather_heads",
    "output_shape",
    "scatter_heads",
    "shutdown",
    "supports_dtype",
    "supports_world_size",
    "unsupported_reason",
]


def _transport(group=None) -> UlyssesGroup | None:
    """The transport for ``group``, built once, or None to use the process group's own.

    Construction is collective, so every rank must reach this with the same argument. They do:
    the only caller is an exchange, which every rank issues together, and ``create()`` agrees
    its outcome across the ranks before returning.
    """
    process_group = group if group is not None else dist.group.WORLD
    name = process_group.group_name
    if name in _TRANSPORTS:
        return _TRANSPORTS[name]
    with _BUILD_LOCKS_GUARD:
        build_lock = _BUILD_LOCKS.setdefault(name, threading.Lock())
    with build_lock:
        if name not in _TRANSPORTS:
            transport, reason = UlyssesGroup.create_or_reason(process_group=process_group)
            _TRANSPORTS[name] = transport
            if transport is None:
                logger.info(
                    "fast-ulysses has no transport for this group (%s); "
                    "using its all-to-all instead",
                    reason,
                )
            else:
                logger.info(
                    "fast-ulysses is carrying this group's exchanges "
                    "(backend=%s, world_size=%d)",
                    transport.backend,
                    transport.world_size,
                )
        return _TRANSPORTS[name]


def all_to_all_4d(x: torch.Tensor, mode: int = 0, group=None) -> torch.Tensor:
    """``mode=0`` splits heads and gathers sequence; ``mode=1`` is its inverse.

    Returns a new tensor. Falls back to the process group's own all-to-all for any shape no
    transport can carry -- a decision that depends only on the mode, shape, dtype, world size
    and transport, every one of which is the same on every rank.
    """
    world_size = dist.get_world_size(group)
    shape = _fallback.validate_input(x, mode, world_size)
    sizes = tuple(int(size) for size in x.shape)
    transport = _transport(group)
    if transport is not None:
        reason = transport.unsupported_reason(sizes, x.dtype, mode)
        if reason is None:
            return transport._exchange_validated(x, mode, shape, sizes)
        # Asked on every layer of every step, not only on a cold shape, so an unthrottled log
        # would be thousands of identical lines per request.
        key = (transport.pg.group_name, mode, tuple(x.shape), x.dtype)
        if key not in _DECLINED:
            _DECLINED.add(key)
            logger.info(
                "fast-ulysses declines %s mode=%d (%s); using the process group's all-to-all",
                tuple(x.shape),
                mode,
                reason,
            )
    return _fallback._all_to_all_4d_validated(x, mode, group, world_size, shape)


def scatter_heads(x: torch.Tensor, group=None) -> torch.Tensor:
    """``[B, S_local, H, D]`` -> ``[B, S_global, H/ws, D]``."""
    return all_to_all_4d(x, 0, group)


def gather_heads(x: torch.Tensor, group=None) -> torch.Tensor:
    """``[B, S_global, H_local, D]`` -> ``[B, S_local, H_local*ws, D]``."""
    return all_to_all_4d(x, 1, group)


def unsupported_reason(shape, dtype: torch.dtype, mode: int = 0, group=None) -> str | None:
    """Why a shape would fall back, or None if a transport carries it."""
    try:
        _fallback.output_shape(shape, mode, dist.get_world_size(group))
    except (TypeError, ValueError) as error:
        return str(error)
    transport = _transport(group)
    if transport is None:
        return "no transport could be built for this group"
    return transport.unsupported_reason(shape, dtype, mode)


def output_shape(shape, mode: int = 0, group=None) -> tuple[int, ...]:
    """The shape an exchange of ``shape`` would return, without needing a tensor to ask."""
    return _fallback.output_shape(shape, mode, dist.get_world_size(group))


def backend(group=None) -> str:
    """``"mlx5"``, ``"p2p"``, or ``"fallback"`` when the process group carries it itself."""
    transport = _transport(group)
    return "fallback" if transport is None else transport.backend


def shutdown() -> None:
    """Release every transport. Collective: every rank must call this before any rank exits."""
    for name in list(_TRANSPORTS):
        transport = _TRANSPORTS[name]
        if transport is not None:
            transport.destroy()
        del _TRANSPORTS[name]
    _DECLINED.clear()
    _BUILD_LOCKS.clear()
