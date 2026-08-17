from __future__ import annotations

import os
import threading
import warnings
from typing import Self

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from ._C import SUPPORTED_WORLD_SIZES, supports_world_size


class UlyssesGroup:
    """Equal-split, inference-only Ulysses all-to-all on one bound CUDA stream."""

    def __init__(
        self,
        process_group=None,
        device=None,
        stream: torch.cuda.Stream | None = None,
    ):
        """Build a group, or raise with a reason every rank agrees on.

        Prefer :meth:`create` when there is a fallback. Catching this is only safe because
        the outcome is agreed before anything irreversible happens -- see ``_build``.
        """
        reason = self._build(process_group, device, stream)
        if reason is not None:
            raise RuntimeError(reason)

    @classmethod
    def create(
        cls,
        process_group=None,
        device=None,
        stream: torch.cuda.Stream | None = None,
    ) -> Self | None:
        """The group, or ``None`` on **every** rank if any rank could not build one.

        This is the entry point for a caller with a fallback. `except: use something else`
        around the constructor is not that: mlx5 setup fails per rank, not per job --
        `select_nic` hands each rank a different NIC, so one missing IPv4 GID raises on one
        rank and leaves the other seven blocked in a collective it has already left. The
        return value here is agreed, so a caller that falls back on ``None`` falls back on
        every rank or on none.
        """
        group = cls.__new__(cls)
        reason = group._build(process_group, device, stream)
        if reason is None:
            return group
        warnings.warn(
            f"fast-ulysses is unavailable on every rank: {reason}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    def _build(self, process_group, device, stream) -> str | None:
        """Construct, returning an agreed reason instead of raising. None means success.

        The order is the whole point. Every rank-local check runs before the first
        collective; the configuration and the local outcome are agreed in one gather; and the
        native constructor -- the only step that can fail on some ranks and not others -- is
        followed by an outcome gather placed strictly before the connect handshake. After the
        handshake it would be too late: the ranks that got a transport would already be
        waiting inside it for one that never arrives.
        """
        # A partially constructed group cannot be closed collectively from __del__.
        self._destroyed = True
        self._group = None
        self.backend = None
        self.pg = process_group or dist.group.WORLD
        self.rank = dist.get_rank(self.pg)
        self.world_size = dist.get_world_size(self.pg)

        local_reason = self._prepare_local(device, stream)
        wire = (
            local_reason,
            None if local_reason else self.device.index,
            None if local_reason else self._environment(),
        )
        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, wire, group=self.pg)

        failures = [entry[0] for entry in gathered if entry[0] is not None]
        if failures:
            return "; ".join(failures)
        configs = [entry[2] for entry in gathered]
        if any(config != configs[0] for config in configs[1:]):
            return f"rank-inconsistent RDMA/NIC configuration: {configs!r}"
        devices = [entry[1] for entry in gathered]
        enable_rdma, nics = configs[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            symm_mem.enable_symm_mem_for_group(self.pg.group_name)

        native_reason = None
        try:
            self._group = torch.classes.fast_ulysses.UlyssesGroup(
                self.pg.group_name,
                self.rank,
                self.world_size,
                self.device.index,
                devices,
                enable_rdma,
                list(nics),
            )
        except Exception as error:  # noqa: BLE001 -- reported to every rank below
            native_reason = f"rank {self.rank}: {error}"

        outcomes: list = [None] * self.world_size
        dist.all_gather_object(outcomes, native_reason, group=self.pg)
        native_failures = [entry for entry in outcomes if entry is not None]
        if native_failures:
            self._abandon()
            return "; ".join(native_failures)

        self.backend = self._group.backend()
        if self.backend == "mlx5":
            peers: list = [None] * self.world_size
            dist.all_gather_object(peers, self._group.connection_info(), group=self.pg)
            self._group.connect(peers)
        self._output_pool: dict[tuple, torch.Tensor] = {}
        self._allocation_ordinal = 0
        self._destroyed = False
        return None

    def _prepare_local(self, device, stream) -> str | None:
        """Every check that needs no collective, so a failure here reaches the gather."""
        if not supports_world_size(self.world_size):
            return (
                f"rank {self.rank}: world size {self.world_size} is not one of "
                f"{tuple(SUPPORTED_WORLD_SIZES)}"
            )
        resolved = torch.device("cuda" if device is None else device)
        if resolved.type != "cuda":
            return f"rank {self.rank}: device must be CUDA"
        if resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        self.device = resolved
        torch.cuda.set_device(self.device)

        selected = stream or torch.cuda.current_stream(self.device)
        if torch.device(selected.device) != self.device:
            return f"rank {self.rank}: stream is on the wrong GPU"
        self.stream = selected
        self._owner_thread = threading.get_ident()
        return None

    @staticmethod
    def _environment() -> tuple:
        """The settings that affect collective construction, read in one place."""
        nics_text = os.environ.get("FAST_ULYSSES_NICS", "")
        return (
            not os.environ.get("FAST_ULYSSES_DISABLE_RDMA"),
            tuple(nics_text.split(",")) if nics_text else (),
        )

    def _abandon(self) -> None:
        """Release a group no rank will use, without the collective shutdown.

        destroy() barriers, and a rank whose native constructor threw has no group to barrier
        with. The native destroy is local and safe here because no workspace exists yet, so
        close_imports has nothing to coordinate -- and skipping it would leak the verbs
        context, PD, CQ and QPs for the process lifetime, since the uncoordinated destructor
        deliberately retains them.
        """
        if self._group is not None:
            try:
                self._group.destroy()
            except Exception:  # noqa: BLE001 -- teardown of an already-failed build
                pass
            self._group = None
        self._destroyed = True

    def unsupported_reason(self, shape, dtype: torch.dtype, mode: int = 0) -> str | None:
        """Why this shape cannot be exchanged, or None if it can.

        Pure and collective-free, so unlike catching allocate_output it is safe both to call
        on the hot path and to branch on: the answer depends only on the mode, the shape, the
        dtype, the world size and the transport, every one of which is the same on every
        rank. A caller that skips a call on the strength of it skips it everywhere.
        """
        self._check_alive()
        reason = self._group.unsupported_reason([int(size) for size in shape], dtype, mode)
        return reason or None

    def supports(self, shape, dtype: torch.dtype, mode: int = 0) -> bool:
        return self.unsupported_reason(shape, dtype, mode) is None

    def output_shape(self, shape, mode: int = 0) -> tuple[int, ...]:
        """The shape allocate_output would produce, without needing a tensor to ask."""
        self._check_alive()
        return tuple(self._group.output_shape_for([int(size) for size in shape], mode))

    def allocate_output(self, x: torch.Tensor, mode: int = 0) -> torch.Tensor:
        self._check_execution_context("allocate_output")
        if mode not in (0, 1):
            raise ValueError(f"mode must be 0 or 1, got {mode!r}")

        descriptor = (
            self._allocation_ordinal,
            mode,
            tuple(x.shape),
            str(x.dtype),
            x.numel() * x.element_size(),
        )
        self._require_rank_consistent("output allocation", descriptor)
        self._allocation_ordinal += 1

        output = self._group.allocate_output(x, mode)
        if self.backend == "mlx5":
            peers = [None] * self.world_size
            dist.all_gather_object(
                peers, self._group.buffer_info(output), group=self.pg
            )
            self._group.connect_buffer(output, peers)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.pg, device_ids=[self.device.index])
        return output

    def all_to_all_4d(
        self,
        x: torch.Tensor,
        mode: int = 0,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a 4-D all-to-all into ``out`` or an internal workspace.

        The automatic workspace is overwritten by the next call with the same
        mode, shape, and dtype. Pass an explicit ``out`` when multiple results
        with the same geometry must remain live simultaneously.
        """
        self._check_execution_context("all_to_all_4d")
        if mode not in (0, 1):
            raise ValueError(f"mode must be 0 or 1, got {mode!r}")
        if out is None:
            key = (mode, tuple(x.shape), x.dtype)
            out = self._output_pool.get(key)
            if out is None:
                out = self.allocate_output(x, mode)
                self._output_pool[key] = out
        if self.backend == "p2p":
            # Native P2P copies are stream-asynchronous. Tell the caching allocator not to
            # recycle the input while the bound stream can still be reading it.
            x.record_stream(self.stream)
        self._group.all_to_all_4d(x, out, mode, self.stream.cuda_stream)
        if self.backend == "mlx5":
            # The cross-quad half finishes on the host, in the completion poll, while the closing
            # barrier is on the stream. This lines the two back up. It carries its own NVTX range
            # because it is the last of the three host waits in a call and, like the other two,
            # shows on no CUDA timeline.
            with torch.cuda.nvtx.range("fu::trailing_sync"):
                self.stream.synchronize()
        return out

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._check_owner_stream("destroy")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("destroy is unsupported during CUDA Graph capture")

        self.stream.synchronize()
        dist.barrier(group=self.pg, device_ids=[self.device.index])
        self._group.close_imports()
        dist.barrier(group=self.pg, device_ids=[self.device.index])
        self._group.destroy()
        self._output_pool.clear()
        self._destroyed = True

    def __enter__(self) -> Self:
        self._check_alive()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.destroy()
        return False

    def __del__(self) -> None:
        try:
            if not getattr(self, "_destroyed", True):
                warnings.warn(
                    "UlyssesGroup was not explicitly destroyed; native resources may leak. "
                    "Use it as a context manager or call destroy() collectively on every rank.",
                    ResourceWarning,
                    stacklevel=2,
                )
        except (AttributeError, TypeError):
            # Interpreter teardown can clear imported modules before object finalizers run.
            return

    def _require_rank_consistent(self, what: str, value) -> None:
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, value, group=self.pg)
        if any(candidate != gathered[0] for candidate in gathered[1:]):
            raise RuntimeError(f"rank-inconsistent {what}: {gathered!r}")

    def _check_execution_context(self, operation: str) -> None:
        self._check_owner_stream(operation)
        if torch.is_grad_enabled():
            # What this guards is the output workspace: it is handed back by
            # identity and overwritten by the next call with the same geometry,
            # so a recorded graph could later read a buffer that no longer
            # holds what it saw. Recording is what has to be off, and no_grad
            # stops it as completely as inference_mode does. The native side
            # separately refuses an input that requires grad.
            raise RuntimeError(
                f"{operation} requires autograd to be off; run under "
                "torch.inference_mode() or torch.no_grad()"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{operation} is unsupported during CUDA Graph capture")
        compiler = getattr(torch, "compiler", None)
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling) and is_compiling():
            raise RuntimeError(f"{operation} is unsupported under torch.compile/export")

    def _check_owner_stream(self, operation: str) -> None:
        self._check_alive()
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError(f"{operation} must run on the group owner thread")
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError(
                f"{operation} must run with {self.device} as current device"
            )
        current = torch.cuda.current_stream(self.device)
        if current.cuda_stream != self.stream.cuda_stream:
            raise RuntimeError(
                f"{operation} must run on the stream bound at construction"
            )

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("group is destroyed")


__all__ = ["UlyssesGroup"]
