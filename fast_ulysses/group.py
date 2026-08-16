from __future__ import annotations

import os
import threading
import warnings
from typing import Self

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


class UlyssesGroup:
    """Equal-split, inference-only Ulysses all-to-all on one bound CUDA stream."""

    def __init__(
        self,
        process_group=None,
        device=None,
        stream: torch.cuda.Stream | None = None,
    ):
        # A partially constructed group cannot be closed collectively from __del__.
        self._destroyed = True
        self.pg = process_group or dist.group.WORLD
        self.rank = dist.get_rank(self.pg)
        self.world_size = dist.get_world_size(self.pg)
        self.device = torch.device("cuda" if device is None else device)
        if self.device.type != "cuda":
            raise ValueError("device must be CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(self.device)

        selected = stream or torch.cuda.current_stream(self.device)
        if torch.device(selected.device) != self.device:
            raise ValueError("stream is on the wrong GPU")
        self.stream = selected
        self._owner_thread = threading.get_ident()

        local = torch.tensor([self.device.index], device=self.device)
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local, group=self.pg)
        devices = [int(value.item()) for value in gathered]

        # These settings affect collective construction. Refuse a mixed configuration before
        # entering the native transport so every rank observes the same failure.
        enable_rdma = not os.environ.get("FAST_ULYSSES_DISABLE_RDMA")
        nics_text = os.environ.get("FAST_ULYSSES_NICS", "")
        nics = tuple(nics_text.split(",")) if nics_text else ()
        self._require_rank_consistent(
            "RDMA/NIC configuration",
            (enable_rdma, nics),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            symm_mem.enable_symm_mem_for_group(self.pg.group_name)
        self._group = torch.classes.fast_ulysses.UlyssesGroup(
            self.pg.group_name,
            self.rank,
            self.world_size,
            self.device.index,
            devices,
            enable_rdma,
            list(nics),
        )
        self.backend = self._group.backend()
        if self.backend == "mlx5":
            peers = [None] * self.world_size
            dist.all_gather_object(peers, self._group.connection_info(), group=self.pg)
            self._group.connect(peers)
        self._output_pool: dict[tuple, torch.Tensor] = {}
        self._allocation_ordinal = 0
        self._destroyed = False

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
            # barrier is on the stream. This lines the two back up.
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
