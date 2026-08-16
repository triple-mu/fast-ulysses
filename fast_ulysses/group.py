from __future__ import annotations

import os
import warnings

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


class UlyssesGroup:
    """Equal-split, inference-only Ulysses all-to-all."""

    def __init__(self, process_group=None, device=None):
        self.pg = process_group or dist.group.WORLD
        self.rank = dist.get_rank(self.pg)
        self.world_size = dist.get_world_size(self.pg)
        self.device = torch.device("cuda" if device is None else device)
        if self.device.type != "cuda":
            raise ValueError("device must be CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(self.device)

        local = torch.tensor([self.device.index], device=self.device)
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local, group=self.pg)
        devices = [int(value.item()) for value in gathered]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            symm_mem.enable_symm_mem_for_group(self.pg.group_name)
        # The environment is read here and passed down; the extension reads none itself.
        nics = os.environ.get("FAST_ULYSSES_NICS", "")
        self._group = torch.classes.fast_ulysses.UlyssesGroup(
            self.pg.group_name,
            self.rank,
            self.world_size,
            self.device.index,
            devices,
            not os.environ.get("FAST_ULYSSES_DISABLE_RDMA"),
            nics.split(",") if nics else [],
        )
        self.backend = self._group.backend()
        if self.backend == "mlx5":
            peers = [None] * self.world_size
            dist.all_gather_object(peers, self._group.connection_info(), group=self.pg)
            self._group.connect(peers)
        self._output_pool = {}
        self._destroyed = False

    def allocate_output(self, x: torch.Tensor, mode: int = 0) -> torch.Tensor:
        self._check_alive()
        output = self._group.allocate_output(x, mode)
        if self.backend == "mlx5":
            peers = [None] * self.world_size
            dist.all_gather_object(peers, self._group.buffer_info(output), group=self.pg)
            self._group.connect_buffer(output, peers)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.pg, device_ids=[self.device.index])
        return output

    def all_to_all_4d(
        self,
        x: torch.Tensor,
        mode: int = 0,
        out: torch.Tensor | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Run a 4-D all-to-all into ``out`` or an internal workspace.

        The automatic workspace is overwritten by the next call with the same
        mode, shape, and dtype. Pass an explicit ``out`` when multiple results
        with the same geometry must remain live simultaneously.
        """
        self._check_alive()
        if mode not in (0, 1):
            raise ValueError(f"mode must be 0 or 1, got {mode!r}")
        if out is None:
            key = (mode, tuple(x.shape), x.dtype)
            out = self._output_pool.get(key)
            if out is None:
                out = self.allocate_output(x, mode)
                self._output_pool[key] = out
        selected = stream or torch.cuda.current_stream(self.device)
        if torch.device(selected.device) != self.device:
            raise ValueError("stream is on the wrong GPU")
        self._group.all_to_all_4d(x, out, mode, selected.cuda_stream)
        if self.backend == "mlx5":
            # The cross-quad half finishes on the host, in the completion poll, while the closing
            # barrier is on the stream. This lines the two back up.
            selected.synchronize()
        return out

    def destroy(self):
        if self._destroyed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.pg, device_ids=[self.device.index])
        self._group.destroy()
        self._output_pool.clear()
        self._destroyed = True

    def _check_alive(self):
        if self._destroyed:
            raise RuntimeError("group is destroyed")


__all__ = ["UlyssesGroup"]
