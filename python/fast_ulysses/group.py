"""The Python surface: a constructor, two collectives, and an output allocator.

Everything that survives a call -- the symmetric windows, the cached plans, the staging buffers --
lives in C++ (src/group.cc). What stays here is what has no C++ equivalent: the process group's
name, the comm stream, and ``AsyncCollectiveTensor``.
"""

from __future__ import annotations

import warnings

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._functional_collectives import AsyncCollectiveTensor

from . import _C


class CompletedHandle:
    """What the async call returns on a build whose libtorch has no ``c10d::register_work``. With
    no registry to bind the completion event to, the C++ side has already made the caller's stream
    wait on it, so ``wait()`` only unwraps -- a distinct type so that is visible."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def wait(self) -> torch.Tensor:
        return self._tensor

    def __repr__(self) -> str:
        return f"CompletedHandle({tuple(self._tensor.shape)}, {self._tensor.dtype})"


class UlyssesGroup:
    """Ulysses all-to-all over NVLink, moved by the copy engines.

    Construction is collective over ``process_group``. Windows are allocated on the first call that
    needs one and cached, so every rank must issue the same sequence of shapes. Contract:
    docs/api.md.
    """

    def __init__(
        self,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | str | int | None = None,
        *,
        require_nvlink: bool = True,
    ) -> None:
        """
        Args:
            process_group: the group this collective runs over; ``None`` uses ``dist.group.WORLD``.
            device: this rank's CUDA device, as anything ``torch.device`` accepts; ``None`` and a
                device with no index both use the current device.
            require_nvlink: refuse a group whose GPUs are not all NVLink-joined. ``False`` is for
                measuring that case, not for running in it.
        """
        pg = process_group if process_group is not None else dist.group.WORLD
        self.pg = pg
        self.rank = dist.get_rank(pg)
        self.world_size = dist.get_world_size(pg)
        # Normalised, because everything below indexes `device.index`: a bare "cuda:0" string has
        # no .index attribute, and torch.device("cuda") has one that is None. Both used to fail
        # several lines later, on a message about neither the device nor this constructor.
        device = torch.device("cuda" if device is None else device)
        if device.type != "cuda":
            raise ValueError(f"device must be a CUDA device, got {device}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        torch.cuda.set_device(device)

        if require_nvlink:
            self._require_nvlink()

        # Registers the group with torch's symmetric-memory bootstrap, which rendezvous needs. It
        # is idempotent, and deprecated from torch 2.13 (where rendezvous resolves the group by
        # itself) -- calling it anyway is what keeps 2.10 working.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            symm_mem.enable_symm_mem_for_group(pg.group_name)

        self._handle = torch.classes.fast_ulysses.UlyssesGroup(
            pg.group_name, int(self.rank), int(self.world_size), int(device.index)
        )
        self._destroyed = False

        # High-priority stream for the async call only; the sync call stays on the caller's stream,
        # since routing it through here costs two event hops. The two streams are NOT ordered
        # against each other, which is why the C++ side gives them separate windows.
        _, greatest = torch.cuda.Stream.priority_range()
        self._comm_stream = torch.cuda.Stream(device=device, priority=greatest)

    def _require_nvlink(self) -> None:
        """Every rank checks the same set of devices, so the answer is identical on all of them and
        the refusal is collective in practice."""
        local = torch.tensor([self.device.index], dtype=torch.int64, device=self.device)
        gathered = [torch.zeros_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local, group=self.pg)
        devices = [int(t.item()) for t in gathered]
        if len(set(devices)) != len(devices):
            return  # more than one rank per GPU: not a topology this check is about
        problem = _C.check_nvlink(devices)
        if problem:
            raise RuntimeError(problem)

    def all_to_all_4d(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        out: torch.Tensor | None = None,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> torch.Tensor:
        """mode 0 scatters heads and gathers sequence; mode 1 inverts it.

        Returns a tensor the caller owns, with no lifetime rules. ``out`` from ``empty_output()``
        skips the copy-out; any other contiguous CUDA tensor of the output shape is copied into.

        ``seq_splits[p]`` / ``head_splits[p]`` are rank p's sequence and head shard: pass both or
        neither, identical on every rank and matching the shape handed in. Neither means even
        shards. Collective: every rank must issue the same sequence of shapes.
        """
        return torch.ops.fast_ulysses.all_to_all_4d(
            self._handle, x, mode, seq_splits, head_splits, out
        )

    def all_to_all_4d_async(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        out: torch.Tensor | None = None,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> torch.Tensor | CompletedHandle:
        """``all_to_all_4d`` on this group's comm stream, returning immediately.

        The result is an ``AsyncCollectiveTensor``: ``.wait()`` returns the plain tensor, and so
        does the first use by any aten op -- either way the caller's stream waits on the comm
        stream's completion event GPU-side, and the host does not block. A view op re-wraps without
        waiting.

        Wait on, or use, every result: a dropped one leaves its entry in torch's work registry, and
        ``out=`` is the hole, since reading your own ``out`` never touches the registry. On a
        libtorch with no ``c10d::register_work`` this returns a ``CompletedHandle`` instead --
        correct, no overlap, and a distinct type so it is visible.
        """
        caller = torch.cuda.current_stream()
        with torch.cuda.stream(self._comm_stream):
            y = torch.ops.fast_ulysses.all_to_all_4d_staged(
                self._handle,
                x,
                mode,
                seq_splits,
                head_splits,
                out,
                caller.stream_id,
                caller.device_index,
            )
        registered = _C.register_stream_completion(y, self._comm_stream.cuda_stream)
        # Two streams touch the output: one allocated it, the other wrote it, and the caching
        # allocator must know about the cross-stream use before it may recycle the block.
        y.record_stream(self._comm_stream)
        y.record_stream(caller)
        return AsyncCollectiveTensor(y) if registered else CompletedHandle(y)

    def empty_output(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> torch.Tensor:
        """A buffer shaped like ``all_to_all_4d(x, mode=...)``'s output, in symmetric memory.

        Passing it back as ``out=`` makes the peers write it directly, which removes the copy-out.
        You own it: keep it across steps, free it when you like, and it may outlive the group.
        COLLECTIVE, so allocate outside the loop -- and use one buffer per concurrent call, since a
        call overwrites it.
        """
        return torch.ops.fast_ulysses.empty_output(self._handle, x, mode, seq_splits, head_splits)

    def _timed(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """BENCHMARKS ONLY: the copying call with CUDA events between its stages, which sum to the
        whole call because they are strictly ordered on one stream. Reading the events syncs."""
        out, stages = torch.ops.fast_ulysses.all_to_all_4d_timed(
            self._handle, x, mode, seq_splits, head_splits
        )
        names = ("barrier_in", "transfer", "barrier_out", "copy_out")
        return out, dict(zip(names, (float(v) for v in stages), strict=True))

    def destroy(self) -> None:
        """Release the group's windows and its transfer stream. Collective: dropping a window is.
        Buffers from ``empty_output()`` are unaffected -- they are yours."""
        if self._destroyed:
            return
        # Drain the comm stream first: dist.barrier only syncs the caller's current stream, so an
        # unwaited async call could still be writing a window that is about to be freed.
        self._comm_stream.synchronize()
        dist.barrier(group=self.pg)
        self._handle.destroy()
        self._destroyed = True
