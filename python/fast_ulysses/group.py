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
    """Ulysses all-to-all moved by the copy engines.

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
        backend: str = "pitched",
    ) -> None:
        """
        Args:
            process_group: the group this collective runs over; ``None`` uses ``dist.group.WORLD``.
            device: this rank's CUDA device, as anything ``torch.device`` accepts; ``None`` and a
                device with no index both use the current device.
            require_nvlink: refuse a group whose GPUs are not all NVLink-joined. Set ``False`` for
                the experimental packed PCIe backend.
            backend: ``"pitched"`` keeps the NVLink-optimised strided-copy path. ``"packed"``
                locally packs/unpacks around flat peer copies and is the experimental PCIe path.
        """
        # Before anything that can throw: a refused construction -- a non-NVLink group, a device
        # that is not CUDA -- still leaves an object for __del__ to run on, with nothing to
        # release.
        self._destroyed = True
        if backend not in ("pitched", "packed"):
            raise ValueError(f"backend must be 'pitched' or 'packed', got {backend!r}")
        self.backend = backend
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

        # High-priority stream for the async call only; the sync call stays on the caller's stream,
        # since routing it through here costs two event hops. The two streams are NOT ordered
        # against each other, which is why the C++ side gives them separate windows.
        _, greatest = torch.cuda.Stream.priority_range()
        self._comm_stream = torch.cuda.Stream(device=device, priority=greatest)
        self._destroyed = False

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
        skips the copy-out on the pitched backend and packed mode 0; packed mode 1 must locally
        unpack into it. Any other contiguous CUDA tensor of the output shape is copied into.

        ``seq_splits[p]`` / ``head_splits[p]`` are rank p's sequence and head shard: pass both or
        neither, identical on every rank and matching the shape handed in. Neither means even
        shards. Collective: every rank must issue the same sequence of shapes.
        """
        if self.backend == "packed":
            if seq_splits is not None or head_splits is not None:
                raise ValueError(
                    "the packed backend currently supports even shards only; omit seq_splits "
                    "and head_splits"
                )
            if x.ndim == 4 and x.shape[0] != 1:
                raise ValueError(
                    f"the packed backend currently requires batch=1, got batch={x.shape[0]}"
                )
            if torch.is_grad_enabled() and x.requires_grad:
                raise RuntimeError(
                    "the packed backend is currently inference-only and does not support "
                    "autograd; use backend='pitched' for differentiable calls"
                )
            if out is None:
                return torch.ops.fast_ulysses.all_to_all_4d_packed(
                    self._handle, x, mode, None, None
                )
            torch.ops.fast_ulysses.all_to_all_4d_packed_out(self._handle, x, mode, None, None, out)
            return out
        if out is None:
            return torch.ops.fast_ulysses.all_to_all_4d(
                self._handle, x, mode, seq_splits, head_splits
            )
        # Refused rather than silently wrong, for the same reason the async call refuses: the
        # out-variant mutates a buffer the caller already owns, so it carries no autograd formula
        # and what comes back is that buffer -- backward() would run to completion and leave
        # x.grad as None. The returned value cannot signal it either, since nothing distinguishes
        # a caller who wanted no gradient from one whose gradient was dropped.
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError(
                "all_to_all_4d(out=...) does not support autograd: the out-variant is a mutating "
                "op with no backward, so the gradient would be dropped without an error. Drop "
                "out=, or run under torch.no_grad()."
            )
        # Two ops, one method: the out-variant declares the alias it really has, which is what
        # keeps the functional one differentiable. See the note in src/bindings.cc.
        torch.ops.fast_ulysses.all_to_all_4d_out(
            self._handle, x, mode, seq_splits, head_splits, out
        )
        return out

    def all_to_all_4d_async(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        out: torch.Tensor | None = None,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
        lend: bool = False,
    ) -> torch.Tensor | CompletedHandle:
        """``all_to_all_4d`` on this group's comm stream, returning immediately.

        The result is an ``AsyncCollectiveTensor``: ``.wait()`` returns the plain tensor, and so
        does the first use by any aten op -- either way the caller's stream waits on the comm
        stream's completion event GPU-side, and the host does not block. A view op re-wraps without
        waiting.

        ``lend=True`` gets ``out=``'s saving without a buffer of your own: the peers write a window
        this group holds, the result IS that window, and there is no copy-out. The group rotates
        through four windows per dtype, so at most four lent results may be alive at once -- a
        fifth call raises rather than overwriting one, and the count is per rank, so every rank has
        to drop its results at the same point in its own program. Mutually exclusive with ``out=``.
        Like the rest of this method it is not differentiable.

        Wait on, or use, every result: a dropped one leaves its entry in torch's work registry, and
        ``out=`` is the hole, since reading your own ``out`` never touches the registry. On a
        libtorch with no ``c10d::register_work`` this returns a ``CompletedHandle`` instead --
        correct, no overlap, and a distinct type so it is visible.
        """
        if self.backend == "packed":
            raise RuntimeError(
                "the packed backend currently supports synchronous all_to_all_4d only"
            )
        # Refused rather than silently ignored: both ask for the zero-copy path, and picking one
        # for the caller would leave a lend=True that quietly did nothing.
        if out is not None and lend:
            raise ValueError(
                "all_to_all_4d_async takes out= or lend=True, not both: each is a way of getting "
                "the zero-copy path, one with a buffer you own and one with a window the group "
                "lends. Drop whichever you did not mean."
            )
        # Refused rather than silently wrong. AsyncCollectiveTensor is built with
        # _make_wrapper_subclass(..., requires_grad=elem.requires_grad), which makes the wrapper a
        # LEAF: autograd runs above the subclass and never sees the wrapped tensor's history, so
        # backward() would deposit the gradient on the wrapper and leave x.grad as None.
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError(
                "all_to_all_4d_async does not support autograd: its result is an "
                "AsyncCollectiveTensor, which is a leaf, so the gradient would be dropped without "
                "an error. Use all_to_all_4d()."
            )
        # This group's device, not the current one: an aten op on `x` runs on the current stream
        # of x's device whatever device is current, and that stream is what the transfer has to be
        # ordered against. With another device current, the no-argument form names a stream on it,
        # which orders nothing here and cannot take record_stream() for this tensor.
        caller = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self._comm_stream):
            if out is None:
                y = torch.ops.fast_ulysses.all_to_all_4d_staged(
                    self._handle,
                    x,
                    mode,
                    seq_splits,
                    head_splits,
                    caller.stream_id,
                    caller.device_index,
                    lend,
                )
            else:
                torch.ops.fast_ulysses.all_to_all_4d_staged_out(
                    self._handle,
                    x,
                    mode,
                    seq_splits,
                    head_splits,
                    out,
                    caller.stream_id,
                    caller.device_index,
                )
                y = out
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
        if self.backend == "packed":
            if seq_splits is not None or head_splits is not None:
                raise ValueError(
                    "the packed backend currently supports even shards only; omit seq_splits "
                    "and head_splits"
                )
            if x.ndim == 4 and x.shape[0] != 1:
                raise ValueError(
                    f"the packed backend currently requires batch=1, got batch={x.shape[0]}"
                )
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
        if self.backend == "packed":
            raise RuntimeError("_timed is not implemented for the packed backend")
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

    def __del__(self) -> None:
        # A group that is dropped rather than destroyed still has to outlive its own transfers.
        # The handle's C++ destructor waits on the internal transfer stream but knows nothing of
        # this one, so an unwaited async call could still be writing a window as it is freed.
        # There is no barrier here on purpose: a finalizer is no place for a collective, which is
        # why destroy() and not this is the supported path.
        if not self._destroyed:
            self._comm_stream.synchronize()
