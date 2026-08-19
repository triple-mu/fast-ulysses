from __future__ import annotations

import os
import threading
import warnings
from typing import Self

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from ._C import SUPPORTED_WORLD_SIZES, supports_world_size
from ._fallback import validate_input


class UlyssesGroup:
    """One transport, the symmetric-memory pool its outputs come from, and its registrations.

    This is the implementation behind the module-level functions, which is where callers should
    normally enter. It is public because a caller that owns several process groups may want to
    hold the transports itself, and because ``destroy()`` is collective and therefore has to be
    reachable.
    """

    def __init__(self, process_group=None, device=None):
        """Build a group, or raise with a reason every rank agrees on.

        Prefer :meth:`create` when there is a fallback. Catching this is only safe because
        the outcome is agreed before anything irreversible happens -- see ``_build``.
        """
        reason = self._build(process_group, device)
        if reason is not None:
            raise RuntimeError(reason)

    @classmethod
    def create(cls, process_group=None, device=None) -> Self | None:
        """The group, or ``None`` on **every** rank if any rank could not build one.

        This is the entry point for a caller with a fallback. `except: use something else`
        around the constructor is not that: mlx5 setup fails per rank, not per job --
        `select_nic` hands each rank a different NIC, so one missing IPv4 GID raises on one
        rank and leaves the other seven blocked in a collective it has already left. The
        return value here is agreed, so a caller that falls back on ``None`` falls back on
        every rank or on none.
        """
        group, reason = cls.create_or_reason(process_group, device)
        if reason is not None:
            warnings.warn(
                f"fast-ulysses is unavailable on every rank: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
        return group

    @classmethod
    def create_or_reason(
        cls, process_group=None, device=None
    ) -> tuple[Self | None, str | None]:
        """:meth:`create`, with the agreed reason instead of a warning, for a caller that logs."""
        group = cls.__new__(cls)
        reason = group._build(process_group, device)
        return (group, None) if reason is None else (None, reason)

    def _build(self, process_group, device) -> str | None:
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
        self.pg = process_group if process_group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.pg)
        self.world_size = dist.get_world_size(self.pg)

        try:
            local_reason = self._prepare_local(device)
        except Exception as error:  # noqa: BLE001 -- agreed with every rank below
            local_reason = f"rank {self.rank}: local setup failed: {error}"
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

        setup_reason = None
        local_backend = None
        connection_info = None
        try:
            local_backend = self._group.backend()
            if local_backend == "mlx5":
                connection_info = self._group.connection_info()
        except Exception as error:  # noqa: BLE001 -- agreed with every rank below
            setup_reason = f"rank {self.rank}: transport setup failed: {error}"

        setup: list = [None] * self.world_size
        dist.all_gather_object(
            setup, (setup_reason, local_backend, connection_info), group=self.pg
        )
        setup_failures = [entry[0] for entry in setup if entry[0] is not None]
        backends = [entry[1] for entry in setup]
        if not setup_failures and any(backend != backends[0] for backend in backends[1:]):
            setup_failures.append(f"rank-inconsistent transport backends: {backends!r}")
        if setup_failures:
            self._abandon()
            return "; ".join(setup_failures)

        self.backend = backends[0]
        finish_reason = None
        try:
            if self.backend == "mlx5":
                self._group.connect([entry[2] for entry in setup])

            # Outputs are allocated here, from a pool of this group's own. Two of torch's presets
            # matter and are repeated rather than inherited from get_mem_pool(), which returns a
            # process-wide pool shared with anything else using symmetric memory: block identity
            # has to be decided by this group's own call sequence alone.
            self._pool = torch.cuda.MemPool(
                symm_mem.get_mempool_allocator(self.device),
                # Lending space to unrelated allocations would desynchronise rank-local reuse.
                use_on_oom=False,
                # One segment per output keeps an address tied to exactly one peer allocation.
                no_split=True,
            )
        except Exception as error:  # noqa: BLE001 -- agreed with every rank below
            finish_reason = f"rank {self.rank}: transport finalization failed: {error}"

        finish: list = [None] * self.world_size
        dist.all_gather_object(finish, finish_reason, group=self.pg)
        finish_failures = [entry for entry in finish if entry is not None]
        if finish_failures:
            self._abandon()
            return "; ".join(finish_failures)

        self._geometries: set[tuple] = set()
        self._support_reasons: dict[tuple, str | None] = {}
        self._previous_stream = None
        self._destroyed = False
        return None

    def _prepare_local(self, device) -> str | None:
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
        # Not a stream: an exchange runs on whichever stream the caller is on, like any other
        # torch op. This is the thread that issues the collectives, and two threads issuing them
        # into one process group is a hang however the streams are arranged.
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
        with. The native destroy is local and safe here because no output is registered yet --
        and skipping it would leak the verbs context, PD, CQ and QPs for the process lifetime,
        since the uncoordinated destructor deliberately retains them.
        """
        if self._group is not None:
            try:
                self._group.destroy()
            except Exception:  # noqa: BLE001 -- teardown of an already-failed build
                pass
            self._group = None
        self._destroyed = True

    # ------------------------------------------------------------------- queries --
    def unsupported_reason(self, shape, dtype: torch.dtype, mode: int = 0) -> str | None:
        """Why this shape cannot be exchanged, or None if it can.

        Pure and collective-free, so unlike catching the exchange it is safe both to call on the
        hot path and to branch on: the answer depends only on the mode, the shape, the dtype,
        the world size and the transport, every one of which is the same on every rank. A caller
        that skips a call on the strength of it skips it everywhere.
        """
        self._check_alive()
        sizes = tuple(int(size) for size in shape)
        key = (mode, sizes, dtype)
        if key not in self._support_reasons:
            reason = self._group.unsupported_reason(list(sizes), dtype, mode)
            self._support_reasons[key] = reason or None
        return self._support_reasons[key]

    def supports(self, shape, dtype: torch.dtype, mode: int = 0) -> bool:
        return self.unsupported_reason(shape, dtype, mode) is None

    def output_shape(self, shape, mode: int = 0) -> tuple[int, ...]:
        """The shape an exchange would produce, without needing a tensor to ask."""
        self._check_alive()
        return tuple(self._group.output_shape_for([int(size) for size in shape], mode))

    # ------------------------------------------------------------------ exchange --
    def exchange(self, x: torch.Tensor, mode: int = 0) -> torch.Tensor:
        """Run a 4-D all-to-all and return the result, which the caller owns."""
        output_shape = validate_input(x, mode, self.world_size)
        sizes = tuple(int(size) for size in x.shape)
        if reason := self.unsupported_reason(sizes, x.dtype, mode):
            raise ValueError(reason)
        return self._exchange_validated(x, mode, output_shape, sizes)

    def _exchange_validated(
        self,
        x: torch.Tensor,
        mode: int,
        output_shape: tuple[int, int, int, int],
        sizes: tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Hot path after the module dispatcher has validated geometry and support."""
        self._check_owner()
        self._order_stream()
        self._agree_geometry(sizes, x.dtype, mode)
        output = self._new_output(output_shape, x.dtype)
        self._register(output, sizes, mode)
        self._group.all_to_all_4d(x, output, mode)
        return output

    def _agree_geometry(self, sizes: tuple, dtype: torch.dtype, mode: int) -> None:
        """Once per geometry, check that every rank is asking for the same one.

        Strictly before the first allocation for it. Everything downstream -- which block the
        pool hands back, and therefore which call is the one that has to rendezvous -- follows
        from the sequence of allocations, so a single rank-dependent shape does not merely fail
        one call: it desynchronises the pools and every later call decides differently on
        different ranks. Failing here leaves the group usable.

        Best effort, and deliberately so. It gathers only when a geometry is new, so it sees a
        divergence in which at least one rank is meeting its shape for the first time -- which
        is the first divergence of a run, and the one worth a message. Two ranks that each issue
        a *different but already seen* shape reach no collective here and deadlock instead, in
        whichever barrier they get to first. That is the documented failure model: every rank
        must issue the same sequence of shapes, and violating it hangs. Making this complete
        would cost a collective on every call, which is more than the diagnostic is worth.
        """
        key = (mode, sizes, dtype)
        if key in self._geometries:
            return
        self._require_rank_consistent(
            "exchange geometry", (len(self._geometries), mode, sizes, str(dtype))
        )
        self._geometries.add(key)

    def _new_output(self, shape, dtype: torch.dtype) -> torch.Tensor:
        with torch.cuda.use_mem_pool(self._pool):
            return torch.empty(shape, dtype=dtype, device=self.device)

    def _register(self, output: torch.Tensor, input_sizes: tuple, mode: int) -> None:
        """Make an output exchangeable, once per allocation and geometry.

        The native side is the single record of what has been registered, so this asks it rather
        than keeping a second one that could drift from it. Everything past the question is
        collective and every rank must reach it together. That relies on the documented contract
        that ranks retain and drop corresponding outputs in the same pattern; a mixed reuse/new
        decision diverges inside the native rendezvous and cannot be diagnosed afterward without
        adding a control collective to every call.
        """
        sizes = list(input_sizes)
        reason = None
        registered = False
        info = None
        try:
            registered = self._group.register_output(output, sizes, mode)
            if registered and self.backend == "mlx5":
                info = self._group.buffer_info(output)
        except Exception as error:  # noqa: BLE001 -- agreed with every rank below
            reason = f"rank {self.rank}: output registration failed: {error}"
        if not registered and reason is None:
            return

        outcomes: list = [None] * self.world_size
        dist.all_gather_object(outcomes, (reason, info), group=self.pg)
        failures = [entry[0] for entry in outcomes if entry[0] is not None]
        if failures:
            self._raise_collective_failure(failures)

        if self.backend == "mlx5":
            connect_reason = None
            try:
                self._group.connect_buffer(output, [entry[1] for entry in outcomes])
            except Exception as error:  # noqa: BLE001 -- agreed with every rank below
                connect_reason = f"rank {self.rank}: output connection failed: {error}"
            outcomes = [None] * self.world_size
            dist.all_gather_object(outcomes, connect_reason, group=self.pg)
            failures = [entry for entry in outcomes if entry is not None]
            if failures:
                self._raise_collective_failure(failures)

    def _raise_collective_failure(self, failures: list[str]) -> None:
        message = "; ".join(failures)
        # Unlike construction failure, this group may own registered buffers and prior CUDA
        # work. Every rank just completed the same outcome gather, so the normal collective
        # teardown is both reachable and required before registrations can be released safely.
        try:
            self.destroy()
        except Exception as error:  # noqa: BLE001 -- preserve the agreed root failure
            raise RuntimeError(f"{message}; coordinated teardown failed: {error}") from error
        raise RuntimeError(message)

    # ---------------------------------------------------------------------- life --
    def destroy(self) -> None:
        if self._destroyed:
            return
        self._check_owner()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("destroy is unsupported during CUDA Graph capture")
        # One barrier, not the two-phase import shutdown this used to need: no peer mapping of
        # ours is left to close in a coordinated order, because torch's symmetric memory owns
        # the peer mappings and outlives the transport. All this has to say is that no peer is
        # still writing into a buffer whose registration is about to go away.
        stream = torch.cuda.current_stream(self.device)
        previous = self._previous_stream
        if previous is not None and previous.cuda_stream != stream.cuda_stream:
            stream.wait_stream(previous)
        stream.synchronize()
        dist.barrier(group=self.pg, device_ids=[self.device.index])
        self._group.destroy()
        self._previous_stream = None
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

    # --------------------------------------------------------------------- rules --
    def _require_rank_consistent(self, what: str, value) -> None:
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, value, group=self.pg)
        if any(candidate != gathered[0] for candidate in gathered[1:]):
            raise RuntimeError(f"rank-inconsistent {what}: {gathered!r}")

    def _order_stream(self) -> torch.cuda.Stream:
        """Order a stream switch after all work already submitted to the previous stream."""
        stream = torch.cuda.current_stream(self.device)
        previous = self._previous_stream
        if previous is not None and previous.cuda_stream != stream.cuda_stream:
            stream.wait_stream(previous)
        self._previous_stream = stream
        return stream

    def _check_owner(self) -> None:
        self._check_alive()
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("this group must be used from the thread that built it")
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError(f"an exchange must run with {self.device} as current device")

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("group is destroyed")


__all__ = ["UlyssesGroup"]
