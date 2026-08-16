"""Correctness worker. Run under torchrun with 2, 4 or 8 ranks.

- every supported shape and dtype, both modes, against the NCCL reference in benchmark.py;
- the rejection paths;
- back-to-back calls with two ranks deliberately skewed, which is where a missing or mis-ordered
  barrier shows up. That check is only worth the tearing it can see, so it is armed: the same
  pattern runs once over raw symmetric-memory copies with no barrier at all, and MUST tear. A
  run whose control stays clean is reported BLIND, not passing.

The skewed ranks are one per quad: on the mlx5 backend the payload reaches same-quad peers by IPC
copy and cross-quad peers through the NIC, and only the cross-quad half can race ahead of a
reader. A skew inside one quad leaves that half untested.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from benchmark import nccl_mode0, nccl_mode1
from fast_ulysses import UlyssesGroup

# About 130 us at a 1.5 GHz clock, against a host gap between two calls of about 10 us.
SKEW_CYCLES = 200_000
ROUNDS = 400
SKEWED_RANKS = (0, 5)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def reference(x: torch.Tensor, mode: int, ws: int) -> torch.Tensor:
    """The NCCL path the benchmark already validates against, allocated fresh."""
    send = torch.empty(x.numel(), dtype=x.dtype, device=x.device)
    recv = torch.empty_like(send)
    b, s, h, d = x.shape
    if mode == 0:
        out = torch.empty((b, s * ws, h // ws, d), dtype=x.dtype, device=x.device)
        return nccl_mode0(x, send, recv, out, ws)
    out = torch.empty((b, s // ws, h * ws, d), dtype=x.dtype, device=x.device)
    return nccl_mode1(x, send, recv, out, ws)


def shapes(ws: int) -> list[tuple[int, int, int, int]]:
    """(batch, seq_local, heads_global, dim). The mlx5 backend takes batch 1 only."""
    cases = [(1, 8, 2 * ws, 16), (1, 128, ws, 64), (1, 592, 7 * ws, 128)]
    if os.environ.get("FAST_ULYSSES_DISABLE_RDMA"):
        cases += [(2, 16, 2 * ws, 32), (3, 64, ws, 128)]
    return cases


def check_shapes(group: UlyssesGroup, device: int, ws: int) -> None:
    for dtype in (torch.bfloat16, torch.float16):
        for batch, seq, heads, dim in shapes(ws):
            for mode in (0, 1):
                shape = (batch, seq * ws, heads // ws, dim) if mode else (batch, seq, heads, dim)
                x = torch.randn(shape, dtype=dtype, device=device)
                want = reference(x, mode, ws)
                out = group.allocate_output(x, mode)
                group.all_to_all_4d(x, mode, out=out)
                torch.cuda.synchronize()
                check(torch.equal(out, want),
                      f"{dtype} {tuple(shape)} mode={mode}: "
                      f"{int((out != want).sum())} of {out.numel()} elements differ")


def check_repeated(group: UlyssesGroup, device: int, ws: int, mode: int, rank: int) -> int:
    """Back-to-back calls with one rank per quad skewed. Returns the number of torn rounds.

    The payload cycles through 1..128 rather than counting up: a round that lands a neighbouring
    iteration's bytes is invisible when consecutive values are close.
    """
    shape = (1, 256 * ws, 4, 128) if mode else (1, 256, 4 * ws, 128)
    x = torch.empty(shape, dtype=torch.bfloat16, device=device)
    out = group.allocate_output(x, mode)
    torn = 0
    for i in range(ROUNDS):
        value = float(i % 128 + 1)
        x.fill_(value)
        if rank in SKEWED_RANKS:
            torch.cuda._sleep(SKEW_CYCLES)
        group.all_to_all_4d(x, mode, out=out)
        if not bool((out == value).all()):
            torn += 1
    torch.cuda.synchronize()
    return torn


def control(device: int, ws: int, rank: int, pg) -> int:
    """The same pattern over raw peer copies with no barrier. Must tear, or the check is blind."""
    slice_numel = 256 * 4 * 128
    x = torch.empty(slice_numel * ws, dtype=torch.bfloat16, device=device)
    window = symm_mem.empty(slice_numel * ws, dtype=torch.bfloat16, device=device)
    handle = symm_mem.rendezvous(window, pg.group_name)
    peers = [handle.get_buffer(p, (slice_numel * ws,), torch.bfloat16) for p in range(ws)]
    window.zero_()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[device])
    torn = 0
    for i in range(ROUNDS):
        value = float(i % 128 + 1)
        x.fill_(value)
        if rank in SKEWED_RANKS:
            torch.cuda._sleep(SKEW_CYCLES)
        for step in range(ws):
            peer = rank ^ step
            peers[peer][rank * slice_numel:(rank + 1) * slice_numel].copy_(
                x[peer * slice_numel:(peer + 1) * slice_numel], non_blocking=True)
        if not bool((window == value).all()):
            torn += 1
    torch.cuda.synchronize()
    dist.barrier(device_ids=[device])
    return torn


def check_rejections(group: UlyssesGroup, device: int, ws: int) -> None:
    """Every rank runs these, so no rank is left alone in a collective."""
    good = torch.randn((1, 64, 4 * ws, 128), dtype=torch.bfloat16, device=device)
    out = group.allocate_output(good, 0)

    def raises(fn, what: str) -> None:
        try:
            fn()
        except (RuntimeError, ValueError, TypeError):
            return
        FAILURES.append(f"{what} was accepted")

    if ws > 1:
        bad = torch.randn((1, 64, 4 * ws + 1, 128), dtype=torch.bfloat16, device=device)
        raises(lambda: group.all_to_all_4d(bad, 0, out=out),
               "a head count not divisible by world_size")
    raises(lambda: group.all_to_all_4d(good.float(), 0, out=out), "a float32 input")
    raises(lambda: group.all_to_all_4d(good.transpose(1, 2), 0, out=out),
           "a non-contiguous input")
    raises(lambda: group.all_to_all_4d(good.clone().requires_grad_(), 0, out=out),
           "an input that requires grad")
    raises(lambda: group.all_to_all_4d(good, 0, out=good), "an unallocated output")
    raises(lambda: group.all_to_all_4d(out.view(1, 64, 4 * ws, 128), 0, out=out),
           "an input overlapping the output")
    raises(lambda: group.all_to_all_4d(good, 2, out=out), "an invalid mode")
    if group.backend == "mlx5":
        # heads*dim*itemsize over all ranks is 65536, one byte past what an interleaved MKey's
        # stride can hold. Accepted by every verbs call and then silently wrong, so it has to be
        # refused before any of them.
        wide = torch.randn((1, 64, 256, 128), dtype=torch.bfloat16, device=device)
        raises(lambda: group.allocate_output(wide, 0), "a head stride over the MKey limit")


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(1234 + rank)

    group = UlyssesGroup(device=local_rank)
    if rank == 0:
        skewed = tuple(r for r in SKEWED_RANKS if r < ws)
        print(f"# world_size={ws} backend={group.backend} rounds={ROUNDS} "
              f"skew={SKEW_CYCLES} skewed_ranks={skewed}")

    check_shapes(group, local_rank, ws)
    check_rejections(group, local_rank, ws)
    for mode in (0, 1):
        torn = check_repeated(group, local_rank, ws, mode, rank)
        check(torn == 0, f"mode={mode}: {torn} of {ROUNDS} skewed back-to-back rounds tore")

    torn_control = 0 if ws == 1 else control(local_rank, ws, rank, dist.group.WORLD)
    armed = ws == 1 or torn_control > 0
    armed_all = [None] * ws
    failures = [None] * ws
    dist.all_gather_object(armed_all, armed)
    dist.all_gather_object(failures, FAILURES)
    group.destroy()
    dist.destroy_process_group()

    if rank == 0:
        for r, messages in enumerate(failures):
            for message in messages:
                print(f"FAIL rank {r}: {message}")
        if not any(armed_all):
            print("BLIND: the unbarriered control never tore, so the race check saw nothing")
        print("FAILED" if any(failures) or not any(armed_all) else "PASSED")
    sys.exit(1 if any(failures) or not any(armed_all) else 0)


if __name__ == "__main__":
    main()
