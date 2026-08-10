"""Does the NVLink probe say what NVML says, and does the constructor act on it?

    torchrun --nproc_per_node=8 test/distributed/nvlink.py

Three tiers, in that order:

  1. The COMPARATOR, on synthetic matrices, before the machine is touched at all. ``verdict()`` is
     a pure function and is called on four inputs whose answers are known, one of them a mismatch
     and one of them the fail-open below. The real comparison cannot prove the comparator works --
     on a healthy machine its two arguments are trivially equal -- so this is what proves it can
     distinguish, the way ce_ordering.py's injected fault proves its reader can.
  2. ``_C.nvlink_matrix`` over every visible device, against a second implementation of the same
     question written here against NVML through pynvml. Then rank uniformity: group.py's
     ``_require_nvlink`` is documented as giving every rank the same answer, and an answer that
     differs by rank is a RuntimeError that differs by rank, which is a hang.
  3. The constructor itself, with ``require_nvlink=True`` over the ranks' own devices. A machine
     only ever exercises one direction of this: it either accepts an NVLink group or refuses a
     non-NVLink one.

THE FAIL-OPEN is what tier 2 exists for. ``check_nvlink`` returns an empty string -- "no basis to
refuse" -- whenever the matrix is nullopt, so a probe that breaks into returning nullopt ADMITS a
PCIe group instead of refusing it, and prints nothing anywhere. If NVML can answer at all the
oracle returns a matrix, so a None from the C++ against an answering oracle is exactly that state,
and it fails. The reverse -- a link claimed or missed -- is a plain pair-by-pair mismatch.

The oracle looks its handles up by the PCI bus id built from ``torch.cuda.get_device_properties``,
which is the string src/nvlink.cc builds from ``cudaGetDeviceProperties``: BDF MATCHING IS SHARED
between the two and is therefore not what this compares. What it compares is the LINK SWEEP --
walking links until the device says there are no more -- and the switch fold, where every GPU with
an active link into an NVSwitch fabric lands in one clique.

One deliberate asymmetry with src/nvlink.cc: where it does ``continue`` on a per-link NVML error,
the oracle returns None. A test oracle either answers fully or says nothing, otherwise a driver
quirk that makes one remote-type query fail turns into a false MISMATCH and fails a healthy
machine.

BLIND EXITS 0 HERE, which ce_ordering.py and window_race.py do not do. Their blindness is a defect
of the test and every test machine can arm them; this one's is a property of the machine -- no
pynvml, no usable NVML, or one visible GPU, which leaves the matrix without an off-diagonal pair to
compare. test_distributed.py turns an ``NVLINK BLIND`` line into a pytest skip, so a blind run is
still not a silent pass.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from fast_ulysses import _C, UlyssesGroup


def oracle(devices: list[int]) -> dict[tuple[int, int], bool] | None:
    """``{(i, j): joined by NVLink}`` from NVML through pynvml, or None when it cannot be asked.

    Same contract as ``nvlink_matrix()``, and deliberately not the same code: this is the second
    opinion. None on ANY per-link error, including one src/nvlink.cc skips over.
    """
    try:
        import pynvml
    except ImportError:
        return None
    # A pynvml too old for these entry points cannot ask either, and the missing NAME is an
    # AttributeError rather than an NVMLError -- uncaught, it would fail the worker on a machine
    # that simply has nothing to say, which is the one outcome this file must never produce.
    if any(
        not hasattr(pynvml, n)
        for n in (
            "nvmlDeviceGetHandleByPciBusId",
            "nvmlDeviceGetNvLinkState",
            "nvmlDeviceGetNvLinkRemoteDeviceType",
            "nvmlDeviceGetNvLinkRemotePciInfo",
            "NVML_NVLINK_DEVICE_TYPE_GPU",
            "NVML_NVLINK_DEVICE_TYPE_SWITCH",
            "NVML_ERROR_NOT_SUPPORTED",
            "NVML_ERROR_INVALID_ARGUMENT",
        )
    ):
        return None
    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 -- any init failure is "cannot ask", not a test failure
        return None

    by_bdf, handles = {}, {}
    for i in devices:
        prop = torch.cuda.get_device_properties(i)
        # getattr: the PCI triple is not part of the 2.10 floor's documented surface. Missing it
        # means the oracle cannot build a bus id and has nothing to say.
        bdf = tuple(
            getattr(prop, a, None) for a in ("pci_domain_id", "pci_bus_id", "pci_device_id")
        )
        if None in bdf:
            return None
        by_bdf[bdf] = i
        try:
            handles[i] = pynvml.nvmlDeviceGetHandleByPciBusId(
                f"{bdf[0]:08x}:{bdf[1]:02x}:{bdf[2]:02x}.0"
            )
        except pynvml.NVMLError:
            return None

    linked = {(i, j): i == j for i in devices for j in devices}
    on_switch = set()
    for i, h in handles.items():
        answered = False
        for link in range(32):
            try:
                active = pynvml.nvmlDeviceGetNvLinkState(h, link)
            except pynvml.NVMLError as err:
                # NOT_SUPPORTED at link 0 is a device with no NVLink; INVALID_ARGUMENT past the
                # last link is a device with some. Both are the device ANSWERING.
                if err.value not in (
                    pynvml.NVML_ERROR_NOT_SUPPORTED,
                    pynvml.NVML_ERROR_INVALID_ARGUMENT,
                ):
                    return None
                answered = True
                break
            answered = True
            if not active:
                continue
            try:
                kind = pynvml.nvmlDeviceGetNvLinkRemoteDeviceType(h, link)
            except pynvml.NVMLError:
                return None
            if kind == pynvml.NVML_NVLINK_DEVICE_TYPE_SWITCH:
                on_switch.add(i)
            elif kind == pynvml.NVML_NVLINK_DEVICE_TYPE_GPU:
                try:
                    info = pynvml.nvmlDeviceGetNvLinkRemotePciInfo(h, link)
                except pynvml.NVMLError:
                    return None
                j = by_bdf.get((info.domain, info.bus, info.device))
                if j is not None:
                    linked[(i, j)] = linked[(j, i)] = True
        if not answered:
            return None

    # One fabric per node: every GPU with a switch link reaches every other one.
    for i in on_switch:
        for j in on_switch:
            linked[(i, j)] = True
    return linked


def verdict(
    mine: dict[tuple[int, int], bool] | None, ref: dict[tuple[int, int], bool] | None
) -> tuple[str, str]:
    """PASS, FAIL or BLIND for one pair of matrices. Pure, so `controls()` can arm it."""
    if ref is None:
        if mine is None:
            return "BLIND", "neither probe answered: no usable NVML on this machine"
        return "BLIND", "NVML cannot be read from Python here, so there is nothing to compare to"
    if mine is None:
        return (
            "FAIL",
            "NVML answered but _C.nvlink_matrix returned None -- check_nvlink then returns an "
            "empty string for every group, so a PCIe group is admitted rather than refused",
        )
    if set(mine) != set(ref):
        return "FAIL", f"the two matrices cover different pairs: {sorted(set(mine) ^ set(ref))}"
    wrong = [k for k in sorted(ref) if bool(mine[k]) != bool(ref[k])]
    if wrong:
        detail = ", ".join(
            f"cuda:{i}/cuda:{j} probe says {mine[(i, j)]}, NVML says {ref[(i, j)]}"
            for i, j in wrong
        )
        return "FAIL", f"the two matrices disagree on {len(wrong)} of {len(ref)} pairs -- {detail}"
    if not any(i != j for i, j in ref):
        return (
            "BLIND",
            "one visible GPU: the matrix has no off-diagonal pair, so nothing was compared",
        )
    return "PASS", f"{len(ref)} pairs agree"


def controls() -> list[str]:
    """Four synthetic inputs whose verdicts are known, run on every rank before the machine is
    touched. A comparison whose two sides are trivially equal -- which is what a healthy machine
    produces -- cannot show that the comparator would notice a difference; these can. They also
    exercise the BLIND branch, so it is not dead code on a machine that never goes blind."""
    agree = {(0, 0): True, (1, 1): True, (0, 1): False, (1, 0): False}
    flipped = dict(agree)
    flipped[(0, 1)] = True
    cases = [
        ("two matrices that agree", verdict(agree, agree)[0], "PASS"),
        ("one off-diagonal pair flipped", verdict(flipped, agree)[0], "FAIL"),
        ("the fail-open: no matrix while NVML answers", verdict(None, agree)[0], "FAIL"),
        ("no matrix from either side", verdict(None, None)[0], "BLIND"),
    ]
    return [f"{name} -> {got}, expected {want}" for name, got, want in cases if got != want]


def fail_open_pin() -> list[str]:
    """The fail-open, pinned in process. A device index that cannot be probed makes the matrix
    None, and check_nvlink over it then returns an EMPTY STRING -- the caller is allowed through.
    That is deliberate and documented in include/fast_ulysses/nvlink.hpp: "cannot say" is not
    "refuse", and the real refusal comes from a matrix that ANSWERED and holds a False pair. Pinned
    so that changing it -- to fail closed, or to fabricating a matrix -- is a noticed change."""
    out = []
    if _C.nvlink_matrix([999]) is not None:
        out.append("nvlink_matrix over a device that cannot be probed answered instead of None")
    admitted = _C.check_nvlink([0, 999])
    if admitted != "":
        out.append(f"check_nvlink over a device that cannot be probed refused: {admitted!r}")
    return out


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())

    problems = controls() + fail_open_pin()
    if problems:
        state, message = "FAIL", "; ".join(problems)
    else:
        visible = list(range(torch.cuda.device_count()))
        mine, ref = _C.nvlink_matrix(visible), oracle(visible)
        state, message = verdict(mine, ref)

    # Every rank decides for itself and the worst answer wins, so the ranks leave together: a rank
    # that exited here alone would leave the others in the collectives below.
    code = torch.tensor([{"PASS": 0, "BLIND": 1, "FAIL": 2}[state]], device=dev)
    dist.all_reduce(code, op=dist.ReduceOp.MAX)
    worst = int(code.item())
    if state != "PASS":
        print(f"[rank {rank}] {state}: {message}", flush=True)
    if worst != 0:
        word = "FAIL" if worst == 2 else "BLIND"
        if rank == 0:
            print(
                f"NVLINK {word}: {message if state == word else 'see the per-rank lines above'}",
                flush=True,
            )
        dist.destroy_process_group()
        raise SystemExit(1 if worst == 2 else 0)

    # --- rank uniformity ---------------------------------------------------------------------
    # The one part of this that has to be distributed at all. A matrix that differs by rank makes
    # the constructor's refusal differ by rank, and a refusal on some ranks only is a hang.
    pairs = sorted(mine)
    row = torch.tensor([mine[p] for p in pairs], dtype=torch.uint8, device=dev)
    rows = [torch.zeros_like(row) for _ in range(ws)]
    dist.all_gather(rows, row)
    if any(not torch.equal(r, rows[0]) for r in rows):
        if rank == 0:
            print("NVLINK FAIL: the ranks do not agree on the matrix", flush=True)
        dist.destroy_process_group()
        raise SystemExit(1)

    # --- the constructor -----------------------------------------------------------------------
    local = torch.tensor([dev.index], dtype=torch.int64, device=dev)
    gathered = [torch.zeros_like(local) for _ in range(ws)]
    dist.all_gather(gathered, local)
    devices = [int(t.item()) for t in gathered]
    unlinked = [(i, j) for i in devices for j in devices if i != j and not ref[(i, j)]]

    raised, reason, group = 0, "", None
    try:
        group = UlyssesGroup(require_nvlink=True)
    except RuntimeError as exc:
        raised, reason = 1, str(exc)
    # AFTER the catch, never inside the try: a collective issued from inside it would be one the
    # ranks that did not raise never reach.
    flag = torch.tensor([raised], device=dev)
    dist.all_reduce(flag)
    refusals = int(flag.item())
    # destroy() is collective as well -- a barrier, then a window release every rank has to reach --
    # so it may only run when EVERY rank has a group to destroy. A refusal on some ranks only is
    # exactly what the FAIL below reports, and destroying on the ranks that built one would hang
    # there instead, leaving that report unreachable.
    if group is not None and refusals == 0:
        group.destroy()

    if len(set(devices)) != len(devices):
        # group.py skips the check when the ranks share GPUs: that is not a topology this is about.
        ok = True
        line = f"NVLINK PASS ({message}; the constructor tier did not run: the ranks share GPUs)"
    elif unlinked:
        ok = refusals == ws and "not joined by NVLink" in reason
        i, j = unlinked[0]
        line = (
            f"NVLINK PASS (refused a non-NVLink group: cuda:{i} / cuda:{j})"
            if ok
            else f"NVLINK FAIL: {ws - refusals}/{ws} ranks built a group over cuda:{i} / cuda:{j}, "
            f"which NVML says is not NVLink-joined ({reason or 'no error raised'})"
        )
    else:
        ok = refusals == 0
        line = (
            f"NVLINK PASS (accepted an NVLink group, {len(devices)} devices)"
            if ok
            else f"NVLINK FAIL: {refusals}/{ws} ranks refused a group NVML says is NVLink-joined "
            f"({reason})"
        )

    if rank == 0:
        print(line, flush=True)
    dist.destroy_process_group()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
