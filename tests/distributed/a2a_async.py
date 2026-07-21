"""torchrun correctness check for the async a2a API (comm-stream execution + overlap).

    torchrun --nproc_per_node=8 tests/distributed/a2a_async.py

Checks that (1) async results bitwise-match the sync op, (2) compute submitted between launch and
wait() (the overlap window) does not corrupt the result, (3) sync and async calls interleave safely
on the shared comm stream, and (4) in-flight q/k/v async calls with distinct tags stay independent.
Grouped-handshake coverage: (5) a barrier=False/False/True q/k/v group publishes all three
results on both the base and CE paths, and (6) a deep pile of undrained barrier=False CE calls
published by one final barrier=True call (regression for the per-call CE join events).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    pg = dist.group.WORLD
    group = UlyssesGroup(process_group=pg, initial_pool_bytes=1 << 30)

    b, s_local, d = 1, 512, 128
    n_global = 4 * ws
    torch.manual_seed(7 + rank)
    fails = 0

    def check(name, ok):
        nonlocal fails
        fails += not ok
        if rank == 0:
            print(f"{'OK ' if ok else 'FAIL'} ws={ws} {name}", flush=True)
        dist.barrier()

    # 1) async == sync (bitwise), with dummy compute inside the overlap window
    x = torch.randn(b, s_local, n_global, d, device=dev, dtype=torch.bfloat16)
    a = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    ref = group.all_to_all_single_4d(x, mode=0, tag="sync").clone()
    h = group.all_to_all_single_4d_async(x, mode=0, tag="async")
    for _ in range(8):
        a = a @ a  # main-stream compute overlapping the comm stream
    got = h.wait()
    torch.cuda.synchronize()
    check("async == sync (mode0, overlap compute)", torch.equal(got, ref))

    # 2) three in-flight async calls (q/k/v pattern, distinct tags)
    q = torch.randn(b, s_local, n_global, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(b, s_local, n_global, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(b, s_local, n_global, d, device=dev, dtype=torch.bfloat16)
    rq = group.all_to_all_single_4d(q, mode=0, tag="rq").clone()
    rk = group.all_to_all_single_4d(k, mode=0, tag="rk").clone()
    rv = group.all_to_all_single_4d(v, mode=0, tag="rv").clone()
    hq = group.all_to_all_single_4d_async(q, mode=0, tag="aq")
    hk = group.all_to_all_single_4d_async(k, mode=0, tag="ak")
    for _ in range(4):
        a = a @ a
    hv = group.all_to_all_single_4d_async(v, mode=0, tag="av")
    oq, ok_, ov = hq.wait(), hk.wait(), hv.wait()
    torch.cuda.synchronize()
    check(
        "3 in-flight async q/k/v",
        torch.equal(oq, rq) and torch.equal(ok_, rk) and torch.equal(ov, rv),
    )

    # 3) mode1 roundtrip through async: a2a(mode0) then async a2a(mode1) restores the input
    y = group.all_to_all_single_4d(x, mode=0, tag="rt0")
    h1 = group.all_to_all_single_4d_async(y, mode=1, tag="rt1")
    back = h1.wait()
    torch.cuda.synchronize()
    check("mode1 async roundtrip == input", torch.equal(back, x))

    # 4) barrier=False grouping on the base path: q/k defer their handshake, v carries it;
    #    all three results must be published after the barrier-carrying handle's wait().
    hq = group.all_to_all_single_4d_async(q, mode=0, tag="gq", barrier=False)
    hk = group.all_to_all_single_4d_async(k, mode=0, tag="gk", barrier=False)
    hv = group.all_to_all_single_4d_async(v, mode=0, tag="gv", barrier=True)
    oq, ok_, ov = hq.wait(), hk.wait(), hv.wait()
    torch.cuda.synchronize()
    check(
        "barrier=False group (q/k deferred, v publishes)",
        torch.equal(oq, rq) and torch.equal(ok_, rk) and torch.equal(ov, rv),
    )

    # 5) CE grouping: q/k defer, v publishes -- same contract as the base path, and the CE
    #    fan-out's join events must chain correctly across the deferred calls.
    cq = group.all_to_all_single_4d_ce(q, mode=0, tag="cq").clone()
    ck = group.all_to_all_single_4d_ce(k, mode=0, tag="ck").clone()
    cv = group.all_to_all_single_4d_ce(v, mode=0, tag="cv").clone()
    hq = group.all_to_all_single_4d_ce_async(q, mode=0, tag="caq", barrier=False)
    hk = group.all_to_all_single_4d_ce_async(k, mode=0, tag="cak", barrier=False)
    hv = group.all_to_all_single_4d_ce_async(v, mode=0, tag="cav", barrier=True)
    oq, ok_, ov = hq.wait(), hk.wait(), hv.wait()
    torch.cuda.synchronize()
    check(
        "CE barrier=False group (q/k deferred, v publishes)",
        torch.equal(oq, cq) and torch.equal(ok_, ck) and torch.equal(ov, cv),
    )

    # 6) CE deferred deep pile: many undrained barrier=False CE calls before one publishing
    #    call. Regression for the per-call join events -- the old shared CEResources events
    #    deadlocked within a handful of undrained groups when the host ran far ahead of the
    #    device (a pending wait could resolve against a later re-record).
    w = torch.randn(1, 8, ws, 32, device=dev, dtype=torch.bfloat16)
    wref = group.all_to_all_single_4d_ce(w, mode=0, tag="pileref").clone()
    for _ in range(20):
        group.all_to_all_single_4d_ce_async(w, mode=0, tag="pile", barrier=False)
    hw = group.all_to_all_single_4d_ce_async(w, mode=0, tag="pile2", barrier=True)
    got_w = hw.wait()
    torch.cuda.synchronize()
    check("CE deferred deep pile (20 undrained groups)", torch.equal(got_w, wref))

    _ = a  # keep the dummy compute live

    nfail = torch.tensor([fails], device=dev)
    dist.all_reduce(nfail)
    if rank == 0:
        print("ALL PASS" if nfail.item() == 0 else f"FAILED {int(nfail.item())}", flush=True)
    group.destroy()
    dist.destroy_process_group()
    if nfail.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
