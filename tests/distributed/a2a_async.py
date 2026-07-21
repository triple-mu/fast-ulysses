"""torchrun correctness check for the async a2a API (comm-stream execution + overlap).

    torchrun --nproc_per_node=8 tests/distributed/a2a_async.py

Checks that (1) async results bitwise-match the sync op, (2) compute submitted between launch and
wait() (the overlap window) does not corrupt the result, (3) sync and async calls interleave safely
on the shared comm stream, and (4) in-flight q/k/v async calls with distinct tags stay independent.
Grouped-handshake coverage: (5) signal_wait without a prior arrive raises on every rank alike,
(6) a barrier=False/False/True q/k/v group publishes all three results, (7) a CE barrier=False
group published by signal_arrive_async/signal_wait alone (no fast_barrier), and (8) an epoch-wrap
loop (>256 arrive/wait pairs, crossing the low-byte wrap and the skip-zero branch).
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

    # 4) signal_wait BEFORE any signal_arrive must raise -- identically on every rank, no
    #    hang. MUST run before any arrive: csig_ready_ latches on the first arrive and the
    #    error path is unreachable afterwards. ws==1 signal ops are no-ops, so skip there.
    if ws > 1:
        try:
            group.signal_wait()
            raised = False
        except RuntimeError:
            raised = True
        check("signal_wait without arrive raises", raised)

    # 5) barrier=False grouping on the base path: q/k defer their handshake, v carries it;
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

    # 6) CE group published by the consumer signal alone: three barrier=False CE copies,
    #    one CE-written arrive on the comm stream, one poll kernel on the main stream --
    #    the whole group crosses ranks without a single fast_barrier.
    cq = group.all_to_all_single_4d_ce(q, mode=0, tag="cq").clone()
    ck = group.all_to_all_single_4d_ce(k, mode=0, tag="ck").clone()
    cv = group.all_to_all_single_4d_ce(v, mode=0, tag="cv").clone()
    hq = group.all_to_all_single_4d_ce_async(q, mode=0, tag="caq", barrier=False)
    hk = group.all_to_all_single_4d_ce_async(k, mode=0, tag="cak", barrier=False)
    hv = group.all_to_all_single_4d_ce_async(v, mode=0, tag="cav", barrier=False)
    group.signal_arrive_async()
    group.signal_wait()
    oq, ok_, ov = hq.wait(), hk.wait(), hv.wait()
    torch.cuda.synchronize()
    check(
        "CE barrier=False group + signal arrive/wait",
        torch.equal(oq, cq) and torch.equal(ok_, ck) and torch.equal(ov, cv),
    )

    # 7) epoch wrap: >256 arrive/wait pairs cross the low-byte wrap (and the skip-zero
    #    branch). Small tensor; verify every 50 rounds and at the end.
    w = torch.randn(1, 8, ws, 32, device=dev, dtype=torch.bfloat16)
    wref = group.all_to_all_single_4d_ce(w, mode=0, tag="wrapref").clone()
    wrap_ok = True
    for i in range(300):
        hw = group.all_to_all_single_4d_ce_async(w, mode=0, tag="wrap", barrier=False)
        group.signal_arrive_async()
        group.signal_wait()
        if i % 50 == 49 or i == 299:
            got_w = hw.wait()
            torch.cuda.synchronize()
            wrap_ok = wrap_ok and torch.equal(got_w, wref)
            dist.barrier()  # keep peers' next-round writes from racing this round's read
    check("signal epoch wrap (300 rounds)", wrap_ok)

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
