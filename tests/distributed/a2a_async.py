"""torchrun correctness check for the async a2a API (comm-stream execution + overlap).

    torchrun --nproc_per_node=8 tests/distributed/a2a_async.py

Checks that (1) async results bitwise-match the sync op, (2) compute submitted between launch and
wait() (the overlap window) does not corrupt the result, (3) sync and async calls interleave safely
on the shared comm stream, and (4) in-flight q/k/v async calls with distinct tags stay independent.
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
