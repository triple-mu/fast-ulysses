# Benchmarks

**Status: `pending`.** The v0.2 numbers will be taken in one pass, on the machines and at the time
the maintainer schedules; the tables below are the shape they will take. Everything needed to
reproduce them is here.

## Method

Every number must be taken under `tools/exclusive.sh`, which refuses to start until the requested
GPUs are free, binds the run to them, samples throughout, and prints `EXCLUSIVE` or `CONTENDED`. A
`CONTENDED` run is not a result. Numbers from different machines are not compared.

bf16, `mode=0`, medians over 25 iterations after 8 warm-up calls, milliseconds.

Shapes are the attention inputs of two real models, QKV packed into one collective so the last dim
is `3 * head_dim`. `s` includes a 227-token text tail, so it does not divide by the group size.

| label | s | heads | 3·head_dim |
|---|---|---|---|
| wan-720p | 75827 | 40 | 384 |
| wan-480p | 32987 | 40 | 384 |
| h3-t2va-5s | 38051 | 56 | 384 |

One machine, everything, with the environment recorded next to the numbers:

```bash
benchmark/collect.sh <label>          # e.g. b200-node1; writes benchmark-results/<label>.log
```

It gates on `pytest test/` -- a number from a build that fails its own tests only looks like data
-- then runs every mode through `tools/exclusive.sh`, and refuses to call the run complete if any
mode failed. Individually:

```bash
./tools/exclusive.sh 0,1,2,3,4,5,6,7 -- torchrun --nproc_per_node=8 \
    benchmark/bench_a2a.py --mode {stages|zerocopy|sweep|link|overlap|padding}
```

## Machines

| machine | fabric | status |
|---|---|---|
| 8×H100 | NVLink | pending |
| 8×H200 | NVSwitch | pending |
| 8×B200 | NVLink | pending |
| 8×B300 | NVLink | pending |
| 8×RTX PRO 6000 | PCIe, 2 sockets | pending — a different question; see "Why not PCIe" |

8×A100 appears in the v0.1 section below and is not re-measured here: the cluster these numbers
come from has no A100.

## Where the time goes

`BASE` is `torch.distributed`'s path: permute, `all_to_all_single`, permute. `raw` is
`all_to_all_single` alone — same bytes, no relayout, result in the wrong layout, so it is a
transport floor rather than an alternative. `a2a` against `transfer` is the like-for-like pair.

| GPU | shape | perm_in | a2a | perm_out | BASE | barr_in | transfer | barr_out | copy_out | OURS | raw | CE/raw | relayout% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pending | | | | | | | | | | | | | |

## Hiding the collective under compute

`hidden% = (serial − concurrent) / a2a_alone`, against a concurrent 3-GEMM chain shaped like
to_q/k/v. This is the claim the zero-SM design exists to support. Read >100% as "fully hidden": the
metric can exceed 100 because the serial arrangement pays a launch cost the concurrent one avoids.

| GPU | gemm alone | a2a alone | hidden |
|---|---|---|---|
| pending | | | |

## Removing the sequence padding

Rounding a sequence up to a multiple of the group size keeps every rank the same length, which is
what lets the baseline stay on its flat path; the padded tokens then ride through attention and the
collective on every layer of every step. Per-rank `seq_splits` accepts shards differing by one token
instead. The pad is at most `world_size − 1` tokens, so the ratio is the number to read, not the
absolute times.

| GPU | shape | base padded | base unpadded | base cost | ours padded | ours unpadded | ours cost |
|---|---|---|---|---|---|---|---|
| pending | | | | | | | |

## The zero-copy path

`out=` from `empty_output()` removes the `copy_out` stage. Its share of the copying call is the
number to look at before reaching for it.

| GPU | shape | copying | zero-copy | saving | copy_out share of copying |
|---|---|---|---|---|---|
| pending | | | | | |

## Where this stops being the right tool

The barriers cost what they cost regardless of payload, so their share is what says at which
message size to use something else. Most of that cost is rank arrival skew, which any
synchronisation pays somewhere — read it as a floor, not as something to optimise away.

| GPU | s_local | MB/rank | barr_in | transfer | barr_out | copy_out | barriers % | GB/s crossed |
|---|---|---|---|---|---|---|---|---|
| pending | | | | | | | | |

## What the fabric can do

Flat 64 MiB peer copies, measured through torch symmetric memory rather than through this
operator, so the ceiling is established independently of what is judged against it. `transfer`
over the bytes that actually cross a link, against the per-flow number, is how much of the fabric
the collective is using.

| GPU | flows | ms | GB/s per flow | GB/s aggregate |
|---|---|---|---|---|
| pending | | | | |

## Why not PCIe

RTX PRO 6000 is measured for a different reason: it is PCIe with the GPUs split across two CPU
sockets, which is the topology the constructor refuses. Three things are recorded there — that
`require_nvlink=True` does refuse it and names the pair, what the operator does anyway with
`require_nvlink=False`, and the same-socket versus cross-socket contrast that
[design.md](design.md) rests on.

| measurement | result |
|---|---|
| pending | |

## Carried over from v0.1

These were measured on the NVSHMEM backend. The transfer path is unchanged — the same pitched
copies over the same kind of VMM-mapped peer memory — and a same-machine A/B on 2×H200 put the
`transfer` stage within 0.8% (wan-720p 1.501 → 1.499 ms, wan-480p 0.664 → 0.659, h3 1.061 → 1.061).
They are recorded as context, not as v0.2 results.

- Across A100, H200 and B200 at 8 ranks: **1.7–2.2× the `torch.distributed` path**. 47–60% of the
  baseline is relayout that costs nothing here, and the transfer alone beat a bare
  `all_to_all_single` by 1.12–1.37×.
- The collective hid essentially completely under a concurrent GEMM chain: 86% on B200, ~105% on
  A100.
- Dropping the sequence padding was free (1.00×); the baseline paid 5–8% for the same change.

## Alternatives tried and not adopted

| alternative | result |
|---|---|
| One stream per peer, and everything on one stream | Both slower than remote-serialised with the own share on the caller's stream: 2.273 / 1.345 / **1.175** ms at wan-720p, 4×H200. |
| Sequential peer order instead of XOR-shift | XOR-shift pairs ranks without coordination; measured at ~14% in a sibling implementation. |
| `cudaMemcpy3DBatchAsync` | 0.82 ms, and 1.35 ms with `cudaMemcpyFlagPreferOverlapWithCompute`, against the plain `cudaMemcpy3DAsync` used instead. Also rejects the legacy default stream. |
| Fusing single-row copies into 3D | 0.67 → 2.24 ms at `b=2`. Fusion is applied only to multi-row copies for this reason. |
| Contiguous per-sender segments instead of strided writes | ~9% below contiguous in a sibling implementation, and not available here regardless: the window is the returned tensor, so there is no local pass to interleave afterwards. |
| `cuStreamWriteValue64` / `cuStreamWaitValue64` instead of the spin kernel | Concurrent-GEMM overlap fell from +34% to −28%. See [design.md](design.md) for why `CAN_FLUSH_REMOTE_WRITES`, which an earlier note blamed, is not the reason. |
| Publishing the closing flag as a copy-engine write on the transfer stream (NCCL's kernel-less shape) | Measured 3× slower than the barrier kernel it would replace (7× 8 B peer copies ≈ 16 µs against 4–6 µs), and it only moves the publish — the wait stays in a kernel. |
| Double-buffering the window to drop the opening barrier | Sound, but worth ~0.7% at the design point: it only helps the copying path, costs 2× window memory, and invalidates the premise of the one adversarial test. |
| Cross-socket staging through pinned host memory | 2× the link bandwidth on an Intel PCIe machine, 10% on an AMD one — but it needs the receiver to pull, so it is a new transport with a second handshake. Not built; see [design.md](design.md). |

## What is not measured

- **End-to-end model impact.** These are microbenchmarks — warm L2, no neighbours competing for
  bandwidth. Removing the padding saves attention work and memory this cannot see.
- **Beyond `world_size = 8`,** or across nodes.
