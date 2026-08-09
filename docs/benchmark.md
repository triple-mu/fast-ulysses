# Benchmarks

Measured 2026-08-08 on a ComputeLab Slurm cluster. Four of the five GPU models are in; **H100 is
still queued** and its rows say `pending`.

## Method

Every number is taken under `tools/exclusive.sh`, which refuses to start until the requested GPUs
are free, samples throughout, and prints `EXCLUSIVE` or `CONTENDED`. Every run below reported
`EXCLUSIVE`. Numbers from different machines are not compared except where the table says so.

- **Whole node, 8 GPUs**, allocated exclusively through Slurm with a GPU health gate.
- **One binary.** A single wheel built for `80;90;100;120` was installed on every machine
  (`sha256 5e4e3747…`), so nothing was recompiled between generations.
- **One image**, `nvcr.io/nvidia/pytorch:26.07-py3` — torch `2.13.0a0+9186a08b2c.nv26.07`, CUDA 13.3.
- **Correctness first.** `pytest test/` passed on each machine before any measurement; a number
  from a build that fails its own tests only looks like data.
- **Two nodes per model.** v0.1 had to withdraw a published row when a second node of the same
  model disagreed by 5×. Where the two nodes agree, one is shown and the spread is noted; where
  they do not, both are shown and nothing is averaged.

bf16, `mode=0`, medians over 25 iterations after 8 warm-up calls, milliseconds. Shapes are the
attention inputs of two real models, QKV packed into one collective so the last dim is
`3 * head_dim`. `s` includes a 227-token text tail, so it does not divide by the group size.

| label | s | heads | 3·head_dim | MB/rank at ws=8 |
|---|---|---|---|---|
| wan-720p | 75827 | 40 | 384 | 291 |
| wan-480p | 32987 | 40 | 384 | 127 |
| h3-t2va-5s | 38051 | 56 | 384 | 205 |

```bash
benchmark/collect.sh <label>          # everything, with the environment recorded next to it
./tools/exclusive.sh 0,1,2,3,4,5,6,7 -- torchrun --nproc_per_node=8 \
    benchmark/bench_a2a.py --mode {stages|zerocopy|sweep|link|overlap|padding}
```

Raw logs, one per (model, node), are archived outside this repo at
`<cluster scratch>/nvidia/fu-bench/results/`. Each carries its own fingerprint header: node,
driver, torch, CUDA, `nvidia-smi topo -m`, SM clocks before and after, and the commit.

## Machines

| machine | fabric | status |
|---|---|---|
| 8×H200 | NVLink | measured, two nodes agree within 3.9% |
| 8×B200 | NVLink | measured, two nodes agree within 7.9% |
| 8×B300 SXM6 | NVLink | measured, two nodes agree within 2.6% |
| 8×RTX PRO 6000 | PCIe, 2 sockets | measured, **two nodes disagree by 5×** — see "Why not PCIe" |
| 8×H100 | NVLink | pending — queued |

8×A100 appears in the v0.1 section and is not re-measured: this cluster has none.

## Where the time goes

`BASE` is `torch.distributed`'s path: permute, `all_to_all_single`, permute. `raw` is
`all_to_all_single` alone — same bytes, no relayout, result in the wrong layout, so it is a
transport floor rather than an alternative. `a2a` against `transfer` is the like-for-like pair.

ws=8. The last column is how far the second node's `OURS` was from the first's.

| GPU | shape | perm_in | a2a | perm_out | BASE | barr_in | transfer | barr_out | copy_out | OURS | vs BASE | raw | CE/raw | relayout% | node 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H200 | wan-720p | 0.358 | 0.826 | 0.385 | **1.569** | 0.011 | 0.680 | 0.025 | 0.143 | **0.860** | **1.82×** | 0.859 | 1.26× | 47.3% | 0.8% |
| H200 | wan-480p | 0.162 | 0.378 | 0.171 | **0.711** | 0.010 | 0.319 | 0.035 | 0.068 | **0.431** | **1.65×** | 0.391 | 1.23× | 46.9% | 3.9% |
| H200 | h3-t2va-5s | 0.254 | 0.590 | 0.273 | **1.117** | 0.010 | 0.490 | 0.019 | 0.100 | **0.619** | **1.80×** | 0.609 | 1.24× | 47.2% | 1.1% |
| B200 | wan-720p | 0.347 | 0.478 | 0.369 | **1.194** | 0.030 | 0.410 | 0.013 | 0.097 | **0.550** | **2.17×** | 0.510 | 1.24× | 60.0% | 1.1% |
| B200 | wan-480p | 0.158 | 0.246 | 0.164 | **0.568** | 0.025 | 0.193 | 0.028 | 0.048 | **0.294** | **1.93×** | 0.263 | 1.36× | 56.7% | 7.1% |
| B200 | h3-t2va-5s | 0.247 | 0.365 | 0.262 | **0.874** | 0.025 | 0.289 | 0.046 | 0.072 | **0.431** | **2.03×** | 0.379 | 1.31× | 58.3% | 7.9% |
| B300 | wan-720p | 0.344 | 0.474 | 0.366 | **1.184** | 0.017 | 0.411 | 0.021 | 0.099 | **0.548** | **2.16×** | 0.493 | 1.20× | 59.9% | 1.3% |
| B300 | wan-480p | 0.156 | 0.227 | 0.163 | **0.546** | 0.011 | 0.199 | 0.044 | 0.048 | **0.302** | **1.81×** | 0.242 | 1.21× | 58.4% | 2.6% |
| B300 | h3-t2va-5s | 0.245 | 0.347 | 0.261 | **0.854** | 0.013 | 0.295 | 0.015 | 0.071 | **0.395** | **2.16×** | 0.360 | 1.22× | 59.3% | 2.0% |
| H100 | | | | | pending | | | | | | | | | | |

Across three NVLink generations the shape is the same: **1.65–2.17× the `torch.distributed` path**,
of which 46.9–60.0% is relayout that costs nothing here, and the transfer alone beats a bare
`all_to_all_single` by 1.20–1.36×.

The node-2 spread is not in the transport. Where it reaches 7–8% (B200's two smaller shapes) the
`transfer` stage agrees to within 2% and the difference is in the barriers — that is rank arrival
skew, which moves between the opening and closing handshake and does not change what was moved.

## How much of the fabric this uses

Flat 64 MiB peer copies, measured through torch symmetric memory rather than through this operator,
so the ceiling is established independently of what is judged against it. The last column is
`transfer` at wan-720p over the bytes that actually cross a link (7/8 of 291 MB), against the
single-flow number.

| GPU | 1 flow | 8 flows, per flow | 8 flows, aggregate | a2a achieves | of the ceiling |
|---|---|---|---|---|---|
| H200 | 378.7 GB/s | 376.0 | 3008 GB/s | 375 GB/s | **99%** |
| B200 | 691.7 GB/s | 689.4 | 5515 GB/s | 621 GB/s | **90%** |
| B300 | 692.1 GB/s | 686.9 | 5495 GB/s | 620 GB/s | **90%** |
| H100 | pending | | | | |

Per-flow bandwidth does not drop when all eight flows run at once, on any of them: the fabric is
not the contended resource. At 90–99% of what a flat copy achieves there is essentially nothing
left to schedule, which is the useful way to read the `transfer` column above.

## The zero-copy path

`out=` from `empty_output()` removes the `copy_out` stage. The saving is close to that stage but
consistently a little under it (0.09–0.12 ms saved against a 0.10–0.14 ms `copy_out`), which is
what you would expect: the two are separate measurements of the whole call, not a subtraction.

The last column is `copy_out` as a share of the four timed stages, which is what the benchmark
prints; it is not `copy_out / copying`.

| GPU | shape | copying | zero-copy | saved | speedup | copy_out, share of the timed stages |
|---|---|---|---|---|---|---|
| H200 | wan-720p | 0.911 | 0.790 | 0.121 | **1.15×** | 16.6% |
| H200 | wan-480p | 0.418 | 0.351 | 0.067 | **1.19×** | 16.1% |
| H200 | h3-t2va-5s | 0.627 | 0.521 | 0.106 | **1.20×** | 16.3% |
| B200 | wan-720p | 0.554 | 0.464 | 0.090 | **1.19×** | 18.4% |
| B200 | wan-480p | 0.267 | 0.224 | 0.043 | **1.19×** | 17.9% |
| B200 | h3-t2va-5s | 0.387 | 0.319 | 0.068 | **1.21×** | 18.3% |
| B300 | wan-720p | 0.567 | 0.475 | 0.093 | **1.20×** | 17.6% |
| B300 | wan-480p | 0.306 | 0.267 | 0.039 | **1.14×** | 15.6% |
| B300 | h3-t2va-5s | 0.388 | 0.320 | 0.068 | **1.21×** | 18.2% |
| H100 | | | | pending | | |

The range is 1.14–1.21×, and the low end is B300's wan-480p, whose `copy_out` share is also the
smallest at 15.6% — the two move together, which is the check that the saving is the copy-out.

There is a second reason to prefer it, which the times above do not show: `copy_out` is a
same-device copy, and a same-device copy competes with compute for SMs where a peer copy does not
(see [design.md](design.md)). The zero-copy path is the one that is actually zero-SM end to end.

## Hiding the collective under compute

`hidden% = (serial − concurrent) / a2a_alone`, against a concurrent 3-GEMM chain shaped like
to_q/k/v. This is the claim the zero-SM design exists to support.

**This metric has real spread.** The two nodes of the same model differ by up to 35 points, and the
benchmark prints the min–max of its own eight alternating samples. Read it as a range.

| GPU | gemm alone | a2a alone | serial | concurrent | hidden (node 1) | hidden (node 2) |
|---|---|---|---|---|---|---|
| H200 | 1.793 | 0.876 | 2.971 | 2.320 | **74%** | 71% |
| B200 | 0.978 | 0.535 | 1.633 | 1.312 | **60%** | 95% |
| B300 | 0.865 | 0.533 | 1.469 | 1.219 | **47%** | 50% |
| H100 | | | | | pending | |

H200 hides more than the Blackwell parts because there is more GEMM to hide under: its
`a2a_alone / gemm_alone` is 0.49 against B200's 0.55 and B300's 0.62. The metric is bounded by how
much compute is available, not only by how well the collective gets out of the way.

## Removing the sequence padding

Rounding a sequence up to a multiple of the group size keeps every rank the same length, which is
what lets the baseline stay on its flat path; the padded tokens then ride through attention and
the collective on every layer of every step. Per-rank `seq_splits` accepts shards differing by one
token instead. The pad is at most `world_size − 1` tokens, so the ratio is the number to read.

| GPU | shape | base padded | base unpadded | base cost | ours padded | ours unpadded | ours cost |
|---|---|---|---|---|---|---|---|
| H200 | wan-720p | 1.565 | 1.666 | 1.06× | 0.869 | 0.871 | **1.00×** |
| H200 | wan-480p | 0.708 | 0.757 | 1.07× | 0.418 | 0.420 | **1.01×** |
| H200 | h3-t2va-5s | 1.118 | 1.189 | 1.06× | 0.649 | 0.652 | **1.00×** |
| B200 | wan-720p | 1.187 | 1.251 | 1.05× | 0.533 | 0.537 | **1.01×** |
| B200 | wan-480p | 0.541 | 0.571 | 1.06× | 0.265 | 0.272 | **1.03×** |
| B200 | h3-t2va-5s | 0.851 | 0.896 | 1.05× | 0.390 | 0.394 | **1.01×** |
| B300 | wan-720p | 1.178 | 1.238 | 1.05× | 0.534 | 0.538 | **1.01×** |
| B300 | wan-480p | 0.538 | 0.569 | 1.06× | 0.267 | 0.273 | **1.02×** |
| B300 | h3-t2va-5s | 0.843 | 0.891 | 1.06× | 0.394 | 0.395 | **1.00×** |
| H100 | | | pending | | | | |

Dropping the pad is free here (1.00–1.03×, every shape on every machine) and costs the baseline
5–7%, because it cannot stay on flat `all_to_all_single` once shards differ at all.

## At world_size 4

The tables above are ws=8 throughout. `collect.sh` also runs ws=4, which had not been published.
One node, four of its eight B200s, certified exclusive by `tools/exclusive.sh` (`VERDICT:
EXCLUSIVE, 2 samples, max 4/4 procs`). torch 2.11 + CUDA 13, one node only, so there is no
second-node column.

| GPU | shape | perm_in | a2a | perm_out | BASE | barr_in | transfer | barr_out | copy_out | OURS | vs BASE | raw | relayout% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B200 | wan-720p | 0.681 | 0.843 | 0.732 | **2.256** | 0.011 | 0.600 | 0.011 | 0.186 | **0.808** | **2.79×** | 0.861 | 62.6% |
| B200 | wan-480p | 0.303 | 0.414 | 0.324 | **1.041** | 0.009 | 0.290 | 0.009 | 0.086 | **0.394** | **2.64×** | 0.429 | 60.3% |
| B200 | h3-t2va-5s | 0.482 | 0.644 | 0.516 | **1.642** | 0.011 | 0.429 | 0.011 | 0.133 | **0.584** | **2.81×** | 0.655 | 60.8% |

The ratio is HIGHER than at ws=8 (2.6-2.8x against 1.9-2.2x), and the reason is the baseline, not
this operator. A permute costs the whole tensor whatever the group size, while the collective
itself moves only `(ws-1)/ws` of it -- 3/4 here against 7/8 at ws=8. So the relayout the baseline
pays and this path does not is a larger share of a smaller total. Read the `relayout%` column, not
the multiplier, when comparing across group sizes.

## Where this stops being the right tool

The barriers cost what they cost regardless of payload, so their share says at which message size
to use something else. Most of that cost is rank arrival skew, which any synchronisation pays
somewhere — read it as a floor, not as something to optimise away. µs, ws=8, 40 heads, d=384.

| GPU | s_local | MB/rank | barr_in | transfer | barr_out | copy_out | total | barriers | GB/s crossed |
|---|---|---|---|---|---|---|---|---|---|
| B200 | 16 | 0.5 | 12.2 | 35.1 | 17.0 | 6.1 | 70.4 | **41.5%** | 12.3 |
| B200 | 256 | 7.9 | 32.3 | 52.1 | 11.1 | 8.2 | 103.6 | **41.9%** | 132.2 |
| B200 | 1024 | 31.5 | 31.5 | 89.0 | 20.4 | 12.3 | 153.1 | **33.9%** | 309.3 |
| B200 | 4096 | 125.8 | 27.6 | 205.0 | 25.6 | 47.1 | 305.2 | **17.4%** | 537.2 |
| B200 | 9478 | 291.2 | 33.5 | 408.4 | 13.2 | 99.3 | 554.4 | **8.4%** | 623.8 |
| B200 | 16384 | 503.3 | 31.7 | 653.5 | 21.3 | 161.8 | 868.3 | **6.1%** | 673.9 |
| H200 | 16 | 0.5 | 28.3 | 29.9 | 13.0 | 4.8 | 76.1 | **54.4%** | 14.4 |
| H200 | 9478 | 291.2 | 13.3 | 689.2 | 31.7 | 142.9 | 877.2 | **5.1%** | 369.7 |
| B300 | 16 | 0.5 | 12.4 | 41.2 | 11.1 | 7.2 | 71.9 | **32.6%** | 10.4 |
| B300 | 9478 | 291.2 | 11.2 | 403.3 | 27.4 | 99.3 | 541.2 | **7.1%** | 631.8 |

The share depends on the machine as much as on the size, so read the column, not a rule of thumb.
At half a megabyte per rank the handshakes are 33–54% of the call; at 31.5 MB they are 19–34%; at
the wan-720p working point (291 MB) they are 5.1–8.4%, and at 503 MB, 2.1–6.4%. B200 carries the
highest share throughout because its `barr_in` sits around 30 µs against 11–13 µs on the other two
— that is arrival skew on that pair of nodes, not a property of the part.

This operator is built for the long-sequence video DiT case, hundreds of MB per rank, which is the
end of that range where the handshakes stop mattering.

## Why not PCIe

RTX PRO 6000 is measured for a different reason: PCIe, GPUs split 4/4 across two CPU sockets, which
is the topology the constructor refuses. Three things came out of it.

**The refusal works.** With the default `require_nvlink=True`, `UlyssesGroup()` raised
`cuda:0 and cuda:1 are not joined by NVLink`, and `doctor` printed an all-`N` matrix — while
`pytest test/` passed 51/51 on the same machine, since the tests construct their groups with the
check off. This is the first confirmation on real hardware that the NVML fix holds: before it,
"this GPU has no NVLink" was read as "NVML cannot say", and the guard stayed inert on exactly the
topology it exists for.

That evidence is in `pro6000-refusal-smc521ge-0080.log`, the run that was *stopped* by the
refusal. The two tables below come from separate runs with `--allow-non-nvlink`, whose logs
therefore contain no refusal at all.

**The cost of ignoring it, with `--allow-non-nvlink`** — and here the two nodes disagree by 5×, so
nothing is averaged:

| node | PCIe layout, socket 0 | BASE | transfer | OURS | vs BASE | raw |
|---|---|---|---|---|---|---|
| smc521ge-0040 | 2+2 | 14.933 | 22.848 | 24.090 | 0.62× | 14.026 |
| smc521ge-0080 | 3+1 | 17.525 | 122.250 | 122.750 | 0.14× | 16.540 |

**The fabric is not what differs.** Flat 64 MiB peer copies give 54.6 and 55.9 GB/s single-flow,
and 52.1 GB/s per flow with all eight running, on *both* nodes. What differs is what the collective
extracts from it: 11.2 GB/s on one node and 2.1 GB/s on the other — 20% and 4% of the same ceiling,
against 90–99% on every NVLink machine.

So the PCIe deficit is not bandwidth. It is that the pitched (strided) copy pattern degrades badly
there, and by an amount that depends on the PCIe switch layout: the fast node has its socket-0 GPUs
in two pairs, the slow one has three on one switch and one alone.

This also revises a v0.1 conclusion. v0.1 measured 119.2 ms on one RTX PRO 6000 node, could not
reproduce it on a second, and withdrew the row as a bad node. It reproduces here at 122.25 ms, with
a second node at 22.85 ms against v0.1's surviving 22.4 ms. **Both modes are real.** v0.1 also
recorded that "a pitched copy runs at 1.00–1.10× a flat one, never slower, on either path" — that
was measured on different PCIe machines and does not hold here, where the gap is 25×.

## Carried over from v0.1

Measured on the NVSHMEM backend, kept as context. A same-machine A/B on 2×H200 put the `transfer`
stage within 0.8% across the backend swap, and the v0.2 numbers above land within 1% of the v0.1
ones on the machines both saw (B200 wan-720p: v0.1 BASE 1.193 / OURS 0.554, v0.2 1.194 / 0.550).

- 8×A100-SXM4-80GB: 1.72–1.87× the `torch.distributed` path, CE/raw 1.12–1.37×.
- The collective hid ~105% under a concurrent GEMM chain on A100 (read >100% as fully hidden: the
  serial arrangement pays a launch cost the concurrent one avoids).

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
- **Beyond `world_size = 8`,** or across nodes. ws=4 is above; ws=2 is not measured.
- **H100**, which was still queued when the rest was taken.
