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

```bash
./tools/exclusive.sh 0,1,2,3,4,5,6,7 -- torchrun --nproc_per_node=8 benchmark/bench_a2a.py
./tools/exclusive.sh 0,1,2,3,4,5,6,7 -- torchrun --nproc_per_node=8 benchmark/bench_a2a.py --overlap
./tools/exclusive.sh 0,1,2,3,4,5,6,7 -- torchrun --nproc_per_node=8 benchmark/bench_a2a.py --padding
```

## Machines

| machine | fabric | status |
|---|---|---|
| 8×H200 | NVSwitch | pending |
| 8×A100-SXM4-80GB | NVLink | pending |
| 8×B200 | NVLink | pending |

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

| GPU | shape | copying | zero-copy | saving |
|---|---|---|---|---|
| pending | | | | |

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
| `cuStreamWriteValue64` / `cuStreamWaitValue64` instead of the spin kernel | Concurrent-GEMM overlap fell from +34% to −28%. The waiting form also needs `CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES`, which is 0 on much of the target hardware. |
| Cross-socket staging through pinned host memory | 2× the link bandwidth on an Intel PCIe machine, 10% on an AMD one — but it needs the receiver to pull, so it is a new transport with a second handshake. Not built; see [design.md](design.md). |

## What is not measured

- **End-to-end model impact.** These are microbenchmarks — warm L2, no neighbours competing for
  bandwidth. Removing the padding saves attention work and memory this cannot see.
- **Beyond `world_size = 8`,** or across nodes.
