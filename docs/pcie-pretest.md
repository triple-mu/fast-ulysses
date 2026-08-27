# PCIe packed-flat pretest

`benchmark/bench_a2a.py --mode pcie-pretest` validates the experimental production PCIe transport
and keeps its earlier layout-only prototype beside it for attribution.

The current copy-engine transport folds the Ulysses relayout into pitched peer copies. That is the
right trade on NVLink, but the pitch collapses effective bandwidth on PCIe. The candidate PCIe
mode pays for one local contiguous pack, then sends one flat chunk to each peer:

```text
[b, s_local, h_global, d]
          local pack by destination
                    |
                    v
[peer, b, s_local, h_local, d] --flat P2P--> peer's final sequence slice
```

The production backend supports mode 0 and mode 1, batch 1, even shards, and synchronous inference.
Mode 1 sends contiguous sequence slices to sender-major receiver staging, then performs one local
unpack. Select it with `UlyssesGroup(require_nvlink=False, backend="packed")`.

The layout-only rows remain deliberately limited to mode 0. They check the complete output
bit-for-bit but have no production cross-rank GPU barrier; the report separately measures those
costs. Rows prefixed `PRODUCTION` call the integrated backend and include its real barriers and
local output work.

## B300: validate the design, not PCIe P2P

B300's GPU peer copies use NVLink/NVSwitch. They cannot be forced through PCIe by choosing a CUDA
stream or copy API. A B300 run therefore validates:

- pack layout and exact output correctness;
- multi-peer ordering and copy-engine scheduling;
- local pack and owned-output costs;
- host PCIe D2H/H2D bandwidth through the pinned-host probes.

It does **not** predict direct GPU-to-GPU PCIe bandwidth. Build for B300 (`sm_103`) and run:

```bash
FAST_ULYSSES_CUDA_ARCH=103 pip install -e . --no-build-isolation

./tools/exclusive.sh 0,1,2,3 -- \
  torchrun --standalone --nproc_per_node=4 benchmark/bench_a2a.py \
  --mode pcie-pretest --iters 25 --warmup 8 --host-mib 64
```

On the local 4xB300 test node, Wan 720p mode 0 produced the following slowest-rank medians on
2026-08-25. These are a regression reference, not a PCIe claim:

| path | time | versus BASE |
| --- | ---: | ---: |
| BASE permute + NCCL + permute | 2.179 ms | 1.00x |
| raw NCCL, prepared layout | 0.833 ms | 2.61x |
| local pack | 0.655 ms | 3.33x |
| flat peer copies, no barrier | 0.832 ms | 2.62x |
| pack + flat peer, no barrier | 1.378 ms | 1.58x |
| PRODUCTION mode 0, zero-copy | 1.284 ms | 1.70x |
| PRODUCTION mode 0, owned output | 1.456 ms | 1.50x |
| PRODUCTION mode 1, owned output | 1.279 ms | 1.70x |

The existing pitched NVLink operator was 0.804 ms on the same four GPUs, so packed-flat must be a
PCIe-specific backend. It must not replace the default NVLink path.

The same run measured pinned-host copies at about 47.7 GB/s D2H and 55.9 GB/s H2D. The benchmark
prints a full-message ideal-duplex projection, but that projection excludes PCIe-switch and root
complex contention, synchronization, CPU-socket routing, and NUMA placement.

The benchmark also runs a real all-rank GPU-to-host-to-next-GPU data path with two pinned buffers.
A later 4xB300 run measured 12.899 ms / 33.9 GB/s per rank for that path, versus a 9.520 ms
single-rank bandwidth projection. It is intentionally not a collective yet: the destination is
scratch owned by the sender process and there is no cross-rank completion handshake. Its purpose
is to reject an uncompetitive host route before implementing shared buffers and synchronization.

## RTX PRO 5000/6000: validate real PCIe P2P

First record topology and P2P capability, then run the same shape:

```bash
nvidia-smi topo -m

./tools/exclusive.sh 0,1,2,3 -- \
  torchrun --standalone --nproc_per_node=4 benchmark/bench_a2a.py \
  --mode pcie-pretest --allow-non-nvlink --iters 25 --warmup 8 --host-mib 64
```

Run two-card groups within one PCIe switch/root complex before crossing CPU sockets. Keep the
shape, dtype, rank count, warmup, and iteration count identical between comparisons.

Use these gates before enabling the production path in a model:

1. `flat peer copies` must recover most of the bandwidth shown by `--mode link`; otherwise peer
   order or topology routing is still the bottleneck.
2. `PRODUCTION packed mode0 owned` should beat BASE by at least 10%. A smaller margin is unlikely to
   survive integration overhead and model-level variance.
3. Prefer direct packed P2P when it passes. Consider pinned-host staging only when P2P is disabled,
   ACS/IOMMU routing is pathological, or a cross-socket group makes direct P2P slower than the
   reported host-path projection.
4. After the microbenchmark passes, profile an actual Q/V/K pipeline. The current packed backend
   is synchronous; QKV async overlap is the next implementation step after real PCIe P2P passes.

Selection is currently explicit: retain `backend="pitched"` on NVLink, use `backend="packed"` on
favorable PCIe P2P, and keep NCCL as the fallback until host staging proves faster on the target
topology.

For MiniMax H3, use the TP-aware separate-QKV block benchmark and the one-command RTX PRO 5000
runner in [h3-packing-test-plan.md](h3-packing-test-plan.md). The older `h3-t2va-5s` decomposition
row fuses QKV into a `d=384` tensor; `--mode h3-block` instead matches vLLM-Omni's three mode-0
calls plus one mode-1 call and can model TP2 x Ulysses2 contention.
