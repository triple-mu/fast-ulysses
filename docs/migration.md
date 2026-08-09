# Migration

Coming from DeepSpeed-Ulysses, sglang's `usp.py` or yunchang: what each call becomes here, what your
layout costs, and what this library does not have.

One structural difference first. All three of those ship an **attention module** —
`DistributedAttention`, `UlyssesAttention`, `LongContextAttention` — that owns the q/k/v exchange,
the attention call and the output exchange. fast-ulysses ships **only the collective**. You replace
one module with two calls and your own attention kernel. `examples/dit_attention.py` is that shape
end to end, forward and backward.

Left-hand cells below were read from: DeepSpeed `deepspeed/sequence/layer.py` and
`deepspeed/runtime/sequence_parallel/ulysses_sp.py`; sglang
`python/sglang/multimodal_gen/runtime/layers/usp.py` and
`runtime/distributed/{communication_op.py,device_communicators/base_device_communicator.py}`;
yunchang `yunchang/comm/all_to_all.py`, `yunchang/ulysses/attn_layer.py`, `yunchang/globals.py` and
`yunchang/hybrid/attn_layer.py`. Read on the default branch on 2026-08-10, not from a pinned
release; yunchang's default branch and its latest tag, 0.6.4, are identical in `comm/all_to_all.py`.
Check your installed version before trusting a row.

## The shape of the replacement

```python
# DeepSpeed
from deepspeed.sequence.layer import DistributedAttention

self.dist_attn = DistributedAttention(local_attn, sp_group, scatter_idx=2, gather_idx=0)
out = self.dist_attn(q, k, v, batch_dim_idx)   # 0 -> (b, s, n, d), 1 -> (s, b, n, d)
```

```python
# here -- qkv packed into the last axis, so two collectives stay two collectives
from fast_ulysses import UlyssesGroup

group = UlyssesGroup(process_group=sp_group)   # collective; build once, not per call

# qkv is this rank's shard, (b, s_local, n_global, 3 * head_dim)
qkv = group.all_to_all_4d(qkv, mode=0)   # -> (b, s_global, n_local, 3 * head_dim)
q, k, v = qkv.chunk(3, dim=-1)
out = local_attn(q, k, v)                # your kernel, on whole sequences
out = group.all_to_all_4d(out, mode=1)   # (b, s_global, n_local, hd) -> (b, s_local, n, hd)
```

## From DeepSpeed-Ulysses

| what you write today | what it becomes here |
| --- | --- |
| `DistributedAttention(local_attn, sp_group, scatter_idx=2, gather_idx=0)` | `UlyssesGroup(process_group=sp_group)` — the collective only; you call `local_attn` yourself |
| `dist_attn(q, k, v, batch_dim_idx=0)` | `all_to_all_4d(t, mode=0)` per tensor, or one call on q/k/v packed into a `3 * head_dim` last axis |
| `_SeqAllToAll.apply(pg, x, 2, g, batch_dim_idx)` | `group.all_to_all_4d(x, mode=0)` |
| `_SeqAllToAll.apply(pg, x, g, 2, batch_dim_idx)` — the output call, indices swapped, `g` in `{0, 1}` | `group.all_to_all_4d(x, mode=1)` |
| three `_SeqAllToAll.apply` calls, one each for q, k, v | one call on a `3 * head_dim` last axis: one handshake instead of three, same bytes |
| `batch_dim_idx=1`, i.e. `(s, b, n, d)` | not accepted — transpose first, see [Layouts](#layouts) |
| `uneven_heads_all2all`, sizes derived from `get_num_kv_heads()`, `assert async_op == False` | `head_splits=[...]` that you pass, on the sync **and** the async call |
| `UlyssesSPAttentionHF` (ALST), strictly `(sl, bs, hc, hs)` | not accepted — transpose in and back out, see [Layouts](#layouts) |
| the `_SeqAllToAll` `torch.autograd.Function` wrapper | nothing: `all_to_all_4d` is differentiable |
| `UlyssesSPDataLoaderAdapter`, `register_with_transformers()`, `UlyssesSPFwdLossBwdWithLogits` | no counterpart — this is a collective, not a training integration |

`DistributedAttention.forward`'s fourth parameter is `batch_dim_idx`, positional and required. If
your call site passes an attention mask there — the DeepSpeed tutorial snippet and
Megatron-DeepSpeed's non-flash branch both do — the layout selection is reading a tensor, and you
should settle what that call actually did before porting it.

## From sglang `usp.py`

| what you write today | what it becomes here |
| --- | --- |
| `_usp_input_all_to_all(x, head_dim=2)` | `group.all_to_all_4d(x, mode=0)` |
| `_usp_output_all_to_all(x, head_dim=2)` | `group.all_to_all_4d(x, mode=1)` |
| `_usp_input_all_to_all(x, head_dim=1)`, i.e. `(b, n, s, d)` | not accepted — transpose first, see [Layouts](#layouts) |
| `_usp_input_all_to_all_qkv` (4D q/k/v) / `_usp_input_all_to_all_packed_qkv` (3D q/k/v) — both pack the three into one collective | one `all_to_all_4d` on `(b, s_local, n_global, 3 * head_dim)`, which is the same packing expressed as a shape |
| `_usp_input_all_to_all_varlen(x, seq_lens, head_dim=2)` and its output form | `seq_splits=seq_lens, head_splits=[n_global // ws] * ws` on the ordinary call; there is no separate varlen entry point |
| `sequence_model_parallel_all_to_all_4D(x, scatter_dim=2, gather_dim=1)` | `group.all_to_all_4d(x, mode=0)` |
| `DistributedAutograd.AllToAll4D.apply(...)` | nothing: `all_to_all_4d` is differentiable |
| `_a2a_staging_buffer(role, shape, dtype, device)`, bypassed when `torch.is_grad_enabled()` | nothing to pass or plumb: windows are cached internally per dtype and stream, with grad on or off |
| `_ipc_all_to_all_4d`, gated to `world_size == 2` | the copy-engine path is the only path, at every `world_size` in `[1, 8]` |

sglang has two implementations of this collective — `usp.py`'s plain functions, with no gradient
support, and `DistributedAutograd.AllToAll4D`, which is differentiable and reaches the same result
through different permutes. Both map to the same two calls here.

## From yunchang / long-context-attention

| what you write today | what it becomes here |
| --- | --- |
| `set_seq_parallel_pg(...)` writing the module-level `PROCESS_GROUP.ULYSSES_PG` | `UlyssesGroup(process_group=sp_pg)` per instance; no global state |
| `UlyssesAttention(sp_pg, scatter_idx=2, gather_idx=1, attn_type=AttnType.FA)` | `UlyssesGroup(process_group=sp_pg)` plus your own attention call — there is no `attn_type` |
| `SeqAllToAll4D.apply(pg, x, 2, 1)` | `group.all_to_all_4d(x, mode=0)` |
| `SeqAllToAll4D.apply(pg, x, 1, 2)` | `group.all_to_all_4d(x, mode=1)` |
| `all_to_all_4D(x, 2, 1, group=pg)` — the plain function, no grad | `group.all_to_all_4d(x, mode=0)`, which does carry grad |
| `all_to_all_5D(x, scatter_idx=3, ...)` on `(bs, s, 3, hc, hs)` | no 5D entry point — fold the qkv axis, see [Layouts](#layouts) |
| `use_pack_qkv=True`, `torch.cat([q, k, v])` on the batch axis — the branch then calls `.continous()`, which is not a tensor method, so it raises `AttributeError` before the collective | that cat works unchanged (`b` becomes `3 * b`); packing into `3 * head_dim` is the same one handshake |
| `use_sync=True` | no counterpart; nothing here inserts a `torch.cuda.synchronize()` |
| `LongContextAttention(..., ring_impl_type=...)`, ulysses × ring | no counterpart. Ulysses only |
| a head count that does not divide `world_size` (fails at yunchang's `reshape`) | `head_splits=[...]` |

yunchang enforces the index pair — only `(2, 1)` and `(1, 2)` are implemented, anything else raises —
so its two directions map one-to-one onto the two modes.

## scatter_idx / gather_idx, and mode

`mode=0` scatters heads and gathers sequence: `(b, s_local, n_global, d) -> (b, s_global, n_local, d)`.
`mode=1` is the inverse.

| what selects the direction today | here |
| --- | --- |
| DeepSpeed `scatter_idx = 2` | `mode=0` |
| DeepSpeed `scatter_idx` in `{0, 1}` | `mode=1` |
| yunchang `scatter_idx=2, gather_idx=1` | `mode=0` |
| yunchang `scatter_idx=1, gather_idx=2` | `mode=1` |
| sglang `_usp_input_all_to_all(x, head_dim=2)` | `mode=0` |
| sglang `_usp_output_all_to_all(x, head_dim=2)` | `mode=1` |
| sglang `scatter_dim=2, gather_dim=1` | `mode=0` |

Two traps in the left column:

- **DeepSpeed's `gather_idx` does not select the layout on the even path.** It is never passed to
  `_generate_layout_params`, and in `single_all_to_all` it is forwarded only to
  `uneven_heads_all2all`; the direction is chosen by the predicate `scatter_idx < 2` alone.
  `gather_idx` matters only because the output call swaps it into `scatter_idx`. Map by
  `scatter_idx`, not by the argument names.
- **yunchang's `all_to_all_4D` docstring contradicts its own signature** (it says the defaults are
  `scatter_idx=1, gather_idx=2`; the signature is `2, 1`). Read the signature.

## Layouts

This library takes **4D `(b, s, n, d)` and nothing else.** No 3D form, no 5D form, no
`batch_dim_idx`, no `head_dim` argument. That is a real constraint and it is where a migration
costs something.

| the layout you have | what it costs here |
| --- | --- |
| `(b, s, n, d)` — sglang `head_dim=2`, yunchang, DeepSpeed `batch_dim_idx=0` | nothing; this is the input |
| `(3, s, n, d)` — a q/k/v stack whose dim 0 is not batch | nothing; `b = 3`, and q, k, v are exchanged independently |
| `(s, n * d)` contiguous 3D | `x.view(1, s, n, d)` — a view, no copy |
| `(b, n, s, d)` — sglang `head_dim=1`, and sglang's default | `x.transpose(1, 2)` in, and back out after the call. Each of those is the size of one of the two permutes this library removes: half the saving goes back if your attention kernel takes the output as a strided view, all of it if it wants contiguous both ways |
| `(s, b, n, d)` — Megatron native, DeepSpeed `batch_dim_idx=1`, ALST `UlyssesSPAttentionHF` | `x.transpose(0, 1)` in and back out. Megatron's flash path already pays exactly this (`rearrange("s b ... -> b s ...").contiguous()`) before DeepSpeed's own permutes, so for that caller it is a wash; for ALST, which stays sequence-first end to end, it is a new cost |
| `(b, s, 3, n, d)` — yunchang's 5D | fold the qkv axis to `(b, s, n, 3 * d)`, one permute, or emit the projection in that order upstream as `examples/dit_attention.py` does |

`.contiguous()` is applied internally, so writing it yourself changes nothing — but a transposed
input is copied either way, by you or by the op.

The 5D fold has a shortcut that looks right and is not. `(b, s, 3, n, d)` reshapes to
`(b, s, 3 * n, d)` for free, but an even split then cuts that merged axis into contiguous chunks, so
a rank receives a mixture of q, k and v heads rather than the same head slice of each.

## Autograd: there is no wrapper to port

DeepSpeed's `_SeqAllToAll` and yunchang's `SeqAllToAll4D` are `torch.autograd.Function`s whose
`backward` re-applies the forward with `scatter_idx` and `gather_idx` swapped. `all_to_all_4d` is
differentiable in C++, so the wrapper is deleted, not translated: the backward is the other `mode`.

The convention differs where it matters. Those libraries express the backward as *swap the two
indices*; here it is *the other mode with the **same** splits*. Identical while shards are even.
Not identical once they are not: `seq_splits` / `head_splits` describe the group's geometry, which
holds whichever way the data moves, unlike `all_to_all_single`'s split sizes, which are one call's
send and receive counts and do swap. See [api.md](api.md).

`examples/dit_attention.py` is the worked case — one DiT attention block, sequence-parallel, an
ordinary `nn.Module`, checked for both output and input gradient against the same block with no
sequence parallelism, then run through five optimiser steps:

```bash
torchrun --nproc_per_node=8 examples/dit_attention.py
```

The **async** form is the exception: `all_to_all_4d_async` is not differentiable and raises on a
grad-requiring input rather than dropping the gradient silently.

## Uneven shards, which is the reason to move

| library | uneven support |
| --- | --- |
| DeepSpeed | heads only, via `uneven_heads_all2all`, with sizes derived from `get_num_kv_heads()` rather than passed by the caller; `assert async_op == False`. No uneven sequence |
| sglang `usp.py` | sequence only. `_usp_*_varlen` take `seq_lens`, the group's per-rank sequence shards with `s_local == seq_lens[rank]` asserted, which is what `seq_splits` is. Heads still have to divide: `assert h_global % world_size == 0` |
| yunchang | none. `hc // world_size` then a `reshape`, so a non-divisible head count fails at the reshape with no explicit check |
| here | `seq_splits` / `head_splits`, both or neither, identical on every rank, on the sync and async calls alike. Even splits are the special case of the same code path, not a fast path beside it |

What that buys: sharding a sequence normally means rounding it up to a multiple of the group size,
and those padded tokens then ride through attention and both collectives on every layer of every
step. Dropping the pad costs the baseline **5–11%** and costs this path **1.00–1.03×** — every
shape, every machine. Table in [benchmark.md](benchmark.md), "Removing the sequence padding".

## The measured baseline is the path you are leaving

The `BASE` column in [benchmark.md](benchmark.md) is not a strawman built for the comparison. It is
sglang's `_usp_input_all_to_all(x, head_dim=2)` restated permute for permute —
`benchmark/bench_a2a.py:baseline_stages`:

```python
y = x.permute(2, 0, 1, 3).contiguous().flatten()
dist.all_to_all_single(recv, y, group=pg)
recv.reshape(ws, h_local, b, s_local, d).permute(2, 0, 3, 1, 4).contiguous().reshape(...)
```

`test/distributed/correctness.py:reference_even` holds this operator **bit-exact** against that same
path, so the two are checked to be the same function before either is timed. The **1.60–2.17×** in
that document, and the per-machine multipliers in the README, are therefore measured against exactly
what a sglang or DeepSpeed caller runs today, and the `relayout%` column — **46.9–60.0%** of the
baseline — is the two permute kernels that do not exist on this side.

Two qualifications, so the number is read for what it is:

- The benchmark measures `mode=0`. Upstream sglang's `mode=1` equivalent, `_usp_output_all_to_all`
  at `head_dim=2`, already replaces its second permute with a fused CUDA kernel — `usp_merge_heads`
  from `sglang.kernels.ops.diffusion.usp_relayout`, JIT-built from `diffusion/usp_relayout.cuh`,
  with `x.permute(2, 1, 0, 3, 4).contiguous()` as its own fallback. Its cost was not measured here,
  so against *current* sglang the saving on the mode-1 half is smaller than the table implies, by an
  unknown amount. Mode 0's two permutes are real on both.
- The multiplier moves with world size, for the baseline's reason rather than this operator's: a
  permute costs the whole tensor at any group size while the collective moves `(ws-1)/ws` of it, so
  at ws=4 the same relayout is a larger share of a smaller total (2.6–2.8× on B200, 1.7–1.9× on
  H100). Compare `relayout%` across group sizes, not the multiplier.

## What this does not have

Check this list before you start, not after.

- **Cross-node.** One node. Nothing routes over a NIC.
- **PCIe.** `UlyssesGroup()` refuses a group whose GPUs are not NVLink-joined; `require_nvlink=False`
  exists to measure that case, not to run in it. Across a CPU socket this path measured 0.62× and
  0.14× of `torch.distributed` on two PCIe nodes — if your group is not NVLink-joined, stay where
  you are.
- **`world_size > 8`.** Structural: `BarPeers::p[8]` in `src/barrier.cu`.
- **Ring attention and the ulysses × ring hybrid.** `LongContextAttention` has no counterpart.
- **An attention kernel.** No `attn_type`, no kernel selection, no q/k/v plumbing — you call your own.
- **`torch.compile` tracing.** A meta kernel propagates shapes under `FakeTensor` and AOTAutograd,
  but a module holding an `UlyssesGroup` graph-breaks: the group is a torchbind object with no
  registered fake class.
- **Autograd on the async form**, and **3D or 5D entry points**.
- **dtypes** beyond `float16` / `bfloat16` / `float32` / `float8_e4m3fn` / `float8_e5m2` / `int8` /
  `uint8`, with `d * elem_size` 16-byte aligned.

## After the switch

- **Construction, `empty_output()` and `destroy()` are collective**, and **every rank must issue the
  same sequence of shapes and dtypes** — a new shape allocates a window collectively. Violating it
  hangs; nothing raises and nothing times out. This is the same discipline the collectives you are
  leaving already require, applied to allocation as well as to the call.
- Build the group **once** and keep it. It holds the windows, the plan cache and the comm stream.
- Pass `out=` from `empty_output()` to remove the copy-out, worth a further 1.14–1.28×. Allocate the
  buffer outside the loop; it is collective. See [quickstart.md](quickstart.md).
