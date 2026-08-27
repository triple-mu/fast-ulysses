# API

Shapes: `b` batch, `d` head dim, `ws` world size. `s_local` / `n_local` are this rank's sequence
and head shard; `s_global = sum(seq_splits)`, `n_global = sum(head_splits)`.

## Collective contract

Violating these hangs the group. Nothing raises and nothing times out.

- **Every rank must issue the same sequence of shapes.** A window is allocated on the first call
  that needs one, and that allocation is collective.
- **Construction, `empty_output()` and `destroy()` are collective** over `process_group`.
- **Every rank must issue the same sequence of dtypes**, for the same reason as shapes: a window is
  allocated per `(role, dtype)`.
- **With autograd, every rank's engine must reach the backward node in the same order.** In
  sequence parallelism it does, because every rank runs the same graph — but it is a new way to
  hang, and the graph holds the group alive, so a graph replayed after `destroy()` raises.
- **One buffer, one call at a time.** A call overwrites the buffer it writes into, so two
  concurrent calls need two `empty_output()` buffers — or `all_to_all_4d_async(lend=True)`, which
  keeps the buffers itself and rotates through them.

## `UlyssesGroup(process_group=None, device=None, *, require_nvlink=True, backend="pitched")`

| Parameter | Meaning |
| --- | --- |
| `process_group` | The group to run over; `None` uses `dist.group.WORLD`. Any subgroup works — there is no restriction on which ranks it contains. |
| `device` | This rank's CUDA device; `None` uses the current device. |
| `require_nvlink` | Refuse a group whose GPUs are not all NVLink-joined. Set `False` when testing the packed PCIe backend. |
| `backend` | `"pitched"` is the NVLink-optimised default. `"packed"` locally packs/unpacks around flat peer copies for PCIe. |

The NVLink check reads the link type from NVML. Both a direct GPU-to-GPU link and two GPUs on one
NVSwitch fabric count. The check **fails open**: when NVML cannot answer — it is missing, or one
device cannot be probed — the group is built with no refusal and no warning, so a PCIe group on
such a machine runs, slowly, instead of being caught here. It is also skipped when two ranks share
a GPU, which is not a topology it can say anything about.

## `fast_ulysses.nvlink_matrix(devices) -> dict[(int, int), bool] | None`

The same probe, on its own: for a list of CUDA device indices, which ordered pairs are NVLink-joined.
`None` when NVML cannot answer at all. Not collective, and it takes no group — `fast-ulysses doctor`
prints it as a matrix.

## `all_to_all_4d(x, *, mode=0, out=None, seq_splits=None, head_splits=None) -> Tensor`

| mode | input | output |
| --- | --- | --- |
| 0 — scatter heads, gather sequence | `(b, s_local, n_global, d)` | `(b, s_global, n_local, d)` |
| 1 — the inverse | `(b, s_global, n_local, d)` | `(b, s_local, n_global, d)` |

`x` is 4D CUDA, one of `float16` / `bfloat16` / `float32` / `float8_e4m3fn` / `float8_e5m2` /
`int8` / `uint8`, with `d * elem_size` 16-byte aligned; `.contiguous()` is applied internally.
Note the alignment rule tightens as the element shrinks: `d % 8` at float16, `d % 4` at float32,
`d % 16` at float8 and int8. Nothing below the check is dtype-specific — the transport moves bytes.

The default **pitched backend is differentiable, except with `out=`.** The vjp of a permutation is the inverse permutation, so the
backward is the other `mode` with the **same** splits. `all_to_all_single`'s backward swaps its
split sizes, because those describe one call's send and receive counts; `seq_splits` /
`head_splits` describe the group's geometry, which holds whichever way the data moves. Backward
always takes the copying path, never `out=`.

A **meta** kernel propagates shapes under `FakeTensor` and AOTAutograd. `torch.compile` over a
module that holds an `UlyssesGroup` still graph-breaks: the group is a torchbind object with no
registered fake class. `empty_output()` refuses to be traced at all, with a message saying why.

**Forward-mode AD works; reverse-mode `torch.func` does not.** A dual input carries its tangent
through a second collective, so `torch.autograd.forward_ad`, `torch.func.jvp` and
`torch.func.linearize` all run, at twice the traffic. `grad` / `vjp` / `jacrev` raise on their own:
the backward is a C++ `torch::autograd::Function`, which functorch cannot lift. `vmap` raises by an
explicit check on a batched input — it would run the collective once per batch element, so ranks
that disagree on the batch size would issue different numbers of collectives and hang. `jacfwd`
raises through that same check, being `vmap` over `jvp`. Map over the batch axis by reshaping it
into `b`, which the collective already carries.

**`out`** decides the speed:

- **absent, or a plain tensor of the output shape** — the transfer lands in a window this group
  keeps, and one flat device-to-device copy moves it out. The result is the caller's, with no
  lifetime rules.
- **a buffer from `empty_output()`** — the peers write it directly and there is no copy-out. Still
  the caller's own buffer, so still no lifetime rules; it is simply overwritten by the next call
  that uses it, like any output buffer.

**`out` is not differentiable, and raises rather than dropping the gradient.** It reaches a
mutating op with no autograd formula, so the result is the buffer itself — `grad_fn` is `None`.
Returning that quietly would be indistinguishable from a caller who wanted no gradient, so a
grad-requiring input under grad mode raises instead. Use `out=` in inference or under
`torch.no_grad()`, where it is accepted; use the plain call inside a training graph.

**`seq_splits[p]` / `head_splits[p]`** are rank p's sequence and head shard. Pass **both or
neither**, identical on every rank, matching the shape handed in. Neither means even shards, and
the scattered axis must then divide (`n_global % ws` for mode 0, `s_global % ws` for mode 1). Both
lets shards differ arbitrarily, which is what lets a caller drop sequence padding.

### Packed PCIe backend

Construct it explicitly:

```python
group = UlyssesGroup(require_nvlink=False, backend="packed")
```

This first production version supports batch 1, even shards, synchronous inference, and both
modes. Mode 0 locally packs by destination and sends one flat chunk to each peer. Mode 1 receives
sender-major flat chunks and locally unpacks the head shards. It rejects autograd, uneven split
lists, batch greater than one, and `all_to_all_4d_async` before starting the collective.

For mode 0, an `out` from `empty_output()` remains a zero-copy output. Mode 1 needs receiver-side
unpack, so its `out` is written locally after communication even when it came from
`empty_output()`.

### Raises

`RuntimeError`, from validation that runs **before the call's first barrier**, so a rejected
argument leaves no rank waiting on peers that did not reject it: the group already `destroy()`ed;
`x` not 4D or not CUDA, or on a device other than the group's; dtype not in the dtype list above;
`d * elem_size` not 16 B-aligned; `mode` not 0 or 1; one of the splits without the other, or splits
contradicting `x`'s shape; no splits and the scattered axis does not divide; every rank's shard
empty, so the call moves no data; `out` not contiguous CUDA, on another device, or its dtype or
shape not the output's; `x` overlapping the window, or `out` partially overlapping it.

`world_size` outside `[1, 8]` is refused by the constructor, so no call reaches it.

## `all_to_all_4d_async(x, *, mode=0, out=None, seq_splits=None, head_splits=None, lend=False)`

The same call on the group's high-priority comm stream, returning immediately.

Available only with `backend="pitched"`; the packed backend currently rejects async calls.

**Not differentiable, and it refuses rather than pretending.** An `AsyncCollectiveTensor` is built
with `requires_grad` on the wrapper, which makes it a leaf: autograd runs above the subclass and
never sees the wrapped tensor's history, so a `backward()` through it would run to completion and
leave `x.grad` as `None`. A grad-requiring input raises instead. Use `all_to_all_4d`.

The result is an `AsyncCollectiveTensor`: `.wait()` returns the plain tensor, and so does the **first use by any
aten op** — either way the caller's stream waits on the comm stream's completion event GPU-side,
and the host does not block. A **view op** does not wait; it re-wraps.

**Wait on, or use, every result.** A dropped one leaves its entry in torch's work registry and its
CUDA event behind; torch prints a count of the survivors at exit. `out=` is the hole, since reading
your own `out` never touches the registry — read the returned wrapper.

The input is staged into a persistent per-`(shape, dtype)` buffer on the caller's stream, so `x` is
never retained cross-stream. That costs one device copy per call. Buffers of one shape form a pool
of **4**, so that many same-shaped calls can be in flight before one waits for another to retire.

### `lend=True`

`out=`'s saving without a buffer of your own: the peers write a window the group holds, the result
**is** that window, and there is no copy-out. Mutually exclusive with `out=` — passing both raises.
Not differentiable, like the rest of this call.

The group keeps **4 windows per dtype and takes them in order**, one per call. Which window a call
fills has to be the same on every rank, since the transfer addresses the peers through that window
and the barrier through its signal pad; a rotation driven by the call count is the same everywhere,
and anything read off local reference counts is not.

So the bound is a rule, not a hint: **at most 4 lent results alive at once, and every rank must drop
them at the same point in its own program.** A fifth live one raises. A rank that dropped a result
its peers still hold does not raise — it fills a window they are not reading, which is the hang this
page opens with.

Use it when a result is consumed and released within the step that issued it, which is what a
split q/k/v exchange does. Keep a result across steps and `empty_output()` with `out=` is the fit.

On a libtorch with no `c10d::register_work` there is no registry to bind to, and this returns a
`CompletedHandle` instead: same `.wait()`, correct results, no overlap. A distinct type, so it is
visible.

The sync and async calls do not share a window, so mixing them is safe.

## `empty_output(x, *, mode=0, seq_splits=None, head_splits=None) -> Tensor`

A buffer shaped like `all_to_all_4d(x, mode=...)`'s output, in symmetric memory. Passing it back as
`out=` removes the copy-out.

**Collective**, so allocate outside the loop and reuse it. It is an ordinary tensor: keep it across
steps, free it when you like, and it may outlive the group.

It is larger than it looks. Under uneven splits the window has to hold the **largest** rank's
output, because the peer offsets only line up while every rank allocates the same size; the tensor
you get is a view of that allocation's dense prefix.

## `destroy() -> None`

Releases the group's windows and its transfer stream. All ranks must call it together. Buffers
from `empty_output()` are unaffected — they are yours.

## Environment

Nothing is read from the environment on the collective path. On a machine whose Fabric Manager
cannot bind NVLink SHARP, NCCL needs `NCCL_NVLS_ENABLE=0`; that affects the `torch.distributed`
bootstrap, not this operator. See [install.md](install.md).
