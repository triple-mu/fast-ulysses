# Quick start

## Wrapping attention

The two calls bracket attention: mode 0 trades the sequence shard for a head shard, mode 1 trades
it back.

```python
import os

import torch
import torch.distributed as dist
from fast_ulysses import UlyssesGroup

dist.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
ws = dist.get_world_size()

group = UlyssesGroup()          # collective; refuses a group that is not NVLink-joined

def attention_block(qkv):       # (b, s_local, n_global, 3 * head_dim)
    qkv = group.all_to_all_4d(qkv, mode=0)      # -> (b, s_global, n_local, 3 * head_dim)
    out = attention(qkv)                        # each rank now holds whole sequences
    return group.all_to_all_4d(out, mode=1)     # -> (b, s_local, n_global, head_dim)
```

Pack q, k and v into one collective when you can: the last dim is just `3 * head_dim`, and one
call moves the same bytes with one handshake instead of three.

## Removing the copy-out

The default returns a tensor the call allocated, which costs one flat device-to-device copy out of
the window. Give it a symmetric buffer instead and the peers write that buffer directly:

```python
buf = group.empty_output(qkv, mode=0)     # collective: allocate ONCE, outside the loop
for step in range(steps):
    y = group.all_to_all_4d(qkv, mode=0, out=buf)   # no copy-out; y is buf
```

`buf` is an ordinary tensor you own. It is overwritten by the next call that uses it, so use one
buffer per concurrent call.

## Overlapping with compute

The transfer uses no SMs, so it can run underneath compute. Submit it, do the compute, then use the
result — the first aten op on it inserts the wait GPU-side.

```python
y = group.all_to_all_4d_async(qkv, mode=0)
h = some_gemm_chain(other_input)     # runs while the copy engines move qkv
y = y.wait()                         # or just use y; the first op waits by itself
```

Wait on, or use, every async result. A dropped one leaves an entry in torch's work registry.

## Dropping the sequence padding

Sharding a sequence usually means rounding it up to a multiple of the group size and padding the
tail, so every rank holds the same length. Those padded tokens then ride through attention and the
collective on every layer of every step. Pass the real per-rank lengths instead:

```python
s = 75827                                    # does not divide by ws
seq_splits = [s // ws + (1 if p < s % ws else 0) for p in range(ws)]
head_splits = [n_global // ws] * ws

y = group.all_to_all_4d(qkv, mode=0, seq_splits=seq_splits, head_splits=head_splits)
```

Both lists or neither, identical on every rank. Uneven is the general path here, not a slow one.

## Under 2-D parallelism

`process_group` takes any subgroup — the sp slice of a tp × sp mesh, with no restriction on which
ranks it contains:

```python
sp_group = mesh["sp"].get_group()
group = UlyssesGroup(process_group=sp_group)
```

## What will hang you

Every rank must issue the same sequence of shapes. The first call with a new shape allocates a
window, and that allocation is collective — if one rank takes a branch the others do not, it waits
there. This is the same discipline `torch.distributed` collectives already require.

## Checking the machine

```bash
fast-ulysses doctor
```

Prints the build, the devices, and the pairwise NVLink matrix. A group spanning a pair marked `N`
is refused at construction.
