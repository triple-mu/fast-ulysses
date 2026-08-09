"""Ulysses sequence parallelism around one DiT attention block, forward and backward.

    torchrun --nproc_per_node=8 examples/dit_attention.py

Each rank holds a slice of the sequence and all of the heads. The first all-to-all trades that for
all of the sequence and a slice of the heads, attention runs on whole sequences, and the second
trades back. Both are differentiable, so this is an ordinary module: no autograd.Function wrapper,
no manual backward.

The run checks itself two ways. It compares against the same block with no sequence parallelism at
all -- one rank holding everything -- for both the output and the input gradient. And it prints the
loss over a few optimiser steps, because a collective can be bit-exact in a unit test and still be
wired into the wrong axis of a real module.

The packing matters more than it looks. q, k and v go through ONE collective as a `3 * head_dim`
last axis: one handshake instead of three, for the same bytes.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fast_ulysses import UlyssesGroup

B, S_LOCAL, HEADS_PER_RANK, HEAD_DIM = 1, 512, 4, 128


class UlyssesAttention(torch.nn.Module):
    """One attention block, sequence-parallel over `group`.

    Input and output are both (b, s_local, n_global * head_dim): the shard the caller already has.
    Everything between the two collectives sees whole sequences and a head shard.
    """

    def __init__(self, group: UlyssesGroup, n_global: int, head_dim: int) -> None:
        super().__init__()
        self.group, self.n_global, self.head_dim = group, n_global, head_dim
        self.qkv = torch.nn.Linear(n_global * head_dim, 3 * n_global * head_dim, bias=False)
        self.proj = torch.nn.Linear(n_global * head_dim, n_global * head_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s_local, _ = x.shape
        ws = self.group.world_size

        # (b, s_local, n_global, 3 * head_dim) -- q, k, v packed into the last axis so the two
        # collectives below stay two collectives.
        qkv = self.qkv(x).view(b, s_local, self.n_global, 3 * self.head_dim)

        # mode 0: sequence shard -> head shard. Differentiable; its backward is mode 1.
        qkv = self.group.all_to_all_4d(qkv, mode=0)  # (b, s_global, n_local, 3 * head_dim)

        q, k, v = qkv.chunk(3, dim=-1)
        # SDPA wants (b, heads, s, d).
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)  # (b, s_global, n_local, head_dim)

        # mode 1: back to a sequence shard, all heads. Its backward is mode 0.
        attn = self.group.all_to_all_4d(attn.contiguous(), mode=1)
        return self.proj(attn.reshape(b, s_local, self.n_global * self.head_dim))


def reference(module: UlyssesAttention, x_global: torch.Tensor) -> torch.Tensor:
    """The same block with no sequence parallelism: one rank, whole sequence, all heads."""
    b, s, _ = x_global.shape
    qkv = module.qkv(x_global).view(b, s, module.n_global, 3 * module.head_dim)
    q, k, v = qkv.chunk(3, dim=-1)
    attn = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    return module.proj(attn.reshape(b, s, module.n_global * module.head_dim))


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)  # same seed on every rank: the weights must match

    n_global = HEADS_PER_RANK * ws
    block = UlyssesAttention(UlyssesGroup(require_nvlink=False), n_global, HEAD_DIM)
    block = block.to(dev, dtype=torch.bfloat16)

    # One global sequence, sharded by rank. Built identically everywhere, then sliced, so the
    # reference below is over exactly the same data.
    s_global = S_LOCAL * ws
    x_global = torch.randn(
        B, s_global, n_global * HEAD_DIM, dtype=torch.bfloat16, device=dev
    ).requires_grad_(True)
    x_local = x_global.detach()[:, rank * S_LOCAL : (rank + 1) * S_LOCAL].requires_grad_(True)

    y_local = block(x_local)
    y_local.sum().backward()

    # The whole point: gather the sharded result and compare, output AND gradient, against the
    # single-rank block. allclose, not equal -- attention is not a permutation, and the two paths
    # reduce in different orders.
    gathered = [torch.empty_like(y_local) for _ in range(ws)]
    dist.all_gather(gathered, y_local.detach())
    grads = [torch.empty_like(x_local.grad) for _ in range(ws)]
    dist.all_gather(grads, x_local.grad)

    y_ref = reference(block, x_global)
    y_ref.sum().backward()

    if rank == 0:
        out_err = (torch.cat(gathered, dim=1).float() - y_ref.detach().float()).abs().max().item()
        grad_err = (torch.cat(grads, dim=1).float() - x_global.grad.float()).abs().max().item()
        tol = 5e-2  # bfloat16 through attention plus two projections
        print(f"ws={ws} s_global={s_global} n_global={n_global}")
        print(f"  output   max |diff| = {out_err:.4f}  {'OK' if out_err < tol else 'FAIL'}")
        print(f"  d/dinput max |diff| = {grad_err:.4f}  {'OK' if grad_err < tol else 'FAIL'}")

    # And a few real steps, because being correct once is not the same as training.
    opt = torch.optim.AdamW(block.parameters(), lr=1e-3)
    target = torch.randn_like(y_local)
    for step in range(5):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(block(x_local), target)
        loss.backward()
        opt.step()
        # Every rank sees the same loss only if the collectives are wired right.
        if rank == 0:
            print(f"  step {step} loss {loss.item():.5f}")

    block.group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
