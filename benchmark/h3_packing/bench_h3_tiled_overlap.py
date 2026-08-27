#!/usr/bin/env python3
"""Benchmark two-way head-tiled Ulysses/attention overlap for MiniMax H3."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fast_ulysses import UlyssesGroup


def attention_context(name: str):
    if name == "auto":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend = {
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
    }[name]
    return sdpa_kernel(backend)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str) -> torch.Tensor:
    with attention_context(backend):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
    return out.transpose(1, 2).contiguous()


def pack_qkv_by_destination(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, world_size: int
) -> torch.Tensor:
    """Lay out Q/K/V so one mode-0 collective returns three contiguous tensors per peer."""
    q_chunks = q.chunk(world_size, dim=2)
    k_chunks = k.chunk(world_size, dim=2)
    v_chunks = v.chunk(world_size, dim=2)
    return torch.cat(
        tuple(
            chunk
            for peer in range(world_size)
            for chunk in (q_chunks[peer], k_chunks[peer], v_chunks[peer])
        ),
        dim=2,
    )


def distribution(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        dist.barrier()
        start.record()
        fn()
        end.record()
        end.synchronize()
        elapsed = torch.tensor(
            [start.elapsed_time(end)], dtype=torch.float64, device=torch.cuda.current_device()
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(float(elapsed.item()))

    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999999) - 1))
        return ordered[index]

    return {
        "min_ms": ordered[0],
        "p50_ms": statistics.median(ordered),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1],
        "mean_ms": statistics.mean(ordered),
    }


def make_ulysses_group(tp_size: int) -> tuple[dist.ProcessGroup, int]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size % tp_size:
        raise ValueError(f"world_size {world_size} must divide TP size {tp_size}")
    ulysses_size = world_size // tp_size
    if ulysses_size != 2:
        raise ValueError("this prototype currently requires Ulysses degree 2")

    local_group = None
    for tp_rank in range(tp_size):
        ranks = [tp_rank + sp_rank * tp_size for sp_rank in range(ulysses_size)]
        candidate = dist.new_group(ranks, backend="nccl")
        if rank in ranks:
            local_group = candidate
    if local_group is None:
        raise RuntimeError(f"rank {rank} was not assigned to a Ulysses group")
    return local_group, ulysses_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=37760)
    parser.add_argument("--model-heads", type=int, default=56)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument(
        "--attention-backend", choices=("cudnn", "flash", "auto"), default="cudnn"
    )
    parser.add_argument("--allow-non-nvlink", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    pg, ulysses_size = make_ulysses_group(args.tensor_parallel_size)

    if args.model_heads % args.tensor_parallel_size:
        raise ValueError("model heads must divide TP size")
    tp_heads = args.model_heads // args.tensor_parallel_size
    if tp_heads % 2:
        raise ValueError("TP-local heads must divide two head tiles")
    if args.sequence_length % ulysses_size:
        raise ValueError("sequence length must divide Ulysses degree")
    tile_heads = tp_heads // 2
    if tile_heads % ulysses_size:
        raise ValueError("each head tile must divide Ulysses degree")

    torch.manual_seed(1101 + rank)
    shape = (1, args.sequence_length // ulysses_size, tp_heads, args.head_dim)
    q, k, v = [torch.randn(shape, dtype=torch.bfloat16, device=device) for _ in range(3)]
    q_tiles = tuple(chunk.contiguous() for chunk in q.chunk(2, dim=2))
    k_tiles = tuple(chunk.contiguous() for chunk in k.chunk(2, dim=2))
    v_tiles = tuple(chunk.contiguous() for chunk in v.chunk(2, dim=2))
    fused_qkv = pack_qkv_by_destination(q, k, v, ulysses_size)
    fused_qkv_tiles = tuple(
        pack_qkv_by_destination(q_tiles[tile], k_tiles[tile], v_tiles[tile], ulysses_size)
        for tile in range(2)
    )

    group = UlyssesGroup(
        process_group=pg,
        device=device,
        require_nvlink=not args.allow_non_nvlink,
        backend="pitched",
    )

    def full_serial() -> torch.Tensor:
        q_all = group.all_to_all_4d(q, mode=0)
        k_all = group.all_to_all_4d(k, mode=0)
        v_all = group.all_to_all_4d(v, mode=0)
        out = attention(q_all, k_all, v_all, args.attention_backend)
        return group.all_to_all_4d(out, mode=1)

    def tiled_serial() -> torch.Tensor:
        outputs = []
        for tile in range(2):
            q_all = group.all_to_all_4d(q_tiles[tile], mode=0)
            k_all = group.all_to_all_4d(k_tiles[tile], mode=0)
            v_all = group.all_to_all_4d(v_tiles[tile], mode=0)
            out = attention(q_all, k_all, v_all, args.attention_backend)
            outputs.append(group.all_to_all_4d(out, mode=1))
        return torch.cat(outputs, dim=2)

    def fused_full_serial() -> torch.Tensor:
        qkv_all = group.all_to_all_4d(fused_qkv, mode=0)
        q_all, k_all, v_all = qkv_all.chunk(3, dim=2)
        out = attention(q_all, k_all, v_all, args.attention_backend)
        return group.all_to_all_4d(out, mode=1)

    def fused_full_with_pack() -> torch.Tensor:
        packed = pack_qkv_by_destination(q, k, v, ulysses_size)
        qkv_all = group.all_to_all_4d(packed, mode=0)
        q_all, k_all, v_all = qkv_all.chunk(3, dim=2)
        out = attention(q_all, k_all, v_all, args.attention_backend)
        return group.all_to_all_4d(out, mode=1)

    def fused_tiled_serial() -> torch.Tensor:
        outputs = []
        for packed in fused_qkv_tiles:
            qkv_all = group.all_to_all_4d(packed, mode=0)
            q_all, k_all, v_all = qkv_all.chunk(3, dim=2)
            out = attention(q_all, k_all, v_all, args.attention_backend)
            outputs.append(group.all_to_all_4d(out, mode=1))
        return torch.cat(outputs, dim=2)

    def fused_tiled_overlap() -> torch.Tensor:
        qkv0 = group.all_to_all_4d_async(fused_qkv_tiles[0], mode=0)
        q0_all, k0_all, v0_all = qkv0.wait().chunk(3, dim=2)

        # Fused tile 1 Q/K/V runs on the CE stream while tile 0 attention occupies the SMs.
        qkv1 = group.all_to_all_4d_async(fused_qkv_tiles[1], mode=0)
        attn0 = attention(q0_all, k0_all, v0_all, args.attention_backend)

        # O0 queues after fused tile 1 Q/K/V, then overlaps tile 1 attention.
        o0 = group.all_to_all_4d_async(attn0, mode=1)
        q1_all, k1_all, v1_all = qkv1.wait().chunk(3, dim=2)
        attn1 = attention(q1_all, k1_all, v1_all, args.attention_backend)
        o1 = group.all_to_all_4d_async(attn1, mode=1)
        return torch.cat((o0.wait(), o1.wait()), dim=2)

    try:
        with torch.inference_mode():
            expected = full_serial()
            candidates = {
                "tiled_serial": tiled_serial(),
                "fused_full_serial": fused_full_serial(),
                "fused_full_with_pack": fused_full_with_pack(),
                "fused_tiled_serial": fused_tiled_serial(),
                "fused_tiled_overlap": fused_tiled_overlap(),
            }
            torch.cuda.synchronize()
            correctness = {}
            for name, actual in candidates.items():
                diff = (actual.float() - expected.float()).abs()
                max_abs = torch.tensor([diff.max().item()], dtype=torch.float64, device=device)
                mean_abs = torch.tensor([diff.mean().item()], dtype=torch.float64, device=device)
                is_close = torch.tensor(
                    [int(torch.allclose(actual, expected, rtol=2e-2, atol=2e-2))],
                    dtype=torch.int32,
                    device=device,
                )
                dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
                dist.all_reduce(mean_abs, op=dist.ReduceOp.SUM)
                dist.all_reduce(is_close, op=dist.ReduceOp.MIN)
                mean_abs /= dist.get_world_size()
                correctness[name] = {
                    "max_abs": float(max_abs.item()),
                    "mean_abs": float(mean_abs.item()),
                    "allclose": bool(is_close.item()),
                }
            if not all(item["allclose"] for item in correctness.values()):
                raise AssertionError(f"tiled attention correctness failed: {correctness}")

            paths = {
                "full_serial": full_serial,
                "tiled_serial": tiled_serial,
                "fused_full_serial": fused_full_serial,
                "fused_full_with_pack": fused_full_with_pack,
                "fused_tiled_serial": fused_tiled_serial,
                "fused_tiled_overlap": fused_tiled_overlap,
            }
            results = {
                name: distribution(fn, args.warmup, args.iters) for name, fn in paths.items()
            }
    finally:
        group.destroy()

    baseline = results["full_serial"]["p50_ms"]
    for stats in results.values():
        stats["versus_full_serial"] = baseline / stats["p50_ms"]
    hidden = (
        results["fused_tiled_serial"]["p50_ms"]
        - results["fused_tiled_overlap"]["p50_ms"]
    )
    gate_pass = (
        results["fused_tiled_overlap"]["p50_ms"] < results["full_serial"]["p50_ms"]
        and results["fused_tiled_overlap"]["p50_ms"]
        < results["fused_tiled_serial"]["p50_ms"]
    )
    report = {
        "shape": {
            "sequence_length": args.sequence_length,
            "model_heads": args.model_heads,
            "tensor_parallel_size": args.tensor_parallel_size,
            "ulysses_degree": ulysses_size,
            "tp_local_heads": tp_heads,
            "tile_heads_before_mode0": tile_heads,
            "tile_heads_in_attention": tile_heads // ulysses_size,
            "head_dim": args.head_dim,
            "dtype": "bfloat16",
            "attention_backend": args.attention_backend,
        },
        "warmup": args.warmup,
        "iterations": args.iters,
        "correctness": correctness,
        "measurements": results,
        "overlap_hidden_ms": hidden,
        "gate_pass": gate_pass,
    }

    if rank == 0:
        print(
            "# MiniMax H3 tiled Ulysses overlap; "
            f"S={args.sequence_length} TP={args.tensor_parallel_size} U={ulysses_size} "
            f"heads={tp_heads} tile_heads={tile_heads} d={args.head_dim} "
            f"attention={args.attention_backend}"
        )
        print("# full serial = 3 mode0 + attention + mode1; owned outputs only")
        print("# fused rows assume QKV norm/RoPE emits destination-major layout; with_pack pays torch.cat")
        print(f"{'path':<22} {'p50 ms':>10} {'p95 ms':>10} {'vs full':>10}")
        print("-" * 56)
        for name, stats in results.items():
            print(
                f"{name:<22} {stats['p50_ms']:10.3f} {stats['p95_ms']:10.3f} "
                f"{stats['versus_full_serial']:9.3f}x"
            )
        print(f"\n# tiled overlap hides {hidden:.3f} ms versus tiled serial")
        print(f"# overlap_gate={'PASS' if gate_pass else 'FAIL'}")
        for name, item in correctness.items():
            print(
                f"# correctness {name}: allclose={item['allclose']} "
                f"max_abs={item['max_abs']:.6g} mean_abs={item['mean_abs']:.6g}"
            )
        if args.json_out:
            output = Path(args.json_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"# JSON: {output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
