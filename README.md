# custom_ulysses_op

基于 **NVSHMEM 对称堆 + NVLink P2P** 的 Ulysses 序列并行 all-to-all 自定义算子。

## 项目介绍

Ulysses 序列并行（DeepSpeed-Ulysses）把超长序列切到多卡上算注意力：进注意力前用一次 all-to-all 把「按序列切分」换成「按注意力头切分」（每张卡拿到全部序列、自己那几个头），出注意力后再换回来。这次 all-to-all 是长序列 / 视频 DiT（如 Wan、HunyuanVideo）训练与推理里的关键通信，序列越长、卡越多，它越是瓶颈。

`custom_ulysses_op` 把这个 4D all-to-all 实现为一个独立、可分发的 **torch custom op**（命名空间 `ulysses`，`torch.ops.ulysses.all_to_all_single_4d`），单节点内绕开 NCCL：在 NVSHMEM 对称堆上分配输出缓冲，直接用 NVLink P2P 把数据写到对端 GPU 显存，再用一个轻量的自定义 NVLink flag barrier 做跨 rank 同步——整条路径不落 host、不走 NCCL collective。

核心特性：

- **双内核路径，自动择优**：
  - **TMA 路径**（sm90+，Hopper/Blackwell）：用 `cp.async.bulk`（TMA copy engine）搬运 + 软件流水。TMA 由拷贝引擎而非 SM 执行搬运，**只占极少 SM**，给计算-通信重叠留出算力。
  - **non-TMA 路径**：SM 向量化直写（512 线程拉满在飞远端写），sm80（A100）等无 TMA 时回退于此。
  - 默认按架构自动选择（见 `use_tma` 三态），也可每次调用显式覆盖。
- **单节点 NVLink P2P**，`world_size ∈ {1, 2, 4, 8}`。
- **均匀（uniform）+ 变长（varlen）** 两类切分：
  - 均匀：序列长 `s` / 头数 `n` 能被 `world_size` 整除。
  - 变长：每个 rank 的 `seq_lens` / `head_splits` 由调用方提供（无运行时 gather）。
- **两个方向**：`mode=0` scatter heads / gather seq（进注意力）；`mode=1` 为其逆（出注意力）。
- 数据类型 `float16` / `bfloat16`；要求 `d * elem_size` 16B 对齐。

## 安装 / 构建

### 依赖

- **NVSHMEM 3.7.0**
- **PyTorch**（CUDA 12 构建）
- **CUDA 12**
- 目标 GPU 架构：sm80（A100）/ sm90（H100）/ sm100（B200）

### 构建命令

```bash
NVSHMEM_HOME=<nvshmem 安装路径> \
CUSTOM_ULYSSES_CUDA_ARCH=90 \
pip install -e . --no-build-isolation
```

- `NVSHMEM_HOME`：NVSHMEM 安装根目录（需含 `include/nvshmem.h` 与 `lib/cmake/nvshmem`）。
- `CUSTOM_ULYSSES_CUDA_ARCH`：目标算力，分号分隔多目标，默认 `80;90;100`。例如 H100 用 `90`，B200 用 `100`，多目标 `80;90;100`。
- `--no-build-isolation`：构建依赖宿主环境里已装好的 PyTorch（CMake 通过它定位 libtorch）。

### 无 NVSwitch fabric 的节点

部分 H100/H200 节点没有 NVSwitch fabric（或带 IB NIC），NVSHMEM 默认会尝试 NVLS 多播堆映射或 IB 远程传输，初始化时可能 segfault。本算子是**单节点 NVLink P2P**，`comm.py` 在 `UlyssesGroup` 构造时已默认 `setdefault`：

```text
NVSHMEM_DISABLE_NVLS=1
NVSHMEM_REMOTE_TRANSPORT=none
```

因此这类节点**无需手动设这两个 env**；若有特殊需求可在构造前自行覆盖。

## 接口说明

包导出一个类 `UlyssesGroup`：

```python
from custom_ulysses_op import UlyssesGroup
```

### `UlyssesGroup(process_group=None, device=None, initial_pool_bytes=2<<30)`

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `process_group` | `torch.distributed.ProcessGroup` 或 `None` | 用于 bootstrap 的进程组；`None` 取 `dist.group.WORLD`。 |
| `device` | `torch.device` 或 `None` | 本 rank 的 CUDA 设备；`None` 取当前设备。 |
| `initial_pool_bytes` | `int` | NVSHMEM 对称堆预留字节数，默认 `2<<30`（2 GiB）。所有预调/调用的临时输出都从这块堆里分配。 |

构造时会用 `dist.broadcast` 分发 NVSHMEM unique id 并 `init_world`，前后各一次 `dist.barrier`，因此**所有 rank 必须一起构造**。

### `all_to_all_single_4d(x, *, mode=0, tag="", seq_lens=None, head_splits=None, use_tma=None) -> Tensor`

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `x` | `Tensor` | 4D CUDA 张量，`float16`/`bfloat16`；内部会 `.contiguous()`。 |
| `mode` | `int` | `0`（scatter heads/gather seq）或 `1`（逆）。 |
| `tag` | `str` | 对称堆缓冲的标签；不同 shape/路径请用不同 tag 以复用各自的输出块。 |
| `seq_lens` | `list[int]` 或 `None` | 变长：各 rank 序列长度，长度须等于 `world_size`。 |
| `head_splits` | `list[int]` 或 `None` | 变长：各 rank 头数，长度须等于 `world_size`。 |
| `use_tma` | `bool` 或 `None` | 内核路径三态选择（见下）。 |

`seq_lens` 与 `head_splits` 必须**同时提供或同时为 `None`**：

- 两者都 `None` → 均匀路径（`s`/`n` 必须被 `world_size` 整除）。
- 两者都给 → 变长路径（split 由调用方提供，无运行时 gather）。

**输入 / 输出形状（`b` 为 batch，`d` 为 head dim）**

均匀（`s_local = s_global / ws`，`n_local = n_global / ws`）：

| mode | 输入 `x` | 输出 |
| --- | --- | --- |
| 0 | `(b, s_local, n_global, d)` | `(b, s_global, n_local, d)` |
| 1 | `(b, s_global, n_local, d)` | `(b, s_local, n_global, d)` |

变长（`S = sum(seq_lens)`，`N = sum(head_splits)`，`me` 为本 rank）：

| mode | 输入 `x` | 输出 |
| --- | --- | --- |
| 0 | `(b, seq_lens[me], N, d)` | `(b, S, head_splits[me], d)` |
| 1 | `(b, S, head_splits[me], d)` | `(b, seq_lens[me], N, d)` |

**`use_tma` 三态**

- `None`（自动）：均匀且 sm90+ → TMA，其余（均匀但 sm<90、或任何变长）→ non-TMA；变长**恒** non-TMA（uneven 序列下 TMA 被迫 `tile_s=1` 反而慢约 1.5x）。
- `True`：强制 TMA（均匀与变长都走 TMA；需要 sm90+，否则 `TORCH_CHECK` 报错）。
- `False`：强制 non-TMA。

**集体硬约束（务必遵守，否则整组 hang）**

- 所有 rank 必须以**相同的 `(shape, mode, use_tma)` 序列**调用本方法。`use_tma` 与 `shape`/`mode` 同等严格——`use_tma` 不一致会让各 rank 走不同 kernel/barrier，并在内部配置缓存的 key（含 `tma` 位）上分叉，导致整组 hang。
- 均匀路径首次见到某 `(shape, mode, use_tma)` 时，会跑一次**集体微基准**选最优 launch 配置并缓存（之后命中直接返回）。
- **`tune` 与 lazy 兜底互斥**：要么所有 rank 预先 `tune` 相同的 shape 集合，要么所有 rank 都依赖首次调用的 lazy 微基准；严禁部分 rank `tune`、部分 rank lazy。

### `tune(shape, *, mode=0, use_tma=None, verbose=False) -> None`

预热某个 shape 的 launch 配置（集体调用，**只预热不改变结果**），用于避免首次 `all_to_all_single_4d` 触发微基准的开销。

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `shape` | `tuple[int, int, int, int]` | all_to_all 的输入物理形状 `(b, x1, x2, d)`：mode0 为 `(b, s_local, n_global, d)`，mode1 为 `(b, s_global, n_local, d)`。 |
| `mode` | `int` | `0` 或 `1`。 |
| `use_tma` | `bool` 或 `None` | 同上三态，被**决议为单一路径**后只预热那条路径（`tma` 位是缓存 key 的一部分）。 |
| `verbose` | `bool` | 打印微基准细节。 |

注意：

- 所有 rank 必须以**相同 `(shape, mode, use_tma)` 序列**调用 `tune`。
- `tune` 只预热**被决议的那一条路径**。若之后对同一 shape 用相反的 `use_tma` 调 `all_to_all_single_4d`，会 miss 缓存并再触发一次 lazy 微基准（同样要求全 rank 一致）。
- 可预调的 shape 数受 `initial_pool_bytes` 限制（每个 shape 在对称堆累积一段临时输出）。

### `destroy() -> None`

释放对称堆资源（内部先 `dist.barrier` 再 destroy）。所有 rank 一起调用。

## 使用方式

最小可运行示例（保存为 `example.py`，用 `torchrun --nproc_per_node=2 example.py` 运行）：

```python
import os

import torch
import torch.distributed as dist

from custom_ulysses_op import UlyssesGroup


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    pg = dist.group.WORLD

    group = UlyssesGroup(process_group=pg, initial_pool_bytes=1 << 30)

    # mode0 均匀：输入 (b, s_local, n_global, d) -> 输出 (b, s_global, n_local, d)
    b, s_local, d = 2, 16, 128
    n_global = 4 * ws  # 须能被 world_size 整除
    x = torch.randn(b, s_local, n_global, d, dtype=torch.bfloat16, device=dev)

    # 可选：预热该 shape 的 launch 配置（所有 rank 须传相同 shape/mode/use_tma）
    group.tune((b, s_local, n_global, d), mode=0, use_tma=None)

    # all-to-all（所有 rank 须以相同 shape/mode/use_tma 序列调用）
    out = group.all_to_all_single_4d(x, mode=0, tag="demo", use_tma=None)
    assert out.shape == (b, s_local * ws, n_global // ws, d)
    if rank == 0:
        print(f"ws={ws} in={tuple(x.shape)} out={tuple(out.shape)}", flush=True)

    # 变长示例：split 由调用方提供，长度须等于 world_size
    seq_lens = [4, 6, 3, 7, 5, 2, 8, 1][:ws]
    head_splits = [2, 3, 1, 4, 2, 5, 1, 3][:ws]
    S, N = sum(seq_lens), sum(head_splits)
    xv = torch.randn(b, seq_lens[rank], N, d, dtype=torch.bfloat16, device=dev)
    outv = group.all_to_all_single_4d(
        xv, mode=0, tag="demo_varlen", seq_lens=seq_lens, head_splits=head_splits
    )
    assert outv.shape == (b, S, head_splits[rank], d)

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

正确性对拍与吞吐基准见 `test/test_correctness.py` 与 `benchmark/bench_uniform.py`：

```bash
# 正确性（逐位对拍 torch permute + all_to_all_single）
torchrun --nproc_per_node=8 test/test_correctness.py

# 吞吐（mode0 bf16，对比 NCCL；CUSTOM_ULYSSES_USE_TMA 控制路径）
PROF_MODE=0 CUSTOM_ULYSSES_USE_TMA=1 torchrun --nproc_per_node=8 benchmark/bench_uniform.py
```

## 性能

### 基准 shape：Wan2.2 5s 720p

以 **Wan2.2（14B 级）5s 720p 视频**的注意力 all-to-all 为基准 shape：

- 720p(1280×720) + 81 帧；VAE 4×8×8 + patch (1,2,2) → 21 个 latent 帧 × 45×80 patch = **序列 N = 75600**。
- **头数 H = 40，head_dim D = 128**，`bf16`，`b = 1`，`mode=0`。
- Ulysses 并行度 = `world_size`：每卡序列 `N/ws`，A2A 后每卡头数 `n_local = 40/ws`。

下表 `ours` 为本算子（`use_tma` 自动 / 显式 non-TMA 两列），`NCCL` 为 `torch.distributed` permute + `all_to_all_single` 参考；格式 `时延 us / 吞吐 GB/s`，`加速` = NCCL 时延 / 本算子（取两路径较优）时延。

**H100 × 8（无 NVSwitch fabric）**

| ws | n_local | ours TMA(auto) | ours non-TMA | NCCL | 加速 |
| --- | --- | --- | --- | --- | --- |
| 2 | 20 | 537 / 361 | 558 / 347 | 1198 / 162 | **2.2×** |
| 4 | 10 | 519 / 280 | 513 / 283 | 730 / 199 | **1.4×** |
| 8 | 5 | 337 / 251 | 355 / 239 | 425 / 199 | **1.3×** |

**H200 × 8（无 NVSwitch fabric）**

| ws | n_local | ours TMA(auto) | ours non-TMA | NCCL | 加速 |
| --- | --- | --- | --- | --- | --- |
| 2 | 20 | 539 / 359 | 546 / 355 | 1136 / 170 | **2.1×** |
| 4 | 10 | 518 / 280 | 499 / 291 | 726 / 200 | **1.4×** |
| 8 | 5 | 361 / 234 | 322 / 263 | 412 / 205 | **1.3×** |

观察：

- **全 `world_size` 稳定超过 NCCL 1.3–2.2×**；`ws` 越小（每卡头块越大、对端越少）领先越多（NCCL 在小 ws 下尤其差）。
- **TMA 与 non-TMA 在此 shape 基本持平**：Wan 头数少，`ws=8` 时 `n_local=5`、连续块仅 `5×128×2≈1.3KB`，TMA 大块突发的优势收窄；`ws=8` 上 non-TMA 反而略优。自动路径（sm90+→TMA）此处够用，对小 `n_local` 也可显式 `use_tma=False`。
- **H100 ≈ H200**：两者同为无 NVSwitch、同级 NVLink，结果基本一致。

### 其它参考（H100 × 8，mode0 bf16，指示性）

| shape | 配置 | ours 吞吐 |
| --- | --- | --- |
| 合成大头数 | H=128, D=128 | ~345 GB/s（≈ ThunderKittens，本算子 kernel 上限） |
| 变长 | uneven s/n | ~132 GB/s |

口径与说明：

- 吞吐 = **per-rank remote bytes / 时间** = `numel * 2 * (ws-1) / ws` ÷ 耗时（与 ThunderKittens benchmark 口径一致，见 `benchmark/bench_uniform.py`）；时延为 20 次迭代的 CUDA event 中位数。
- 数字在**共享 GPU** 上测得，存在邻居噪声；不同节点 / fabric 拓扑会有明显差异，仅供量级参考。复现：`PROF_N=75600 PROF_H=40 PROF_D=128 torchrun --nproc_per_node=8 benchmark/bench_uniform.py`。
