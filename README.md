# fast-ulysses

基于 **NVSHMEM 对称堆 + NVLink P2P** 的 Ulysses 序列并行 all-to-all 自定义算子。

## 项目介绍

Ulysses 序列并行（DeepSpeed-Ulysses）把超长序列切到多卡上算注意力：进注意力前用一次 all-to-all 把「按序列切分」换成「按注意力头切分」（每张卡拿到全部序列、自己那几个头），出注意力后再换回来。这次 all-to-all 是长序列 / 视频 DiT（如 Wan、HunyuanVideo）训练与推理里的关键通信，序列越长、卡越多，它越是瓶颈。

`fast_ulysses` 把这个 4D all-to-all 实现为一个独立、可分发的 **torch custom op**（命名空间 `fast_ulysses`，`torch.ops.fast_ulysses.all_to_all_single_4d`），单节点内绕开 NCCL：在 NVSHMEM 对称堆上分配输出缓冲，直接用 NVLink P2P 把数据写到对端 GPU 显存，再用一个轻量的自定义 NVLink flag barrier 做跨 rank 同步——整条路径不落 host、不走 NCCL collective。

核心特性：

- **双内核路径，运行时择优**：
  - **TMA 路径**（sm90+，Hopper/Blackwell）：用 `cp.async.bulk`（TMA copy engine）搬运 + 软件流水。TMA 由拷贝引擎而非 SM 执行搬运，**只占极少 SM**，给计算-通信重叠留出算力。
  - **non-TMA 路径**：SM 向量化直写（512 线程拉满在飞远端写），sm80（A100）等无 TMA 时回退于此。
  - **`use_tma=None`（自动）时按当前硬件运行时实测两条路径、缓存较快者**（取代离线静态表）；也可每次调用显式覆盖（见 `use_tma` 三态）。
- **单节点 NVLink P2P**，`world_size ∈ [1, 8]`（含奇数，如 3/5/6/7）。
- **均匀（uniform）切分**：序列长 `s` / 头数 `n` 能被 `world_size` 整除。
- **两个方向**：`mode=0` scatter heads / gather seq（进注意力）；`mode=1` 为其逆（出注意力）。
- 数据类型 `float16` / `bfloat16`；要求 `d * elem_size` 16B 对齐。

## 安装 / 构建

### 依赖

- **NVSHMEM 3.7.0**
- **PyTorch**（CUDA 12 构建）
- **CUDA 12**
- 目标 GPU 架构：sm80（A100）/ sm90（H100/H200）/ sm100（B200）

### 构建命令

```bash
NVSHMEM_HOME=<nvshmem 安装路径> \
CUSTOM_ULYSSES_CUDA_ARCH=90 \
pip install -e . --no-build-isolation
```

- `NVSHMEM_HOME`：NVSHMEM 安装根目录（需含 `include/nvshmem.h` 与 `lib/cmake/nvshmem`）。
- `CUSTOM_ULYSSES_CUDA_ARCH`：目标算力，分号分隔多目标，默认 `80;90;100`。例如 H100/H200 用 `90`，B200 用 `100`，多目标 `80;90;100`。
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
from fast_ulysses import UlyssesGroup
```

### `UlyssesGroup(process_group=None, device=None, initial_pool_bytes=2<<30)`

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `process_group` | `torch.distributed.ProcessGroup` 或 `None` | 用于 bootstrap 的进程组；`None` 取 `dist.group.WORLD`。 |
| `device` | `torch.device` 或 `None` | 本 rank 的 CUDA 设备；`None` 取当前设备。 |
| `initial_pool_bytes` | `int` | NVSHMEM 对称堆预留字节数，默认 `2<<30`（2 GiB）。所有调用的临时输出都从这块堆里分配（按 `tag` 复用）。 |

构造时会用 `dist.broadcast` 分发 NVSHMEM unique id 并 `init_world`，前后各一次 `dist.barrier`，因此**所有 rank 必须一起构造**。

### `all_to_all_single_4d(x, *, mode=0, tag="", use_tma=None) -> Tensor`

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `x` | `Tensor` | 4D CUDA 张量，`float16`/`bfloat16`；内部会 `.contiguous()`。 |
| `mode` | `int` | `0`（scatter heads/gather seq）或 `1`（逆）。 |
| `tag` | `str` | 对称堆输出缓冲的标签（按 `tag+shape+dtype` 复用）。**并存的多个结果（如 q/k/v）必须用不同 `tag`**，否则会复用同一块缓冲互相覆盖。 |
| `use_tma` | `bool` 或 `None` | 内核路径三态选择（见下）。 |

**输入 / 输出形状（`b` 为 batch，`d` 为 head dim；`s_local = s_global / ws`，`n_local = n_global / ws`）**

| mode | 输入 `x` | 输出 |
| --- | --- | --- |
| 0 | `(b, s_local, n_global, d)` | `(b, s_global, n_local, d)` |
| 1 | `(b, s_global, n_local, d)` | `(b, s_local, n_global, d)` |

**`use_tma` 三态**

- `None`（自动）：sm<9 → non-TMA；**sm90+ → 首次见到该 shape 时运行时实测 TMA 与 non-TMA 两条路径、缓存较快者**，之后命中直接用最优路径（替代离线静态表，按实际硬件择优）。
- `True`：强制 TMA（需要 sm90+，否则 `TORCH_CHECK` 报错）。
- `False`：强制 non-TMA。

**集体硬约束（务必遵守，否则整组 hang）**

- 所有 rank 必须以**相同的 `(shape, mode, use_tma)` 序列**调用本方法。`use_tma` 与 `shape`/`mode` 同等严格——不一致会让各 rank 走不同 kernel/barrier、在内部缓存 key 上分叉而 hang。
- 首次见到某 `(shape, mode, use_tma)` 时跑一次**懒惰微基准**选最优 launch 配置（自动路径还会比较两条路径）并缓存（之后命中零额外集体开销）。严格 SPMD 下所有 rank 在首次调用一起 miss、一起跑微基准，故 hang-free。

### `all_to_all_single_4d_async(x, *, mode=0, tag="", use_tma=None) -> AsyncA2AHandle`

异步变体：集合操作提交到 group 专属的高优先级 comm stream 后立即返回；此后提交到调用方 stream 的 kernel 可与 a2a 重叠，`handle.wait()`（GPU 侧 event 等待，host 不阻塞）后返回输出。集体约束与同步版相同（同步与异步调用共同计入全 rank 一致的调用序列）。

**与同步调用混用的顺序约束**：`fast_barrier` 的 epoch 是 group 级单调计数，barrier kernel 必须按提交顺序执行——在发起本 group 的下一次同步集合之前，必须先 `wait()` 掉所有未决的 async handle（数据依赖会强制 comm stream 的 barrier 先完成）。

**重叠的现实约束（8×H200 实测）**：直写 scatter 是 SM-resident 大 grid，与 cooperative launch 的 GEMM（如 cuBLAS nvjet coop kernel）互斥——GEMM 运行期间不释放任何 SM slot，a2a 只能等其排空（nsys 实测零重叠）。异步 API 的收益要在非 cooperative 计算窗口或未来 copy-engine/SM-carveout 搬运路径上兑现。

### `destroy() -> None`

释放对称堆资源（内部先 `dist.barrier` 再 destroy）。所有 rank 一起调用。

## 融合算子：QK RMSNorm + RoPE

DiT（如 Wan）在进注意力前对 q/k 做 **RMSNorm + RoPE**，是访存密集的 elementwise。本库把它们做成融合算子，语义严格对齐 Wan：RMSNorm 在 fp32 累加、eps 在 `rsqrt` 内、per-channel 权重、无 bias；RoPE 覆盖全 `head_dim`。融合版 `norm_rope` 全程 fp32（不在 norm 后中途 round，比分开两步更准）。

统一约定：`x` 形状 `(b, seq, n, d)`，`float16`/`bfloat16`，`d` 为 2 的幂且 `≤1024`；`weight`/`cos`/`sin` 为 **fp32** CUDA 张量；`cos`/`sin` 形状 `(seq, d/2)`，按每行的 seq 位置取。

- `mode`：`"per_head"`（沿 `d` 归约，`weight` 形状 `[d]`）或 `"cross_head"`（沿整条 `n*d` 归约，`weight` 形状 `[n*d]`，一个 token 全部 head 共享一个 RMS 标量——Wan 默认）。
- `interleaved`：`True`=GPT-J 相邻配对 `(x[2i],x[2i+1])`；`False`=NeoX 半分配对 `(x[i],x[i+d/2])`。

独立算子（`from fast_ulysses import rms_norm, rope, norm_rope`，单卡、无需 group）：

| API | 说明 |
| --- | --- |
| `rms_norm(x, weight, *, mode, eps=1e-6)` | 独立 RMSNorm |
| `rope(x, cos, sin, *, interleaved=True)` | 独立 RoPE |
| `norm_rope(x, weight, cos, sin, *, mode, interleaved=True, eps=1e-6)` | 融合 norm→rope（一趟读写） |

融进 all-to-all（`UlyssesGroup` 方法）：

### `all_to_all_single_4d_qk(x, weight, cos, sin, *, mode="cross_head", interleaved=True, eps=1e-6, tag="", use_tma=None) -> Tensor`

mode0 Ulysses input all-to-all，**把 q/k 的 norm+rope 融进 scatter kernel 本身**——norm+rope 作为寄存器内 epilogue，在数据被 P2P 写到对端 GPU 前就地完成（边搬边算，省掉独立 norm/rope 的两趟全量 HBM 读写）。融合 kernel 与非 TMA 快速 a2a kernel **访存结构同构**：grid-stride + uint4 向量化 + UNROLL 寄存器预取流水 + threads×unroll×blocks 运行时 autotune（首调按 shape 微基准并缓存，与 a2a 相同的 hang-safe 论证）。

- **cross-head（Wan 默认）**：先跑一个极小的向量化 `[b·s_local]` inv-rms 预归约（读一遍源，~占 a2a 时长 10%），scatter 走纯 elementwise 快速路径（interleaved 时与 plain a2a 完全同构，每线程一个 uint4）。
- **per-head**：无预归约；d-row 的 Σx² 在 kernel 内由 `vecs/2` 个相邻 lane 的段内 warp shuffle 完成（无 smem、无 `__syncthreads`）。
- 约束：`d` 为 2 的幂且 `d·elem_size ≥ 32B`（bf16/fp16 即 d≥16）；per-head 另需 `d·elem_size ≤ 1024B`（行规约须在一个 warp 内）。

输入 `(b, s_local, n_global, d)`（本 rank 持有其序列分片的全部 head），输出 `(b, s_global, n_local, d)`。`cos`/`sin` 形状 `(s_local, d/2)`，需由调用方**预切为本 rank 的全局位置**。`v` 不做 norm/rope，继续用 `all_to_all_single_4d`。集体约束与 `all_to_all_single_4d` 相同。（融合路径走直写内核，不含 TMA——TMA 拷贝引擎无法做搬运途中的变换；`use_tma` 被忽略。）调试：`FAST_ULYSSES_QK_TUNE_VERBOSE=1` 打印每 rank 的 autotune 选择。

### `all_to_all_single_4d_qk2(q, k, weight_q, weight_k, cos, sin, *, mode="cross_head", interleaved=True, eps=1e-6, tag="", use_tma=None) -> (Tensor, Tensor)`

**q、k 一次集体调用**：两次融合 scatter 背靠背 + **一次共享的 quiet + NVLink flag barrier**（对比两次 `all_to_all_single_4d_qk` 省一半同步时延——barrier 是纯时延，共享是净赚）。q/k 同 shape/dtype、共享 cos/sin，各用自己的 norm weight；输出落在 `tag::q` / `tag::k` 两块独立缓冲。结果与两次单调用**逐位一致**。作为单个 op 暴露以保证全 rank 的 barrier 计数 lockstep。

## 使用方式

最小可运行示例（保存为 `example.py`，用 `torchrun --nproc_per_node=2 example.py` 运行）：

```python
import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    pg = dist.group.WORLD

    group = UlyssesGroup(process_group=pg, initial_pool_bytes=1 << 30)

    # mode0：输入 (b, s_local, n_global, d) -> 输出 (b, s_global, n_local, d)
    b, s_local, d = 2, 16, 128
    n_global = 4 * ws  # 须能被 world_size 整除
    x = torch.randn(b, s_local, n_global, d, dtype=torch.bfloat16, device=dev)

    # 首次该 shape 自动跑微基准择优并缓存；所有 rank 须以相同 shape/mode/use_tma 序列调用
    out = group.all_to_all_single_4d(x, mode=0, tag="demo", use_tma=None)
    assert out.shape == (b, s_local * ws, n_global // ws, d)
    if rank == 0:
        print(f"ws={ws} in={tuple(x.shape)} out={tuple(out.shape)}", flush=True)

    # 并存的多个结果（q/k/v）必须用不同 tag，否则复用同一缓冲互相覆盖
    q = torch.randn(b, s_local, n_global, d, dtype=torch.bfloat16, device=dev)
    k = torch.randn(b, s_local, n_global, d, dtype=torch.bfloat16, device=dev)
    oq = group.all_to_all_single_4d(q, mode=0, tag="q")
    ok = group.all_to_all_single_4d(k, mode=0, tag="k")

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
PROF_MODE=0 torchrun --nproc_per_node=8 benchmark/bench_uniform.py
```

## 性能

### 基准 shape：Wan2.2 5s 720p

以 **Wan2.2（14B 级）5s 720p 视频**的注意力 all-to-all 为基准 shape：

- 720p(1280×720) + 81 帧；VAE 4×8×8 + patch (1,2,2) → 21 个 latent 帧 × 45×80 patch = **序列 N = 75600**。
- **头数 H = 40，head_dim D = 128**，`bf16`，`b = 1`。
- Ulysses 并行度 = `world_size`：每卡序列 `N/ws`，A2A 后每卡头数 `n_local = 40/ws`。

下表 `ours` 为本算子 `use_tma=None`（运行时路径择优）的吞吐，`NCCL` 为 `torch.distributed` permute + `all_to_all_single` 参考；吞吐口径 GB/s，`加速` = NCCL 时延 / 本算子时延。

**H200 × 8（无 NVSwitch fabric，空闲机实测）**

| ws | n_local | ours mode0 | ours mode1 | NCCL | 加速 |
| --- | --- | --- | --- | --- | --- |
| 2 | 20 | 355 | 355 | 171 | **2.1×** |
| 4 | 10 | 301 | 306 | 202 | **1.5×** |
| 8 | 5 | 301 | 301 | 203 | **1.5×** |

观察：

- **全 `world_size` 稳定超过 NCCL 1.5–2.1×**；`ws` 越小（每卡头块越大、对端越少）领先越多。
- **运行时路径择优让 mode0/mode1 都用到各自更快的路径**：在 H200 上该 shape 两个方向都自动选中 non-TMA（比强制 TMA 显著更快），无需离线表或手工指定。
- 不同节点 / fabric 拓扑会有明显差异；上表为单节点 8×H200 空闲机实测，仅供量级参考。

口径与说明：

- 吞吐 = **per-rank remote bytes / 时间** = `numel * 2 * (ws-1) / ws` ÷ 耗时（与 ThunderKittens benchmark 口径一致，见 `benchmark/bench_uniform.py`）；时延为 20 次迭代的 CUDA event 中位数。
- 复现：`PROF_N=75600 PROF_H=40 PROF_D=128 PROF_MODE=0 torchrun --nproc_per_node=8 benchmark/bench_uniform.py`。

### QK 融合算子（8×H200，bf16，Wan n_global=40 / d=128，cross-head + interleaved）

`benchmark/bench_qk_fused.py`（ws=8，ms/iter）：`a2a` 为纯搬运下界；`unfused` = 独立 rms_norm + rope + a2a（Wan 现状）；`fused` = `all_to_all_single_4d_qk`；`qk2` = q、k 一次调用（共享 barrier）。

| seq_global | a2a | unfused | fused | fused/unfused | qk2 | qk2 vs 2×fused |
| --- | --- | --- | --- | --- | --- | --- |
| 20480 | 0.109 | 0.216 | 0.150 | **1.44×** | 0.281 | 0.94× |
| 46080 | 0.199 | 0.420 | 0.288 | **1.46×** | 0.523 | 0.91× |

- 融合把 norm+rope 的额外开销（相对纯 a2a）从 unfused 的 ~0.11/0.22 ms 压到 ~0.04/0.09 ms。
- 对 Wan 实际的 q+k 场景：`qk2` 0.523 ms vs unfused 的 2×0.420=0.840 ms → **1.61×**。
- 复现：`torchrun --nproc_per_node=8 benchmark/bench_qk_fused.py`。
