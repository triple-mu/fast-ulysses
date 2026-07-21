# A2A kernel：TMA vs non-TMA 深层性能分析（8×H200）

环境：hyper00 节点，8×H200（NV18/NVSwitch，450 GB/s 单向/卡），容器 `sglang-diffusion-ulysess`，CUDA 12.8。
口径：per-rank NVLink egress = `numel*2*(ws-1)/ws ÷ t`。基准 shape：mode0，ws=8，b=1，H=128，D=128，bf16。
复现材料：`harness/a2a_harness.cu`（忠实复刻两个 kernel，逐位对拍通过；含 `--queued`/`mp` 等隔离模式）、`lib_experiment.patch`（对库的临时实验改动，已还原）、文本数据 `reports/`（原始 `.ncu-rep`/`.csv` 体积大，归档在仓库外，需要时联系维护者）。
本报告经 4 视角对抗审查修正过一轮；审查指出的因果缺口（lib vs harness 数字不闭合）由后续 `mp` 实验最终闭合。

## 0. 现象校准（bench_uniform，每迭代含 quiet+barrier）

| GB/s | N=16K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|
| ws=8 mode0 non-TMA | **302** | **321** | **314** | **290** | **299** |
| ws=8 mode0 TMA | 167 | 180 | 298 | 285 | 275 |
| ws=4 mode0 non-TMA | 334 | 319 | 307 | 321 | 317 |
| ws=4 mode0 TMA | 305 | **341** | **342** | **333** | 316 |
| ws=8 mode1 non-TMA | 301 | 314 | 311 | 292 | 287 |
| ws=8 mode1 TMA | 156 | 178 | 242 | 285 | **308** |

Wan shape（H=40, N=75600, mode0, ws=8，用户真实负载）：non-TMA **305** vs TMA 261。
「non-TMA 差一些」仅在 ws=4 中大 N 与 mode1 大 N 成立（TMA +4~11%）；ws=8 多数点与 Wan shape 上 non-TMA 显著更快。

## 1. 被否证的假设（附实验）

| 假设 | 实验 | 结果 |
|---|---|---|
| 16B st.global 粒度损失线级效率 | pair GPU0→GPU1 512MB + nvlink Data/Raw 计数 | SM 直写 366-372 ≈ TMA 368-372（DMA 参照 396）；Data=payload；Raw/Data non-TMA 1.133 / TMA 1.130 / DMA 1.064。无实质差别 |
| src 读延迟是 non-TMA 瓶颈 | write-only 变体（去读、地址流不变） | 339 vs 335 GB/s，无差别 |
| SM 发射能力不足 | localdst 变体（全写本地 HBM） | 1613 GB/s，5× 富余 |
| 全双工（链路双向）是主要争用 | ws=2 双向 vs 单向 | 352 vs 356，仅 -1% |
| NVSHMEM 映射类型（VMM/POSIX-FD） | harness `--alloc` × host-paced/queued | 全部与 cudaMalloc 相同 |
| rank 间配置混跑是 bench 崩溃主因 | `--mixn` + 实测 pick | 混跑有害但非主因（N=16K 全 rank 同 pick 仍崩） |
| 坏均衡有滞回 | `seq` 先坏后好配置 | 无滞回，配置内在 |
| NCCL NVLS 多播干扰 | `NCCL_NVLS_ENABLE=0` | 无变化 |

注：ncu stall 计数（lg_throttle 20.7 等）采自锁频串行化 replay（比真实工作点慢 2.3×），且 write-only 变体 lg_throttle 降至 8.4 而吞吐不变——stall 读数是 egress 受限的症状而非成因，仅作定性参考。ncu 按字节计的量（读放大）可靠。

## 2. 三层因果链（每层有独立的干预实验）

### 第 1 层：多对多扇入下存在「聚合在飞远端写」阈值，越界即拥塞塌陷

- non-TMA 并发 factor 曲线呈倒 U + 负斜率：f4=278 → f12=**337** → f32=**193** GB/s；单发送方全平坦（~357）。
- TMA 侧同一旋钮是驻留块数：崩溃配置 tn15/ts1/stg4/bdiv4（13 blk/SM，224 GB/s）用 `--smempad` 压到 3 blk/SM 后**治愈到 364**。
- ws=2 对照排除链路双工；机制定位在 7×7 扇入的接收端/交换机排队（微架构细节不可观测，因果方向由干预实验确立）。

### 第 2 层：相位锁定 lockstep 是触发器

bench_uniform 的计时循环将 20 个迭代（kernel+quiet+barrier）全部入队、不落 host；barrier 链使 8 rank 每迭代同时起跑，prologue 突发同时冲击 fabric。harness `--queued` 复刻后：

| 配置 | host-paced（错相位） | queued（锁相位） |
|---|---|---|
| TMA tn8/ts1/stg8/bdiv2, N=16K | 179.5μs | 452.2μs |
| TMA tn15/ts1/stg4/bdiv4, N=32K | 523μs | 649.0μs（=lib tuner 测得 648.7 ✓） |
| TMA tn16/ts4（16KB tile）, N=32K | 325μs | 325.3μs（单进程下免疫） |
| non-TMA th512/f12, N=32K | 348.6μs | 536.3μs |
| non-TMA th256/un8/f32, N=32K | 350.8μs | 362.6μs（=lib bench 365.5 ✓） |

**non-TMA 的 tuner 本身就在锁相位环境里测量**（microbench 同样入队式），选出的 th256/un8/f32 天然抗锁定——这是 non-TMA 在 bench 里稳定 300-325 的原因。

### 第 3 层（最终根因）：跨进程导入的 peer 映射让 TMA bulk store 失去免疫区

单候选强制实验（`FAST_ULYSSES_TMA_ONLY_BIGTILE=1`，全 rank 同 pick）证明 lib 里连"免疫配置"也跑 453μs——于是用 `mp` 模式（fork 8 进程 + cudaIpc 句柄交换，复刻 torchrun 部署形态）做最终隔离：

| 配置（mode0 ws=8 N=16K） | 单进程原生映射（queued） | **多进程 IPC 映射（mp）** | 真实 lib |
|---|---|---|---|
| TMA 大 tile (16,4)/stg8/bdiv2 | **167μs / 351 GB/s** | 451.4μs / 130 | 456.7μs ✓ |
| TMA 小 tile (8,1)/stg8/bdiv2 | 452μs | 450.1μs | 452.8μs ✓ |
| non-TMA th256/un8/f32 (N=32K) | 362.6μs | 361.8μs / 325 | 365.5μs ✓ |

**结论：`cp.async.bulk.tensor` 写跨进程导入的 peer 内存（cudaIpc 与 NVSHMEM 的 VMM-FD 导入同病）在锁相位扇入下塌陷到 ~130 GB/s，且对 tile/stages/blocks 配置不敏感——TMA 的免疫区只存在于单进程原生映射；`st.global` 直写对进程拓扑完全不敏感（361.8 vs 362.6）。** 大 N/ws=4 部分恢复（mp (15,1)@N=256K=275 GB/s、ws=4=329，均与 bench 一致）。

推论：仓库原来的 stages=4/bdiv=4/ts=1"实测 DiT 最优"默认值在部署域（多进程）其实是合理的；把单进程 harness 的最优配置（大 tile 深流水）搬进库反而使小 N 更糟（129 GB/s，实测验证）——**错域调优**。

## 3. 次级问题（真实存在，但非主因）

1. mode0 默认 tile_n=n_local-1 有 **1.78× DRAM 读放大**（ncu 239MB vs 134MB；解析式 (7×30+16)/128=1.77 吻合；机制：第二 n-tile 在 src 侧到 n_global 才裁剪、dst 侧在 n_local 裁剪，93% 读入即弃）。不直接压吞吐（单发 355 不变，~26μs @HBM），属效率/功耗问题。
2. autotune per-rank 独立选择（`resolve_config` 无跨 rank 协商）：N=32K 实测 pick 混合 tn15/tn16；harness 复刻混跑劣化至两 uniform 之间（439-489μs vs 405/527）。
3. TMA microbench 无 OOR probe（non-TMA 有）：候选 smem 超限会直接抛异常（实验中触发过 crash）。

## 4. 结论（回答「non-TMA 为什么差一些」）

1. **写法本身没有差别**：单点写、线级效率、发射能力全部等价（§1）。
2. **在这台 8×H200 的真实部署形态（torchrun 多进程 + NVSHMEM 堆）下，TMA 路径受硬件/驱动层的「跨进程映射 × bulk store × 锁相位扇入」塌陷限制**，small-N ws=8 被压到 126-180 GB/s；non-TMA 完全不受此影响，稳定 300-325。auto 模式选 non-TMA 是对的。
3. 「non-TMA 差一些」成立的区间（ws=4 中大 N：TMA 341-342 vs 319-307；mode1 256K：308 vs 287）：扇入较小或 tile 相对较大时 IPC-TMA 未塌陷，bulk 传输在拥塞下的优势兑现，+4~11%。
4. **能力上限意义上 TMA 确实更高**（单进程 362-371 vs non-TMA 335-341，且 SM 占用 1.56% vs 47.6%），但在多进程部署里兑现不了 ws=8 小中 N 的部分。
5. non-TMA 已接近其结构位置：pair 上限 370（DMA 的 93%）→ 扇入争用 ~335 → 锁相位安全点 ~324。

## 5. 可行动项

1. （低成本、正收益）**tuner 跨 rank 一致化**：rank0 决策广播或 allreduce 取 max 再选，消除 pick 发散。
2. （低成本）**TMA microbench 加 OOR probe**（同 non-TMA 的 probe 模式）。
3. （效率）mode0 默认 tile_n 改用整除 tile，消除 1.78× 读放大。
4. （方向性）TMA 路径想在 ws=8 小 N 翻身需要驱动/硬件层面解决跨进程映射的 bulk-store 塌陷（可给 NVIDIA 提 bug：最小复现 = `a2a_harness mp` vs `--queued` 对照，同配置 451μs vs 167μs）；或改用 copy-engine（cudaMemcpyPeerAsync batch）路径试探。
5. Wan shape（n_local=5, s_local=9450）：行 bytes 仅 1.25KB 且 s_local 整除性差，即使在单进程域 TMA 候选也难做大 tile —— **继续用 non-TMA（auto 现状）**。

## 6. 对 e2e（sglang Wan）的含义

- 纯通信：现状（auto→non-TMA，305 GB/s）已是此平台可达的合理水平，无行动项。
- 重叠：TMA 的 SM 占用 1.56% 依然是与计算重叠的正确形态，但 cooperative GEMM 不释放 SM slot 的问题（见 memory: coop-gemm-blocks-a2a-overlap）对 1-thread block 同样存在；且多进程塌陷使其带宽劣势明显，重叠收益需在非 cooperative 窗口且大 N 场景再评估。

## 7. 未闭合项（诚实边界）

1. 「跨进程映射 × TMA bulk store」塌陷的微架构机理（页表属性差异？异步代理的地址翻译路径？）在 ncu/nsys 观测能力之外；行为级证据（mp vs queued 同配置 2.7× 差）已足够扎实，可作为 driver bug report 的最小复现。
2. `mp` 用 legacy cudaIpc，NVSHMEM 用 cuMem FD 导入；两者都复现同样的数字（450≈453/456），等价类成立，但未逐一枚举所有导入方式。
3. mode1/ws=4 未逐点做 mp 对照（其 bench 行为与机制预测一致：ws=4 mp 329 ≈ bench 342）。
