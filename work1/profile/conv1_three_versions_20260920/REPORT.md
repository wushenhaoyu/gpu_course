# Conv1 + Pool1 三版本 NCU 报告

## 范围与环境

本报告只比较 LeNet-5 的第一个 `Conv(1,6,5x5) + ReLU + MaxPool(2x2)` 阶段；输入批量为 1000，三个版本都使用相同的输入及权重。设备为 NVIDIA A800（SM80），CUDA 12.4，Nsight Compute 2024.1。

每份 `full_*.ncu-rep` 使用 `--set full`，并额外采集 `PmSampling` 与 `PmSampling_WarpStates`。在当前 NCU 版本中，这个集合列出约 1,187 行 metric 请求，实际导出 1,226 个带数值/实例的 metric 字段；这是 NCU App 常规“Full”分析所需的完整报告。`source_*.ncu-rep` 使用 `--set source --section SourceCounters`，用于 source/assembly 对照。

| 标签 | 实现 | kernel |
|---|---|---|
| direct | 一个线程计算一个 pool 输出，4 个卷积结果放寄存器后立即 MaxPool | `conv1_pool1_kernel` |
| smem_c | scalar implicit GEMM，整个 pre-pool C (32x576) 放 shared memory 后 pool | `conv1_igemm_smem_c_pool_kernel` |
| tiled | 96-column 分块、K=8 双缓冲 scalar implicit GEMM，C tile 放 shared memory 后 pool | `conv1_igemm_tiled_pool_kernel` |

## 可在 NCU App 直接打开的报告

| 版本 | 全指标报告 | Source Counters 报告 |
|---|---|---|
| direct | `reports/full_direct_b1000.ncu-rep` | `reports/source_direct_b1000.ncu-rep` |
| smem_c | `reports/full_smem_c_b1000.ncu-rep` | `reports/source_smem_c_b1000.ncu-rep` |
| tiled | `reports/full_tiled_b1000.ncu-rep` | `reports/source_tiled_b1000.ncu-rep` |

在本机 NCU App 中将这六个 `.ncu-rep` 拷贝过去后用 `File -> Open` 打开即可。每个 `full` 报告只有目标 kernel 的一次 capture，便于横向比较。注意 NCU 还能枚举大量 raw counter 的 suffix/instance 派生名；它们不是可一次性采集的独立指标，因此没有把约 40 万个派生命名空间逐一加入 replay。

## 关键指标

| 指标 | direct | smem_c | tiled |
|---|---:|---:|---:|
| Duration (ms) | 0.1451 | 0.4114 | 0.3918 |
| SM Throughput (% peak) | 51.73 | 60.27 | 83.41 |
| Memory Throughput (% peak) | 90.98 | 90.58 | 87.02 |
| L1 hit rate (%) | 96.22 | 8.57 | 84.32 |
| DRAM Throughput (% peak) | 1.73 | 0.60 | 0.77 |
| Executed instructions | 29.38 M | 56.58 M | 94.68 M |
| Registers / thread | 32 | 30 | 79 |
| Theoretical occupancy (%) | 100.0 | 25.0 | 37.5 |
| Achieved occupancy (%) | 85.24 | 23.73 | 36.50 |
| Largest stall | LG throttle 19.48% | MIO throttle 6.17% | MIO throttle 3.92% |

完整字段导出为 JSON：

- `analysis/metrics_all_direct_b1000.json`
- `analysis/metrics_all_smem_c_b1000.json`
- `analysis/metrics_all_tiled_b1000.json`

汇总和 NCU rule-engine 原始解释分别在 `analysis/metrics_key_direct_b1000_smem_c_b1000.txt`、`analysis/details_*.txt` 与 `analysis/stall_hotspots_*.txt`。

## 解读

1. **direct 最快**：0.1451 ms；分块版本仍慢 2.70x，完整 C 的 shared-memory 版本慢 2.83x。对于这个很小的 Conv1，额外的 implicit-GEMM 变换、C 写回/读取和同步成本超过了数据复用收益。
2. **direct 的下一优先级是 global-load 合并访问**：NCU 报告 global load 平均仅 11.9 B/sector、32.26 M 个 excessive sectors，给出约 48%--57% 的局部优化估计；同时 LG throttle 为主 stall。这个估计不能直接相加或等同最终加速比，但明确表明访存布局是主问题。
3. **smem_c 被 shared-memory 容量限制**：其动态 shared memory 约 80.19 KiB，occupancy 降至约 24%，并有约 14.98 M 个 excessive shared-memory wavefronts；其中 shared memory 读访问仍不够规整，MIO throttle 成为主 stall。
4. **tiled 恢复了多数 L1 命中率但寄存器代价高**：其 24.64 KiB dynamic shared memory 比 smem_c 小得多，但 79 registers/thread 使 occupancy 仅约 36.5%。它还有 4.20 M excessive global sectors（规则引擎估计其影响显著）。
5. **作为 Tensor Core 前置结构，tiled 仍有价值**：N=96/K=8 的 tile 与显式 shared-memory stage 可改造成 WMMA/`mma.sync` 路径；但在转换完成前，scalar tiled GEMM 不应替换 direct baseline。

## Pool bank-conflict 后续实验

初始 tiled 版的 pool epilogue 让同一 warp 同时跨两个 24-float convolution 行读取 `c=2*pc`；其两条 `LDS.64` 均为 2-way bank conflict。将 epilogue 改为“一个 warp 只处理一个 `(channel, pooled-row)`，仅 lanes 0--11 有效”后：

| 指标 | 初始 tiled | warp-owned pool |
|---|---:|---:|
| Duration (ms) | 0.3918 | 0.4018 |
| Shared excessive wavefronts | 1,026,000 | 450,000 |
| Shared bank conflicts | 1,027,985 | 452,395 |

对应报告为 `reports/full_tiled_warp_pool_b1000.ncu-rep`，全量字段为 `analysis/metrics_all_tiled_warp_pool_b1000.json`。冲突下降约 56%，但因每 warp 只有 12 个有效 lane，时间反而增加约 2.6%。这说明 bank conflict 是真实问题，但 tiled 标量版的总瓶颈还包括 lane 利用率、MIO 压力与较高寄存器数；最终方案应使用 WMMA/mma tile 或重新设计线程到 pool 输出的映射，而不是仅增加 padding。

## 复现

执行命令保存在 `commands.txt`。采集时将 Python 环境的 `bin` 放到 `PATH` 前面，并设置独立的 `TMPDIR`，避免多用户机器上的 NCU lock 冲突。
