# `conv_pool_relu_kernel` (conv1 调用) Profiling Report

**Kernel:** `conv_pool_relu_kernel(const float*, const float*, const float*, float*, int×7)`
**Target GPU:** NVIDIA A800-SXM4-80GB (108 SM, CC 8.0, 2.0 TB/s HBM2e, 19.5 TFLOPS fp32)
**Nsight Compute:** 2024.1.0 (CUDA 12.4)
**Compile flags:** `nvcc -O3 -lineinfo`（torch `cpp_extension.load` 默认，`-gencode` 由框架按 sm_80 选定）
**Profile date:** 2026-09-19
**Run directory:** `profile/v1_convpool/`

被剖析的是 v1 融合版 kernel 的**第一次**发射（conv1：`CI=1, CO=32, IH=IW=24, OH=OW=12`），grid `(180000,1,1)` × block `(256,1,1)`。同一 kernel 的 conv2 调用（grid `(204800,1,1)`）在本次报告中**未**覆盖。

**源码：** `lenet5/cuda/versions/v1_convpool/v1_convpool.cu`（2026-09-19 改名前叫
`iterations/v1_convpool/lenet_kernels.cu`，内容未变）。

---

## 0. Profiling setup

- **Harness:** `harness/prof_driver.py` —— 不是独立 C++ driver，而是直接 import 生产用的 `model_cuda.py`，加载 `versions/v1_convpool/v1_convpool.cu`，跑**一次完整前向**。
  这样做的理由：conv1 的输入张量就是生产路径上真实的那一份（真实的 `[10000,1,28,28]` 连续显存、真实的 dtype/对齐），而不是一个手搓的合成 shape。ncu 自己 replay，所以"只跑一次"足够。
- **Workloads:** Fashion-MNIST 测试集前 10000 张（`t10k-images-idx3-ubyte`），`x: fp32 [10000,1,28,28]`，`torch.no_grad()`。
- **Dispatch paths covered:** `(ci=1, co=32, ih=iw=24, oh=ow=12), grid=(180000,1,1), block=(256,1,1)` —— 仅此一条。
- **Metric-name caveats:** 本机是 sm_80，与 `ncu-report-skill` 里 B200/sm_100 的参考不同：
  - `sm__inst_executed.sum` 不存在 → 用 `smsp__inst_executed.sum`（**量纲是 warp-instruction，不是 thread-instruction**）；
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed` 不存在 → DRAM 占用率 = `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` + `dram__bytes_write.sum.pct_of_peak_sustained_elapsed`；
  - `gpu__time_duration.sum` 单位是 **ns**；
  - "Theoretical Occupancy" 不是 metric，由 `launch__occupancy_limit_*` 推算（见 `analysis/analyze_reports.py`）。

    # Profile (full + PM)
    ncu --set full --section PmSampling --section PmSampling_WarpStates \
        -k "regex:conv_pool_relu_kernel" -c 1 \
        -o profile/v1_convpool/reports/full_v1 \
        python profile/v1_convpool/harness/prof_driver.py

    # Profile (source-level)
    ncu --set source --section SourceCounters \
        -k "regex:conv_pool_relu_kernel" -c 1 \
        -o profile/v1_convpool/reports/source_v1 \
        python profile/v1_convpool/harness/prof_driver.py

环境（跑 ncu 前必须设，否则报 Ninja 缺失 / lock file 冲突 / ncu_report 找不到）：

    export PATH=/data/workspace/haoyu/software/miniconda3/envs/kernel/bin:$PATH
    export TMPDIR=/tmp/haoyu_ncu && mkdir -p $TMPDIR
    export PYTHONPATH=/usr/local/cuda-12.4/nsight-compute-2024.1.0/extras/python

### Artifacts

    profile/v1_convpool/
    ├── REPORT.md                      ← 本文件
    ├── harness/prof_driver.py         ← 单次前向 driver
    ├── reports/full_v1.ncu-rep        ← 全量 + PM sampling
    ├── reports/source_v1.ncu-rep      ← source counters
    └── analysis/
        ├── details_v1.txt             ← ncu 规则引擎原文
        ├── stall_hotspots_v1.txt      ← 逐源码行 stall 归因
        ├── source_page_v1.txt         ← source 页原始 dump
        ├── sass_v1.txt                ← cuobjdump -sass 反汇编
        └── metrics_key_v1.{txt,json}  ← 关键指标提取

---

## 1. Headline numbers

| Metric | conv1 | Source |
|---|---:|---|
| **Duration** | **1.97 ms** | `gpu__time_duration.sum` (ns) |
| SM throughput (% peak) | 61.04% | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| **Memory (L1/TEX) throughput (% peak)** | **95.13%** | `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` |
| L2 throughput (% peak) | 7.77% | `lts__throughput...` |
| DRAM throughput (% peak) | 5.26% | read+write pct |
| L1 hit rate | 97.62% | `l1tex__t_sector_hit_rate.pct` |
| L2 hit rate | 94.40% | `lts__t_sector_hit_rate.pct` |
| Bytes / sector (global ld) | **12.14 of 32** | `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` |
| Global-ld sectors | 923,465,296 | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` |
| 其中 excessive（纯浪费） | **537,600,000 (58%)** | `derived__memory_l2_theoretical_sectors_global_excessive` |
| Executed instructions | 591,840,000 (warp-inst) | `smsp__inst_executed.sum` |
| Issue slots busy | 61.04% | `sm__issue_active...` |
| FMA pipe (% of peak) | 44.25% | `sm__pipe_fma_cycles_active...` |
| **ALU pipe (% of peak)** | **41.06%** | `sm__inst_executed_pipe_alu...` |
| Tensor Core usage | 0% (n/a) | 无 MMA |
| Reg / thread | 40 | `launch__registers_per_thread` |
| Theoretical / Achieved occupancy | 75% / 66.02% | 由 `launch__occupancy_limit_registers` 推算 |
| Waves / SM | 277.78 | `launch__waves_per_multiprocessor` |
| Warp cycles / issued instr | 17.23 | `smsp__average_warp_latency_per_inst_issued.ratio` |

**One-line read:** 这个 kernel **卡在 L1 的 sector 吞吐上** —— L1 请求率 95.30%，而 DRAM 只有 5.26%、L2 只有 7.77%、FMA 只有 44%。而且 58% 的 sector 是浪费的（每 sector 只有 12.14/32 字节有用），根因是**同一条 warp 的 32 个 lane 映射到连续的池化列，于是读 x 时列步长是 2**。降不下去的不是算力，是 L1 每周期只能处理 4 个 sector 这件事。

---

## 2. Per-dimension analysis

### 2.1 SM occupancy & launch geometry

- grid 180,000 blocks × 256 threads = 46,080,000 threads；108 SM → **277.78 waves/SM**，wave 数极多，尾部效应可忽略（见 2.2）。
- 40 registers/thread → `launch__occupancy_limit_registers = 6` blocks（256 线程/块 = 8 warps → 6×8 = 48 warps/SM），**theoretical occupancy 75%**，比硬件上限 64 warps 低 1/4。ncu 规则引擎给出的局部收益估计：`Est. Local Speedup: 25%`。
- **achieved 66.02%**（42.25 warps/SM active），与 75% 的理论值差 9 个点——这 9 个点不是 launch 配置造成的，是 warp 长期停在 stall 上（见 2.3）。
- 这里要注意：**提高 occupancy 不是第一优先级**。kernel 现在是 L1 吞吐瓶颈，L1 已经 95% 满载；再塞更多 warp 进去只会让 L1 队列更长，不会更快。先解决 sector 浪费，occupancy 自动会被"腾出来的延迟"吸收。

### 2.2 Thread-block balance (tail effect)

- `Average SM Active Cycles` 2,240,653.56 vs `Elapsed Cycles` 2,245,357 → **SM 活跃占比 99.79%**，没有空闲气泡。
- 277.78 waves 意味着即使最后一个 wave 完全空跑也只损失 ~0.36% —— 不存在尾部失衡。
- 对照：`Average DRAM Active Cycles` 仅 165,130.8 / 2,245,357 = **7.4%** —— DRAM 绝大多数时间在闲置。

### 2.3 Instruction-level stall analysis

平均每条发射指令之间隔 17.23 cycle（`smsp__average_warp_latency_per_inst_issued.ratio`），拆解如下（`analysis/metrics_key_v1.txt`）：

| stall 原因 | cycles/instr | 占比 |
|---|---:|---:|
| **lg_throttle** | **5.404** | **31.4%** |
| not_selected | 3.738 | 21.7% |
| long_scoreboard | 2.092 | 12.1% |
| wait | 1.687 | 9.8% |
| math_pipe_throttle | 1.340 | 7.8% |
| mio_throttle | 0.420 | 2.4% |

`lg_throttle` 的定义是「L1 的 local/global 指令队列满了」——**这是"内存指令发得太多/太挤"的直接证据，不是延迟问题**。ncu 对此的估计：

> `Est. Local Speedup: 31.36%` — "each warp spends 5.4 cycles being stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full."

逐源码行归因（`analysis/stall_hotspots_v1.txt`，总采样 23 个 (file,line) 条目）：

| 排名 | 采样数 | 行 | 源码 | 主要 stall |
|---:|---:|---:|---|---|
| 0 | 45257 | **78** | `float a00 = b[oc], a01 = b[oc], ...` | lg_throttle 31680 |
| 1 | 31062 | 88 | `a11 = fmaf(xp[(i+1)*IW+j+1], wv, a11)` | lg_throttle 10554, long_scoreboard 8753 |
| 2 | 28465 | 86 | `a01 = fmaf(xp[i*IW+j+1], wv, a01)` | lg_throttle 9984, long_scoreboard 6461 |
| 3 | 20650 | 84 | `const float wv = wp[i*5+j]` | lg_throttle 11388 |
| 4 | 20055 | 69 | `const int oc = (idx/(OH*OW)) % CO;` | not_selected 6027, wait 4685, **math_pipe_throttle 4097** |
| 5 | 19142 | 87 | `a10 = fmaf(xp[(i+1)*IW+j], wv, a10)` | long_scoreboard 8796 |
| 7 | 10416 | 67 | `const int pc = idx % OW;` | not_selected 3632, math_pipe_throttle 2212 |
| 8 | 7232 | 68 | `const int pr = (idx/OW) % OH;` | math_pipe_throttle 1410 |

**⚠️ 修正上一轮的一个错误假设。** 我此前猜"第 78 行有 4 次冗余的 `b[oc]` 全局载入"。**SASS 反汇编证明这是错的**（`analysis/sass_v1.txt`）：

    /*03c0*/  LDG.E.CONSTANT R14, [R14.64] ;     ← 只有 1 条 load
    /*03f0*/  MOV R29, R14 ;  MOV R4, R14 ;  MOV R27, R14 ;   ← 另外 3 份走寄存器

编译器已经做了 CSE。第 78 行之所以采样最多，是因为它是整个除法依赖链之后**第一条**内存指令——warp 一旦解出 index 就全部挤在这里抢 LSU 队列，样本自然堆在这一行。它是**症状**（L1 队列满），不是病因。

同理，第 67–70 行的高采样暴露的是**第二个独立问题**：32 位软件除法。

### 2.4 Tensor Core utilization

0%，n/a。kernel 里没有任何 `mma` 指令，也没有 fp16/tf32 路径——本任务要求与真 fp32 参考对齐到 1e-4，TF32 的 10 位尾数不满足（用户已实测：TF32 0.9335 vs 真 fp32 0.9334 的精度差可以被接受，但那是另一条路线（方案 A），需要先确认 `mma.sync` 是否算"使用 NVIDIA 库"）。

### 2.5 SM utilization timeline

PM Sampling 已采集（`reports/full_v1.ncu-rep`，4 个 pass group，0 dropped samples）。

从周期级指标看形状是 **flat-high**：`SM Active / Elapsed = 99.79%`，`SMSP Active` 2,239,119.79 / 2,245,357 = 99.72%，全程满载、没有爬坡、没有尾部塌陷。瓶颈是稳态的 L1 吞吐，不是启动/收尾开销。`Est. Speedup 25%` 那条 occupancy 提示因此**不是**"wave 之间有空隙"，而是"每个 warp 的 stall 太长时 SM 填不满"。

### 2.6 Memory access pattern

这是全篇的核心。

**per-instruction 视角**：kernel 静态 256 条 SASS 指令，其中 **86 条 `LDG.E.CONSTANT`**（全部是 32-bit 标量，无向量化）：

| 类别 | 条数 | 说明 |
|---|---:|---|
| 权重 `w` | 25 | 基址 `R20`，正好 25 个 tap |
| bias `b` | 1 | 已 CSE |
| 输入 `x` | 60 | 见下 |

x 的 60 条 load 对应**只有 36 个不同的地址**（6×6 输入窗口）。因为 `IW` 是运行时参数，编译器无法证明 `i*IW+j+1` 与 `(i+1)*IW+j` 重合，于是 1.67× 冗余。**修好这一步需要让循环上界/步长变成编译期常量或把窗口搬进 shared memory。**

**per-sector 视角**（真正定性的）：

    warp 数 = 46,080,000 / 32 = 1,440,000
    每个 warp 的 global-load 指令数 = 86
    每个 warp-load 生成的 sector = 923,465,296 / (1,440,000 × 86) = 7.46
    完美合并时 32 lane × 4B = 128 B = 4 个 sector

    → 1.87× 的 sector 浪费，正好对上 12.14/32 = 38% 的字节利用率

**根因**：`idx` 连续 → `pc = idx % OW` 连续 → 输入列 `c0 = 2*pc` 以 **步长 2** 递增。32 个 lane 覆盖 64 个 float = 256 字节 = 8 个 sector，实际只用 128 字节。权重那边（`wp[i*5+j]`，同一 warp 内 `oc` 相同）反而是完全 broadcast 的，没有浪费。

**L1 roofline 核算——这是全篇最有力的一段：**

    L1 需要处理的 sector 数          = 923,465,296
    L1 可用容量 = 108 SM × 2,240,653 活跃周期 × 4 sector/周期 = 967,962,000
    →  923,465,296 / 967,962,000 = 95.4%

实测 `L1/TEX Cache Throughput = 95.30%`。**两者几乎完全相等** —— 这个 kernel 正跑在 L1 sector 处理能力的屋顶上，一个周期都不多。想变快，**只能减少 sector 数量**，没有别的路。

**DRAM / L2 完全不是问题**：L1 命中率 97.62%，L2 命中率 94.40%，DRAM 只跑 5.26%（读 0.79% + 写 4.48%）。`Mem Busy 95.13%` 指的是 L1 那条通路的 busy，不要被名字误导成"内存带宽满了"。**没有任何寄存器溢出**（`Block Limit Shared Mem 32`、无 local memory 相关 stall、DRAM 写只有输出张量那一份）。

### 2.7 Additional findings（ncu 规则引擎原文）

| 规则 | Est. Speedup | 指向 |
|---|---:|---|
| `The memory access pattern for global loads from L1TEX might not be optimal. On average, only 12.1 of the 32 bytes transmitted per sector are utilized` | **59.04%** | stride-2 的 x 访问 |
| `This kernel has uncoalesced global accesses resulting in a total of 537600000 excessive sectors (58% of the total 930400000 sectors)` | **50.07%** | 同上，**同源，两条不能相加** |
| `each warp spends 5.4 cycles stalled waiting for the L1 instruction queue for LG memory operations` | 31.36% | 同上，同源的第三种表述 |
| `Theoretical occupancy (75.0%) is limited by the number of required registers` | 25% | 40 regs/thread |
| `FMA is the highest-utilized pipeline (44.3%) ... should not be a bottleneck` | — | 算力不是瓶颈 |
| Roofline: `achieved 30% of this device's fp32 peak` | — | 与 FMA 44.25% 一致 |

另外 ncu 没点名、但 PIPE 指标暴露出来的：**ALU（整数）pipe 占峰值 41.06%，几乎追平 FMA 的 44.25%**。配合第 67–70 行的 `math_pipe_throttle` 采样，可以确认**索引计算本身消耗掉了和浮点计算同量级的发射带宽**。看 SASS 就更直白——kernel 里有 **5 组 `MUFU.RCP` + 5 个 `I2F` + 5 个 `F2I`**，是 4 次 `idx % X` / `idx / X` 被展开成的完整软件除法（60 条 `IADD3` + 49 条 `IMAD` 里有很大一部分是这个）：

    /*00f0*/  I2F.RP R4, R9 ;
    /*0100*/  MUFU.RCP R4, R4 ;
    /*0130*/  F2I.FTZ.U32.TRUNC.NTZ R3, R2 ;
    ...                                    ← 一组 ≈ 15 条指令，共 4 组
    /*2190*/  ...
    /*21a0*/  ...

对 46M 个线程来说，每条线程跑一遍这 4 组除法是**纯浪费**——它只依赖 `idx`，而一个 warp 的 32 个线程共享同一组常量。

---

## 3. Summary diagnosis

| Factor | conv1 状态 | 影响 |
|---|---|---|
| **L1 sector 吞吐** | **95.30%，已到屋顶（理论值 95.4%）** | **#1，决定性** |
| Sector 浪费率 | 58%（12.14/32 字节） | #1 的成因 |
| lg_throttle stall | 5.404 / 17.23 cycle = 31.4% | #1 的表现 |
| 索引整数运算（ALU pipe） | 41.06% of peak，4 组软件除法 | #2，与 #1 独立 |
| x 载入冗余 | 60 条 load / 36 个地址 = 1.67× | #2 的成因之一 |
| Occupancy | 理论 75%（被 40 reg 限），实测 66.02% | #3，**现在修它无效**（L1 已满） |
| FMA 算力 | 44.25% | 不是瓶颈 |
| DRAM / L2 带宽 | 5.26% / 7.77% | 完全不是瓶颈 |
| 尾部失衡 | SM 活跃 99.79%，277.78 waves | 可忽略 |

一句话：**把 sector 数降下来是第一优先级；把索引除法干掉是第二优先级；occupancy 和 DRAM 都不用管。**

---

## 4. Optimization directions (ranked by impact)

### Priority 1 — 让 warp 内的 lane 沿输入行连续取数（shared memory tiling）

把 `x` 的输入 tile 用**合并访存**一次性搬进 shared memory，然后每个线程从 smem 里读自己那个 stride-2 的 5×5 窗口。

具体地，以 `(blockIdx)` 覆盖一个输出 tile：block 载入 `IH×IW` 的一个连续行段（32 lane 读 32 个连续 float = 4 sector，100% 利用率），`__syncthreads()` 之后每个线程从 smem 取 36 个不同的 x 值。

**为什么这一条能同时解决三件事：**
1. **Sector 数**：global load 从 61 条/warp 降到"每线程 36/32 个 float"的合并载入，sector 利用率从 38% 提到接近 100% → L1 压力降到 ~50% 以下，出屋顶。
2. **冗余 load**：x 的 60 条 → 36 个 smem 读（smem 不按 sector 计费，只按 bank conflict）。
3. **寄存器压力**：x 不再需要 40 个 in-flight load，reg/thread 有下降空间 → occupancy 顺带改善。

**Evidence:**
- `L1/TEX Cache Throughput 95.30%`，理论屋顶 95.4%（见 2.6 的核算）——已无余量。
- 每 sector 只有 `12.14/32` 字节有用；`537,600,000` excessive sectors = 58%。
- ncu 规则 `Est. Speedup: 59.04%` / `50.07%`（同源，取其一）。
- 每 warp-load 生成 `7.46` 个 sector，理想值是 4。

**Expected impact:** conv1 1.97 ms → 乐观 1.2–1.3 ms（ncu 的 50% 是最乐观上界；实际会被 FMA 44% + ALU 41% 的新瓶颈接住）。端到端影响 = conv1 占当前 24.73 ms 的 8%，所以**端到端只有 ~2–3%**——但这是 conv2 的同款修复的前置验证，conv2 占 47%，所以**真正值钱的是把它复用到 conv2**。

**Effort:** medium。要重写 thread mapping：block 覆盖一块输出 tile，加 `__shared__ float tile[]`、协作载入、`__syncthreads()`、边界处理。约 40 行。

### Priority 2 — 干掉运行时整数除法（索引重算）

现在 `pc/pr/oc/im` 四个坐标各自展开成一套 `MUFU.RCP + I2F + F2I` 软件除法（SASS 中 5 组，约 60 条指令/线程），而 `OW=12, OH=12, CO=32` 在 conv1 里是**编译期已知**的。

两条路，建议都做：
- **(a) 用 3D grid 拿掉两层**：`blockIdx.y = im`（图像）、`blockIdx.z = oc`（输出通道），线程内只剩 `pc = tid % OW`、`pr = tid / OW`——而且如果让 `blockDim.x` 恰好等于 `OW`（12），两个都变成免费。
- **(b) 把 shape 变成编译期常量**：给 kernel 加 `template <int OW, int OH, int CO>`，让 `%` 和 `/` 退化成 `IMAD.HI` + 移位（甚至被完全折叠）。这也同时修掉 2.6 里 x 的 1.67× 冗余 load——`IW` 变成常量后编译器就能 CSE 那 24 个重复地址。

**Evidence:**
- `sm__inst_executed_pipe_alu = 41.06% of peak`，逼近 FMA 的 44.25%。
- 源码第 67 行（`idx % OW`）10,416 采样、第 69 行（`idx/(OH*OW) % CO`）20,055 采样，`math_pipe_throttle` 分别 2212 / 4097。
- SASS 实证：5 × `MUFU.RCP`、5 × `I2F`、5 × `F2I`。

**Expected impact:** 静态指令 256 条里能砍掉 ~50–60 条（≈22%），ALU pipe 从 41% 降到 15% 以下。在 Priority 1 把 L1 腾出来之后，这一条会直接变成新的主要收益。单看 conv1 预期 10–15%；**同样适用于 conv2**。

**Effort:** low–medium。改 launch 配置 + 模板参数化 shape，不动算法。

### Priority 3 — 用 `__ldg` / 只读缓存 + 让权重走 `__constant__`

`w` 已经在读同一个地址（warp 内 `oc` 相同，完全 broadcast），但仍是 25 条 `LDG`/线程、每条占一个 LSU 发射槽。conv1 的权重只有 32×25×4 = 3.2 KB，conv2 是 64×32×25×4 = 200 KB。

- conv1：整个权重矩阵塞进 `__constant__`，warp 内广播读 constant cache 不定长 LSU 队列，直接削掉 25/86 = **29% 的内存指令**。
- 若 conv1 有效，conv2 的 200 KB 超过 64 KB constant 上限，需要**分块**（每个 block 只处理一部分 `oc`，把该子集的权重放进 constant 或 smem）。

**Evidence:** 86 条 LDG 中 25 条是权重（`R20` 基址），且 `lg_throttle` 在第 84 行（`const float wv = wp[i*5+j]`）单独贡献 11,388 采样。
**Expected impact:** 内存指令数 −29%，conv1 端到端估计 5–10%。
**Effort:** low（conv1）/ medium（conv2 分块）。

### Priority 4 — 提高 occupancy（**必须排在 P1/P2 之后**）

40 regs/thread 把 occupancy 按在 75%。降到 32 可以到 100%。

**但现在绝对不要动它**：L1 已经 95% 满载，多塞 warp 只会让 `lg_throttle` 队列更长。ncu 那条 `Est. Local Speedup: 25%` 是在假设 L1 不再瓶颈的前提下给出的。

**Evidence:** `launch__occupancy_limit_registers = 6`；achieved 66.02% vs theoretical 75%。
**Effort:** low（`__launch_bounds__(256, 8)`），但**现在做等于白做**。

---

## 5. Confidence & caveats

**有把握的：**
- L1 sector 吞吐是当前瓶颈。95.30% 实测 vs 95.4% 理论核算的吻合度极高，且 DRAM 5.26% / L2 7.77% / FMA 44.25% 三个数字互相印证"不是带宽、不是算力"。
- 58% 的 sector 浪费来自 warp 内 lane 的 stride-2 映射——这是 12.14/32 字节利用率唯一的可能解释。
- 第 78 行不是冗余 load（SASS 已证伪我上一轮的猜测），而是依赖链末端抢 LSU 队列的汇聚点。
- 索引除法是独立的第二瓶颈，有 SASS 直接证据。

**不确定的：**
- **Priority 1 的实际收益**。ncu 的 59.04% 是"若 sector 降到理想值"的上界；smem 方案会引入 bank conflict 和 `__syncthreads()` 开销，且搬数据本身也要访存。我估 1.4–1.6×，需要实测 A/B。
- **smem 能否装下 conv2 的窗口**。conv2 是 32 通道输入、5×5 窗口，tile 尺寸要大得多，可能放不下或需要 specialized 的 tiling。**这个必须在做 P1 之前先算清楚。**
- **"占用率不再是瓶颈"是否在 P1 之后依然成立**——修好 L1 后 FMA（44%）和 ALU（41%）会顶上来，届时的天花板大约是现在的 1/(max(0.44,0.41)) ≈ 2.3×，但如果 P2 也做了，ALU 那一半会被释放。

**profile 回答不了的：**
- conv2 调用（grid 204800）的行为没有覆盖——它占端到端时间的大头（11.69/24.73 ms = 47%），**必须先单独 profile 一遍**再决定 P1 的具体形状。
- PM Sampling 的时间线图没有导出成 ASCII，2.5 节的结论是从周期级均值推的，不是从采样曲线看的。
- 端到端百分比：conv1 只占 24.73 ms 的 8%。本报告的所有"Expected impact"若不加说明都指 conv1 自身；端到端要除以 ~12.5。

---

## 6. Reproduction

    cd /data/workspace/haoyu/code/learn/gpu_course/work1

    export PATH=/data/workspace/haoyu/software/miniconda3/envs/kernel/bin:$PATH
    export TMPDIR=/tmp/haoyu_ncu && mkdir -p $TMPDIR
    export PYTHONPATH=/usr/local/cuda-12.4/nsight-compute-2024.1.0/extras/python

    # 采集
    ncu --set full --section PmSampling --section PmSampling_WarpStates \
        -k "regex:conv_pool_relu_kernel" -c 1 \
        -o profile/v1_convpool/reports/full_v1 \
        python profile/v1_convpool/harness/prof_driver.py

    ncu --set source --section SourceCounters \
        -k "regex:conv_pool_relu_kernel" -c 1 \
        -o profile/v1_convpool/reports/source_v1 \
        python profile/v1_convpool/harness/prof_driver.py

    # 提取关键指标
    python profile/v1_convpool/analysis/analyze_reports.py \
        --run-dir profile/v1_convpool --tag v1

    # 源级 stall 归因
    python profile/v1_convpool/analysis/extract_stall_hotspots.py

    # SASS
    /usr/local/cuda-12.4/bin/cuobjdump -sass \
        -fun _Z21conv_pool_relu_kernelPKfS0_S0_Pfiiiiiii \
        ~/.cache/torch_extensions/py310_cu126/lenet5_26bebf37/lenet5_26bebf37.so \
        > profile/v1_convpool/analysis/sass_v1.txt

    # 对照基线
    ncu --import profile/v1_convpool/reports/full_v1.ncu-rep --page details
