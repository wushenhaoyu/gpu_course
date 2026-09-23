# conv1 优化链: v1 → v2 → v3

**Kernel:** `conv1_smem_kernel(const float*, const float*, const float*, float*)`
**Target GPU:** NVIDIA A800-SXM4-80GB (108 SM, CC 8.0, 2.0 TB/s HBM2e, 19.5 TFLOPS fp32)
**Nsight Compute:** 2024.1.0 (CUDA 12.4) · `nvcc -O3 -lineinfo`
**Profile date:** 2026-09-19
**Run directories:** `profile/v1_convpool/` (v1), `profile/v2_conv1_smem_bcast/` (v2), `profile/v3_conv1_coalesced/` (本目录, v3)

三版 kernel 的源码分别在 `lenet5/cuda/versions/<版本名>/<版本名>.cu` —— 目录名、文件名、
profile 目录名三者逐字相同：`v1_convpool`、`v2_conv1_smem_bcast`、`v3_conv1_coalesced`。
v1 的完整基线报告见 `profile/v1_convpool/REPORT.md`；本文覆盖 v2 与 v3 两轮改动。

---

## 0. 结果总览

| | v1 | v2 | v3 |
|---|---:|---:|---:|
| conv1 源码 | `conv_pool_relu_kernel` | `conv1_smem_kernel` (lane=oc) | `conv1_smem_kernel` (lane=扁平输出) |
| **SM 活跃周期** | 2,244,668 | 1,675,868 | **1,118,225** |
| 相对上一版 | — | **1.34×** | **1.50×** |
| 相对 v1 | 1.00× | 1.34× | **2.01×** |
| bench 实测 | 1.975 ms | 1.471 ms | **0.773 ms** |
| 相对 v1 | — | 1.34× | **2.55×** |
| 占端到端 (23.5 ms) | 8.4% | 6.1% | **3.3%** |
| conv1 自身 FMA 峰值占比 | 44.25% | 40.34% | **62.78%** |

> **关于 Duration。** v3 的 ncu 报告里 `Duration = 5.34 ms` 是**假的**：那一轮 replay
> 的 SM 时钟被 DVFS 钉在 209 MHz（`SM Frequency = 209.29 cycle/usecond`），而 v1/v2 两轮
> 都是 1.14 GHz（879 ps/cycle，两轮完全一致）。周期数与时钟无关，所以本文一律用
> `sm__cycles_elapsed.avg` 做 A/B；实测毫秒数用 `prof.py` 的 bench。
> v3 自洽性检查：1,118,225 cycle ÷ 1.41 GHz (boost) = 0.793 ms，bench 量到 0.773 ms。
> 另一条独立佐证：ncu 报 FMA pipe 62.78%，4.608 GMAC × 2 / (19.5 TFLOPS × 0.6278) = 0.753 ms。

**正确性：** 三版 conv1 的输出**逐位相同**（不是"落在 1e-4 内"）。用 `t10k` 前 64 张图
直接比对三版 `pool1` 张量的 int32 位模式，`np.array_equal` 全为 True，max|diff| = 0。
三版求和顺序逐字一致，改的只是访存路径与线程映射。`--validate` 全层通过，准确率 0.9326。

> 复现口径（`/tmp/bitcmp.py` 的思路）：`model_cuda.load_extension` 对每个源文件建一套
> extension，但 Python 模块名都叫 `lenet5`，同一个进程里只能装一个 —— 所以三版各起一个
> 进程把 `pool1` 存成 `.npy`，再在第四个进程里逐位比。

---

## 1. v1 → v2：把 global load 的 sector 浪费消掉

### 1.1 v1 的病（详见 baseline 报告）

L1/TEX 95.30%，已顶在 sector 吞吐屋顶（理论核算 95.4%）。58% 的 sector 是纯浪费——
`idx` 连续 → `pc` 连续 → 输入列 `c0 = 2*pc` 步长为 2，32 个 lane 覆盖 256 字节只用 128 字节。

### 1.2 v2 的改法

换线程映射：**lane = 输出通道 oc**（32 个 lane 正好铺满 CO=32），warp = 空间片，
一个 block 处理一整张图。x 搬进 smem，之后 36 次窗口读全走 smem；warp 内 32 个 lane
的 (pr, pc) 相同 → x 的 smem 读是**纯广播**；`oc` 来自 lane、`im` 来自 blockIdx，
四组软件除法消失。

### 1.3 v2 修好了什么

| 指标 | v1 | v2 | |
|---|---:|---:|---|
| bytes/sector (global ld) | 12.14 / 32 | **32 / 32** | 完美合并 |
| global-ld sector | 923,465,296 | **1,944,626** | **475×** |
| L1/TEX 吞吐 | 95.30% | 50.92% | 出屋顶 |
| ALU pipe | 41.06% | 7.43% | 软件除法消失 |
| executed instructions | 591.8 M | 207.2 M | −65% |
| smem bank conflict | — | 45,548 | 可忽略 |

### 1.4 v2 弄坏了什么 —— 这一轮最重要的发现

**global store 从完美退化成了 8× 放大。**

| | v1 | v2 | 理想 |
|---|---:|---:|---:|
| global-**st** sector | 5,760,000 | **46,080,000** | 5,760,000 |
| L2 写 sector | 5,760,000 | **46,075,863** | 5,760,000 |
| L2 吞吐 | 7.77% | **72.61%** | — |
| L1 写利用率 | 32/32 B | **4/32 B** | 32/32 B |

ncu 规则原文：`Est. Speedup: 44.56% —— "The memory access pattern for global stores to
L1TEX might not be optimal. On average, only 4.0 of the 32 bytes transmitted per sector
are utilized by each thread."`

原因：lane = oc 时，同一 warp 的 32 个 lane 落到 32 个不同的输出通道平面，地址间隔
`12*12*4 = 576` 字节 —— 一个 sector 只有 4 字节有用。**v2 把瓶颈从 L1 读搬到了 L2 写**，
净收益因此只有 1.34×，远低于我事前估的 ~2×。

第二个新问题：`mio_throttle` 从 0.42 涨到 **15.55 cycle/inst = 33.32 里占 46.7%**，
ncu 给 `Est. Local Speedup: 27.39%`。注意 bank conflict 只有 45,548（0.16%），
所以这不是冲突，是 **smem 指令条数／MIO 队列深度**。

---

## 2. v2 → v3：把 global store 凑成合并

### 2.1 v3 的改法（只改一件事：lane 从"输出通道"换成"扁平输出下标"）

`y` 对一张图是 `[CO][OH][OW]` 连续的 4608 个 float。切成 144 个 chunk，每个 chunk 恰好
32 个连续 float：

    lane -> o = chunk*32 + lane
    oc = o/144,  pos = o%144,  pr = pos/12,  pc = pos%12

- 32 个 lane 写 32 个连续 float = 128 字节 = **4 个 sector，100% 利用**，且 chunk 起点按
  128 字节对齐；
- 144 被 8 个 warp 整除（每 warp 18 轮），**没有 v2 那种 144/32 = 4.5 的分块浪费**；
- 除以 144 和 12 都是编译期常量 → multiply-high，v1 的软件除法不会回来。

### 2.2 效果

| 指标 | v2 | v3 | |
|---|---:|---:|---|
| global-**st** sector | 46,080,000 | **5,760,000** | 回到理论下限 |
| L2 写 sector | 46,075,863 | **5,760,000** | 8× |
| L2 吞吐 | 72.61% | **11.70%** | 不再是瓶颈 |
| 写利用率 | 4/32 B | **32/32 B** | |
| `derived__..._excessive` | — | **0** | 零浪费 sector |

顺带收获（不是这次改动的直接目的，但同样重要）：

| 指标 | v2 | v3 | 原因 |
|---|---:|---:|---|
| Registers / thread | 47 | **32** | 索引模式让寄存器分配器轻松了 |
| Theoretical occupancy | 62.5% | **100%** | 8 blocks/SM，被寄存器而不是 smem 限 |
| Achieved occupancy | 60.20% | **95.52%** | 61.14 warp/SM |
| Stall: mio_throttle | 15.554 | 7.829 | |
| Stall: long_scoreboard | 4.684 | 0.798 | 载入延迟被更多 warp 盖住 |
| Warp cycles / issued instr | 33.32 | 22.51 | |

**SM 活跃周期 1,675,868 → 1,118,225（1.50×）**，bench 1.471 → 0.773 ms。

### 2.3 v3 的代价

| 指标 | v2 | v3 | |
|---|---:|---:|---|
| smem load wavefronts | 28,045,782 | **89,474,216** | **3.2×** |
| 每 warp-position 的 smem 载入 wavefront | 19.5 | **62.1** | |
| executed instructions | 207.2 M | 254.0 M | +23%（每 lane 各自算 o/144、pos/12）|
| ALU pipe | 7.43% | 17.35% | 同上 |

原因：lane = pos 之后 warp 内 32 个 lane 的 pos 不同，x 的 smem 读从"纯广播"变成
"每个 lane 自己的窗口"，而 OW=12 意味着一个 warp 的 32 个 lane 横跨 2.67 个池化行，
每次 LDS.64 落到 ~3 条不同的 128 字节 line 上。

**代价换来的净结果仍然是 +1.50×** —— 说明 L2 写那 8× 放大比 smem 的 3.2× wavefront
更贵。

---

## 3. v3 现在的瓶颈（这就是下一轮的入口）

| SOL | v3 | v2 | v1 |
|---|---:|---:|---:|
| L1/TEX（elapsed） | **76.31%** | 50.92% | 95.13% |
| L1/TEX（active） | **98.50%** | — | — |
| FMA pipe | 62.78% | 40.34% | 44.25% |
| L2 | 11.70% | 72.61% | 7.77% |
| DRAM | 17.60% | 6.77% | 5.26% |

拆 L1 的子单元，最接近饱和的两个是：

    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld  74.09% of peak   ← 主因
    l1tex__lsu_writeback_active                       74.45% of peak

ncu 这一轮只剩**一条** OPT 规则（v2 有三条，其中 store 44.56% 和 L2 写 8× 都已消失）：

> `Est. Local Speedup: 34.77%` — "each warp spends 7.8 cycles being stalled waiting for the
> MIO instruction queue to be not full... **When caused by shared memory accesses, trying
> to use fewer but wider loads can reduce pipeline pressure.**"

逐源码行归因（`analysis/stall_hotspots_v3.txt`，1,440,000 个 warp-position）：

| 排名 | 采样 | 行 | 源码 | 主要 stall |
|---:|---:|---:|---|---|
| 0 | 19,255 | **121** | `const float wv = wp[i*5+j];` | **mio_throttle 12,805** |
| 1 | 12,286 | 125 | `a11 = fmaf(xp[(i+1)*IW+j+1], wv, a11)` | mio 4,469 + math_pipe 2,145 |
| 2 | 11,743 | 124 | `a10 = fmaf(xp[(i+1)*IW+j], wv, a10)` | not_selected 4,331 + math_pipe 3,642 |
| 3 | 10,876 | 95 | `for (i...) s_x[i] = xg[i];` | mio 5,330 + long_scoreboard 2,104 |
| 4 | 8,471 | 102 | 位置循环头（`__syncthreads` 归属到这里） | barrier 7,199 |
| 5 | 7,748 | 123 | `a01 = fmaf(xp[i*IW+j+1], wv, a01)` | not_selected 2,926 + math_pipe 2,642 |
| 7 | 6,577 | 96 | `for (i...) s_w[i] = w[i];` | mio 3,349 |

**排名 0 是权重 LDS**，不是 x。每轮位置要发 25 条标量 LDS 去取 `wp[i*5+j]`（+1 条 bias），
而 v3 里 `oc` 逐 lane 不同，编译器既不能把它们变成广播、也不能提升到循环外。
44 条 LDS 里 26 条是权重，**58% 的 smem 指令花在只占 800 个 float 的权重上**。

---

## 4. 还没做的方向（按可量化收益排序）

### 4.1 权重进寄存器 —— 直接打排名 0

要让 `oc` 在 warp 内**一致**，编译器才能把 25 个权重提升到位置循环外。
改法：warp w 负责 4 个整平面（oc = 4w..4w+3），每个平面用 5 个 chunk 扫 144 个位置
（最后一轮 16/32 lane 有效 → 11% 的 FFMA 浪费）。
代价：+25 个寄存器 → occupancy 从 100% 掉到 ~50%。

**收益不确定**（去掉 59% 的 LDS 指令，但 occupancy 腰斩），必须实测。这是唯一一条
能直接命中 ncu 排名 0 的低成本改动。

### 4.2 回到 lane=oc，用带 padding 的 smem 暂存把 store 做成合并

这条在理论上最漂亮：lane=oc 的 x 是纯广播（19.5 wavefront/warp-position），
权重可以进寄存器，store 通过 `s_y[CO][144+pad]`（pad=1 让 `oc*145+pos` 的 bank 分布
互质）中转，最后整块 flush 成合并写。

预估 L1 每 32 个输出：18(x) + 0(w, 寄存器) + 1(暂存写) + 2(暂存读) + 4(store) ≈ **24**，
对比 v3 的 62.1 → **2.6× 更少的 L1 工作**。

代价：暂存要 `32 × 145 × 4 = 18.6 KB`，加上 s_x 3.1 KB + s_w 3.2 KB ≈ 25 KB/block
→ 实测 `Block Limit Shared Mem = 13`（v3，6.4 KB/block）说明本机 SM 可用 smem 约 84 KB
→ 25 KB 只能放 **3 blocks/SM = 24 warps = 37.5% occupancy**。

**这是 conv1 真正可能的下一大跳，但也是一次重写。**

### 4.3 上限在哪

FMA 已经 62.78%。纯 FMA 地板 = 100 FFMA/warp-position × 1.44 M warp-position ÷ (108 SM
× 4 SMSP) × 2 cycle = **667 K cycle**，当前 1,118 K → **conv1 最多还有 1.68×**。

对端到端：conv1 现在只占 23.5 ms 的 **3.3%**，就算 conv1 再快 1.68×，端到端只省
**0.31 ms = 1.3%**。

---

## 5. 结论 & 下一步

1. conv1 的"conv 算法"版本做到 **2.55×（1.975 → 0.773 ms）**，输出与 v1 逐位相同，
   准确率 0.9326 未变。瓶颈从 L1 读 → L2 写 → L1 smem wavefront 依次搬家，
   两次都靠 ncu 的 SOL 与规则引擎定位，没有一次是靠读源码猜的。
2. **conv1 的性价比已经很低**：3.3% 的端到端占比，理论天花板 1.68×，最多再省 1.3%。
3. 真正值钱的是同一套方法搬到 **conv2（11.69 ms，49.7%）** 和 **fc1（10.68 ms，45.4%）**。
   conv2 是同一个 kernel family，v1 报告的 Priority 1/2/3 对它同样成立，但它的
   tile 是 32 通道输入、5×5 窗口（200 KB 权重），装不装得下 smem **必须先算清楚**。
4. 用户要求的第二版 —— conv1 的 **implicit GEMM** 版本 —— 尚未开始。

---

## 6. Reproduction

    cd /data/workspace/haoyu/code/learn/gpu_course/work1

    export PATH=/data/workspace/haoyu/software/miniconda3/envs/kernel/bin:$PATH
    export TMPDIR=/tmp/haoyu_ncu && mkdir -p $TMPDIR
    export PYTHONPATH=/usr/local/cuda-12.4/nsight-compute-2024.1.0/extras/python

    # 正确性 + 计时 (每一版都要过)
    cd lenet5/cuda
    python model_cuda.py --kernel versions/v3_conv1_coalesced/v3_conv1_coalesced.cu --validate --bench
    python prof.py versions/v3_conv1_coalesced/v3_conv1_coalesced.cu

    # 采集 (注意 prof_driver.py 收的是绝对路径或相对 cuda/ 的路径)
    cd ../..
    K=$PWD/lenet5/cuda/versions/v3_conv1_coalesced/v3_conv1_coalesced.cu
    ncu --set full --section PmSampling --section PmSampling_WarpStates \
        -k "regex:conv1_smem_kernel" -c 1 \
        -o profile/v3_conv1_coalesced/reports/full_v3 \
        python profile/v3_conv1_coalesced/harness/prof_driver.py $K
    ncu --set source --section SourceCounters \
        -k "regex:conv1_smem_kernel" -c 1 \
        -o profile/v3_conv1_coalesced/reports/source_v3 \
        python profile/v3_conv1_coalesced/harness/prof_driver.py $K

    # 提取
    python profile/v3_conv1_coalesced/analysis/cmp_versions.py          # v1/v2/v3 并排
    python profile/v3_conv1_coalesced/analysis/analyze_reports.py \
        --run-dir profile/v3_conv1_coalesced --tag v3
    python profile/v3_conv1_coalesced/analysis/extract_stall_hotspots.py \
        --run-dir profile/v3_conv1_coalesced \
        --report profile/v3_conv1_coalesced/reports/source_v3.ncu-rep --tag v3

    # 规则引擎原文
    ncu --import profile/v3_conv1_coalesced/reports/full_v3.ncu-rep --page details \
        > profile/v3_conv1_coalesced/analysis/details_v3.txt

    # SASS / 资源占用
    /usr/local/cuda-12.4/bin/cuobjdump -res-usage \
        ~/.cache/torch_extensions/py310_cu126/lenet5_425d7530/lenet5_425d7530.so
    /usr/local/cuda-12.4/bin/cuobjdump -sass -fun _Z17conv1_smem_kernelPKfS0_S0_Pf \
        ~/.cache/torch_extensions/py310_cu126/lenet5_425d7530/lenet5_425d7530.so \
        > profile/v3_conv1_coalesced/analysis/sass_v3.txt
