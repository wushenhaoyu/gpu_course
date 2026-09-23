# conv1: 卷积算法 (v3) vs implicit GEMM (v4)

**Kernel A:** `conv1_smem_kernel` — `versions/v3_conv1_coalesced/v3_conv1_coalesced.cu`
**Kernel B:** `conv1_igemm_kernel` — `versions/v4_conv1_implicit_gemm/v4_conv1_implicit_gemm.cu`
**Target GPU:** NVIDIA A800-SXM4-80GB (108 SM, CC 8.0, 19.5 TFLOPS fp32)
**Nsight Compute:** 2024.1.0 (CUDA 12.4) · `nvcc -O3 -lineinfo`
**Profile date:** 2026-09-19 · **Run dir:** `profile/v4_conv1_implicit_gemm/`

用户要求 conv1 出两个版本：一个是卷积算法，一个是 implicit GEMM。这是两者的对比。
卷积算法那条链（v1 → v2 → v3）见 `profile/v3_conv1_coalesced/REPORT.md`。

---

## 0. 结果总览

| | v3 卷积算法 | **v4 implicit GEMM** |
|---|---:|---:|
| 线程映射 | 一个 lane 一个池化输出 | 4 个输出通道 × 1 个池化位置进寄存器 tile |
| 每线程累加器 | 4 (a00/a01/a10/a11) | **16** (4 oc × 4 池化成员) |
| 输出 | 与 v1 逐位相同 | **与 v3 逐位相同** |
| 准确率 | 0.9326 | 0.9326 |
| conv1 bench (clean) | 0.773 ms | **0.680 ms** |
| **交错 A/B 比值** | 1.00× | **1.19×** |
| L1/TEX 吞吐 | 76.31% | **24.08%** |
| ALU pipe | 17.35% | **8.78%** |
| Registers / thread | 32 | 142 |
| Achieved occupancy | 95.52% | 18.27% |
| 端到端占比 | 3.3% | **2.9%** |

**正确性：** v4 的 `pool1` 与 v3 **逐位相同**（int32 位模式，`np.array_equal` = True，
max|diff| = 0，见 `analysis/bitcmp_so.py`）。求和顺序与 v3 逐字一致（kr 外 / ks 内），
初值是 bias，max 的括号顺序也照抄——改的只是访存路径与线程映射。

**一句话：** implicit GEMM 把自己想解决的问题**彻底解决了**——L1/TEX 从 76.31% 掉到
24.08%，浪费 sector 归零，bytes/sector 回到 32/32——但只换回 **1.19×**，因为瓶颈
搬到了 FMA 管道（实测 88.8% busy）。**两版撞的是同一堵墙：conv1 就是 4.608 GMAC 的算力活。**

---

## 1. 把 conv1 写成 GEMM

    C[M][N] = A[M][K] · B[K][N]

      M = CO      = 32     输出通道
      K = KH*KW   = 25     卷积极 (kr, ks)；CI=1 所以没有 ci 维
      N = 576              **pre-pool** 位置 (24×24)，不是池化后的 144

2×2/s2 池化要 4 个 pre-pool 值才能取 max，所以 N 维取的是 576 个 conv 输出位置，
池化在寄存器里就地完成：

    B[(kr,ks)][(r,c)] = x[2*pr + r + kr][2*pc + c + ks]     r,c ∈ {0,1}

B 从不落地——它就是 x 换个索引——这就是 "implicit" 的意思。

**注意 FLOP 数没变。** 每个池化输出仍是 4 个 conv 输出 × 25 MAC = **100 MAC**，
和 v3 的 a00/a01/a10/a11 一模一样。隐式 GEMM 改的只是访存路径，不是算术量。

### 每线程的账

```
block = 128 线程 = 8 个 oc 组 (ty) × 16 个位置组 (tx)
每线程 tile: 4 个输出通道 × 4 个池化成员 = 16 个累加器
一个 block 扫一整张图 (9 轮 × 16 个池化位置)
```

内层一个 k 步（共 25 步，全展开）：

| 项 | 指令 | 说明 |
|---|---|---|
| A 片段 | **1 × `LDS.128`** | `s_wT[k][4*ty .. 4*ty+3]`。warp 内 8 个 ty 覆盖 oc 0..31 = 128 字节 = 恰好一行 → 1 个 wavefront |
| B 片段 | 4 个 x 值 | 2×2 窗口，被上面 **4 个通道共用** ← 这是相对 v3 的核心改进 |
| FFMA | **16** | 4 oc × 4 成员 |

### 权重为什么要转置

A 片段要落在同一行里，权重必须以 `[k][oc]` 布局。转置在 **smem 内部**做，不额外碰 global：

    global --(合并读)--> s_w[oc][k] --(smem 内转置)--> s_wT[k][oc]

两步都无 bank conflict：写 `s_wT[k][oc]` 时 i 连续 → oc 连续 → 连续 float；
读 `s_w[oc*K+k]` 时 i 连续 → oc 连续 → 步长 25，25 与 32 互质。

代价是 smem 从 6.4 KB 涨到 9.4 KB（784 + 800 + 800 + 32）。

---

## 2. ncu 实测

    ncu --set full --section PmSampling --section PmSampling_WarpStates \
        -k "regex:conv1_igemm_kernel" -c 1 -o .../full_v4 ...

| 指标 | v3 | **v4** | 读法 |
|---|---:|---:|---|
| L1/TEX 吞吐 | 76.31% | **24.08%** | **设计目标达成** |
| L2 / DRAM 吞吐 | 11.70 / 17.60% | 18.6 / 14.6% | 都不是瓶颈 |
| global-ld sector | — | 1.899e6 | |
| **excessive sector（纯浪费）** | — | **0** | v1 是 537,600,000 |
| Bytes/sector (global ld) | — | **32 / 32** | 完美合并 |
| Executed instructions | 254.0 M | 174.5 M | −31% |
| Issue slots busy | ~55% | 52.84% | 发射不是瓶颈 |
| ALU pipe | 17.35% | **8.78%** | 索引运算大幅下降 |
| Registers / thread | 32 | 142 | |
| Theoretical / Achieved occupancy | 100% / 95.52% | 18.75% / **18.27%** | |
| Warp cycles / issued instr | 22.51 | **5.485** | v3 在等内存，v4 不等 |

### stall 分解（每 5.485 cycle 的发射间隔里）

| stall 原因 | cycles/instr | 读法 |
|---|---:|---|
| not_selected | 1.088 | 别的 warp 被选中了 —— warp 是**就绪**的 |
| wait | 1.063 | 定长延迟依赖 |
| math_pipe_throttle | 1.021 | **FMA 管道满了** |
| long_scoreboard | **0.533** | 内存延迟：几乎可以不看 |
| mio_throttle | 0.028 | |
| lg_throttle | 0.000065 | v1 这里曾是 5.404 |

**这三条放在一起是决定性的：** `long_scoreboard` 只有 0.533，而
`not_selected + wait + math_pipe_throttle` 吃掉 3.17/5.485。warp **不是在等内存，是在抢
FMA 管道**。v1/v2 时代的 `lg_throttle`（L1 队列满）已经彻底消失。

> **时钟口径警告。** 这轮 replay 的 SM 时钟约 1.12 GHz（ncu 的
> `sm__pipe_fma_cycles_active` 是对 **elapsed** 归一化的，88.82% 是这么来的），
> 不是 boost 的 1.41 GHz。v3 那轮是 boost。**跨版本比较一律用 bench 毫秒或交错
> A/B 比值；ncu 只用来看瓶颈形状，不要用它横向比时间。** 这和 v3 报告里
> `Duration = 5.34 ms` 的假象是同一类问题。

---

## 3. 一次错误的优化 —— 本节是这一轮最重要的一条

`REG:142` + `occupancy 18.27%` 看起来像是教科书式的"延迟盖不住"。照这个思路做了一版：
把 `kr` 从 `#pragma unroll` 改成 `#pragma unroll 1`，活跃 x 窗口从 36 降到 12。

| 版本 | 寄存器 | 占用率 | spill | conv1 中位数 | 相对 v3 |
|---|---:|---:|---|---:|---:|
| v3（卷积算法） | 32 | 95.5% | 无 | 1.0955 ms | 1.00× |
| **v4 全展开** | **142** | **18.3%** | 无 | **0.9186 ms** | **1.19×** |
| v4 + kr 滚动（已回退） | 48 | ~50% | 无 | 0.9445 ms | 1.16× |

**滚动 kr 慢了 2.8%。** 寄存器 142 → 48 的收益是零，代价是丢掉了 nvcc 的软件流水。

nvcc 之所以肯花 142 个寄存器，是因为它在给那条 25 步展开的循环做软件流水——**那份 ILP
是有用的**。stall 分解早就说了：`long_scoreboard` 只有 0.533，内存延迟根本不是问题。
**我把 142 当成病，其实它是药。**

> 测量方法：`analysis/ab_bench.py`。两个版本的 `.so` 是各自独立的 pybind 模块，可以
> `importlib` 加载共存，在**同一个进程**里交错跑（A,B,C,A,B,C…），取各自中位数。
> 这台机器上有别人的进程占着 60 GB 显存、load ~40，分两次跑量到的绝对毫秒里混着争用
> 漂移（clean 时 v4 = 0.680 ms，争用时 = 0.919 ms）；**交错跑的比值才是可信的**。

### 3.1 又做了两次占用率实验，ncu 那条 OPT 是陷阱

ncu 的规则引擎对 v4 只留了一条 OPT：

> `Est. Local Speedup: 11.18%` —— each scheduler issues an instruction every 1.9 cycles…
> allocates an average of **2.92 active warps per scheduler, but only 1.11 warps were eligible**.

它把这个归因于**占用率低**（2.92 warps/scheduler ≈ 18%）。听起来可以直接吃掉这 11%，
于是用 `__launch_bounds__` 做了两次探针——**代码一个字没改**，只加一个属性：

| 版本 | 寄存器 | 占用率 | spill | 干净态 conv1 | 相对 142 版 |
|---|---:|---:|---|---:|---:|
| v4 全展开（保留） | 142 | 18.3% | 无 | **0.6713 ms** | 1.000× |
| `__launch_bounds__(128, 4)` | 128 | 25.0% | **无** (STACK:0) | 0.6693 ms | 1.003× |
| `__launch_bounds__(128, 6)` | 80 | 37.5% | **有** (STACK:376) | 2.5399 ms | **0.264×** |

（干净态取交错跑的 min：1.509/0.671 是本轮机器的两个争用档位，中位数被污染了。）

三条结论：

1. **占用率 18.3% → 25.0%、无 spill，性能逐位不变**（0.6713 vs 0.6693，0.3% 在噪声内）。
   占用率**根本不是限制项**。
2. 再往上压到 80 寄存器就要 spill，**3.8× 慢**——这条路的尽头是悬崖。
3. 合上 kr 滚动那次（REG 48 / occ 50% / 无 spill → 慢 2.8%），三个方向全部为负：
   **142 个寄存器不是"病"，它是 nvcc 拿来做软件流水的成本，而这个 ILP 恰好是收益本身。**

ncu 的 11.18% 是**误归因**：warp 之所以 ineligible，不是因为同批 warp 太少，
而是因为它们**全都堵在 FMA 管道和定长延迟依赖上**（`wait` 1.063 + `math_pipe_throttle`
1.021）。多塞 warp 进去不会凭空多出 FMA 吞吐——这一点 FMA 88.8% 这个数字早就说清楚了。
**规则引擎给的是"症状"，不是"病因"；这里必须用 stall 分解去否掉它。**

---

## 4. 天花板在哪，为什么就此打住

    conv1 总 MAC  = 4608 输出/img × 100 MAC = 460,800 MAC/img
                  × 10000 img = 4.608 GMAC = 9.22 GFLOP
    FMA 地板 (19.5 TFLOPS)               = 0.473 ms
    v4 实测                               = 0.680 ms   (69.5% of nominal peak)
    ncu 报 FMA 管道                        88.8% (at the achieved clock)

两条口径差在时钟（见上面的警告），但结论一致：**conv1 已经贴着 FMA 管道在跑，剩余
头寸约 1.13×。** 继续抠 conv1 的边际收益：

    conv1 占端到端 2.9% (0.680 / 23.483)
    再快 13%  =>  端到端省 0.09 ms = 0.4%

**不值得。** 这个 kernel 到此为止。

顺带记一条**没有走**的路：Winograd F(2×2, 5×5) 能把 100 MAC 降到 36，地板从 0.473 掉到
0.17 ms。但 (a) 它只占端到端 2.9%，省不出 2%；(b) 变换本身的访存与数值误差都要压进
1e-4 的容差里，风险与收益不成比例。

---

## 5. 真正的下一步

`prof.py` 给的全局账（v4，10000 张图）：

| 算子 | ms | 占比 | GFLOP/s | 峰值占比 |
|---|---:|---:|---:|---:|
| conv1 | 0.680 | 2.9% | 13554 | 69.5% |
| **conv2** | **11.720** | **49.9%** | 5592 | 28.7% |
| **fc1** | **10.699** | **45.6%** | **230** | **1.2%** |
| fc2 | 0.345 | 1.5% | 584 | 3.0% |
| fc3 | 0.040 | 0.2% | 421 | 2.2% |
| 合计 | 23.483 | | 3302 | 16.9% |

- **fc1 是 1024→120 的普通 GEMM，跑在峰值的 1.2% 上。** 它是 v0 那个"一个线程一个输出、
  k 循环里两次 global load、零复用"的 kernel。若能做到 70% 峰值就是 0.18 ms，
  **单这一项能省掉全流程 45%**。这是全项目最大的一笔。
- **conv2（49.9%，28.7% 峰值）** 是同一个 kernel family、同样的病，v4 这套 implicit GEMM
  结构对它的收益**远大于** conv1：conv2 是 32 通道输入 × 5×5 = **K=800**，A/B 两个
  片段的复用度是 conv1 (K=25) 的 32 倍。conv1 上这套只值 1.19×，在 conv2 上应该高得多。

**逐位相同的红利：** v4 与 v3 的输出逐位相同，所以换 conv2 的时候可以拿同一套
`bitcmp_so.py` 做逐位回归，不用只看 1e-4 容差。

---

## 6. Reproduction

    cd /data/workspace/haoyu/code/learn/gpu_course/work1

    export PATH=/data/workspace/haoyu/software/miniconda3/envs/kernel/bin:$PATH

    # 正确性 + 准确率
    cd lenet5/cuda
    python model_cuda.py --kernel versions/v4_conv1_implicit_gemm/v4_conv1_implicit_gemm.cu \
        --validate --bench
    python prof.py versions/v4_conv1_implicit_gemm/v4_conv1_implicit_gemm.cu

    # 与 v3 逐位比对 (两个 .so 同进程共存, 不用重编译)
    C=~/.cache/torch_extensions/py310_cu126
    python profile/v4_conv1_implicit_gemm/analysis/bitcmp_so.py \
        $C/lenet5_1e1be70e/lenet5_1e1be70e.so \    # v3
        $C/lenet5_d5e19f72/lenet5_d5e19f72.so      # v4

    # 交错 A/B (抗争用)
    cd ../..
    python profile/v4_conv1_implicit_gemm/analysis/ab_bench.py \
        "v3=$C/lenet5_1e1be70e/lenet5_1e1be70e.so" \
        "v4=$C/lenet5_d5e19f72/lenet5_d5e19f72.so" --reps 5 --inner 50

    # ncu 采集
    export TMPDIR=/tmp/haoyu_ncu && mkdir -p $TMPDIR
    export PYTHONPATH=/usr/local/cuda-12.4/nsight-compute-2024.1.0/extras/python
    K=$PWD/lenet5/cuda/versions/v4_conv1_implicit_gemm/v4_conv1_implicit_gemm.cu
    ncu --set full --section PmSampling --section PmSampling_WarpStates \
        -k "regex:conv1_igemm_kernel" -c 1 \
        -o profile/v4_conv1_implicit_gemm/reports/full_v4 \
        python profile/v4_conv1_implicit_gemm/harness/prof_driver.py $K

    # 提取
    python profile/v4_conv1_implicit_gemm/analysis/analyze_reports.py \
        --run-dir profile/v4_conv1_implicit_gemm --tag v4
    python profile/v4_conv1_implicit_gemm/analysis/cmp_versions.py   # v1/v2/v3/v4 并排

    # SASS / 资源占用
    /usr/local/cuda-12.4/bin/cuobjdump -res-usage \
        $C/lenet5_d5e19f72/lenet5_d5e19f72.so
    /usr/local/cuda-12.4/bin/cuobjdump -sass -fun _Z18conv1_igemm_kernelPKfS0_S0_Pf \
        $C/lenet5_d5e19f72/lenet5_d5e19f72.so \
        > profile/v4_conv1_implicit_gemm/analysis/sass_v4.txt

> **扩展缓存目录名**（内容哈希，与路径无关）：
> `lenet5_d5e19f72` = v4 首次全展开（142 寄存器），
> `lenet5_93f059c4` = v4 + kr 滚动（已回退，保留作 A/B 证据），
> `lenet5_8db498ad` = 回退后的 v4（142 寄存器，本文件当前版本），
> `lenet5_42519b12` / `lenet5_3a033c56` = `__launch_bounds__(128,4)` / `(128,6)` 两次探针（均已回退）。
