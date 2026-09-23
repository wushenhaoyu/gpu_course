# Fused naive conv+pool profiling report

**Kernels:** `conv1_pool1_kernel` and `conv2_pool2_kernel`  
**Target GPU:** NVIDIA A800-SXM4-80GB (108 SM, CC 8.0)  
**Nsight Compute:** 2024.1.0 (CUDA 12.4)  
**Profile date:** 2026-09-20  
**Run directory:** `work1/profile/fused_naive_convpool_20260920/`

## 0. Executive summary

The newly fused kernels are correct, and fusion removes the conv-to-pool global
intermediates. It does **not** solve the dominant remaining problem: the direct
per-pool-output thread mapping generates highly uncoalesced global loads and fills
the L1/TEX load queue.

- `conv1_pool1_kernel`: L1/TEX throughput is **90.99% of peak**, only **11.9/32 B**
  per global-load sector are useful, and **56%** of its load sectors are excessive.
- `conv2_pool2_kernel`: L1/TEX throughput is **86.97%**, only **8.63/32 B** are useful,
  and it spends **21.84 cycles/issued instruction** stalled on the LG queue.
- `conv2_pool2_kernel` is the expensive convolution: **0.958 ms** versus **0.145 ms**
  for conv1 in this NCU collection. It is the first optimization target.

The evidence supports changing the *internal* mapping/data staging (shared-memory
or implicit-GEMM-style tiled convolution) while retaining the current register-local
conv+pool fusion boundary.

## 1. Profiling setup

- Harness: `harness/profile_fused.py`, which loads the production folded weights and
  launches one selected fused kernel. It is intentionally not a timing loop: NCU
  replays the selected launch.
- Workload: FP32 batch 1000, the normal benchmark batch magnitude. `conv1` input is
  `[1000,1,28,28]`; `conv2` consumes `[1000,32,12,12]` produced by the actual fused
  conv1 path.
- Build: PyTorch extensions are JIT-built with `-O3 -lineinfo` in `model_cuda.py`.
- Each kernel has a `--set full` plus PM sampling report and a `--set source` plus
  SourceCounters report. The metric names and occupancy calculation are the existing
  sm_80/A800 variants, not the B200 defaults in the generic skill documentation.

    export PATH=/data/workspace/haoyu/software/miniconda3/envs/kernel/bin:$PATH
    export TMPDIR=/tmp/haoyu_ncu_fused_20260920
    mkdir -p "$TMPDIR"
    ncu --set full --section PmSampling --section PmSampling_WarpStates \
        -k 'regex:conv1_pool1_kernel' -c 1 \
        -o reports/full_conv1_b1000 \
        /data/workspace/haoyu/software/miniconda3/envs/kernel/bin/python \
        harness/profile_fused.py --kernel conv1 --batch 1000

The run-specific `TMPDIR` is required on this shared host because the default
`/tmp/nsight-compute-lock` was already held by another process.

## 2. Headline metrics

| Metric | conv1 + pool1 | conv2 + pool2 |
|---|---:|---:|
| Duration | 0.145 ms | **0.958 ms** |
| SM throughput | 51.74% | 53.09% |
| L1/TEX throughput | **90.99%** | **86.97%** |
| DRAM throughput | 1.71% | 1.07% |
| L1 hit rate | 96.21% | 98.10% |
| FMA pipe | 45.27% | 43.56% |
| Registers / thread | 32 | 31 |
| Theoretical / achieved occupancy | 100% / 85.33% | 100% / 91.84% |
| Useful bytes per global-load sector | 11.90 / 32 | **8.63 / 32** |
| Excessive global-load sectors | 32.256 M (56%) | 98.304 M (33%) |
| LG queue stall / issued instruction | 19.53 cycles | **21.84 cycles** |
| Long-scoreboard stall / issued instruction | 3.07 cycles | **8.88 cycles** |

The low DRAM percentages plus high L1/TEX percentages rule out HBM bandwidth as the
limit. These kernels saturate the on-chip L1/TEX request path instead.

## 3. Diagnosis

### 3.1 Occupancy and balance

Both kernels are lightweight in register use (31–32 registers/thread), have no static
or dynamic shared memory, and reach 100% theoretical occupancy. Achieved occupancy is
also high (85.33% and 91.84%), so increasing occupancy is not the leading fix.

`conv1` launches 18,000 blocks (20.83 waves/SM); `conv2` launches 4,000 blocks
(4.63 waves/SM). The latter has fewer waves, but still enough work for the observed
91.84% achieved occupancy; its bottleneck is instruction-side memory pressure, not an
empty-SM tail.

### 3.2 Memory access and stalls

For each pooled output, one thread computes four overlapping convolution windows.
Neighboring lanes advance in pooled-x by one, so their source x addresses advance by
two floats. Across channels, weights are also repeatedly read without an explicit
cooperative tile. The direct result is poor sector utilization:

- NCU estimates **57.14%** potential speedup from the conv1 load pattern and reports
  32.256 M excessive sectors.
- NCU estimates **63.5%** potential speedup from the conv2 load pattern and reports
  98.304 M excessive sectors.

This appears directly in the source samples. In conv1, the four FMA input-load lines
`conv1_pool1.cu:20–23` account for the main long-scoreboard samples, while the loop's
weight/value loads at lines 16 and 19 produce 8,063 LG-throttle samples combined. In
conv2, the same pattern is much larger: lines `21–25` dominate, with line 25 alone
showing 16,923 long-scoreboard and 15,173 LG-throttle samples.

The rule engine independently reports LG-queue pressure: 19.5 cycles/issued
instruction for conv1 (estimated 9.01% local speedup) and 21.8 for conv2 (13.03%).
There is no spill evidence and no tensor-core work; scalar FP32 FMA is expected for
this correctness-first implementation.

## 4. Ranked next steps

### Priority 1 — Tile conv2 input and weights in shared memory

Keep the current `conv2_pool2` output boundary, but assign a block a spatial/output-
channel tile, cooperatively load an x tile and its corresponding weight tile, then
compute the four pooled members in registers. This directly attacks lines 21–25 and
the 8.63/32 B sector efficiency.

**Evidence:** NCU reports 63.5% estimated improvement for the global-load pattern,
21.84 LG-stall cycles/instruction, and 8.88 long-scoreboard cycles/instruction.

**Risk:** shared-memory layout and output mapping must preserve coalesced loads and
writes; validate `pool2` against the reference after each change.

### Priority 2 — Change lane mapping before adding more arithmetic tricks

Map adjacent lanes to contiguous x/output positions within a tile (or use an
implicit-GEMM N dimension with contiguous columns). The current pool-coordinate map
creates stride-2 input access and wastes 33–56% of sectors. This should be designed
together with Priority 1; adding unrolling alone cannot recover unused sectors.

### Priority 3 — Apply the successful conv2 layout to conv1

Conv1 has the same qualitative issue but is only 0.145 ms in this profile. It should
reuse the improved mapping after conv2 proves it, rather than being optimized first.

## 5. Caveats

- NCU replay changes SM clocks and the GPU is shared. Treat the recorded durations as
  profiling context, not as a replacement for the CUDA-Event median from `cuda/prof.py`.
- This report compares the two fused kernels' bottleneck shapes; it is not an A/B
  measurement against the prior 7-kernel naive implementation.
- The scope excludes FC kernels, whose naive implementation is separately visible in
  `cuda/prof.py` and will need its own profile before optimization.

## 6. Artifacts

- `reports/full_{conv1,conv2}_b1000.ncu-rep`: full NCU reports, openable in NCU UI.
- `reports/source_{conv1,conv2}_b1000.ncu-rep`: source-counter reports.
- `analysis/metrics_key_conv1_b1000_conv2_b1000.{txt,json}`: curated metrics.
- `analysis/stall_hotspots_{conv1,conv2}_b1000.txt`: source-line stall attribution.
- `analysis/details_{conv1,conv2}_b1000.txt`: NCU rule-engine output.
