# Conv1-only: cuDNN vs register-scalar CUDA

同一计算：`[B,1,28,28]` 与 `[32,1,5,5]` 做 valid convolution，带 bias；不含 ReLU，不含 MaxPool。两侧均在 CUDA Event 区间内进行常规输出分配。

环境为 A800，kernel conda 环境，`batch=1000`、warmup=20、reps=100。为保证同 FP32 语义，cuDNN 与 matmul 的 TF32 均关闭。

| 实现 | median (ms) | min (ms) | max (ms) | 相对 cuDNN |
|---|---:|---:|---:|---:|
| custom register-scalar | 0.3922 | 0.3912 | 0.4372 | 1.260x |
| cuDNN FP32 | 0.3113 | 0.3082 | 0.3174 | 1.000x |

最大绝对误差为 `1.669e-6`。因此在该小卷积上，纯标量寄存器 kernel 并未胜过 cuDNN；cuDNN 快约 26%。本实验不代表已融合的 `Conv+ReLU+Pool` 路径，因为融合会避免中间 Conv 输出的 global-memory 写回和再次读取。

机器可读原始结果：`conv1_cudnn_vs_reg_20260920.json`。复现：

```bash
cd /data/workspace/haoyu/code/learn/gpu_course/work1/lenet5/cuda
PATH=/data/workspace/haoyu/software/miniconda3/envs/kernel/bin:$PATH \
  /data/workspace/haoyu/software/miniconda3/envs/kernel/bin/python \
  experiments/bench_conv1_cudnn.py --batch 1000 --warmup 20 --reps 100
```
