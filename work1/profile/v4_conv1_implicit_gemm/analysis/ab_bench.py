#!/usr/bin/env python3
"""同进程交错 A/B 比 conv1 的 ms。

为什么不用 model_cuda.py --bench 来回比:
  1. 两个版本的 .so 是**各自独立的 pybind 模块**, 可以直接 importlib 加载共存 ——
     不用一个版本起一个进程, 省掉一整轮编译;
  2. 这台机器上还有别的进程在跑 (load ~40), 分两次跑量到的差值里混着争用漂移.
     交错跑 (每轮 A,B,C,A,B,C...) 之后取各自的中位数, 三个版本看到的是同一段
     GPU 状态, 差值才可信.

用法:
    python ab_bench.py 标签=路径.so [标签=路径.so ...] [--reps 5] [--inner 50]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys

import torch

CUDA_DIR = "/data/workspace/haoyu/code/learn/gpu_course/work1/lenet5/cuda"
sys.path.insert(0, CUDA_DIR)
import model_cuda as M  # noqa: E402

CONV1 = "conv_pool_k5_k2_s1_s2_cin1_cout32_relu_bias"


def load_so(path: str):
    name = os.path.basename(path).replace(".so", "")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mods", nargs="+", help="标签=路径.so")
    ap.add_argument("--reps", type=int, default=5, help="每版测几轮")
    ap.add_argument("--inner", type=int, default=50, help="每轮内重复几次取均值")
    ap.add_argument("--n", type=int, default=10000, help="图像数")
    args = ap.parse_args()

    pairs = []
    for s in args.mods:
        tag, _, path = s.partition("=")
        pairs.append((tag, load_so(path)))

    model = M.ModelNew()
    M.load_bins(model)
    model = model.cuda().eval()
    w, b = model.conv1.weight, model.conv1.bias

    x = torch.from_numpy(
        M.load_idx_images(os.path.join(M.DATA_DIR, "t10k-images-idx3-ubyte"))[: args.n]
    ).unsqueeze(1).cuda()

    # 先各自 warmup 一遍 (第一次调用里含 cudaMalloc / lazy init, 不算时间)
    for tag, mod in pairs:
        with torch.no_grad():
            getattr(mod, CONV1)(x, w, b)
    torch.cuda.synchronize()

    samples = {tag: [] for tag, _ in pairs}
    for r in range(args.reps):
        for tag, mod in pairs:
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            with torch.no_grad():
                s.record()
                for _ in range(args.inner):
                    getattr(mod, CONV1)(x, w, b)
                e.record()
            torch.cuda.synchronize()
            samples[tag].append(s.elapsed_time(e) / args.inner)

    print(f"图像数 {args.n}   每轮 {args.inner} 次   交错跑 {args.reps} 轮")
    print(f"{'版本':<24}{'中位数 ms':>12}{'最小 ms':>12}   各轮")
    print("-" * 76)
    base = None
    for tag, _ in pairs:
        v = samples[tag]
        med = statistics.median(v)
        if base is None:
            base = med
        print(f"{tag:<24}{med:>12.4f}{min(v):>12.4f}   "
              + " ".join(f"{t:.3f}" for t in v)
              + ("" if tag == pairs[0][0] else f"   {base / med:.3f}x"))


if __name__ == "__main__":
    main()
