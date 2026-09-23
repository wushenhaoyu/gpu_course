#!/usr/bin/env python3
"""逐位比对两版 conv1 的 pool1 输出 —— 直接加载已编译好的 .so, 不触发重编译。

比 /tmp/bitcmp.py 好在: 两版的 .so 是各自独立的 pybind 模块 (扩展名不同), 可以
在**同一个进程**里共存, 不用靠 .npy 中转。省掉一整轮编译。

用法:
    python bicmp_so.py <A.so> <B.so>
"""
import importlib.util
import os
import sys

import numpy as np
import torch

CUDA_DIR = "/data/workspace/haoyu/code/learn/gpu_course/work1/lenet5/cuda"
sys.path.insert(0, CUDA_DIR)
import model_cuda as M  # noqa: E402


def load_so(path):
    name = os.path.basename(path).replace(".so", "")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pool1(mod, x, w, b):
    with torch.no_grad():
        return mod.conv_pool_k5_k2_s1_s2_cin1_cout32_relu_bias(x, w, b)


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    A, B = load_so(a_path), load_so(b_path)

    model = M.ModelNew()
    M.load_bins(model)
    model = model.cuda().eval()

    x = torch.from_numpy(
        M.load_idx_images(os.path.join(M.DATA_DIR, "t10k-images-idx3-ubyte"))[:64]
    ).unsqueeze(1).cuda()
    w, bias = model.conv1.weight, model.conv1.bias

    pa, pb = pool1(A, x, w, bias), pool1(B, x, w, bias)
    torch.cuda.synchronize()

    ia = pa.cpu().numpy().view(np.int32)
    ib = pb.cpu().numpy().view(np.int32)
    eq = np.array_equal(ia, ib)
    print(f"A = {os.path.basename(a_path)}")
    print(f"B = {os.path.basename(b_path)}")
    print(f"shape            : {pa.shape}")
    print(f"int32 逐位相同   : {eq}")
    print(f"max |diff|       : {np.abs(pa.cpu().numpy() - pb.cpu().numpy()).max():.3e}")
    if not eq:
        n = int((ia != ib).sum())
        print(f"不同元素数       : {n} / {ia.size}  ({100.0 * n / ia.size:.4f}%)")
        idx = np.argwhere(ia != ib)[:5]
        for i in idx:
            i = tuple(i)
            print(f"  {i}: A={pa.cpu().numpy()[i]!r}  B={pb.cpu().numpy()[i]!r}")


if __name__ == "__main__":
    main()
