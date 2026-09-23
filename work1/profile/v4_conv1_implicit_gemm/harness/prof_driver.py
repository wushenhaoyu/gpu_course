"""单次前向, 供 ncu 抓 kernel。

不计时、不校验、不做 warmup —— ncu 自己 replay。跑完整条流水线是有意的: 这样每个 kernel
拿到的输入张量就是生产路径上真实的那一份 (而 -k 会把不关心的 kernel 滤掉)。

用法:  python prof_driver.py [kernel_src]
"""
import os
import sys

CUDA_DIR = "/data/workspace/haoyu/code/learn/gpu_course/work1/lenet5/cuda"
sys.path.insert(0, CUDA_DIR)

import numpy as np
import torch

import model_cuda as M

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    CUDA_DIR, "versions/v1_convpool/v1_convpool.cu")
M.ext = M.load_extension(os.path.abspath(src))
ext = M.ext

N = 10000
model = M.ModelNew()
M.load_bins(model)
model = model.cuda().eval()

x = torch.from_numpy(M.load_idx_images(os.path.join(M.DATA_DIR, "t10k-images-idx3-ubyte"))[:N])
x = x.unsqueeze(1).cuda()

with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()

print(f"[driver] {N} imgs -> {tuple(y.shape)}   ext={ext.__name__}   src={os.path.basename(src)}")
