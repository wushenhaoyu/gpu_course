"""Launch one selected fused kernel on the production batch shape for Nsight Compute.

The PyTorch extensions are compiled with -lineinfo by model_cuda.py.  This harness
does no warmup or timing: NCU replays the selected launch itself.
"""
import argparse
import os
import sys

CUDA_DIR = "/data/workspace/haoyu/code/learn/gpu_course/work1/lenet5/cuda"
sys.path.insert(0, CUDA_DIR)

import torch
import model_cuda as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=("conv1", "conv2"), required=True)
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()

    M.MODE = "fused"
    # The harness only launches the selected convolution path; avoid compiling unrelated FC
    # extensions during every NCU replay.
    M.ops = {
        "conv1_pool1": M.load_stage("conv1_pool1", M.FUSED_DIR),
        "conv2_pool2": M.load_stage("conv2_pool2", M.FUSED_DIR),
    }
    model = M.ModelNew().cuda().eval()
    M.load_bins(model)
    x = torch.rand(args.batch, 1, 28, 28, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        p1 = M.ops["conv1_pool1"].forward(x, model.conv1.weight, model.conv1.bias)
        if args.kernel == "conv1":
            y = p1
        else:
            y = M.ops["conv2_pool2"].forward(p1, model.conv2.weight, model.conv2.bias)
    torch.cuda.synchronize()
    print(f"[harness] kernel={args.kernel} batch={args.batch} output={tuple(y.shape)}")


if __name__ == "__main__":
    main()
