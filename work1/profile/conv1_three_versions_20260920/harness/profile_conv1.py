"""Single-launch NCU harness for the three conv1+pool1 implementations."""
import argparse
import sys

CUDA_DIR = "/data/workspace/haoyu/code/learn/gpu_course/work1/lenet5/cuda"
sys.path.insert(0, CUDA_DIR)

import torch
import model_cuda as M


VARIANT_DIR = {
    "direct": M.FUSED_DIR,
    "smem_c": M.IGEMM_SMEM_C_DIR,
    "tiled": M.IGEMM_TILED_SMEM_DIR,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANT_DIR, required=True)
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()

    # Compile/load only the selected conv1 extension; no FC or conv2 noise in replay.
    op = M.load_stage("conv1_pool1", VARIANT_DIR[args.variant])
    model = M.ModelNew().cuda().eval()
    M.load_bins(model)
    x = torch.rand(args.batch, 1, 28, 28, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        y = op.forward(x, model.conv1.weight, model.conv1.bias)
    torch.cuda.synchronize()
    print(f"[harness] variant={args.variant} batch={args.batch} output={tuple(y.shape)}")


if __name__ == "__main__":
    main()
