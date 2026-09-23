#!/usr/bin/env python3
"""Single-kernel NCU harness for naive or V1 LeNet-5 Conv1."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import LeNet5
from model_cuda import NAIVE_DIR, V1_IGEMM_DIR, V2_ROW_WARP_DIR, load_stage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("naive", "v1_igemm", "v2_row_warp"), required=True)
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()
    directory = {"naive": NAIVE_DIR, "v1_igemm": V1_IGEMM_DIR, "v2_row_warp": V2_ROW_WARP_DIR}[args.variant]
    op = load_stage("conv1", directory)
    model = LeNet5().cuda().eval()
    x = torch.rand(args.batch, 1, 28, 28, device="cuda")
    with torch.inference_mode():
        # JIT/loading and allocator warmup; NCU uses -c 1 to collect the next call.
        op.forward(x, model.features[0].weight, model.features[0].bias)
        torch.cuda.synchronize()
        op.forward(x, model.features[0].weight, model.features[0].bias)
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
