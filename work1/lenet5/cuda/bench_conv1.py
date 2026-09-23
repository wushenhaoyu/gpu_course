#!/usr/bin/env python3
"""CUDA Event comparison: naive Conv1 vs V1 direct implicit GEMM Conv1."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import LeNet5
from model_cuda import NAIVE_DIR, V1_IGEMM_DIR, V2_ROW_WARP_DIR, load_stage


def bench(fn, warmup: int, reps: int) -> tuple[float, float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(reps):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            fn()  # normal output allocation is deliberately included on both paths
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    return statistics.median(samples), min(samples), max(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--json-out", type=Path, required=True)
    args = ap.parse_args()
    if min(args.batch, args.reps) <= 0 or args.warmup < 0:
        ap.error("batch and reps must be positive; warmup must be non-negative")

    torch.manual_seed(0)
    model = LeNet5().cuda().eval()
    x = torch.rand(args.batch, 1, 28, 28, device="cuda")
    w, b = model.features[0].weight, model.features[0].bias
    naive = load_stage("conv1", NAIVE_DIR)
    v1 = load_stage("conv1", V1_IGEMM_DIR)
    v2 = load_stage("conv1", V2_ROW_WARP_DIR)
    with torch.inference_mode():
        naive_y = naive.forward(x, w, b)
        error_v1 = (naive_y - v1.forward(x, w, b)).abs().max().item()
        error_v2 = (naive_y - v2.forward(x, w, b)).abs().max().item()

    rows = {}
    for name, fn in (("naive", lambda: naive.forward(x, w, b)),
                     ("v1_igemm", lambda: v1.forward(x, w, b)),
                     ("v2_row_warp", lambda: v2.forward(x, w, b))):
        median, low, high = bench(fn, args.warmup, args.reps)
        rows[name] = {"median_ms": median, "min_ms": low, "max_ms": high}
    result = {
        "batch": args.batch, "warmup": args.warmup, "reps": args.reps,
        "max_abs_error_v1": error_v1, "max_abs_error_v2": error_v2, **rows,
        "v1_over_naive": rows["v1_igemm"]["median_ms"] / rows["naive"]["median_ms"],
        "v2_over_naive": rows["v2_row_warp"]["median_ms"] / rows["naive"]["median_ms"],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
