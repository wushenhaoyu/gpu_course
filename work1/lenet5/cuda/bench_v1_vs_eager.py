#!/usr/bin/env python3
"""Conv1+Bia+Tanh timing: V1 implicit GEMM versus PyTorch eager."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import LeNet5
from model_cuda import V1_IGEMM_DIR, load_stage


def bench(fn, warmup: int, reps: int) -> tuple[float, float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(reps):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    return statistics.median(samples), min(samples), max(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--json-out", type=Path, required=True)
    args = ap.parse_args()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    model = LeNet5().cuda().eval()
    x = torch.rand(args.batch, 1, 28, 28, device="cuda")
    w, b = model.features[0].weight, model.features[0].bias
    v1 = load_stage("conv1", V1_IGEMM_DIR)
    v1_fn = lambda: v1.forward(x, w, b)
    eager_fn = lambda: torch.tanh(F.conv2d(x, w, b))
    with torch.inference_mode():
        max_abs_error = (v1_fn() - eager_fn()).abs().max().item()
    values = {}
    for name, fn in (("v1_igemm", v1_fn), ("torch_eager", eager_fn)):
        median, low, high = bench(fn, args.warmup, args.reps)
        values[name] = {"median_ms": median, "min_ms": low, "max_ms": high}
    result = {
        "batch": args.batch, "warmup": args.warmup, "reps": args.reps,
        "tf32": False, "max_abs_error": max_abs_error, **values,
        "v1_over_eager": values["v1_igemm"]["median_ms"] / values["torch_eager"]["median_ms"],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
