"""Naive CUDA execution for LeNet-5 with PyTorch-owned parameters.

Extensions are stateless: every forward call receives the current module weight
and bias tensors. Therefore normal PyTorch ``load_state_dict`` is the sole
weight-loading path; no .bin conversion or duplicated CUDA-side state exists.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import LeNet5


HERE = Path(__file__).resolve().parent
NAIVE_DIR = HERE / "naive"
V1_IGEMM_DIR = HERE / "v1_igemm"
V2_ROW_WARP_DIR = HERE / "v2_row_warp"
STAGES = ("conv1", "pool1", "conv2", "pool2", "conv3", "fc1", "fc2")


def load_stage(stage: str, source_dir: Path) -> object:
    """Build/load exactly one stateless CUDA extension file."""
    src = source_dir / f"{stage}.cu"
    digest = hashlib.md5(src.read_bytes()).hexdigest()[:10]
    return load(
        name=f"lenet5_naive_{stage}_{digest}",
        sources=[str(src)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=False,
    )


def load_naive_extensions(conv1_dir: Path = NAIVE_DIR) -> dict[str, object]:
    """Build/load one extension per CUDA file; no extension owns parameters."""
    return {stage: load_stage(stage, conv1_dir if stage == "conv1" else NAIVE_DIR) for stage in STAGES}


class CudaLeNet5(LeNet5):
    """Same parameter names/shapes as :class:`LeNet5`, with CUDA kernel forward."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(num_classes=num_classes)
        self.ops: dict[str, object] | None = None

    def load_kernels(self, conv1_variant: str = "naive") -> "CudaLeNet5":
        if conv1_variant == "naive":
            conv1_dir = NAIVE_DIR
        elif conv1_variant == "v1_igemm":
            conv1_dir = V1_IGEMM_DIR
        elif conv1_variant == "v2_row_warp":
            conv1_dir = V2_ROW_WARP_DIR
        else:
            raise ValueError("conv1_variant must be 'naive', 'v1_igemm', or 'v2_row_warp'")
        self.ops = load_naive_extensions(conv1_dir)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ops is None:
            raise RuntimeError("call load_kernels() before CUDA forward")
        if not x.is_cuda or x.dtype != torch.float32 or not x.is_contiguous():
            raise ValueError("x must be contiguous CUDA float32")
        # Parameters remain ordinary PyTorch Parameters. load_state_dict(),
        # optimizers and checkpointing work unchanged; kernels only consume them.
        x = self.ops["conv1"].forward(x, self.features[0].weight, self.features[0].bias)
        x = self.ops["pool1"].forward(x)
        x = self.ops["conv2"].forward(x, self.features[3].weight, self.features[3].bias)
        x = self.ops["pool2"].forward(x)
        x = self.ops["conv3"].forward(x, self.features[6].weight, self.features[6].bias)
        x = x.flatten(1)
        x = self.ops["fc1"].forward(x, self.classifier[1].weight, self.classifier[1].bias)
        return self.ops["fc2"].forward(x, self.classifier[3].weight, self.classifier[3].bias)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conv1-variant", choices=("naive", "v1_igemm", "v2_row_warp"), default="naive")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    try:
        torch.manual_seed(0)
        reference = LeNet5().cuda().eval()
        cuda_model = CudaLeNet5().cuda().eval()
        # A single normal PyTorch operation transfers every layer's parameters.
        cuda_model.load_state_dict(reference.state_dict())
        cuda_model.load_kernels(args.conv1_variant)
        x = torch.randn(args.batch, 1, 28, 28, device="cuda")
        with torch.inference_mode():
            expected = reference(x)
            actual = cuda_model(x)
        torch.cuda.synchronize()
        max_abs_error = (expected - actual).abs().max().item()
        print(f"batch={args.batch}, max_abs_error={max_abs_error:.3e}")
        if args.json_out:
            args.json_out.write_text(json.dumps({"batch": args.batch, "conv1_variant": args.conv1_variant, "max_abs_error": max_abs_error}) + "\n")
    except Exception as exc:
        if args.json_out:
            args.json_out.write_text(json.dumps({"error": repr(exc)}) + "\n")
        raise


if __name__ == "__main__":
    main()
