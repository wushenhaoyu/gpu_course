#!/usr/bin/env python3
"""Export every metric recorded in each full NCU report to JSON."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/usr/local/cuda-12.4/nsight-compute-2024.1.0/extras/python")
import ncu_report


def metric_value(metric):
    try:
        return metric.value()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, action="append", required=True)
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for report, tag in zip(args.report, args.tag):
        action = ncu_report.load_report(str(report)).range_by_idx(0).action_by_idx(0)
        values = {}
        for name in action.metric_names():
            values[name] = metric_value(action[name])
        out = args.out_dir / f"metrics_all_{tag}.json"
        out.write_text(json.dumps(values, indent=2, default=str) + "\n")
        print(f"{tag}: {len(values)} metrics -> {out}")


if __name__ == "__main__":
    main()
