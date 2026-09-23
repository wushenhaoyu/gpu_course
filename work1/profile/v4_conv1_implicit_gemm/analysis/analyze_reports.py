#!/usr/bin/env python3
"""Extract curated key metrics from the run's .ncu-rep files.

Usage:
    python analyze_reports.py --run-dir <dir> --tag <tag> [--tag <tag2> ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/usr/local/cuda-12.4/nsight-compute-2024.1.0/extras/python")
import ncu_report  # noqa: E402

# 有些 SOL 指标在 sm_80 / NCU 2024.1 上没有同名 raw metric, 用 lambda 现算。
def _dram_pct(a):
    return (a["dram__bytes_read.sum.pct_of_peak_sustained_elapsed"].value()
            + a["dram__bytes_write.sum.pct_of_peak_sustained_elapsed"].value())


def _theo_occ(a):
    # NCU 的 SOL "Theoretical Occupancy" = 各占用上限里最紧的那个 × 每块 warp 数 / SM 上限(64)。
    limits = [a[f"launch__occupancy_limit_{k}"].value()
              for k in ("blocks", "registers", "shared_mem", "warps")]
    warps_per_block = a["launch__block_size"].value() / 32.0
    return min(limits) * warps_per_block / 64.0 * 100.0


def _ns_to_ms(a):
    return a["gpu__time_duration.sum"].value() / 1e6


KEY = [
    ("Duration (ms)",                     _ns_to_ms,                                                       "ms"),
    ("SM Throughput (% peak)",            "sm__throughput.avg.pct_of_peak_sustained_elapsed",              "%"),
    ("Memory Throughput (% peak)",        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "%"),
    ("L1/TEX Throughput (% peak)",        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",           "%"),
    ("L2 Throughput (% peak)",            "lts__throughput.avg.pct_of_peak_sustained_elapsed",             "%"),
    ("DRAM Throughput (% peak)",          _dram_pct,                                                       "%"),
    ("L1 Hit Rate (%)",                   "l1tex__t_sector_hit_rate.pct",                                  "%"),
    ("L2 Hit Rate (%)",                   "lts__t_sector_hit_rate.pct",                                    "%"),
    ("Bytes/Sector (global ld)",          "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio", "B of 32"),
    ("Global-ld Sectors",                 "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",                "sector"),
    ("  of which excessive",              "derived__memory_l2_theoretical_sectors_global_excessive",       "sector"),
    ("Executed Instructions",             "smsp__inst_executed.sum",                                       "inst"),
    ("Issue Slots Busy (%)",              "sm__issue_active.avg.pct_of_peak_sustained_elapsed",            "%"),
    ("FMA Pipe (% of peak)",              "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",  "%"),
    ("ALU Pipe (% of peak)",              "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",   "%"),
    ("Registers / thread",                "launch__registers_per_thread",                                  "reg"),
    ("Theoretical Occupancy (%)",         _theo_occ,                                                       "%"),
    ("Achieved Occupancy (%)",            "sm__warps_active.avg.pct_of_peak_sustained_active",             "%"),
    ("Waves / SM",                        "launch__waves_per_multiprocessor",                              "wave"),
    ("Warp Cycles / Issued Instr",        "smsp__average_warp_latency_per_inst_issued.ratio",              "cycle"),
    ("Stall: lg_throttle",                "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio", "cycle"),
    ("Stall: long_scoreboard",            "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "cycle"),
    ("Stall: mio_throttle",               "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio", "cycle"),
    ("Stall: math_pipe_throttle",         "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "cycle"),
    ("Stall: wait",                       "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", "cycle"),
    ("Stall: not_selected",               "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio", "cycle"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None,
                    help="default: <run-dir>/reports/full_<tag>.ncu-rep")
    ap.add_argument("--tag", action="append", required=True)
    args = ap.parse_args()

    out_dir = args.run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {}
    for tag in args.tag:
        rep = args.report or (args.run_dir / "reports" / f"full_{tag}.ncu-rep")
        r = ncu_report.load_report(str(rep))
        a = r.range_by_idx(0).action_by_idx(0)
        d = {}
        for label, mname, unit in KEY:
            try:
                d[label] = mname(a) if callable(mname) else a[mname].value()
            except Exception:
                d[label] = None
        d["__unit__"] = {label: unit for label, _, unit in KEY}
        tables[tag] = d

    # side-by-side text
    labels = [l for l, _, _ in KEY]
    with open(out_dir / f"metrics_key_{'_'.join(args.tag)}.txt", "w") as f:
        f.write(f"{'metric':<34}" + "".join(f"{t:>18}" for t in args.tag) + "\n")
        f.write("-" * (34 + 18 * len(args.tag)) + "\n")
        for label in labels:
            row = f"{label:<34}"
            for t in args.tag:
                v = tables[t][label]
                row += f"{(f'{v:,.4g}' if isinstance(v, (int, float)) else '-'):>18}"
            f.write(row + "\n")
    with open(out_dir / f"metrics_key_{'_'.join(args.tag)}.json", "w") as f:
        json.dump(tables, f, indent=2)

    print(open(out_dir / f"metrics_key_{'_'.join(args.tag)}.txt").read())


if __name__ == "__main__":
    main()
