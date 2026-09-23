#!/usr/bin/env python3
"""把多个 conv1 版本的关键 ncu 指标并排放在一起。

用法:
    python cmp_versions.py --rep ../../../profile/v1_convpool/reports/full_v1.ncu-rep:v1 \
                           --rep ../../../profile/v2_conv1_smem_bcast/reports/full_v2.ncu-rep:v2 ...
    或者直接 python cmp_versions.py            # 用默认的三个版本
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import load_action  # noqa: E402

W1 = Path("/data/workspace/haoyu/code/learn/gpu_course/work1")

DEFAULT = [
    (W1 / "profile/v1_convpool/reports/full_v1.ncu-rep", "v1"),
    (W1 / "profile/v2_conv1_smem_bcast/reports/full_v2.ncu-rep", "v2"),
    (W1 / "profile/v3_conv1_coalesced/reports/full_v3.ncu-rep", "v3"),
    (W1 / "profile/v4_conv1_implicit_gemm/reports/full_v4.ncu-rep", "v4"),
]


def dram_pct(a):
    return (a["dram__bytes_read.sum.pct_of_peak_sustained_elapsed"].value()
            + a["dram__bytes_write.sum.pct_of_peak_sustained_elapsed"].value())


def theo_occ(a):
    limits = [a[f"launch__occupancy_limit_{k}"].value()
              for k in ("blocks", "registers", "shared_mem", "warps")]
    return min(limits) * (a["launch__block_size"].value() / 32.0) / 64.0 * 100.0


M = [
    ("Duration (ms)", lambda a: a["gpu__time_duration.sum"].value() / 1e6, "{:,.4f}"),
    ("Elapsed cycles", "sm__cycles_elapsed.avg", "{:,.0f}"),
    ("SM Throughput %", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "{:.2f}"),
    ("L1/TEX Throughput %", "l1tex__throughput.avg.pct_of_peak_sustained_elapsed", "{:.2f}"),
    ("L2 Throughput %", "lts__throughput.avg.pct_of_peak_sustained_elapsed", "{:.2f}"),
    ("DRAM Throughput %", dram_pct, "{:.2f}"),
    ("L1 Hit Rate %", "l1tex__t_sector_hit_rate.pct", "{:.2f}"),
    ("L2 Hit Rate %", "lts__t_sector_hit_rate.pct", "{:.2f}"),
    ("  global-ld sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "{:,.0f}"),
    ("  global-st sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "{:,.0f}"),
    ("  L2 rd sectors", "lts__t_sectors_srcunit_tex_op_read.sum", "{:,.0f}"),
    ("  L2 wr sectors", "lts__t_sectors_srcunit_tex_op_write.sum", "{:,.0f}"),
    ("  smem wavefronts", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum", "{:,.0f}"),
    ("  smem bank conflicts", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", "{:,.0f}"),
    ("Bytes/Sector (gld)", "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio", "{:.2f}"),
    ("Inst Executed (warp)", "smsp__inst_executed.sum", "{:,.0f}"),
    ("Issue Slots Busy %", "sm__issue_active.avg.pct_of_peak_sustained_elapsed", "{:.2f}"),
    ("FMA Pipe % of peak", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed", "{:.2f}"),
    ("ALU Pipe %", "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active", "{:.2f}"),
    ("Registers / thread", "launch__registers_per_thread", "{:,.0f}"),
    ("Shared mem / block", "launch__shared_mem_per_block_static", "{:,.0f}"),
    ("Theoretical Occupancy %", theo_occ, "{:.1f}"),
    ("Achieved Occupancy %", "sm__warps_active.avg.pct_of_peak_sustained_active", "{:.2f}"),
    ("Warp Cycles / Issued Instr", "smsp__average_warp_latency_per_inst_issued.ratio", "{:.2f}"),
    ("Stall: mio_throttle", "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio", "{:.3f}"),
    ("Stall: long_scoreboard", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "{:.3f}"),
    ("Stall: lg_throttle", "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio", "{:.3f}"),
    ("Stall: short_scoreboard", "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", "{:.3f}"),
    ("Stall: math_pipe_throttle", "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "{:.3f}"),
    ("Stall: wait", "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", "{:.3f}"),
    ("Stall: not_selected", "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio", "{:.3f}"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", action="append", default=None, help="path:tag, 可重复")
    ap.add_argument("-o", "--out", type=Path, default=HERE / "cmp_conv1_versions.txt")
    args = ap.parse_args()

    pairs = []
    if args.rep:
        for s in args.rep:
            p, _, t = s.rpartition(":")
            pairs.append((Path(p), t))
    else:
        pairs = [(p, t) for p, t in DEFAULT if p.exists()]

    acts = []
    for p, t in pairs:
        acts.append((t, load_action(p)))

    lines = []
    w = 30
    lines.append(f"{'metric':<{w}}" + "".join(f"{t:>18}" for t, _ in acts))
    lines.append("-" * (w + 18 * len(acts)))
    for label, mname, fmt in M:
        row = f"{label:<{w}}"
        for _, a in acts:
            try:
                v = mname(a) if callable(mname) else a[mname].value()
                row += f"{fmt.format(v):>18}"
            except Exception:
                row += f"{'-':>18}"
        lines.append(row)

    txt = "\n".join(lines)
    args.out.write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
