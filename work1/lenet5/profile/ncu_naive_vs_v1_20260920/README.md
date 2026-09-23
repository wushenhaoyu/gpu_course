# Conv1 NCU reports: naive vs V1 implicit GEMM

Both valid reports use `CUDA_VISIBLE_DEVICES=1`, batch 1000, and `ncu --set full`. Each captures exactly one Conv1 kernel.

| Implementation | NCU report | Captured kernel |
|---|---|---|
| naive | `reports/full_naive_b1000_conv1.ncu-rep` | `kernel` |
| V1 implicit GEMM | `reports/full_v1_igemm_b1000_conv1.ncu-rep` | `conv1_igemm_v1_kernel` |

`reports/full_naive_b1000.ncu-rep` is intentionally excluded: it was a first profiling attempt with an overly broad kernel regex and captured PyTorch's input-randomization kernel instead of Conv1.

## B=1 cache-aware V2

`reports/full_v2_compact_row_b1_conv1.ncu-rep` profiles the current V2 `blockDim=(24,6)` row-local mapping. Compared with V1 at B=1, its L2 theoretical global sectors decrease from 17,496 to 16,752, and excessive sectors decrease from 3,456 to 480. `full_v2_row_warp_b1_conv1.ncu-rep` is an earlier inactive-lane V2 experiment, not the current implementation.
