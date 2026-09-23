# V1 direct implicit GEMM

Conv1 only, with `A[6,25] × B_implicit[25,B×576] → C[6,B×576]`. A `6×32` CUDA block tile maps `threadIdx.y` to GEMM M and `threadIdx.x` to GEMM N. Reduction K=25 stays in registers. No shared memory, explicit im2col buffer, Tensor Core, or pooling fusion is used in V1.
