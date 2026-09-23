# V2: cache-aware row-warp implicit GEMM

No shared memory. `blockDim=(24,6)` computes every channel of one complete output row. A physical warp can span channels but never output rows, so every input load remains row-local. This removes V1's `24 values from row r + 8 values from row r+1` input access pattern without inactive lanes.
