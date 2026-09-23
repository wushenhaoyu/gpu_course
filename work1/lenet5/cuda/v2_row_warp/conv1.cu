#include <torch/extension.h>
#include <cuda_runtime.h>

// V2 cache-aware direct implicit GEMM for LeNet-5 Conv1.
// A block owns all six channels of one output row. Its compact 24x6 layout
// keeps every warp within that output row (a warp can cross channels, never
// rows), so each B-operand load remains row-local without inactive lanes.
constexpr int IH = 28, IW = 28, CO = 6, R = 5, S = 5, OH = 24, OW = 24;

__global__ void conv1_igemm_v2_row_warp_kernel(const float* __restrict__ x,
                                                const float* __restrict__ a,
                                                const float* __restrict__ bias,
                                                float* __restrict__ c) {
    const int ox = threadIdx.x;
    const int m = threadIdx.y;
    const int block_row = blockIdx.x;
    const int batch = block_row / OH;
    const int oy = block_row - batch * OH;
    float acc = bias[m];
#pragma unroll
    for (int k = 0; k < R * S; ++k) {
        const int kr = k / S;
        const int ks = k - kr * S;
        acc = fmaf(a[m * (R * S) + k],
                   x[(size_t)batch * IH * IW + (oy + kr) * IW + ox + ks], acc);
    }
    c[((size_t)batch * CO + m) * OH * OW + oy * OW + ox] = tanhf(acc);
}

torch::Tensor forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32, "float32 only");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && bias.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(x.dim() == 4 && x.size(1) == 1 && x.size(2) == IH && x.size(3) == IW, "x must be [B,1,28,28]");
    TORCH_CHECK(weight.dim() == 4 && weight.size(0) == CO && weight.size(1) == 1 && weight.size(2) == R && weight.size(3) == S && bias.dim() == 1 && bias.size(0) == CO, "Conv1 parameter shape mismatch");
    auto output = torch::empty({x.size(0), CO, OH, OW}, x.options());
    // 144 active threads. CUDA warps may span two channels, but this block
    // contains exactly one output row, so no warp can cross an input row.
    dim3 block(OW, CO);
    dim3 grid(x.size(0) * OH);
    conv1_igemm_v2_row_warp_kernel<<<grid, block>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>());
    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1_igemm_v2_row_warp_kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward, "V2 row-warp direct implicit GEMM Conv1 + Tanh"); }
