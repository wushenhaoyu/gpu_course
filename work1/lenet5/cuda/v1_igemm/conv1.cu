#include <torch/extension.h>
#include <cuda_runtime.h>

// V1 implicit GEMM for standard LeNet-5 Conv1.
//
// A = filter  [M, K] = [K_out=6, C_in*R*S=25]
// B = im2col  [K, N] = [25, batch*Oh*Ow]
// C = output  [M, N] = [6, batch*576]
//
// A block owns a 6x32 C tile. threadIdx.y chooses M (output channel),
// threadIdx.x chooses N (a batch/spatial output column), and each thread
// reduces the full K dimension in registers. B is implicit: its values are
// gathered directly from x, never materialized as global im2col storage.
constexpr int CI = 1, IH = 28, IW = 28, CO = 6, R = 5, S = 5;
constexpr int OH = 24, OW = 24, K_REDUCE = CI * R * S;
constexpr int TILE_M = CO, TILE_N = 32;

__global__ void conv1_igemm_v1_kernel(const float* __restrict__ x,
                                       const float* __restrict__ a,
                                       const float* __restrict__ bias,
                                       float* __restrict__ c,
                                       int columns) {
    const int m = threadIdx.y;                         // GEMM M: output channel
    const int n_col = blockIdx.x * TILE_N + threadIdx.x; // GEMM N: batch/spatial column
    if (n_col >= columns) return;

    const int batch = n_col / (OH * OW);
    const int spatial = n_col - batch * (OH * OW);
    const int oy = spatial / OW;
    const int ox = spatial - oy * OW;
    float acc = bias[m];

#pragma unroll
    for (int k = 0; k < K_REDUCE; ++k) {
        const int ci = k / (R * S);
        const int kr = (k / S) % R;
        const int ks = k % S;
        const float av = a[m * K_REDUCE + k];
        const float bv = x[((size_t)batch * CI + ci) * IH * IW + (oy + kr) * IW + ox + ks];
        acc = fmaf(av, bv, acc);
    }
    // Convert GEMM C[m,n_col] back to NCHW output order.
    c[((size_t)batch * CO + m) * OH * OW + spatial] = tanhf(acc);
}

torch::Tensor forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32, "float32 only");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && bias.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(x.dim() == 4 && x.size(1) == CI && x.size(2) == IH && x.size(3) == IW, "x must be [B,1,28,28]");
    TORCH_CHECK(weight.dim() == 4 && weight.size(0) == CO && weight.size(1) == CI && weight.size(2) == R && weight.size(3) == S && bias.dim() == 1 && bias.size(0) == CO, "Conv1 parameter shape mismatch");
    auto output = torch::empty({x.size(0), CO, OH, OW}, x.options());
    const int columns = x.size(0) * OH * OW;
    const dim3 block(TILE_N, TILE_M);
    const dim3 grid((columns + TILE_N - 1) / TILE_N);
    conv1_igemm_v1_kernel<<<grid, block>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), columns);
    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1_igemm_v1_kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward, "V1 direct implicit-GEMM Conv1 + Tanh"); }
