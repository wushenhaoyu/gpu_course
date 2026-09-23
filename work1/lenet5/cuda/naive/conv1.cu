#include <torch/extension.h>
#include <cuda_runtime.h>

// [B,1,28,28] -> [B,6,24,24], bias + Tanh.
__global__ void kernel(const float* x, const float* w, const float* b, float* y, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int ox = i % 24, oy = (i / 24) % 24, oc = (i / 576) % 6, n = i / (6 * 576);
    const float* xp = x + (size_t)n * 28 * 28 + oy * 28 + ox;
    const float* wp = w + oc * 25;
    float acc = b[oc];
#pragma unroll
    for (int ky = 0; ky < 5; ++ky)
#pragma unroll
        for (int kx = 0; kx < 5; ++kx) acc = fmaf(xp[ky * 28 + kx], wp[ky * 5 + kx], acc);
    y[i] = tanhf(acc);
}
torch::Tensor forward(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda(), "CUDA tensors required");
    auto y = torch::empty({x.size(0), 6, 24, 24}, x.options());
    const int total = x.size(0) * 6 * 24 * 24;
    kernel<<<(total + 255) / 256, 256>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), total);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "conv1 launch failed"); return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
