#include <torch/extension.h>
#include <cuda_runtime.h>

// [B,120] -> [B,84], bias + Tanh.
__global__ void kernel(const float* x, const float* w, const float* b, float* y, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    const int o = i % 84, n = i / 84; float acc = b[o];
#pragma unroll
    for (int k = 0; k < 120; ++k) acc = fmaf(x[(size_t)n * 120 + k], w[o * 120 + k], acc);
    y[i] = tanhf(acc);
}
torch::Tensor forward(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    auto y = torch::empty({x.size(0), 84}, x.options()); const int total = x.size(0) * 84;
    kernel<<<(total + 255) / 256, 256>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), total);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fc1 launch failed"); return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
