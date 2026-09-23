#include <torch/extension.h>
#include <cuda_runtime.h>

// [B,84] -> [B,10], final logits without activation.
__global__ void kernel(const float* x, const float* w, const float* b, float* y, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    const int o = i % 10, n = i / 10; float acc = b[o];
#pragma unroll
    for (int k = 0; k < 84; ++k) acc = fmaf(x[(size_t)n * 84 + k], w[o * 84 + k], acc);
    y[i] = acc;
}
torch::Tensor forward(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    auto y = torch::empty({x.size(0), 10}, x.options()); const int total = x.size(0) * 10;
    kernel<<<(total + 255) / 256, 256>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), total);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fc2 launch failed"); return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
