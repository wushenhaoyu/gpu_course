#include <torch/extension.h>
#include <cuda_runtime.h>

// [B,6,24,24] -> [B,6,12,12], fixed 2x2 average pool.
__global__ void kernel(const float* x, float* y, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int ox = i % 12, oy = (i / 12) % 12, c = (i / 144) % 6, n = i / (6 * 144);
    const float* p = x + ((size_t)n * 6 + c) * 24 * 24 + (2 * oy) * 24 + 2 * ox;
    y[i] = 0.25f * (p[0] + p[1] + p[24] + p[25]);
}
torch::Tensor forward(torch::Tensor x) {
    auto y = torch::empty({x.size(0), 6, 12, 12}, x.options()); const int total = x.size(0) * 6 * 144;
    kernel<<<(total + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), total);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "pool1 launch failed"); return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
