#include <torch/extension.h>
#include <cuda_runtime.h>

// [B,16,8,8] -> [B,16,4,4], fixed 2x2 average pool.
__global__ void kernel(const float* x, float* y, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int ox = i % 4, oy = (i / 4) % 4, c = (i / 16) % 16, n = i / (16 * 16);
    const float* p = x + ((size_t)n * 16 + c) * 64 + (2 * oy) * 8 + 2 * ox;
    y[i] = 0.25f * (p[0] + p[1] + p[8] + p[9]);
}
torch::Tensor forward(torch::Tensor x) {
    auto y = torch::empty({x.size(0), 16, 4, 4}, x.options()); const int total = x.size(0) * 16 * 16;
    kernel<<<(total + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), total);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "pool2 launch failed"); return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
