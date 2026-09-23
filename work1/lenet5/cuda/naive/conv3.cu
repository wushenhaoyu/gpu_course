#include <torch/extension.h>
#include <cuda_runtime.h>

// [B,16,4,4] -> [B,120,1,1], bias + Tanh.
__global__ void kernel(const float* x, const float* w, const float* b, float* y, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int oc = i % 120, n = i / 120; float acc = b[oc];
#pragma unroll
    for (int ic = 0; ic < 16; ++ic)
#pragma unroll
        for (int k = 0; k < 16; ++k) acc = fmaf(x[((size_t)n * 16 + ic) * 16 + k], w[(oc * 16 + ic) * 16 + k], acc);
    y[i] = tanhf(acc);
}
torch::Tensor forward(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    auto y = torch::empty({x.size(0), 120, 1, 1}, x.options()); const int total = x.size(0) * 120;
    kernel<<<(total + 255) / 256, 256>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), total);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "conv3 launch failed"); return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
