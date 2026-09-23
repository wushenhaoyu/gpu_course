#include <torch/extension.h>
#include <cuda_runtime.h>

// [B,6,12,12] -> [B,16,8,8], bias + Tanh.
__global__ void kernel(const float* x, const float* w, const float* b, float* y, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int ox = i % 8, oy = (i / 8) % 8, oc = (i / 64) % 16, n = i / (16 * 64);
    float acc = b[oc];
#pragma unroll
    for (int ic = 0; ic < 6; ++ic)
#pragma unroll
        for (int ky = 0; ky < 5; ++ky)
#pragma unroll
            for (int kx = 0; kx < 5; ++kx)
                acc = fmaf(x[((size_t)n * 6 + ic) * 144 + (oy + ky) * 12 + ox + kx], w[(oc * 6 + ic) * 25 + ky * 5 + kx], acc);
    y[i] = tanhf(acc);
}
torch::Tensor forward(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    auto y = torch::empty({x.size(0), 16, 8, 8}, x.options()); const int total = x.size(0) * 16 * 64;
    kernel<<<(total + 255) / 256, 256>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), total);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "conv2 launch failed"); return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
