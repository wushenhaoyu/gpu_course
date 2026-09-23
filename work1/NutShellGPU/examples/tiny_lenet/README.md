# TinyLeNet 样例

NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education  
Version 1.10  
Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN  
Improved by: Tonghui Ming  
References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018  
September 2026


这是 LeNet 风格的三层可训练网络，共 9,098 个参数：

`28x28 灰度输入 → Conv(1,8,3x3) → ReLU → MaxPool(2x2) → Conv(8,16,3x3) → ReLU → MaxPool(2x2) → FC(784,10)`。

输入为 FP32 NCHW，像素除以 255；卷积步幅 1、补零 1、权重 OIHW。全连接权重 OI，输出十个 logits。该网络不是原始 LeNet-5 的完整复现。

## 随包文件

- `assets`：已训练权重、模型布局及 Fashion-MNIST 官方测试集前三张输入与标签。
- `fixture`：同一模型与输入的 NTAS1 程序、模拟内存布局及独立参考张量。
- `kernels.cu`、`cuda_api.py`、`run_cuda.py`：CUDA 前向计算、Driver/NVRTC 接口和执行入口。
- `test_example.py`：独立 NumPy 参考、逐层比较、分类检查与进程外计时。


## 执行测试

在项目根目录执行。Python 需要 NumPy；CUDA 测试需要 NVIDIA 驱动与 CUDA Toolkit 的 NVRTC。可通过 `NVRTC_LIBRARY` 指定其动态库绝对路径，Windows 也可设置 `CUDA_PATH`。

```sh
python examples/tiny_lenet/test_example.py --backend sim --runner build/model_runner --out tiny_lenet_test
python examples/tiny_lenet/test_example.py --backend cuda --out tiny_lenet_cuda_test
python examples/tiny_lenet/test_example.py --backend both --runner build/model_runner --out tiny_lenet_both_test
```

Windows 多配置构建的 runner 通常位于 `build/Release/model_runner.exe`。CUDA 使用 Driver API 与 NVRTC，不调用 cuDNN/cuBLAS；测试不需要 PyTorch。

`results.json` 保存各层最大绝对误差、预测、标签、样例准确率与每个进程的 `program_seconds`。模拟器另记录 `cycles`、`core_clock_ghz` 和 `simulated_seconds = cycles / (core_clock_ghz × 10^9)`。后者是虚拟时间，不能代替实际运行时间。

程序计时包含启动、Python 导入（CUDA 入口）、CUDA 编译/初始化、数据读取、执行、逐层结果写出和退出。NumPy 参考及正确性比较在计时区间之外。CUDA 测试和模拟器测试分别完整启动进程，默认顺序执行。
