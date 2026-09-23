# NutShellGPU 1.1

NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education  
Version 1.10  
Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN  
Improved by: Tonghui Ming  
References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018  
September 2026


版本：**1.1.0**。入口材料为 [改进日志](CHANGELOG.md)、[时序与修复说明](NutShellGPU_spec/06_1.1时序与修复说明.md) 和 [测试报告](TEST_REPORT.md)。

## 模拟器

- `NutShellGPU_sim/func`：NTAS1 功能语义、SIMT 栈、内存与 CTA 资源管理。
- `NutShellGPU_sim/timing`：六阶段接口、五类执行单元、到期写回、顺序依赖互锁、寄存器 bank 仲裁及分层访存时序。
- `NutShellGPU_sim/configs/timing_v11.ini`：1.1 配置示例；所有延时为可配置的模拟值。
- `NutShellGPU_sim/tests`：指令、正确性、时序及多结构模型测试。
- `NutShellGPU_hw`：保留原 RTL 代码；1.1 的新增时序机制实现于 C++ 模拟器，未声明 RTL 同步实现或通过综合验证。
- `NutShellGPU_spec`：原六册说明及 1.1 补充册；实现差异以补充册为准。

## 构建和测试

需要 C++17 编译器及 CMake 3.10 或以上。构建只依赖标准 C++ 库。

```sh
cmake -S NutShellGPU_sim -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

显式舍入测试要求禁用 `fast-math`。CMake 已为 GCC/Clang 设置 `-frounding-math -ffp-contract=off`，为 MSVC 设置 `/utf-8 /fp:strict`。

完整模型夹具的运行方法：

```sh
build/model_runner fixture_directory output_directory NutShellGPU_sim/configs/timing_v11.ini
python NutShellGPU_sim/tests/check_model.py fixture_directory output_directory
```

Windows 多配置生成器的可执行文件通常位于 `build/Release/`。验证脚本需要 Python 和 NumPy；生成独立参考输出还需要 PyTorch，参考计算不参与模拟器执行。

测试报告分别记录模拟周期、逐层数值误差和分类准确率，并列出模拟器样本数与 GPU 评估数据集规模。

## 程序运行总时间

用 Python 标准库从进程外测量启动至退出的总时间：

```sh
python NutShellGPU_sim/tests/benchmark_process.py --runner build/model_runner --fixture fixture_directory --out process_measurements --repeats 3
```

记录每次总时间、退出码、模拟周期、输入样本数及可执行文件/夹具哈希。计时包含装载、初始化、模拟、逐层结果输出和退出；不包含编译、夹具生成和独立正确性检查。该命令使用模拟器默认配置。详细口径见 [程序总时间测量说明](NutShellGPU_spec/07_程序总时间测量.md)。

模拟统计同时输出 `simulated_seconds`，由周期数和配置的虚拟频率换算，不能替代程序运行总时间。

## TinyLeNet 示例

[三层 TinyLeNet 样例](examples/tiny_lenet/README.md)包含已训练参数、三个输入、CUDA 代码、NTAS1 夹具及逐层测试。可分别运行 CUDA 或模拟器，也可顺序运行两者。测试输出程序总时间、模拟周期、模拟运行时间及分类结果。
