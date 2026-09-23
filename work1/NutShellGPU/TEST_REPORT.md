# NutShellGPU 1.1.0 测试报告

NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education  
Version 1.10  
Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN  
Improved by: Tonghui Ming  
References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018  
September 2026


模型评测只使用公开已训练权重：MLP、CNN、TinyVGG。所有延迟均为模拟值。

## 验证方法

三种模型分别在仅正确性修复基线及 1.1 中执行同一 NTAS1 程序，所有计算经过 CycleSim。输入取官方测试集中每个类别首次出现的图片，共 10 张，索引为 [19, 2, 1, 13, 6, 8, 4, 9, 18, 0]，不按预测结果筛选。每层与独立 CPU FP32 参考比较，atol=rtol=1e-4；分类索引必须一致，CTA 全部完成。

程序总时间由父进程从启动子进程之前到其退出之后测量，含输入/权重/指令装载、初始化、模拟、逐层导出、统计及退出。每项完整运行一次，未清理系统文件缓存，未在计时期间训练或运行 GPU 测评；结果属于初步性能对照。编译、生成输入和独立结果检查不计入此区间。使用 Intel Core i5-12400F，Windows，Zig 0.13.0 C++17，-O3 -march=native -frounding-math -ffp-contract=off。

## 运行时间、准确率与模拟周期

| 模型 | 版本 | 程序总时间 / 秒 | 10 张准确率 | 总模拟周期 | 模拟时间 / 毫秒 |
|---|---|---:|---:|---:|---:|
| MLP | 1.0-correctness | 1.833 | 100.0% | 369,940 | 0.369940 |
| MLP | 1.1 | 664.017 | 100.0% | 113,043,915 | 113.043915 |
| CNN | 1.0-correctness | 7.346 | 100.0% | 1,224,380 | 1.224380 |
| CNN | 1.1 | 1149.303 | 100.0% | 192,774,640 | 192.774640 |
| TinyVGG | 1.0-correctness | 43.971 | 100.0% | 5,433,840 | 5.433840 |
| TinyVGG | 1.1 | 3775.964 | 100.0% | 501,691,267 | 501.691267 |

“1.0-correctness”为仅正确性修复基线，并非上游发布标签。两版时序规则不同。模拟时间按周期数及 1 GHz 虚拟频率换算。

### CUDA 与模拟器设备执行时间对照

| 模型 | CUDA 官方测试 10000 张准确率 | CUDA 10000 张 GPU 执行时间 / 秒 | CUDA 折算 10 张 / 毫秒 | CUDA 对应 10 张实测 / 毫秒 | 1.1 模拟 10 张 / 毫秒 |
|---|---:|---:|---:|---:|---:|
| MLP | 88.29% | 1.560397 | 1.560397 | 1.664000 | 113.043915 |
| CNN | 92.38% | 4.989463 | 4.989463 | 4.989696 | 192.774640 |
| TinyVGG | 93.21% | 6.779156 | 6.779156 | 6.815232 | 501.691267 |

CUDA 使用本轮直接循环实现，FP32 FMA、补零布局、累加顺序、各层 grid/block 与模拟器夹具一致，batch=1。不调用官方数学计算库。计时包括前向与 argmax，使用预热后的 CUDA Graph 和 CUDA Event，累加 10000 张执行区间；不包括初始化、编译、数据传输和文件输出。每张测一次，折算 10 张时间 = 10000 张时间 / 1000。“对应 10 张实测”取模拟器使用的相同索引。

此表对齐的是设备计算阶段的计时范围，不是程序端到端时间，也不是相同硬件的测试。实机为 RTX 4060（24 SM、32 线程 warp、每 SM 最多 48 warp）；模拟器为 16 NSM、32 线程 warp、每 NSM 最多 64 warp、1 GHz。实机未限制 SM 数或锁频，缓存、流水线、寄存器分配和访存延迟仍不同。CUDA Graph 提交、事件精度与运行频率会影响实测值。

10000 张预测与参考完全一致，logits 通过 atol=rtol=1e-4 检查。对应模拟器的 10 张，全部层张量最大绝对误差为 0；分类均正确。原始结果见 CUDA_ALIGNED_BENCHMARK.json。此次 GPU 计时不能替代上表的程序总时间。

## GPU 全量分类评估

在 RTX 4060 上使用自写 CUDA 运算，未载入官方数学计算库。全量包括官方训练 60,000 张与官方测试 10,000 张。官方测试准确率单列。模拟器运行 10 张，GPU 分别运行 70,000 与官方测试 10,000 张。

| 模型 | 全量 70,000 张 | 官方测试 10,000 张 | 官方测试最差类别召回率 |
|---|---:|---:|---:|
| MLP | 91.6914% | 88.29% | 73.50% |
| CNN | 96.1171% | 92.38% | 78.30% |
| TinyVGG | 94.5514% | 93.21% | 77.60% |

## 回归与复现

原有 315 项回归检查再次通过：decode 141、saxpy 9、diverge 57、repairs 35、runtime 28、timing 35、builtin 10；新增 9 项模拟时间换算检查通过。使用编译器直接构建和运行，未实际运行 CMake/CTest。

计时测试启动时冻结可执行文件及源码。后补的模拟秒数输出不改变执行规则，另经回归检查验证。程序计时脚本记录原始时间、退出码、周期、模拟秒数、虚拟频率与哈希。来源、构建校验值、原始结果及各类别召回率见 TEST_EVIDENCE.json 与 PRETRAINED_BENCHMARK.json。

公开权重来源：

- MLP，266,610 参数：https://huggingface.co/sadhaklal/mlp-fashion-mnist ，revision d6b7c50ffcc0f5d2ec5571330ccf722cf0b9d95f。
- CNN，421,642 参数：https://huggingface.co/wasif-mlops/fashion-mnist ，revision 99058b9822796b7bc471deb50b3cdbc7e3334f83。
- TinyVGG，142,794 参数：https://github.com/SatvikPraveen/FashionMNIST-Analysis ，revision 97cfc658264db29bf3bcb8a6cb330c89069223a7。

上述外部权重文件不在本包重复分发。已执行检查为表中列出的软件模拟器测试，未执行 RTL 综合验证。
