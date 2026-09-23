# NutShellGPU 1.1.0 改进日志

NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education  
Version 1.10  
Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN  
Improved by: Tonghui Ming  
References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018  
September 2026


## 程序运行时间测量

新增 `NutShellGPU_sim/tests/benchmark_process.py`，使用 Python 标准库在进程外测量完整运行时间，保存重复测量、退出码、模拟周期、样本数、平台和文件哈希。补充说明第 07 册定义计时边界和复现方法。该入口不改变模拟器执行规则；正确性检查独立于计时区间执行。

统计新增 `simulated_seconds`，按 `cycles / (core_clock_ghz × 10^9)` 换算虚拟运行时间，保留纳秒级数值。程序实际运行时间仍单独测量。新增频率与周期换算测试。

## 新增时序机制

| 项目 | 改动 | 主要位置 |
|---|---|---|
| T01 到期写回 | 发射、收集不提前修改架构状态；ready_cycle 到期且取得提交端口才执行功能语义 | timing/cycle_sim.cpp：collector_tick、writeback_tick |
| T02 指令差异 | 搬运、整数、乘法、除法、浮点、SFU、DP、shuffle、控制流分别设置延时 | timing/timing_model.hpp：execution_latency |
| T03 依赖互锁 | 每 warp 未提交指令持有互锁，覆盖数据、谓词、宽寄存器和控制依赖 | timing/cycle_sim.hpp：CycWarpState；cycle_sim.cpp：fetch_tick、issue_tick |
| T04 寄存器组冲突 | GPR bank 端口竞争、重复源广播、跨 collector 轮询仲裁 | timing/cycle_sim.cpp：collector_tick |
| T05 执行资源 | 延时与启动间隔分离；有限执行槽、lane 数吞吐限制和提交端口竞争 | timing/cycle_sim.cpp：init、collector_tick、writeback_tick |
| T06 内存时序 | 扇区合并、shared bank 冲突、L1/L2 时序 tag、有限 MSHR、NoC 带宽、DRAM bank/行与总线 | timing/cycle_sim.cpp：memory_ready；timing_model.hpp：TimingCache |
| T07 配置与统计 | 模拟 ns 到周期换算、配置合法性检查、1.1 统计字段和带宽单位修正 | func/func_sim.hpp：Config；cycle_sim.cpp：validate_timing_config、CycStats::to_json |

所有延时为可配置的模拟值。当前实现包含每 warp 顺序互锁、I-Cache 命中假设和内存资源预约模型，接口说明见补充册。

## 正确性修复（含先前修复）

以下路径相对于 `NutShellGPU_sim/`，逻辑位置以函数名标示。

| ID | 问题与修复 | 文件／逻辑位置 |
|---|---|---|
| F01 | 多模块启动不能因同名或相同入口混用模块，按加载的模块身份执行 | func/func_sim.cpp：launch、launch_by_name、fetch_inst |
| F02 | 支持非零 code_offset，并拒绝非法启动尺寸、入口和超额资源 | func/func_sim.cpp：launch |
| F03 | 初始驻留之外的 CTA 必须继续补发；未发完不能误判完成 | func/func_sim.cpp：dispatch_pending、sync；func_sim.hpp：is_done |
| F04 | 静态＋动态共享内存、寄存器、warp 和 CTA 使用统一分配与回收账本 | func/func_sim.cpp：dispatch_cta、sync；timing/cycle_sim.cpp：cta_retire_tick |
| F05 | 非法／未对齐 PC 和代码越界返回明确错误 | func/func_sim.cpp：fetch_inst |
| F06 | 可重用屏障按代登记到达状态，同一代每 warp 登记一次 | func/func_sim.cpp：bar_sync；func_sim.hpp：BarrierState |
| F07 | FP32 算术不再错误地按整数舍入；编译设置保留舍入模式 | isa/ntisa.hpp：round_fp32；func/func_sim.cpp：浮点分支；CMakeLists.txt |
| F08 | RZ 读取恒零；SHFL 先快照所有源 lane，再提交各 lane 的结果 | func/func_sim.hpp：WarpState::reg；func_sim.cpp：execute_inst、OP_SHFL |
| F09 | 稀疏页号保留 64 位；跨页访问、复制、clear 的页指针生命周期正确 | func/func_sim.hpp：PageMem |
| F10 | 禁止重复取同一条未提交指令；在途状态排空后才复用 CTA/warp 槽 | timing/cycle_sim.cpp：fetch_tick、cta_retire_tick |
| F11 | LDC 的宽度按 ldc_width 解码，不误用通用 mem_width 的对齐检查 | isa/ntisa.hpp：decode |
| F12 | 每次 run 使用增量预算；超预算失败；占用率按时间累计 | timing/cycle_sim.cpp：run、tick、collect_stats |
| F13 | flat 解码偏移与 lane 基址互逆，local 地址包含 NSM 身份；宽访存高字与 local/const 路径统一，访问执行边界检查 | func/func_sim.cpp：dispatch_cta、decode_flat、load_mem、store_mem、execute_inst 的 LD/ST 分支 |
| F14 | 原子读写失败向调用方传播，未实现的原子宽度明确拒绝；CVTA 地址转换不产生访存请求 | func/func_sim.cpp：atom_op；timing/cycle_sim.cpp：collector_tick |

## 实现范围

新增时序机制位于 C++ 模拟器；RTL 文件保留原实现。延迟参数为模拟值。接口和统计字段见第 06、07 册。


## 预训练模型跨版本补充实测（2026-09-17）

MLP、CNN、TinyVGG 使用下载的固定权重、相同 NTAS 与同一组官方测试图，每类一张共 10 张。程序时间为进程启动至退出，含装载、初始化、模拟、逐层导出和退出，每项实测一次。全量准确率另在 RTX 4060 上用自写 CUDA 测量，无官方数学计算库。

| 模型 | 版本 | 程序总时间 / 秒 | 10 张准确率 | 总模拟周期 | 模拟时间 / 毫秒 |
|---|---|---:|---:|---:|---:|
| MLP | 1.0-correctness | 1.833 | 100.0% | 369,940 | 0.369940 |
| MLP | 1.1 | 664.017 | 100.0% | 113,043,915 | 113.043915 |
| CNN | 1.0-correctness | 7.346 | 100.0% | 1,224,380 | 1.224380 |
| CNN | 1.1 | 1149.303 | 100.0% | 192,774,640 | 192.774640 |
| TinyVGG | 1.0-correctness | 43.971 | 100.0% | 5,433,840 | 5.433840 |
| TinyVGG | 1.1 | 3775.964 | 100.0% | 501,691,267 | 501.691267 |

| 模型 | 全量 70,000 张 | 官方测试 10,000 张 | 官方测试最差类别召回率 |
|---|---:|---:|---:|
| MLP | 91.6914% | 88.29% | 73.50% |
| CNN | 96.1171% | 92.38% | 78.30% |
| TinyVGG | 94.5514% | 93.21% | 77.60% |

模拟器逐层结果在 atol=rtol=1e-4 内符合独立参考，分类索引一致，CTA 完整完成。“1.0-correctness”是之前整理的仅正确性修复基线，不是上游发布标签。两版时序规则不同，周期差和墙钟差不代表相同模拟精度下的加速。所有延迟为模拟值。

70,000 张全量含训练数据；模拟器运行 10 张，GPU 分别运行 70,000 与官方测试 10,000 张。原始证据、来源与构建哈希见 PRETRAINED_BENCHMARK.json。

## CUDA 设备执行时间对照

| 模型 | CUDA 官方测试 10000 张准确率 | CUDA 10000 张 GPU 执行时间 / 秒 | CUDA 折算 10 张 / 毫秒 | CUDA 对应 10 张实测 / 毫秒 | 1.1 模拟 10 张 / 毫秒 |
|---|---:|---:|---:|---:|---:|
| MLP | 88.29% | 1.560397 | 1.560397 | 1.664000 | 113.043915 |
| CNN | 92.38% | 4.989463 | 4.989463 | 4.989696 | 192.774640 |
| TinyVGG | 93.21% | 6.779156 | 6.779156 | 6.815232 | 501.691267 |

CUDA 使用本轮直接循环实现，FP32 FMA、补零布局、累加顺序、各层 grid/block 与模拟器夹具一致，batch=1。不调用官方数学计算库。计时包括前向与 argmax，使用预热后的 CUDA Graph 和 CUDA Event，累加 10000 张执行区间；不包括初始化、编译、数据传输和文件输出。每张测一次，折算 10 张时间 = 10000 张时间 / 1000。“对应 10 张实测”取模拟器使用的相同索引。

此表对齐的是设备计算阶段的计时范围，不是程序端到端时间，也不是相同硬件的测试。实机为 RTX 4060（24 SM、32 线程 warp、每 SM 最多 48 warp）；模拟器为 16 NSM、32 线程 warp、每 NSM 最多 64 warp、1 GHz。实机未限制 SM 数或锁频，缓存、流水线、寄存器分配和访存延迟仍不同。CUDA Graph 提交、事件精度与运行频率会影响实测值。

10000 张预测与参考完全一致，logits 通过 atol=rtol=1e-4 检查。对应模拟器的 10 张，全部层张量最大绝对误差为 0；分类均正确。原始结果见 CUDA_ALIGNED_BENCHMARK.json。此次 GPU 计时不能替代上表的程序总时间。
