# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.10
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers,
#             General-Purpose Graphics Processor Architecture,
#             Morgan & Claypool, 2018
# September 2026
# =============================================================================
# CMakeLists.txt - NutShellGPU 模拟器构建配置
# -----------------------------------------------------------------------------
# 参考: NutShellGPU_spec 05_模拟器实现契约.md §M1 (项目结构)
#
# 本文件定义了三个构建目标:
#   1) ntisa   - 头文件库 (ISA 解码器/语义, 被 func_sim 与 cycle_sim 共享)
#   2) func_sim - 功能模拟器 (S2, 无时序, 立即完成所有指令与访存)
#   3) cycle_sim - 周期级模拟器 (S3, 带流水线/记分板/缓存/NoC/DRAM 时序)
#
# 另外构建三个测试可执行文件:
#   t_decode  - 指令解码器单元测试 (02 册 §9 测试向量)
#   t_saxpy   - SAXPY 端到端测试 (01 册 §9)
#   t_diverge - SIMT 栈发散/重汇聚快照测试 (03 册 §4)
# =============================================================================

# CMake 最低版本要求 3.10 (支持 C++17 标准)
cmake_minimum_required(VERSION 3.10)
project(NutShellGPU_sim VERSION 1.1.0 LANGUAGES CXX)

# 使用 C++17 标准 (structured bindings, constexpr if 等特性)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC 必须加 /utf-8: 本项目源码为 UTF-8 (无 BOM) 且含大量中文注释。
# 若不加, MSVC 按系统码页 (GBK) 解析: UTF-8 汉字 3 字节被 GBK 两两配对,
# 注释行末尾汉字数为奇数时会吞掉行尾换行符, 下一行代码被并入注释,
# 枚举成员/case 标签 "凭空消失", 引发 C2065/C2051/C4065 错误雪崩。
if(MSVC)
    add_compile_options(/utf-8 /fp:strict)
else()
    add_compile_options(-frounding-math -ffp-contract=off)
endif()

# -----------------------------------------------------------------------------
# 头文件库 ntisa: 仅提供 include 路径, 无需编译
# ntisa.hpp 包含 NTAS1 指令集的解码器/编码器/语义辅助函数 (02 册 §3-§7)
# 作为 INTERFACE 库, 所有依赖它的目标都能自动获得 include 路径
# -----------------------------------------------------------------------------
add_library(ntisa INTERFACE)
target_include_directories(ntisa INTERFACE ${CMAKE_SOURCE_DIR})

# -----------------------------------------------------------------------------
# 功能模拟器 (S2): 参考 05 册 §M2
# 功能模拟器不建模时序: 每条指令立即执行, 访存立即完成
# 核心实现 SIMT 栈状态机 (03 册 §4)、指令执行 (02 册 §5)、内存系统 (04 册 §3)
# -----------------------------------------------------------------------------
add_executable(func_sim
    func/func_sim.cpp   # 功能模拟器核心: 指令执行/SIMT栈/内存访问/CTA调度
    func/main.cpp)      # 入口: SAXPY 演示与内置测试 (解码/SAXPY/发散)
target_link_libraries(func_sim PRIVATE ntisa)
# MSVC: /W3 警告级别3, /wd4819 抑制 Unicode 编码警告
# GCC/Clang: -Wall 全警告, -Wno-unused 抑制未使用变量警告
if(MSVC)
    target_compile_options(func_sim PRIVATE /W3 /wd4819)
else()
    target_compile_options(func_sim PRIVATE -Wall -Wno-unused)
endif()

# -----------------------------------------------------------------------------
# 周期级模拟器 (S3): 参考 05 册 §M3, 03 册 §6-§9, 04 册 §5-§8
# 1.1 周期级模型: F-D-I-R-E-W, 顺序依赖互锁, 操作数 bank 仲裁
# 缓存层次 L1D→L2→DRAM、有限资源预约和链路带宽; 延时为模拟值
# 复用 func_sim.cpp 中的功能执行逻辑来保证正确性
# -----------------------------------------------------------------------------
add_executable(cycle_sim
    timing/cycle_sim.cpp  # 周期级实现: 流水线各阶段/缓存/NoC/DRAM
    timing/main.cpp       # 入口: SAXPY 周期级仿真, 输出 stats.json
    func/func_sim.cpp)    # 复用功能模拟器: 指令执行/内存访问的正确性基础
target_link_libraries(cycle_sim PRIVATE ntisa)
if(MSVC)
    target_compile_options(cycle_sim PRIVATE /W3 /wd4819)
else()
    target_compile_options(cycle_sim PRIVATE -Wall -Wno-unused)
endif()

# -----------------------------------------------------------------------------
# 测试套件: enable_testing() 允许 ctest 运行
# 每个测试都链接 func_sim.cpp 以获得完整的模拟器功能
# -----------------------------------------------------------------------------
enable_testing()

# t_decode: 解码器单元测试, 验证 NTAS1 各格式的编码/解码往返一致性
add_executable(t_decode       tests/t_decode.cpp    func/func_sim.cpp)
# t_saxpy: SAXPY 端到端测试, 验证单warp/多CTA/部分warp场景的正确性
add_executable(t_saxpy        tests/t_saxpy.cpp     func/func_sim.cpp)
# t_diverge: SIMT栈测试, 验证分支发散/重汇聚/嵌套发散/屏障/共享内存
add_executable(t_diverge      tests/t_diverge.cpp    func/func_sim.cpp)
foreach(t t_decode t_saxpy t_diverge)
    target_link_libraries(${t} PRIVATE ntisa)
    if(MSVC)
        target_compile_options(${t} PRIVATE /W3 /wd4819)
    else()
        target_compile_options(${t} PRIVATE -Wall -Wno-unused)
    endif()
    add_test(NAME ${t} COMMAND ${t})
endforeach()

add_executable(t_runtime_v2 tests/t_runtime_v2.cpp func/func_sim.cpp timing/cycle_sim.cpp)
target_link_libraries(t_runtime_v2 PRIVATE ntisa)
add_test(NAME t_runtime_v2 COMMAND t_runtime_v2)

# 1.1: 独立机制回归与多结构模型验证入口。
add_executable(t_timing_v11 tests/t_timing_v11.cpp func/func_sim.cpp timing/cycle_sim.cpp)
target_link_libraries(t_timing_v11 PRIVATE ntisa)
add_test(NAME t_timing_v11 COMMAND t_timing_v11)
add_executable(t_repairs tests/t_repairs.cpp func/func_sim.cpp)
target_link_libraries(t_repairs PRIVATE ntisa)
add_test(NAME t_repairs COMMAND t_repairs)
add_executable(model_runner tests/model_runner.cpp func/func_sim.cpp timing/cycle_sim.cpp)
target_link_libraries(model_runner PRIVATE ntisa)
add_test(NAME cycle_builtin COMMAND cycle_sim --test)

add_executable(t_stats_time tests/t_stats_time.cpp func/func_sim.cpp timing/cycle_sim.cpp)
target_link_libraries(t_stats_time PRIVATE ntisa)
add_test(NAME t_stats_time COMMAND t_stats_time)

# 1.1: 随包提供的残差/深度可分离夹具, 与独立参考逐层比对。
find_package(PythonInterp 3 QUIET)
if(PYTHONINTERP_FOUND)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy"
        RESULT_VARIABLE NUMPY_IMPORT_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(NUMPY_IMPORT_RESULT EQUAL 0)
        add_test(NAME tiny_lenet_example COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../examples/tiny_lenet/test_example.py
            --backend sim --runner $<TARGET_FILE:model_runner>
            --out ${CMAKE_CURRENT_BINARY_DIR}/model_tests/tiny_lenet)
        foreach(model residual_small depthwise_medium)
            add_test(NAME ${model}_simulate COMMAND model_runner
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/${model}
                ${CMAKE_CURRENT_BINARY_DIR}/model_tests/${model})
            add_test(NAME ${model}_compare COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_model.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/${model}
                ${CMAKE_CURRENT_BINARY_DIR}/model_tests/${model})
            set_tests_properties(${model}_compare PROPERTIES DEPENDS ${model}_simulate)
        endforeach()
    endif()
endif()
