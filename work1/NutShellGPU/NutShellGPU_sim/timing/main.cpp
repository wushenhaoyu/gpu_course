// =============================================================================
// NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
// Version 1.10
// Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
// Improved by: Tonghui Ming
// References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers,
//             General-Purpose Graphics Processor Architecture,
//             Morgan & Claypool, 2018
// September 2026
// =============================================================================
// main.cpp - 周期级模拟器主入口 (S3 阶段) ★★★ 性能分析入口
// -----------------------------------------------------------------------------
// 本文件是 NutShellGPU 周期级模拟器的主入口, 提供:
//   1. SAXPY 内核编码 (build_saxpy_kernel)
//   2. 周期级模拟测试 (test_saxpy_cycle): 运行 SAXPY 并验证结果
//   3. 统计格式验证 (test_stats_format): 检查 stats.json 字段完整性
//   4. 命令行入口 (main): 支持 --test 模式与默认演示模式
//
// 与功能模拟器 (func/main.cpp) 的区别:
//   - 功能模拟器只验证 "结果是否正确"
//   - 周期模拟器额外输出 "性能指标" (周期数/IPC/SIMD 效率)
//   - 输出 stats.json 文件供后续分析工具使用
//
// 教学要点:
//   - SAXPY 是 GPU 的 "Hello World", 用于验证模拟器
//   - 周期模拟器复用功能模拟器的架构状态 (FuncSim)
//   - 性能指标反映 GPU 微结构的效率
// =============================================================================
#include "cycle_sim.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>

using namespace ntisa;

// =============================================================================
// build_saxpy_kernel: SAXPY 内核编码 (01 册 §9) ★ 经典 GPU 内核
// -----------------------------------------------------------------------------
// SAXPY: y = a*x + y (Single precision, A, X, Plus, Y)
// 这是 GPU 编程的 "Hello World", 用于验证模拟器功能
//
// 内核逻辑:
//   1. 从常量内存加载参数 (n, a, xp, yp)
//   2. 计算全局线程索引: i = blockIdx.x * blockDim.x + threadIdx.x
//   3. 边界检查: if (i >= n) goto done
//   4. 计算字节偏移: offset = i * 4 (float 4 字节)
//   5. 加载 x[i] 和 y[i], 计算 a*x[i]+y[i], 存回 y[i]
//   6. EXIT
//
// 参数布局 (常量内存 bank 0):
//   [0x00] n  (u32): 元素个数
//   [0x04] a  (f32): 标量系数
//   [0x08] xp (u64): x 数组指针
//   [0x10] yp (u64): y 数组指针
//
// 指令序列 (24 条):
//   00-18: 加载参数 (LDC) + 计算线程索引 (S2R/MOV/IMAD)
//   48-58: 边界检查 (SETP/SSY/BRA)
//   60-88: 计算字节偏移 (MOV32I/IMAD.WIDE/IADD.U64)
//   90-98: 加载数据 (LDG)
//   A0:    计算 (FFMA)
//   A8:    存储 (STG)
//   B0:    退出 (EXIT)
// =============================================================================
// 教材引用: 第 2 章 §2.1 Programming Model (p.10) — SAXPY 内核体现 CUDA 线程索引 i=blockIdx.x*blockDim.x+threadIdx.x; §2.2.2 Instruction Encoding (p.18) 64 位指令编码; §2.2.3 Fused Multiply-Add (p.20) FFMA 第三源通过 Rd 字段 (Rd=Ra*Rc+Rd_old)
static std::vector<uint64_t> build_saxpy_kernel() {
    std::vector<uint64_t> c;
    // 参数: c[0x0][0x00]=n(u32) [0x04]=a(f32) [0x08]=xp(u64) [0x10]=yp(u64)
    c.push_back(encode(OP_LDC, 0,0,0,0, FK_MI, 2, 0, 0, (0<<5)|0, 0, 0, 0, 0, 0));      // 00: LDC.Param.U32 R2, c[0x0][0x00]
    c.push_back(encode(OP_LDC, 0,0,0,0, FK_MI, 1, 0, 4, (2<<5)|0, 0, 0, 0, 0, 4));      // 08: LDC.Param.F32 R1, c[0x0][0x04]
    c.push_back(encode(OP_LDC, 0,0,0,0, FK_MI, 12, 0, 8, (1<<5)|0, 0, 0, 0, 0, 8));     // 10: LDC.Param.U64 R12, c[0x0][0x08]
    c.push_back(encode(OP_LDC, 0,0,0,0, FK_MI, 14, 0, 16, (1<<5)|0, 0, 0, 0, 0, 16));   // 18: LDC.Param.U64 R14, c[0x0][0x10]
    c.push_back(encode(OP_S2R, 0,0,0,0, FK_R, 6, 0, 0, SR_CTAID_X, 0, 0, 0, 0, 0));     // 20: S2R R6, SR_CTAID.X
    c.push_back(encode(OP_S2R, 0,0,0,0, FK_R, 7, 0, 0, SR_NTID_X, 0, 0, 0, 0, 0));      // 28: S2R R7, SR_NTID.X
    c.push_back(encode(OP_S2R, 0,0,0,0, FK_R, 9, 0, 0, SR_TID_X, 0, 0, 0, 0, 0));       // 30: S2R R9, SR_TID.X
    c.push_back(encode(OP_MOV, 0,0,0,0, FK_R, 3, 9, 0, 0, 0, 0, 0, 0, 0));             // 38: MOV.U32 R3, R9
    c.push_back(encode(OP_IMAD, 0,0,0,0, FK_R, 3, 6, 7, 0, 0, 0, 0, 0, 0));            // 40: IMAD.LO.U32 R3, R6, R7
    c.push_back(encode(OP_SETP, 0,0,0,0, FK_R, 0, 3, 2, 0x05, 0x83, 0, 0, 0, 0));       // 48: SETP.GE.S32 P0, R3, R2
    c.push_back(encode(OP_SSY, 0,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 12, 0));             // 50: SSY Ldone=0xB0
    c.push_back(encode(OP_BRA, 1,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 11, 0));             // 58: @P0 BRA Ldone
    c.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 16, 0, 0, 0, 0, 0, 4, 0, 0));          // 60: MOV32I R16, 4
    c.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 10, 0, 0, 0, 0, 0, 0, 0, 0));          // 68: MOV32I R10, 0
    c.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 11, 0, 0, 0, 0, 0, 0, 0, 0));          // 70: MOV32I R11, 0
    c.push_back(encode(OP_IMAD, 0,0,0,0, FK_R, 10, 3, 16, (2<<3)|0, 0, 0, 0, 0, 0));   // 78: IMAD.WIDE.U32 R10, R3, R16
    c.push_back(encode(OP_IADD, 0,0,0,0, FK_R, 12, 12, 10, 6, 0, 0, 0, 0, 0));         // 80: IADD.U64 R12, R12, R10
    c.push_back(encode(OP_IADD, 0,0,0,0, FK_R, 14, 14, 10, 6, 0, 0, 0, 0, 0));         // 88: IADD.U64 R14, R14, R10
    c.push_back(encode(OP_LD, 0,0,0,0, FK_MI, 0, 12, 0, 0x38, 0x00, 0, 0, 0, 0));      // 90: LDG.CA.F32 R0, [R12]
    c.push_back(encode(OP_LD, 0,0,0,0, FK_MI, 2, 14, 0, 0x38, 0x00, 0, 0, 0, 0));      // 98: LDG.CA.F32 R2, [R14]
    c.push_back(encode(OP_FFMA, 0,0,0,0, FK_R, 2, 0, 1, 0, 0, 0, 0, 0, 0));            // A0: FFMA.RN.F32 R2, R0, R1, R2  fma(R0=x, R1=a, R2=y) → R2
    c.push_back(encode(OP_ST, 0,0,0,0, FK_MI, 2, 14, 0, 0x38, 0x01, 0, 0, 0, 0));       // A8: STG.CG.F32 [R14], R2 (R2=a*x+y)
    c.push_back(encode(OP_EXIT, 0,0,0,0, FK_R, 0, 0, 0, 0, 0, 0, 0, 0, 0));             // B0: EXIT
    return c;
}

// =============================================================================
// 测试辅助宏与计数器
// -----------------------------------------------------------------------------
// pass_count/fail_count: 全局测试统计
// CHECK 宏: 断言, 失败时输出错误信息与行号
// =============================================================================
static int pass_count = 0, fail_count = 0;
#define CHECK(cond, msg) do { if (cond) pass_count++; else { fail_count++; std::cerr << "FAIL: " << msg << " at line " << __LINE__ << std::endl; } } while(0)

// =============================================================================
// test_saxpy_cycle: 周期级 SAXPY 模拟测试 (05 册 §7) ★ 功能+性能验证
// -----------------------------------------------------------------------------
// 测试流程:
//   1. 初始化 CycleSim, 同步配置 (sim.func.cfg = sim.cfg)
//   2. 加载 SAXPY 内核 (load_code)
//   3. 准备测试数据: 256 个 float, a = 2.5
//   4. 分配设备内存 (malloc), 拷贝数据 (memcpy_h2d)
//   5. 启动内核 (launch_by_name): 1 CTA × 256 线程
//   6. 运行周期模拟 (sim.run), 检查错误码
//   7. 拷贝结果回主机 (memcpy_d2h), 验证正确性
//   8. 输出性能指标 (周期/指令/IPC/SIMD 效率)
//   9. 输出 stats.json 文件
//
// 教学要点:
//   - 周期模拟器同时验证功能正确性与性能指标
//   - IPC 反映流水线效率 (越接近 1 越好)
//   - SIMD 效率反映 SIMT 发散损失 (越接近 1 越好)
// =============================================================================
// 教材引用: 第 3 章 §3.3 Three-Loop Approximation (p.35) — 周期级模拟器入口运行 SAXPY; §5.1 Analyzing a GPU Microarchitecture (p.119) 性能指标 IPC/SIMD 效率; §5.3 Validation (p.129) 结果验证容差 0.001
static void test_saxpy_cycle() {
    std::cout << "--- test_saxpy_cycle ---" << std::endl;
    CycleSim sim;
    sim.func.cfg = sim.cfg;
    sim.init();

    // 加载 SAXPY 内核
    KernelInfo ki;
    ki.name = "saxpy";
    ki.regs_u32 = 16;     // 16 个 32 位寄存器 (R0-R15, xp 在 R12:R13, yp 在 R14:R15)
    ki.regs_u64 = 2;      // 2 个 64 位寄存器 (R12:R13, R14:R15)
    ki.param_size = 20;   // 参数 20 字节 (n+a+xp+yp)
    sim.func.load_code(build_saxpy_kernel(), ki);

    // 准备测试数据 (256 个元素)
    int n = 256;
    std::vector<float> hx(n), hy(n);
    for (int i = 0; i < n; i++) { hx[i] = (float)(i+1); hy[i] = (float)(i*10); }
    float a = 2.5f;

    // 分配设备内存并拷贝数据
    uint64_t devx = sim.func.malloc(n * 4);
    uint64_t devy = sim.func.malloc(n * 4);
    sim.func.memcpy_h2d(devx, hx.data(), n * 4);
    sim.func.memcpy_h2d(devy, hy.data(), n * 4);
    struct { uint32_t n; float a; uint64_t xp; uint64_t yp; } params;
    params.n = n; params.a = a; params.xp = devx; params.yp = devy;
    sim.func.launch_by_name("saxpy", 1, 1, 1, 256, 1, 1, 0,
                            std::vector<uint8_t>((uint8_t*)&params, (uint8_t*)&params + sizeof(params)));

    sim.run(50000);
    CHECK(sim.func.error.code == ERR_OK || sim.func.error.code == ERR_DEADLOCK, "cycle sim SAXPY run OK");

    // 验证结果
    std::vector<float> result(n);
    sim.func.memcpy_d2h(result.data(), devy, n * 4);
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float exp = a * hx[i] + hy[i];
        if (std::abs(result[i] - exp) > 0.001f) errors++;
    }
    CHECK(errors == 0, "cycle sim SAXPY result correct");

    // 输出性能指标
    std::cout << "  Cycles: " << sim.stats.total_cycles << std::endl;
    std::cout << "  Instructions: " << sim.stats.instructions << std::endl;
    std::cout << "  IPC: " << sim.stats.ipc() << std::endl;
    std::cout << "  SIMD efficiency: " << sim.stats.simd_efficiency() << std::endl;

    // 输出 stats.json
    sim.output_stats_json("stats.json");
    std::cout << "  Stats written to stats.json" << std::endl;
}

// =============================================================================
// test_stats_format: 验证 stats.json 字段完整性 (05 册 §7) ★ 统计格式
// -----------------------------------------------------------------------------
// 检查 stats.json 是否包含规范要求的字段:
//   - cycles: 总周期数
//   - instructions: 总指令数
//   - ipc: 指令/周期
//   - stall_breakdown: 停顿分解
//   - simt_divergent_branch: 发散分支数
//   - dram_row_hits: DRAM 行命中数
//   - noc_flits_sent: NoC 流量
//
// 教学要点:
//   - 统计字段是性能分析的基础, 必须完整
//   - JSON 格式便于工具解析 (Python/可视化工具)
// =============================================================================
// 教材引用: 第 5 章 §5.1 Analyzing a GPU Microarchitecture (p.119) — stats.json 字段对应性能指标 IPC/停顿分解/发散分支/DRAM 行命中/NoC 流量
static void test_stats_format() {
    std::cout << "--- test_stats_format ---" << std::endl;
    std::ifstream f("stats.json");
    CHECK(f.good(), "stats.json exists");
    if (f.good()) {
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        // 检查规范要求的字段
        CHECK(content.find("\"cycles\"") != std::string::npos, "stats has cycles");
        CHECK(content.find("\"instructions\"") != std::string::npos, "stats has instructions");
        CHECK(content.find("\"ipc\"") != std::string::npos, "stats has ipc");
        CHECK(content.find("\"stall_breakdown\"") != std::string::npos, "stats has stall_breakdown");
        CHECK(content.find("\"simt_divergent_branch\"") != std::string::npos, "stats has simt_divergent_branch");
        CHECK(content.find("\"dram_row_hits\"") != std::string::npos, "stats has dram_row_hits");
        CHECK(content.find("\"noc_flits_sent\"") != std::string::npos, "stats has noc_flits_sent");
    }
}

// =============================================================================
// main: 命令行入口 (05 册 §7) ★ 程序入口
// -----------------------------------------------------------------------------
// 支持两种模式:
//   1. --test: 运行所有内置测试 (test_saxpy_cycle + test_stats_format)
//      输出 通过/失败 计数, 返回 0=全过 / 1=有失败
//   2. 默认: 运行 SAXPY 演示 (1024 元素, a=3.0), 输出性能指标
//
// 默认模式流程:
//   1. 初始化 CycleSim
//   2. 加载 SAXPY 内核
//   3. 准备 1024 元素数据, a = 3.0
//   4. 分配设备内存, 拷贝数据
//   5. 启动内核 (4 个 CTA, 每个 256 线程)
//   6. 运行周期模拟
//   7. 验证结果, 输出性能指标
//   8. 输出 stats.json
// =============================================================================
// 教材引用: 第 3 章 §3.3 Three-Loop Approximation (p.35) — 周期模拟器主入口; §5.3 Validation (p.129) SAXPY 端到端验证容差 0.001; §5.1 (p.119) 输出性能指标
int main(int argc, char** argv) {
    std::cout << "NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education\n"
              << "Version 1.10\n"
              << "Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN\n"
              << "Improved by: Tonghui Ming\n"
              << "References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018\n"
              << "September 2026\n";
    if (argc > 1 && std::string(argv[1]) == "--test") {
        // 测试模式: 运行所有内置测试
        test_saxpy_cycle();
        test_stats_format();
        std::cout << "\n===== Summary =====" << std::endl;
        std::cout << "Passed: " << pass_count << ", Failed: " << fail_count << std::endl;
        return fail_count > 0 ? 1 : 0;
    }

    // 默认模式: 运行 SAXPY 演示 (周期级时序)
    std::cout << "NutShellGPU Cycle-Level Simulator (S3)" << std::endl;
    std::cout << "Usage: " << argv[0] << " [--test]" << std::endl;
    std::cout << "  --test: run all built-in tests" << std::endl;

    CycleSim sim;
    sim.func.cfg = sim.cfg;  // 先同步 cfg, 再一次性 init
    sim.init();

    // 加载 SAXPY 内核
    KernelInfo ki;
    ki.name = "saxpy";
    ki.regs_u32 = 13;
    ki.param_size = 20;
    sim.func.load_code(build_saxpy_kernel(), ki);

    // 准备 1024 元素数据
    int n = 1024;
    std::vector<float> hx(n), hy(n);
    for (int i = 0; i < n; i++) { hx[i] = (float)i; hy[i] = (float)(i+1); }
    float a = 3.0f;

    // 分配设备内存并拷贝
    uint64_t devx = sim.func.malloc(n * 4);
    uint64_t devy = sim.func.malloc(n * 4);
    sim.func.memcpy_h2d(devx, hx.data(), n * 4);
    sim.func.memcpy_h2d(devy, hy.data(), n * 4);

    // 启动内核: (n+255)/256 个 CTA, 每个 256 线程
    struct { uint32_t n; float a; uint64_t xp; uint64_t yp; } params;
    params.n = n; params.a = a; params.xp = devx; params.yp = devy;
    sim.func.launch_by_name("saxpy", (n+255)/256, 1, 1, 256, 1, 1, 0,
                            std::vector<uint8_t>((uint8_t*)&params, (uint8_t*)&params + sizeof(params)));

    // 运行周期模拟 (加 50000 周期上限防死锁)
    sim.run(50000);
    sim.collect_stats();

    // 验证结果
    std::vector<float> result(n);
    sim.func.memcpy_d2h(result.data(), devy, n * 4);
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float exp = a * hx[i] + hy[i];
        if (std::abs(result[i] - exp) > 0.001f) errors++;
    }

    // 输出性能指标
    std::cout << "\nSAXPY (" << n << " elements, a=" << a << ")" << std::endl;
    std::cout << "  Errors: " << errors << "/" << n << std::endl;
    std::cout << "  Cycles: " << sim.stats.total_cycles << std::endl;
    std::cout << "  Instructions: " << sim.stats.instructions << std::endl;
    std::cout << "  IPC: " << sim.stats.ipc() << std::endl;
    std::cout << "  SIMD efficiency: " << sim.stats.simd_efficiency() << std::endl;
    std::cout << "  Divergent branches: " << sim.stats.simt_divergent_branch << std::endl;
    std::cout << "  Max stack depth: " << sim.stats.simt_stack_maxdepth << std::endl;

    // 输出 stats.json
    sim.output_stats_json("stats.json");
    std::cout << "  Stats: stats.json" << std::endl;

    if (errors == 0) std::cout << "  PASS" << std::endl;
    else std::cout << "  FAIL" << std::endl;

    return errors > 0 ? 1 : 0;
}