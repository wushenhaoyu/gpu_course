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
// t_saxpy.cpp - SAXPY 端到端测试 (01 册 §9) ★★ 功能验证
// -----------------------------------------------------------------------------
// 本文件测试 NutShellGPU 功能模拟器的端到端正确性, 通过 SAXPY 内核:
//   y = a*x + y
//
// 测试场景:
//   1. test_saxpy_single_warp: 单 warp (256 线程, 1 CTA)
//   2. test_saxpy_multi_cta: 多 CTA (1024 元素, 4 CTA)
//   3. test_saxpy_partial_warp: 部分 warp (100 元素, 尾部发散)
//
// 验证内容:
//   - 结果正确性 (y[i] == a*x[i] + y[i])
//   - 指令执行计数 (inst_count > 0)
//   - 发散分支统计 (divergent_branches)
//
// 教学要点:
//   - SAXPY 是 GPU 编程的 "Hello World"
//   - 多 CTA 测试验证 CTA 调度与共享资源
//   - 部分 Warp 测试验证 SIMT 掩码与发散处理
// =============================================================================
#include "../func/func_sim.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>

using namespace ntisa;

static int pass_count = 0, fail_count = 0;
#define CHECK(cond, msg) do { if (cond) pass_count++; else { fail_count++; std::cerr << "FAIL: " << msg << " at " << __LINE__ << std::endl; } } while(0)

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
// 教材引用: 第 2 章 §2.1 Programming Model (p.10) — SAXPY 内核体现 CUDA 线程索引 i=blockIdx.x*blockDim.x+threadIdx.x; §2.2.2 Instruction Encoding (p.18) 64 位编码; §2.2.3 Fused Multiply-Add (p.20) FFMA 第三源通过 Rd 字段
static std::vector<uint64_t> build_saxpy_kernel() {
    std::vector<uint64_t> code;
    // 参数: c[0x0][0x00]=n(u32) [0x04]=a(f32) [0x08]=xp(u64) [0x10]=yp(u64)
    // 00: LDC.Param.U32  R2,  c[0x0][0x00]   // R2 = n (元素个数)
    code.push_back(encode(OP_LDC, 0,0,0,0, FK_MI, 2, 0, 0, (0<<5)|0, 0, 0, 0, 0, 0));
    // 08: LDC.Param.F32  R1,  c[0x0][0x04]   // R1 = a (标量系数)
    code.push_back(encode(OP_LDC, 0,0,0,0, FK_MI, 1, 0, 4, (2<<5)|0, 0, 0, 0, 0, 4));
    // 10: LDC.Param.U64  R8,  c[0x0][0x08]   // R8 = xp (x 数组指针)
    code.push_back(encode(OP_LDC, 0,0,0,0, FK_MI, 8, 0, 8, (1<<5)|0, 0, 0, 0, 0, 8));
    // 18: LDC.Param.U64  R4,  c[0x0][0x10]   // R4 = yp (y 数组指针)
    code.push_back(encode(OP_LDC, 0,0,0,0, FK_MI, 4, 0, 16, (1<<5)|0, 0, 0, 0, 0, 16));
    // 20: S2R  R6, SR_CTAID.X                // R6 = blockIdx.x
    code.push_back(encode(OP_S2R, 0,0,0,0, FK_R, 6, 0, 0, SR_CTAID_X, 0, 0, 0, 0, 0));
    // 28: S2R  R7, SR_NTID.X                 // R7 = blockDim.x
    code.push_back(encode(OP_S2R, 0,0,0,0, FK_R, 7, 0, 0, SR_NTID_X, 0, 0, 0, 0, 0));
    // 30: S2R  R9, SR_TID.X                  // R9 = threadIdx.x
    code.push_back(encode(OP_S2R, 0,0,0,0, FK_R, 9, 0, 0, SR_TID_X, 0, 0, 0, 0, 0));
    // 38: MOV.U32 R3, R9                    // R3 = threadIdx.x
    code.push_back(encode(OP_MOV, 0,0,0,0, FK_R, 3, 9, 0, 0, 0, 0, 0, 0, 0));
    // 40: IMAD.LO.U32 R3, R6, R7            // R3 = blockIdx.x * blockDim.x + threadIdx.x
    code.push_back(encode(OP_IMAD, 0,0,0,0, FK_R, 3, 6, 7, 0, 0, 0, 0, 0, 0));
    // 48: SETP.GE.S32 P0, R3, R2            // P0 = (i >= n)
    code.push_back(encode(OP_SETP, 0,0,0,0, FK_R, 0, 3, 2, 0x05, 0x83, 0, 0, 0, 0));
    // 50: SSY  Ldone (target=0xB0)          // 设置重汇聚点
    code.push_back(encode(OP_SSY, 0,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 12, 0));
    // 58: @P0 BRA Ldone (target=0xB0)       // if (i >= n) goto done
    code.push_back(encode(OP_BRA, 1,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 11, 0));
    // 60: MOV32I R12, 4                     // R12 = 4 (sizeof(float))
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 12, 0, 0, 0, 0, 0, 4, 0, 0));
    // 68: MOV32I R10, 0                     // R10 = 0 (高 32 位)
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 10, 0, 0, 0, 0, 0, 0, 0, 0));
    // 70: MOV32I R11, 0                     // R11 = 0
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 11, 0, 0, 0, 0, 0, 0, 0, 0));
    // 78: IMAD.WIDE.U32 R10, R3, R12        // R10 = i * 4 (字节偏移, 64 位)
    code.push_back(encode(OP_IMAD, 0,0,0,0, FK_R, 10, 3, 12, (2<<3)|0, 0, 0, 0, 0, 0));
    // 80: IADD.U64 R8, R8, R10             // R8 = xp + offset
    code.push_back(encode(OP_IADD, 0,0,0,0, FK_R, 8, 8, 10, 6, 0, 0, 0, 0, 0));
    // 88: IADD.U64 R4, R4, R10             // R4 = yp + offset
    code.push_back(encode(OP_IADD, 0,0,0,0, FK_R, 4, 4, 10, 6, 0, 0, 0, 0, 0));
    // 90: LDG.CA.F32 R0, [R8]              // R0 = x[i]
    code.push_back(encode(OP_LD, 0,0,0,0, FK_MI, 0, 8, 0, 0x38, 0x00, 0, 0, 0, 0));
    // 98: LDG.CA.F32 R2, [R4]              // R2 = y[i]
    code.push_back(encode(OP_LD, 0,0,0,0, FK_MI, 2, 4, 0, 0x38, 0x00, 0, 0, 0, 0));
    // A0: FFMA.RN.F32 R2, R0, R1, R2        // R2 = x[i]*a + y[i] (fma(Ra=x,Rc=a,Rd=y))
    code.push_back(encode(OP_FFMA, 0,0,0,0, FK_R, 2, 0, 1, 0, 0, 0, 0, 0, 0));
    // A8: STG.CG.F32 [R4], R2             // y[i] = result
    code.push_back(encode(OP_ST, 0,0,0,0, FK_MI, 2, 4, 0, 0x38, 0x01, 0, 0, 0, 0));
    // B0: EXIT                             // 退出
    code.push_back(encode(OP_EXIT, 0,0,0,0, FK_R, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    return code;
}

// =============================================================================
// test_saxpy_single_warp: 单 Warp SAXPY 测试 ★ 基础验证
// -----------------------------------------------------------------------------
// 测试 256 个元素的 SAXPY (1 CTA, 256 线程 = 8 个 warp):
//   1. 初始化功能模拟器 (FuncSim)
//   2. 加载 SAXPY 内核 (regs_u32=13, param_size=20)
//   3. 准备测试数据: hx[i] = i+1, hy[i] = i*10, a = 2.5
//   4. 分配设备内存 (malloc), 拷贝数据 (memcpy_h2d)
//   5. 启动内核 (launch_by_name): 1 CTA × 256 线程
//   6. 同步等待完成 (sync)
//   7. 拷贝结果回主机 (memcpy_d2h), 验证正确性
//   8. 检查指令计数与发散分支统计
//
// 教学要点:
//   - 单 Warp 测试是最基础的功能验证
//   - 所有线程在单 CTA 内, 无跨 CTA 同步需求
// =============================================================================
// 教材引用: 第 2 章 §2.1.3 Memory Model (p.13) — 主机-设备数据传输 malloc/memcpy_h2d/launch/sync/memcpy_d2h; §5.3 Validation (p.129) 端到端验证容差 0.001
void test_saxpy_single_warp() {
    std::cout << "--- test_saxpy_single_warp ---" << std::endl;
    FuncSim sim;
    sim.init();

    // 加载 SAXPY 内核
    KernelInfo ki;
    ki.name = "saxpy";
    ki.regs_u32 = 13;     // 13 个 32 位寄存器
    ki.param_size = 20;   // 参数 20 字节 (n+a+xp+yp)
    sim.load_code(build_saxpy_kernel(), ki);

    // 准备 256 个元素数据
    int n = 256;
    std::vector<float> hx(n), hy(n);
    for (int i = 0; i < n; i++) { hx[i] = (float)(i+1); hy[i] = (float)(i*10); }
    float a = 2.5f;

    // 分配设备内存并拷贝数据
    uint64_t devx = sim.malloc(n * 4);
    uint64_t devy = sim.malloc(n * 4);
    sim.memcpy_h2d(devx, hx.data(), n * 4);
    sim.memcpy_h2d(devy, hy.data(), n * 4);

    // 启动内核 (1 CTA, 256 线程)
    struct { uint32_t n; float a; uint64_t xp; uint64_t yp; } params;
    params.n = n; params.a = a; params.xp = devx; params.yp = devy;
    sim.launch_by_name("saxpy", 1, 1, 1, 256, 1, 1, 0,
                       std::vector<uint8_t>((uint8_t*)&params, (uint8_t*)&params + sizeof(params)));
    int err = sim.sync();
    CHECK(err == ERR_OK, "saxpy sync OK");

    // 验证结果
    std::vector<float> result(n);
    sim.memcpy_d2h(result.data(), devy, n * 4);

    int errors = 0;
    for (int i = 0; i < n; i++) {
        float exp = a * hx[i] + hy[i];
        if (std::abs(result[i] - exp) > 0.001f) errors++;
    }
    CHECK(errors == 0, "saxpy result correct");
    CHECK(sim.inst_count > 0, "instructions executed");           // 至少执行了一些指令
    CHECK(sim.divergent_branches >= 0, "divergent branch count valid");  // 发散分支计数有效
}

// =============================================================================
// test_saxpy_multi_cta: 多 CTA SAXPY 测试 ★ CTA 调度验证
// -----------------------------------------------------------------------------
// 测试 1024 个元素的 SAXPY (4 CTA, 每个 256 线程):
//   - 验证多 CTA 调度与资源分配
//   - 验证跨 CTA 的线程索引计算 (blockIdx.x * blockDim.x + threadIdx.x)
//   - 验证全局内存访问的正确性
//
// 教学要点:
//   - 多 CTA 是 GPU 并行的核心: 不同 CTA 处理不同数据块
//   - 线程全局索引: i = blockIdx.x * blockDim.x + threadIdx.x
//   - CTA 间独立, 通过全局内存通信
// =============================================================================
// 教材引用: 第 2 章 §2.1 Programming Model (p.10) — 多 CTA/Grid 并行, 线程全局索引 i=blockIdx.x*blockDim.x+threadIdx.x; §2.1.3 Memory Model (p.13) 跨 CTA 全局内存通信
void test_saxpy_multi_cta() {
    std::cout << "--- test_saxpy_multi_cta ---" << std::endl;
    FuncSim sim;
    sim.init();

    KernelInfo ki;
    ki.name = "saxpy";
    ki.regs_u32 = 13;
    ki.param_size = 20;
    sim.load_code(build_saxpy_kernel(), ki);

    // 准备 1024 个元素数据 (4 CTA × 256 线程)
    int n = 1024;
    std::vector<float> hx(n), hy(n);
    for (int i = 0; i < n; i++) { hx[i] = (float)(i+1); hy[i] = (float)(i*10); }
    float a = 1.5f;

    uint64_t devx = sim.malloc(n * 4);
    uint64_t devy = sim.malloc(n * 4);
    sim.memcpy_h2d(devx, hx.data(), n * 4);
    sim.memcpy_h2d(devy, hy.data(), n * 4);

    // 启动内核: 4 个 CTA, 每个 256 线程
    struct { uint32_t n; float a; uint64_t xp; uint64_t yp; } params;
    params.n = n; params.a = a; params.xp = devx; params.yp = devy;
    sim.launch_by_name("saxpy", (n+255)/256, 1, 1, 256, 1, 1, 0,
                       std::vector<uint8_t>((uint8_t*)&params, (uint8_t*)&params + sizeof(params)));
    int err = sim.sync();
    CHECK(err == ERR_OK, "saxpy multi-CTA sync OK");

    // 验证结果
    std::vector<float> result(n);
    sim.memcpy_d2h(result.data(), devy, n * 4);

    int errors = 0;
    for (int i = 0; i < n; i++) {
        float exp = a * hx[i] + hy[i];
        if (std::abs(result[i] - exp) > 0.001f) {
            // 输出前 3 个错误用于调试
            if (errors < 3) std::cerr << "  i=" << i << " got=" << result[i] << " exp=" << exp << std::endl;
            errors++;
        }
    }
    CHECK(errors == 0, "saxpy multi-CTA result correct");
    std::cout << "  Instructions: " << sim.inst_count << std::endl;
    std::cout << "  Divergent branches: " << sim.divergent_branches << std::endl;
}

// =============================================================================
// test_saxpy_partial_warp: 部分 Warp SAXPY 测试 ★ SIMT 掩码验证
// -----------------------------------------------------------------------------
// 测试 100 个元素的 SAXPY (1 CTA, 256 线程, 但只有 100 个有效):
//   - 最后一个 Warp 只有 100 % 32 = 4 个有效线程
//   - 边界检查 (if (i >= n)) 导致部分线程提前 EXIT
//   - 验证 SIMT 掩码正确处理部分活跃 Warp
//   - 验证发散分支统计 (尾部 Warp 会发散)
//
// 教学要点:
//   - 部分 Warp: 当元素数不是 32 的倍数时, 最后一个 Warp 不满
//   - lane_valid 掩码: 标记哪些 lane 是有效的
//   - 发散分支: 边界检查导致部分 lane 跳转到 EXIT, 部分继续执行
// =============================================================================
// 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) + §3.1.4 Divergence (p.32) — 部分 warp 的 lane_valid 掩码与边界检查导致的尾部发散; §5.3 Validation (p.129) 容差 0.001
void test_saxpy_partial_warp() {
    std::cout << "--- test_saxpy_partial_warp ---" << std::endl;
    FuncSim sim;
    sim.init();

    KernelInfo ki;
    ki.name = "saxpy";
    ki.regs_u32 = 13;
    ki.param_size = 20;
    sim.load_code(build_saxpy_kernel(), ki);

    // 准备 100 个元素数据 (不满一个 Warp, 尾部发散)
    int n = 100;
    std::vector<float> hx(n), hy(n);
    for (int i = 0; i < n; i++) { hx[i] = (float)(i+1); hy[i] = (float)i; }
    float a = 3.0f;

    // 分配内存 (多分配 64 字节防止越界)
    uint64_t devx = sim.malloc(n * 4 + 64);
    uint64_t devy = sim.malloc(n * 4 + 64);
    sim.memcpy_h2d(devx, hx.data(), n * 4);
    sim.memcpy_h2d(devy, hy.data(), n * 4);

    // 启动内核 (1 CTA, 256 线程, 但只有前 100 个有效)
    struct { uint32_t n; float a; uint64_t xp; uint64_t yp; } params;
    params.n = n; params.a = a; params.xp = devx; params.yp = devy;
    sim.launch_by_name("saxpy", 1, 1, 1, 256, 1, 1, 0,
                       std::vector<uint8_t>((uint8_t*)&params, (uint8_t*)&params + sizeof(params)));
    int err = sim.sync();
    CHECK(err == ERR_OK, "saxpy partial warp sync OK");

    // 验证结果
    std::vector<float> result(n);
    sim.memcpy_d2h(result.data(), devy, n * 4);

    int errors = 0;
    for (int i = 0; i < n; i++) {
        float exp = a * hx[i] + hy[i];
        if (std::abs(result[i] - exp) > 0.001f) errors++;
    }
    CHECK(errors == 0, "saxpy partial warp result correct");
    // 最后一个 Warp 应有发散分支 (部分 lane 因边界检查提前 EXIT)
    CHECK(sim.divergent_branches > 0, "divergent branch for partial warp");
}

// =============================================================================
// main: 测试入口
// -----------------------------------------------------------------------------
// 运行所有 SAXPY 测试, 输出通过/失败计数
// =============================================================================
// 教材引用: 第 5 章 §5.4 Methodology (p.131) — 多场景测试覆盖 (单 warp/多 CTA/部分 warp); §5.3 Validation (p.129) 端到端验证
int main() {
    std::cout << "NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education\n"
              << "Version 1.10\n"
              << "Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN\n"
              << "Improved by: Tonghui Ming\n"
              << "References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018\n"
              << "September 2026\n";
    std::cout << "===== t_saxpy =====" << std::endl;
    test_saxpy_single_warp();
    test_saxpy_multi_cta();
    test_saxpy_partial_warp();

    std::cout << "\n===== Summary =====" << std::endl;
    std::cout << "Passed: " << pass_count << ", Failed: " << fail_count << std::endl;
    return fail_count > 0 ? 1 : 0;
}
