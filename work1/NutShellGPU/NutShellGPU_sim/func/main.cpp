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
// main.cpp - 功能模拟器主入口 + 测试框架 (05 册 §3)
// =============================================================================
// 本文件是功能模拟器 (S2) 的主入口, 提供两个功能:
//   1. SAXPY 演示: 运行 SAXPY 内核, 验证模拟器基本功能
//   2. 测试套件: 运行解码/SAXPY/分支发散等单元测试
//
// 运行方式:
//   ./NutShellGPU_func           → 运行 SAXPY 演示
//   ./NutShellGPU_func --test    → 运行所有测试
//
// 参考: NutShellGPU_spec 01 册 §9 (SAXPY 示例), 05 册 §3 (测试框架)
// 教材引用: 第 2 章 §2.1 Programming Model (p.10) & 第 5 章 §5.4 Methodology (p.131) — 本文件含 SAXPY 内核编码 (线程索引 i=blockIdx.x*blockDim.x+threadIdx.x) 和断言驱动测试框架, 对应教材编程模型与测试方法学
// =============================================================================
#include "func_sim.hpp"
#include "../isa/ntas_enc.hpp"   // test_vectors(): 解码器测试标准向量
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>

using namespace ntisa;

// =============================================================================
// 测试断言宏
// -----------------------------------------------------------------------------
// 简易测试框架: 用宏实现断言, 统计通过/失败数
// ASSERT(cond, msg): 断言条件为真
// ASSERT_EQ(a, b, msg): 断言 a == b
// 教材引用: 第 5 章 §5.4 Methodology (p.131) — ASSERT/ASSERT_EQ 实现断言驱动测试方法学, 统计通过/失败计数并打印定位信息
// =============================================================================
static int test_pass = 0, test_fail = 0;
#define ASSERT(cond, msg) do { \
    if (cond) { test_pass++; } \
    else { test_fail++; std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) == (b)) { test_pass++; } \
    else { test_fail++; std::cerr << "FAIL: " << msg << " got=" << (a) << " exp=" << (b) << " at " << __FILE__ << ":" << __LINE__ << std::endl; } \
} while(0)

// =============================================================================
// SAXPY 内核 (01 册 §9) ★ 经典 GPU 教学内核
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
// 教材引用: 第 2 章 §2.1 Programming Model (p.10) & 第 2 章 §2.2.2 Instruction Encoding (p.18) — saxpy_code() 编码 SAXPY 内核 (y=a*x+y), 体现 CUDA 线程层次与 64 位指令编码
// =============================================================================
static std::vector<uint64_t> saxpy_code() {
    std::vector<uint64_t> code;
    // The SAXPY example from 01 册 §9
    // params: c[0x0][0x00]=n(u32) [0x04]=a(f32) [0x08]=xp(u64) [0x10]=yp(u64)

    // ---- 加载参数 ----
    // 教材引用: 第 2 章 §2.1.3 Memory Model (p.13) & 第 4 章 §4.1.1 Constant Memory (p.68) — LDC 从常量内存读取内核参数 (n/a/xp/yp), 对应教材主机-设备参数传递与 LDC 指令
    // 00: LDC.Param.U32  R2,  c[0x0][0x00]   // R2 = n (元素个数)
    code.push_back(encode(OP_LDC, 0,0,0,0, FK_MI, 2, 0, 0, (0<<5)|0, 0, 0, 0, 0, 0));
    // 08: LDC.Param.F32  R1,  c[0x0][0x04]   // R1 = a (标量系数)
    code.push_back(encode(OP_LDC, 0,0,0,0, FK_MI, 1, 0, 4, (2<<5)|0, 0, 0, 0, 0, 4));
    // 10: LDC.Param.U64  R8,  c[0x0][0x08]   // R8:R9 = xp (x 数组指针)
    code.push_back(encode(OP_LDC, 0,0,0,0, FK_MI, 8, 0, 8, (1<<5)|0, 0, 0, 0, 0, 8));
    // 18: LDC.Param.U64  R4,  c[0x0][0x10]   // R4:R5 = yp (y 数组指针)
    code.push_back(encode(OP_LDC, 0,0,0,0, FK_MI, 4, 0, 16, (1<<5)|0, 0, 0, 0, 0, 16));

    // ---- 计算全局线程索引 i = blockIdx.x * blockDim.x + threadIdx.x ----
    // 教材引用: 第 2 章 §2.1 Programming Model (p.10) — 通过 S2R 读取 SR_CTAID.X/SR_NTID.X/SR_TID.X 并用 IMAD 计算 i=blockIdx.x*blockDim.x+threadIdx.x, 对应教材 CUDA 线程索引公式
    // 20: S2R  R6, SR_CTAID.X     // R6 = blockIdx.x (CTA 坐标)
    code.push_back(encode(OP_S2R, 0,0,0,0, FK_R, 6, 0, 0, SR_CTAID_X, 0, 0, 0, 0, 0));
    // 28: S2R  R7, SR_NTID.X      // R7 = blockDim.x (每 CTA 线程数)
    code.push_back(encode(OP_S2R, 0,0,0,0, FK_R, 7, 0, 0, SR_NTID_X, 0, 0, 0, 0, 0));
    // 30: S2R  R9, SR_TID.X       // R9 = threadIdx.x (线程在 CTA 内坐标)
    code.push_back(encode(OP_S2R, 0,0,0,0, FK_R, 9, 0, 0, SR_TID_X, 0, 0, 0, 0, 0));
    // 38: MOV.U32 R3, R9          // R3 = threadIdx.x
    code.push_back(encode(OP_MOV, 0,0,0,0, FK_R, 3, 9, 0, 0, 0, 0, 0, 0, 0));
    // 40: IMAD.LO.U32 R3, R6, R7  // R3 = blockIdx.x * blockDim.x + threadIdx.x = i
    code.push_back(encode(OP_IMAD, 0,0,0,0, FK_R, 3, 6, 7, 0, 0, 0, 0, 0, 0));

    // ---- 边界检查: if (i >= n) goto done ----
    // 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) & 第 3 章 §3.1.4 Divergence (p.32) — SETP+SSY+@P0 BRA 是 SIMT 谓词保护分支典型序列: SSY 设重汇聚点, @P0 BRA 仅越界线程跳走 (其余线程继续 fall-through)
    // 48: SETP.GE.S32 P0, R3, R2  // P0 = (i >= n)
    code.push_back(encode(OP_SETP, 0,0,0,0, FK_R, 0, 3, 2, 0x05, 0x83, 0, 0, 0, 0));
    // 50: SSY  Ldone              // 设置重汇聚点 (分支前必须调用)
    code.push_back(encode(OP_SSY, 0,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 12, 0));
    // 58: @P0 BRA Ldone           // 谓词保护分支: 超出边界的线程跳转到 done
    code.push_back(encode(OP_BRA, 1,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 11, 0));

    // ---- 计算字节偏移: offset = i * 4 ----
    // 60: MOV32I R12, 4           // R12 = 4 (float 大小)
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 12, 0, 0, 0, 0, 0, 4, 0, 0));
    // 68: MOV32I R10, 0           // R10 = 0 (用于 WIDE 乘法的高位)
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 10, 0, 0, 0, 0, 0, 0, 0, 0));
    // 70: MOV32I R11, 0           // R11 = 0
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 11, 0, 0, 0, 0, 0, 0, 0, 0));
    // 78: IMAD.WIDE.U32 R10, R3, R12  // R10:R11 = i * 4 (64 位字节偏移)
    code.push_back(encode(OP_IMAD, 0,0,0,0, FK_R, 10, 3, 12, (2<<3)|0, 0, 0, 0, 0, 0));

    // ---- 计算实际地址: addr = base + offset ----
    // 80: IADD.U64 R8, R8, R10    // R8:R9 = xp + offset (x[i] 的地址)
    code.push_back(encode(OP_IADD, 0,0,0,0, FK_R, 8, 8, 10, 6, 0, 0, 0, 0, 0));
    // 88: IADD.U64 R4, R4, R10    // R4:R5 = yp + offset (y[i] 的地址)
    code.push_back(encode(OP_IADD, 0,0,0,0, FK_R, 4, 4, 10, 6, 0, 0, 0, 0, 0));

    // ---- 加载、计算、存储: y[i] = a * x[i] + y[i] ----
    // 教材引用: 第 2 章 §2.2.3 Fused Multiply-Add (p.20) — FFMA.RN.F32 R2 = a*x[i]+y[i] 是单次舍入融合乘加, 第三源 (y[i]) 通过 Rd 字段隐式传递; LDG/STG 完成 global 内存读写
    // 90: LDG.CA.F32 R0, [R8]     // R0 = x[i] (从 global 内存加载)
    code.push_back(encode(OP_LD, 0,0,0,0, FK_MI, 0, 8, 0, 0x38, 0x00, 0, 0, 0, 0));
    // 98: LDG.CA.F32 R2, [R4]     // R2 = y[i]
    code.push_back(encode(OP_LD, 0,0,0,0, FK_MI, 2, 4, 0, 0x38, 0x00, 0, 0, 0, 0));
    // A0: FFMA.RN.F32 R2, R0, R1, R2  // R2 = a * x[i] + y[i] (融合乘加)
    code.push_back(encode(OP_FFMA, 0,0,0,0, FK_R, 2, 0, 1, 0, 0, 0, 0, 0, 0));
    // A8: STG.CG.F32 [R4], R2    // y[i] = R2 (存回 global 内存)
    code.push_back(encode(OP_ST, 0,0,0,0, FK_MI, 2, 4, 0, 0x38, 0x01, 0, 0, 0, 0));
    // B0: EXIT                    // 线程退出
    code.push_back(encode(OP_EXIT, 0,0,0,0, FK_R, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    return code;
}

// =============================================================================
// 测试 1: 解码器测试 (02 册 §9)
// -----------------------------------------------------------------------------
// 使用 ntas_enc.hpp 中定义的标准测试向量验证解码器正确性
// 验证: decode(word) → Inst → encode_inst(Inst) == word (往返一致性)
// 教材引用: 第 5 章 §5.3 Validation (p.129) & 第 2 章 §2.2.2 Instruction Encoding (p.18) — test_decode 验证 decode/encode_inst 往返一致性 (round-trip), 是模拟器验证的基础测试
// =============================================================================
static void test_decode() {
    std::cout << "=== test_decode ===" << std::endl;
    auto vecs = test_vectors();  // 获取标准测试向量
    for (const auto& v : vecs) {
        Inst ins;
        int err = decode(v.word, v.pc, ins);  // 解码
        ASSERT_EQ(err, v.expect_err, std::string("decode error for: ") + v.desc);
        if (err == ERR_OK) {
            // 验证往返一致性: 解码后再编码, 应得到原始指令字
            uint64_t reencoded = encode_inst(ins);
            ASSERT_EQ(reencoded, v.word, std::string("round-trip mismatch for: ") + v.desc);
        }
        std::cout << "  OK: " << v.desc << std::endl;
    }
    std::cout << "test_decode: " << test_pass << " passed, " << test_fail << " failed" << std::endl;
}

// =============================================================================
// 测试 2: SAXPY 端到端测试 (01 册 §9)
// -----------------------------------------------------------------------------
// 完整测试 SAXPY 内核: 加载代码→分配内存→传参→启动→同步→验证结果
// 验证: y[i] == a * x[i] + y[i] (浮点精度 0.001)
// 教材引用: 第 5 章 §5.3 Validation (p.129) & 第 2 章 §2.1.3 Memory Model (p.13) — test_saxpy 是端到端验证: malloc/memcpy_h2d/launch/sync/memcpy_d2h, 覆盖主机 API 全流程并断言结果正确性
// =============================================================================
static void test_saxpy() {
    std::cout << "\n=== test_saxpy ===" << std::endl;
    int saved_pass = test_pass, saved_fail = test_fail;

    FuncSim sim;
    sim.init();  // 初始化模拟器 (创建 NSM)

    // 加载 SAXPY 内核代码
    KernelInfo ki;
    ki.name = "saxpy";
    ki.regs_u32 = 13; // R0-R12
    ki.regs_u64 = 2;
    ki.static_smem = 0;
    ki.param_size = 20;
    ki.bar_slots = 0;
    sim.load_code(saxpy_code(), ki);

    // 准备测试数据 (256 个 float, 1 个 CTA = 1 个 warp)
    int n = 256;
    std::vector<float> hx(n), hy(n);
    for (int i = 0; i < n; i++) {
        hx[i] = (float)i;
        hy[i] = (float)(i * 2);
    }
    float a = 2.0f;

    // 在设备端分配内存 (模拟 cudaMalloc)
    uint64_t devx = sim.malloc(n * 4);
    uint64_t devy = sim.malloc(n * 4);

    // 拷贝数据到设备 (模拟 cudaMemcpy H2D)
    sim.memcpy_h2d(devx, hx.data(), n * 4);
    sim.memcpy_h2d(devy, hy.data(), n * 4);

    // 设置内核参数 (通过常量内存传递)
    struct {
        uint32_t n;
        float a;
        uint64_t xp;
        uint64_t yp;
    } params;
    params.n = n;
    params.a = a;
    params.xp = devx;
    params.yp = devy;

    // 启动内核: grid=(1,1,1), block=(256,1,1) → 256 线程 = 8 个 warp
    sim.launch_by_name("saxpy", (n+255)/256, 1, 1, 256, 1, 1, 0,
                       std::vector<uint8_t>((uint8_t*)&params, (uint8_t*)&params + sizeof(params)));

    // 同步等待内核完成 (模拟 cudaDeviceSynchronize)
    int err = sim.sync();
    ASSERT_EQ(err, ERR_OK, "saxpy sync");

    // 拷贝结果回主机 (模拟 cudaMemcpy D2H)
    std::vector<float> result(n);
    sim.memcpy_d2h(result.data(), devy, n * 4);

    // 验证结果正确性
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = a * hx[i] + hy[i];
        if (std::abs(result[i] - expected) > 0.001f) {
            if (errors < 5) {
                std::cerr << "  MISMATCH at " << i << ": got " << result[i] << " exp " << expected << std::endl;
            }
            errors++;
        }
    }
    ASSERT_EQ(errors, 0, "saxpy result correctness");
    ASSERT_EQ(sim.inst_count > 0, true, "saxpy executed instructions");

    std::cout << "test_saxpy: " << (test_pass - saved_pass) << " passed, "
              << (test_fail - saved_fail) << " failed" << std::endl;
    std::cout << "  Instructions executed: " << sim.inst_count << std::endl;
    std::cout << "  Divergent branches: " << sim.divergent_branches << std::endl;
}

// =============================================================================
// 测试 3: 分支发散测试 (03 册 §4) ★ SIMT 栈验证
// -----------------------------------------------------------------------------
// 测试 SIMT 栈处理分支发散的正确性:
//   - 32 个 lane, 按 lane 奇偶性分支 (偶数走一路, 奇数走另一路)
//   - 验证发散后 SIMT 栈深度增加, 重汇聚后恢复
//   - 通过快照 (snap) 记录 SIMT 栈状态变化
//
// 测试内核:
//   00: S2R R0, SR_LANEID       // R0 = lane id
//   08: LOP.AND R0, R0, 1       // R0 = lane & 1 (奇偶性)
//   10: SETP.NE P0, R0, 0       // P0 = (lane is odd)
//   18: SSY Ljoin               // 设置重汇聚点
//   20: @P0 BRA Lodd            // 奇数 lane 跳到 Lodd
//   28: MOV32I R1, 0xAA         // 偶数 lane: R1 = 0xAA
//   30: Lodd: MOV32I R1, 0xBB   // 奇数 lane: R1 = 0xBB
//   38: Ljoin: EXIT             // 重汇聚点: 所有 lane 在此汇聚退出
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) & 第 5 章 §5.3 Validation (p.129) — test_diverge 验证 SIMT 栈处理分支发散/重汇聚的正确性, 通过栈快照 (snap) 与发散统计数确认 32 lane 按 lane&1 分路后重汇聚
// =============================================================================
static void test_diverge() {
    std::cout << "\n=== test_diverge ===" << std::endl;
    int saved_pass = test_pass, saved_fail = test_fail;

    FuncSim sim;
    sim.init();
    sim.enable_snaps(true);  // 启用 SIMT 栈快照记录

    // 构建测试内核 (手工编码的 NTAS1 指令)
    std::vector<uint64_t> code;
    // 00: S2R R0, SR_LANEID
    code.push_back(encode(OP_S2R, 0,0,0,0, FK_R, 0, 0, 0, SR_LANEID, 0, 0, 0, 0, 0));
    // 08: LOP.AND R0, R0, 1  (X[1:0]=0 AND, X[2]=1 imm, imm in c=1)
    code.push_back(encode(OP_LOP, 0,0,0,0, FK_R, 0, 0, 1, 0x04, 0, 0, 0, 0, 0));
    // 10: SETP.NE.S32 P0, R0, RZ (NE=1, S32=0, Y=0x83; c=31=RZ=0)
    code.push_back(encode(OP_SETP, 0,0,0,0, FK_R, 0, 0, 31, 0x01, 0x83, 0, 0, 0, 0));
    // 18: SSY Ljoin (target=0x38, PC=0x18, imm26=(0x38-0x18)/8=4)
    code.push_back(encode(OP_SSY, 0,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 4, 0));
    // 20: @P0 BRA Lodd (target=0x30, PC=0x20, imm26=(0x30-0x20)/8=2)
    code.push_back(encode(OP_BRA, 1,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 2, 0));
    // 28: MOV32I R1, 0xAA  (even path: 偶数 lane 执行)
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 1, 0, 0, 0, 0, 0, (int32_t)0xAA, 0, 0));
    // 30: Lodd: MOV32I R1, 0xBB  (odd path: 奇数 lane 执行)
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 1, 0, 0, 0, 0, 0, (int32_t)0xBB, 0, 0));
    // 38: Ljoin: EXIT  (重汇聚点: 所有 lane 在此汇聚退出)
    code.push_back(encode(OP_EXIT, 0,0,0,0, FK_R, 0, 0, 0, 0, 0, 0, 0, 0, 0));

    KernelInfo ki;
    ki.name = "diverge";
    ki.regs_u32 = 2;
    sim.load_code(code, ki);

    // 启动内核: 1 个 CTA, 32 个线程 (1 个 warp)
    sim.launch_by_name("diverge", 1, 1, 1, 32, 1, 1, 0, {});

    int err = sim.sync();
    ASSERT_EQ(err, ERR_OK, "diverge sync");

    // 检查 SIMT 栈快照
    ASSERT_EQ(sim.snaps.size() > 0, true, "snapshots recorded");
    std::cout << "  Snapshots: " << sim.snaps.size() << std::endl;
    for (const auto& snap : sim.snaps) {
        std::cout << "    " << snap.event << " @0x" << std::hex << snap.pc << std::dec
                  << " stack_depth=" << snap.stack.size()
                  << " mask=0x" << std::hex << snap.exec_mask << std::dec << std::endl;
    }

    // 验证发散统计
    std::cout << "  Divergent branches: " << sim.divergent_branches << std::endl;
    ASSERT_EQ(sim.divergent_branches, 1, "one divergent branch");
    ASSERT_EQ(sim.max_stack_depth >= 2, true, "stack depth >= 2 after divergence");

    std::cout << "test_diverge: " << (test_pass - saved_pass) << " passed, "
              << (test_fail - saved_fail) << " failed" << std::endl;
}

// =============================================================================
// 主函数: 入口点
// =============================================================================
// 命令行:
//   无参数   → 运行 SAXPY 演示 (1024 元素)
//   --test   → 运行所有测试
// 教材引用: 第 5 章 §5.4 Methodology (p.131) & 第 2 章 §2.1.3 Memory Model (p.13) — main() 是模拟器主入口, 通过断言驱动测试或 SAXPY 演示验证模拟器功能, 覆盖主机 API 全流程
// =============================================================================
int main(int argc, char** argv) {
    std::cout << "NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education\n"
              << "Version 1.10\n"
              << "Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN\n"
              << "Improved by: Tonghui Ming\n"
              << "References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018\n"
              << "September 2026\n";
    if (argc > 1 && std::string(argv[1]) == "--test") {
        // 运行所有测试
        test_decode();
        test_saxpy();
        test_diverge();

        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Total: " << test_pass << " passed, " << test_fail << " failed" << std::endl;
        return test_fail > 0 ? 1 : 0;
    }

    // 默认: 运行 SAXPY 演示
    std::cout << "NutShellGPU Functional Simulator (S2)" << std::endl;
    std::cout << "Usage: " << argv[0] << " [--test]" << std::endl;
    std::cout << "  --test: run all built-in tests" << std::endl;

    FuncSim sim;
    sim.init();

    KernelInfo ki;
    ki.name = "saxpy";
    ki.regs_u32 = 13;
    ki.param_size = 20;
    sim.load_code(saxpy_code(), ki);

    int n = 1024;
    std::vector<float> hx(n), hy(n);
    for (int i = 0; i < n; i++) { hx[i] = (float)i; hy[i] = (float)(i+1); }
    float a = 3.0f;

    uint64_t devx = sim.malloc(n * 4);
    uint64_t devy = sim.malloc(n * 4);
    sim.memcpy_h2d(devx, hx.data(), n * 4);
    sim.memcpy_h2d(devy, hy.data(), n * 4);

    struct { uint32_t n; float a; uint64_t xp; uint64_t yp; } params;
    params.n = n; params.a = a; params.xp = devx; params.yp = devy;

    sim.launch_by_name("saxpy", (n+255)/256, 1, 1, 256, 1, 1, 0,
                       std::vector<uint8_t>((uint8_t*)&params, (uint8_t*)&params + sizeof(params)));

    sim.sync();

    std::vector<float> result(n);
    sim.memcpy_d2h(result.data(), devy, n * 4);

    int errors = 0;
    for (int i = 0; i < n; i++) {
        float exp = a * hx[i] + hy[i];
        if (std::abs(result[i] - exp) > 0.001f) errors++;
    }

    std::cout << "\nSAXPY (" << n << " elements, a=" << a << ")" << std::endl;
    std::cout << "  Errors: " << errors << "/" << n << std::endl;
    std::cout << "  Instructions: " << sim.inst_count << std::endl;
    std::cout << "  Divergent branches: " << sim.divergent_branches << std::endl;
    std::cout << "  Max stack depth: " << sim.max_stack_depth << std::endl;

    if (errors == 0) std::cout << "  PASS" << std::endl;
    else std::cout << "  FAIL" << std::endl;

    return errors > 0 ? 1 : 0;
}
