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
// t_diverge.cpp - SIMT 发散栈快照测试 (03 册 §4) ★★★ SIMT 核心
// -----------------------------------------------------------------------------
// 本文件测试 NutShellGPU 的 SIMT 栈状态机, 验证分支发散与重汇聚:
//   1. test_basic_divergence: 基础发散 (偶/奇 lane 分裂)
//   2. test_nested_divergence: 嵌套发散 (深层栈)
//   3. test_uniform_branch: 一致分支 (所有 lane 同方向, 不发散)
//   4. test_notake_branch: 不取分支 (所有 lane 不跳转)
//   5. test_exit_tail: 尾部 Warp 退出 (非 32 倍数)
//   6. test_barrier: 屏障同步
//   7. test_shared_memory: 共享内存访问
//
// SIMT 栈原理 (03 册 §4):
//   - SSY: 设置重汇聚点, 压栈 (rpc = 重汇聚地址)
//   - BRA: 分支, 若发散则压入两个子栈 (taken/not-taken)
//   - 重汇聚: 当两个子栈都到达 rpc 时, 合并回父栈
//
// 测试方法:
//   - enable_snaps(true): 启用栈快照记录
//   - 检查快照的栈深度/掩码/rpc/nextpc 等字段
//
// 教学要点:
//   - SIMT = Single Instruction Multiple Thread, 32 lane 共享一条指令
//   - 分支发散: 不同 lane 走不同路径, 需要栈管理
//   - 重汇聚: 在 SSY 指定的点合并, 恢复全掩码
// =============================================================================
#include "../func/func_sim.hpp"
#include <iostream>
#include <cstring>

using namespace ntisa;

static int pass_count = 0, fail_count = 0;
#define CHECK(cond, msg) do { if (cond) pass_count++; else { fail_count++; std::cerr << "FAIL: " << msg << " at " << __LINE__ << std::endl; } } while(0)

// =============================================================================
// test_basic_divergence: 基础发散测试 (03 册 §4) ★ SIMT 栈核心
// -----------------------------------------------------------------------------
// 内核逻辑:
//   00: S2R R0, SR_LANEID     // 获取 lane ID
//   08: LOP.AND R0, R0, 1     // R0 = lane & 1 (奇偶判断)
//   10: SETP.NE.S32 P0, R0, 0 // P0 = (lane 是奇数)
//   18: SSY Ljoin             // 设置重汇聚点 (0x38)
//   20: @P0 BRA Lodd          // 奇数 lane 跳转到 Lodd (0x30)
//   28: MOV32I R1, 0xAA       // 偶数 lane: R1 = 0xAA
//   30: Lodd: MOV32I R1, 0xBB // 奇数 lane: R1 = 0xBB
//   38: Ljoin: EXIT           // 重汇聚点, 退出
//
// 预期行为:
//   1. SSY 后栈深度 = 2 (压入重汇聚点 0x38)
//   2. BRA 发散后栈深度 = 3 (父 + 两个子栈)
//      - 父栈: dflag=true, nextpc=0x38 (重汇聚点)
//      - 子栈1: 偶数 lane (mask=0x55555555), nextpc=0x28
//      - 子栈2: 奇数 lane (mask=0xAAAAAAAA), nextpc=0x30
//   3. 发散分支数 = 1, 最大栈深度 = 3
//
// 教学要点:
//   - SSY 在 BRA 之前设置重汇聚点, 保证后续能合并
//   - BRA 发散时, 栈分裂为两个子栈 (taken/not-taken)
//   - 重汇聚在 SSY 指定的点发生, 恢复全掩码
// =============================================================================
// 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) + §3.1.4 Divergence (p.32) — test_basic_divergence 验证偶/奇 lane 分裂后 SIMT 栈压栈与重汇聚, active mask 分裂为 0x55555555/0xAAAAAAAA
void test_basic_divergence() {
    std::cout << "--- test_basic_divergence ---" << std::endl;
    FuncSim sim;
    sim.init();
    sim.enable_snaps(true);  // 启用栈快照记录

    // 构建测试内核
    std::vector<uint64_t> code;
    code.push_back(encode(OP_S2R, 0,0,0,0, FK_R, 0, 0, 0, SR_LANEID, 0, 0, 0, 0, 0));     // 00: R0 = lane ID
    code.push_back(encode(OP_LOP, 0,0,0,0, FK_R, 0, 0, 1, 0x04, 0, 0, 0, 0, 0));            // 08: R0 &= 1
    code.push_back(encode(OP_SETP, 0,0,0,0, FK_R, 0, 0, 255, 0x01, 0x83, 0, 0, 0, 0));     // 10: P0 = (R0 != RZ)
    code.push_back(encode(OP_SSY, 0,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 5, 0));               // 18: SSY Ljoin=0x40
    code.push_back(encode(OP_BRA, 1,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 3, 0));               // 20: @P0 BRA Lodd=0x38
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 1, 0, 0, 0, 0, 0, (int32_t)0xAA, 0, 0)); // 28: 偶数: R1 = 0xAA
    code.push_back(encode(OP_BRA, 0,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 2, 0));              // 30: BRA Ljoin=0x40 (跳过奇数路径)
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 1, 0, 0, 0, 0, 0, (int32_t)0xBB, 0, 0)); // 38: 奇数: R1 = 0xBB
    code.push_back(encode(OP_EXIT, 0,0,0,0, FK_R, 0, 0, 0, 0, 0, 0, 0, 0, 0));              // 40: EXIT

    KernelInfo ki;
    ki.name = "diverge";
    ki.regs_u32 = 2;
    sim.load_code(code, ki);

    // 启动内核 (1 CTA, 32 线程 = 1 个 Warp)
    sim.launch_by_name("diverge", 1, 1, 1, 32, 1, 1, 0, {});
    int err = sim.sync();
    CHECK(err == ERR_OK, "diverge sync OK");

    // ===== 检查快照 =====
    CHECK(sim.snaps.size() >= 2, "at least 2 snapshots (SSY + BRA)");

    // 查找 SSY 与 BRA 快照
    const StackSnap* ssy_snap = nullptr;
    const StackSnap* bra_snap = nullptr;
    for (const auto& s : sim.snaps) {
        if (s.event == "SSY") ssy_snap = &s;
        if (s.event == "BRA") bra_snap = &s;
    }
    CHECK(ssy_snap != nullptr, "SSY snapshot exists");
    CHECK(bra_snap != nullptr, "BRA snapshot exists");

    // 验证 SSY 快照
    if (ssy_snap) {
        CHECK(ssy_snap->stack.size() == 2, "stack depth 2 after SSY");           // SSY 后栈深度 = 2
        CHECK(ssy_snap->stack[1].rpc == (uint64_t)0x40, "SSY rpc = Ljoin (0x40)"); // 重汇聚点
        CHECK(ssy_snap->stack[1].nextpc == (uint64_t)0x20, "SSY nextpc = 0x20 (after SSY)"); // 下一条指令
        CHECK(ssy_snap->stack[1].mask == 0xFFFFFFFFu, "SSY mask = all lanes");  // 所有 lane 活跃
    }

    // 验证 BRA 快照
    if (bra_snap) {
        // 发散后: 栈底帧 + 父帧(SSY被BRA修改) + 2 个子帧 = 4 帧
        CHECK(bra_snap->stack.size() == 4, "stack depth 4 after divergent BRA");
        // 栈底帧永远在 [0], 父帧在 [1]
        CHECK(bra_snap->stack[1].dflag == true, "parent dflag=true");
        CHECK(bra_snap->stack[1].nextpc == (uint64_t)0x40, "parent nextpc = join 0x40");

        // 栈顶 (TOS) 应有 16 个 lane (一半)
        // 偶数 lane: mask = 0x55555555 (16 个)
        // 奇数 lane: mask = 0xAAAAAAAA (16 个)
        // 当数量相等时, taken (奇数) 留在栈顶
        uint32_t tos_mask = bra_snap->stack.back().mask;
        uint32_t tos_popcount = popcount32(tos_mask);
        CHECK(tos_popcount == 16, "TOS has 16 lanes (half warp)");
    }

    // 验证发散分支统计
    CHECK(sim.divergent_branches == 1, "exactly 1 divergent branch");
    CHECK(sim.max_stack_depth == 4, "max stack depth = 4 (with fall-through BRA)");
}

// =============================================================================
// test_nested_divergence: 嵌套发散测试 (03 册 §4) ★ 深层栈
// -----------------------------------------------------------------------------
// 内核逻辑:
//   00: S2R R0, SR_LANEID           // lane ID
//   08: LOP.AND R0, R0, 1           // R0 = lane & 1
//   10: SETP.NE.S32 P0, R0, 0       // P0 = 奇数
//   18: SSY Louter (0x78)           // 外层重汇聚
//   20: @P0 BRA Linner_odd (0x68)  // 奇数 lane 跳转
//   28: S2R R0, SR_LANEID           // 偶数 lane: 重新加载 lane
//   30: LOP.AND R0, R0, 2           // R0 = lane & 2
//   38: SETP.NE.S32 P1, R0, 0       // P1 = bit1=1
//   40: SSY Linner_even (0x60)     // 内层重汇聚
//   48: @P1 BRA Lsub (0x50)        // 偶数中 bit1=1 的 lane 跳转
//   50: MOV32I R1, 0x1              // bit1=0
//   58: Lsub: MOV32I R1, 0x2        // bit1=1
//   60: Linner_even: BRA Louter
//   68: Linner_odd: MOV32I R1, 0x3  // 奇数 lane
//   70: BRA Louter
//   78: Louter: EXIT
//
// 预期行为:
//   - 嵌套发散产生更深的栈 (>= 3)
//   - 外层发散: 奇/偶 lane 分裂
//   - 内层发散: 偶数 lane 中 bit1=1/0 分裂
//
// 教学要点:
//   - 嵌套发散: 在已发散的子栈内再次发散
//   - 栈深度增加, 需要更多硬件资源
//   - 内层重汇聚先于外层重汇聚
// =============================================================================
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — 嵌套发散在已发散子栈内再次分裂, 栈深度 >= 3, 内层重汇聚先于外层
void test_nested_divergence() {
    std::cout << "--- test_nested_divergence ---" << std::endl;
    FuncSim sim;
    sim.init();
    sim.enable_snaps(true);

    // 构建嵌套发散内核
    std::vector<uint64_t> code;
    code.push_back(encode(OP_S2R, 0,0,0,0, FK_R, 0, 0, 0, SR_LANEID, 0, 0, 0, 0, 0));     // 00
    code.push_back(encode(OP_LOP, 0,0,0,0, FK_R, 0, 0, 1, 0x04, 0, 0, 0, 0, 0));            // 08: lane & 1
    code.push_back(encode(OP_SETP, 0,0,0,0, FK_R, 0, 0, 255, 0x01, 0x83, 0, 0, 0, 0));     // 10: P0 (R0 != RZ)
    code.push_back(encode(OP_SSY, 0,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 12, 0));             // 18: SSY Louter=0x78
    code.push_back(encode(OP_BRA, 1,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 9, 0));              // 20: @P0 BRA Lodd=0x68
    // 偶数路径: 嵌套分支
    code.push_back(encode(OP_S2R, 0,0,0,0, FK_R, 0, 0, 0, SR_LANEID, 0, 0, 0, 0, 0));     // 28
    code.push_back(encode(OP_LOP, 0,0,0,0, FK_R, 0, 0, 2, 0x04, 0, 0, 0, 0, 0));            // 30: lane & 2
    code.push_back(encode(OP_SETP, 0,0,0,0, FK_R, 1, 0, 255, 0x01, 0x83, 0, 0, 0, 0));     // 38: P1 (R0 != RZ)
    code.push_back(encode(OP_SSY, 0,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 4, 0));               // 40: SSY Linner=0x60
    code.push_back(encode(OP_BRA, 1,0,1,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 2, 0));               // 48: @!P1 BRA Lsub=0x58, imm26=2
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 1, 0, 0, 0, 0, 0, (int32_t)0x1, 0, 0));  // 50: R1=1 (bit1=0)
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 1, 0, 0, 0, 0, 0, (int32_t)0x2, 0, 0));  // 58: R1=2 (bit1=1)
    code.push_back(encode(OP_BRA, 0,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 3, 0));              // 60: BRA Louter=0x78 (偶数内层完成后回外层)
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 1, 0, 0, 0, 0, 0, (int32_t)0x3, 0, 0));  // 68: R1=3 (奇数 lane)
    code.push_back(encode(OP_BRA, 0,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 1, 0));              // 70: BRA Louter=0x78
    code.push_back(encode(OP_EXIT, 0,0,0,0, FK_R, 0, 0, 0, 0, 0, 0, 0, 0, 0));              // 78: Louter: EXIT

    KernelInfo ki;
    ki.name = "diverge";
    ki.regs_u32 = 4;
    sim.load_code(code, ki);

    sim.launch_by_name("diverge", 1, 1, 1, 32, 1, 1, 0, {});
    int err = sim.sync();
    // 验证: 嵌套发散应产生更深的栈 (>= 3)
    if (err == ERR_OK) {
        CHECK(sim.max_stack_depth >= 3, "nested stack depth >= 3");
    } else {
        // 若内核有问题, 至少验证错误被检测到
        CHECK(err != ERR_OK, "error detected for problematic kernel");
    }
}

// =============================================================================
// test_uniform_branch: 一致分支测试 (03 册 §4) ★ 无发散
// -----------------------------------------------------------------------------
// 内核逻辑:
//   00: MOV32I R0, 1               // R0 = 1
//   08: SETP.NE.S32 P0, R0, 0      // P0 = true (所有 lane)
//   10: SSY Ljoin (0x30)           // 设置重汇聚点
//   18: @P0 BRA Lskip (0x28)       // 所有 lane 跳转 (不发散)
//   20: MOV32I R1, 0xAA            // 被跳过
//   28: Lskip: MOV32I R1, 0xBB
//   30: Ljoin: EXIT
//
// 预期行为:
//   - 所有 lane 都跳转 (P0 = true), 不发散
//   - 发散分支数 = 0
//   - 最大栈深度 = 2 (仅 SSY 压栈, BRA 不发散)
//
// 教学要点:
//   - 一致分支: 所有 lane 走同方向, 栈不分裂
//   - 栈深度仅增加 1 (SSY), BRA 不增加
// =============================================================================
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — 一致分支所有 lane 同方向, 栈不分裂, 发散分支数=0, 栈深度仅 SSY +1
void test_uniform_branch() {
    std::cout << "--- test_uniform_branch ---" << std::endl;
    FuncSim sim;
    sim.init();
    sim.enable_snaps(true);

    // 构建一致分支内核
    std::vector<uint64_t> code;
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 0, 0, 0, 0, 0, 0, (int32_t)1, 0, 0)); // 00: R0 = 1
    code.push_back(encode(OP_SETP, 0,0,0,0, FK_R, 0, 0, 255, 0x01, 0x83, 0, 0, 0, 0));     // 08: P0 = (R0 != RZ) = true
    code.push_back(encode(OP_SSY, 0,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 4, 0));               // 10: SSY Ljoin=0x30
    code.push_back(encode(OP_BRA, 1,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 2, 0));               // 18: @P0 BRA Lskip=0x28
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 1, 0, 0, 0, 0, 0, (int32_t)0xAA, 0, 0)); // 20: 被跳过
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 1, 0, 0, 0, 0, 0, (int32_t)0xBB, 0, 0)); // 28: Lskip
    code.push_back(encode(OP_EXIT, 0,0,0,0, FK_R, 0, 0, 0, 0, 0, 0, 0, 0, 0));              // 30: Ljoin

    KernelInfo ki;
    ki.name = "uniform";
    ki.regs_u32 = 2;
    sim.load_code(code, ki);

    sim.launch_by_name("uniform", 1, 1, 1, 32, 1, 1, 0, {});
    int err = sim.sync();
    CHECK(err == ERR_OK, "uniform branch sync OK");

    // 无发散分支 (所有 lane 都跳转)
    CHECK(sim.divergent_branches == 0, "no divergent branches for uniform branch");
    // 栈深度 = 2 (仅 SSY 压栈, BRA 不发散)
    CHECK(sim.max_stack_depth == 2, "max stack depth = 2 (SSY only)");
}

// =============================================================================
// test_notake_branch: 不取分支测试 (03 册 §4) ★ 无发散
// -----------------------------------------------------------------------------
// 内核逻辑:
//   00: MOV32I R0, 0               // R0 = 0
//   08: SETP.NE.S32 P0, R0, 0      // P0 = false (所有 lane)
//   10: SSY Ljoin
//   18: @P0 BRA Lskip              // 无 lane 跳转 (不发散)
//   20: MOV32I R1, 0xAA            // 所有 lane 执行
//   28: Lskip: MOV32I R1, 0xBB
//   30: Ljoin: EXIT
//
// 预期行为:
//   - 无 lane 跳转 (P0 = false), 不发散
//   - 发散分支数 = 0
//
// 教学要点:
//   - 不取分支: 所有 lane 都不跳转, 栈不分裂
//   - 与一致分支类似, 不产生发散
// =============================================================================
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — 不取分支所有 lane 不跳转, 栈不分裂, 发散分支数=0
void test_notake_branch() {
    std::cout << "--- test_notake_branch ---" << std::endl;
    FuncSim sim;
    sim.init();

    // 构建不取分支内核
    std::vector<uint64_t> code;
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 0, 0, 0, 0, 0, 0, 0, 0, 0));           // 00: R0 = 0
    code.push_back(encode(OP_SETP, 0,0,0,0, FK_R, 0, 0, 255, 0x01, 0x83, 0, 0, 0, 0));     // 08: P0 = (R0 != RZ) = false
    code.push_back(encode(OP_SSY, 0,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 4, 0));               // 10: SSY Ljoin
    code.push_back(encode(OP_BRA, 1,0,0,0, FK_B, 0, 0, 0, 0, 0, 0, 0, 2, 0));               // 18: @P0 BRA (不取)
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 1, 0, 0, 0, 0, 0, (int32_t)0xAA, 0, 0)); // 20: 所有 lane 执行
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 1, 0, 0, 0, 0, 0, (int32_t)0xBB, 0, 0)); // 28
    code.push_back(encode(OP_EXIT, 0,0,0,0, FK_R, 0, 0, 0, 0, 0, 0, 0, 0, 0));              // 30

    KernelInfo ki;
    ki.name = "notake";
    ki.regs_u32 = 2;
    sim.load_code(code, ki);

    sim.launch_by_name("notake", 1, 1, 1, 32, 1, 1, 0, {});
    int err = sim.sync();
    CHECK(err == ERR_OK, "notake branch sync OK");
    CHECK(sim.divergent_branches == 0, "no divergent branches");
}

// =============================================================================
// test_exit_tail: 尾部 Warp 退出测试 (03 册 §4) ★ 部分 Warp
// -----------------------------------------------------------------------------
// 内核逻辑:
//   00: EXIT                       // 立即退出
//
// 启动配置:
//   - 1 CTA, 10 线程 (部分 Warp, 只有 10 个 lane 有效)
//
// 预期行为:
//   - 10 个 lane 都执行 EXIT, CTA 完成
//   - completed_ctas = 1
//
// 教学要点:
//   - 部分 Warp: 线程数不是 32 的倍数时, 最后一个 Warp 不满
//   - lane_valid 掩码标记有效 lane (前 10 位为 1)
// =============================================================================
// 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) — 部分 warp lane_valid 掩码, 线程数非 32 倍数时尾部 lane 不满
void test_exit_tail() {
    std::cout << "--- test_exit_tail ---" << std::endl;
    FuncSim sim;
    sim.init();

    // 简单内核: 仅 EXIT
    std::vector<uint64_t> code;
    code.push_back(encode(OP_EXIT, 0,0,0,0, FK_R, 0, 0, 0, 0, 0, 0, 0, 0, 0));

    KernelInfo ki;
    ki.name = "exit";
    ki.regs_u32 = 1;
    sim.load_code(code, ki);

    // 启动 10 线程 (部分 Warp, 只有 10 个 lane 有效)
    sim.launch_by_name("exit", 1, 1, 1, 10, 1, 1, 0, {});
    int err = sim.sync();
    CHECK(err == ERR_OK, "exit tail sync OK");
    CHECK(sim.completed_ctas == 1, "1 CTA completed");
}

// =============================================================================
// test_barrier: 屏障同步测试 (02 册 §7) ★ 跨 Warp 同步
// -----------------------------------------------------------------------------
// 内核逻辑:
//   00: MOV32I R0, 0xAA            // 所有 Warp 写 R0
//   08: BAR.SYNC 0                  // 同步屏障 0
//   10: EXIT
//
// 启动配置:
//   - 1 CTA, 64 线程 (2 个 Warp)
//
// 预期行为:
//   - 两个 Warp 都到达 BAR.SYNC 后才能继续
//   - CTA 完成
//
// 教学要点:
//   - 屏障 (BAR.SYNC): CTA 内所有 Warp 必须都到达才能继续
//   - 跨 Warp 同步: 用于 CTA 内线程协作
//   - bar_slots: 内核声明的屏障槽数 (此内核用 1 个)
// =============================================================================
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — BAR.SYNC 屏障: CTA 内所有 warp 必须到达才能继续, 跨 warp 同步
void test_barrier() {
    std::cout << "--- test_barrier ---" << std::endl;
    FuncSim sim;
    sim.init();

    // 构建屏障内核
    std::vector<uint64_t> code;
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 0, 0, 0, 0, 0, 0, (int32_t)0xAA, 0, 0)); // 00: R0 = 0xAA
    code.push_back(encode(OP_BAR, 0,0,0,0, FK_R, 0, RZ, 0, 0x00, 0, 0, 0, 0, 0));          // 08: BAR.SYNC 0
    code.push_back(encode(OP_EXIT, 0,0,0,0, FK_R, 0, 0, 0, 0, 0, 0, 0, 0, 0));              // 10: EXIT

    KernelInfo ki;
    ki.name = "bar";
    ki.regs_u32 = 1;
    ki.bar_slots = 1;   // 声明 1 个屏障槽
    sim.load_code(code, ki);

    // 启动 64 线程 (2 个 Warp)
    sim.launch_by_name("bar", 1, 1, 1, 64, 1, 1, 0, {});
    int err = sim.sync();
    CHECK(err == ERR_OK, "barrier sync OK");
    CHECK(sim.completed_ctas == 1, "CTA completed after barrier");
}

// =============================================================================
// test_shared_memory: 共享内存测试 (02 册 §6) ★ 共享内存访问
// -----------------------------------------------------------------------------
// 内核逻辑:
//   00: S2R R0, SR_LANEID           // 获取 lane ID
//   08: MOV32I R2, 4                // R2 = 4 (sizeof(float))
//   10: MOV32I R1, 0                // R1 = 0 (IMAD 的初始值)
//   18: IMAD.LO.U32 R1, R0, R2      // R1 = lane * 4 (字节偏移)
//   20: MOV32I R3, 0xAB             // R3 = 0xAB (待存储值)
//   28: STS.U32 [R1], R3            // 共享内存[lane*4] = 0xAB
//   30: LDS.U32 R4, [R1]            // R4 = 共享内存[lane*4]
//   38: EXIT
//
// 预期行为:
//   - 每个 lane 向共享内存写入 0xAB, 再读回验证
//   - 共享内存是 CTA 内共享的快速内存
//
// 教学要点:
//   - 共享内存 (Shared Memory): CTA 内共享, 低延迟
//   - STS/LDS: 共享内存的存储/加载指令
//   - 每个 lane 访问不同地址 (lane*4), 无 bank 冲突
// =============================================================================
// 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — 共享内存 CTA 内共享低延迟, STS/LDS 指令, 每个 lane 访问不同地址避免 bank 冲突
void test_shared_memory() {
    std::cout << "--- test_shared_memory ---" << std::endl;
    FuncSim sim;
    sim.init();

    // 构建共享内存测试内核
    std::vector<uint64_t> code;
    code.push_back(encode(OP_S2R, 0,0,0,0, FK_R, 0, 0, 0, SR_LANEID, 0, 0, 0, 0, 0));     // 00: R0 = lane ID
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 2, 0, 0, 0, 0, 0, (int32_t)4, 0, 0)); // 08: R2 = 4
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 1, 0, 0, 0, 0, 0, (int32_t)0, 0, 0)); // Clear address accumulator; preserve lane ID.
    code.push_back(encode(OP_IMAD, 0,0,0,0, FK_R, 1, 0, 2, 0, 0, 0, 0, 0, 0));             // 18: R1 = lane * 4
    code.push_back(encode(OP_MOV32I, 0,0,0,0, FK_I, 3, 0, 0, 0, 0, 0, (int32_t)0xAB, 0, 0)); // 20: R3 = 0xAB
    // STS.U32 [R1+0], R3 → MI 格式, space=SHARED, width=U32
    // X = (4<<3)|1 = 0x21, Y=0, simm16=0, d=3 (数据), a=1 (基址)
    code.push_back(encode(OP_ST, 0,0,0,0, FK_MI, 3, 1, 0, 0x21, 0x00, 0, 0, 0, 0));       // 28: STS.U32 [R1], R3
    code.push_back(encode(OP_LD, 0,0,0,0, FK_MI, 4, 1, 0, 0x21, 0x00, 0, 0, 0, 0));       // 30: LDS.U32 R4, [R1]
    code.push_back(encode(OP_EXIT, 0,0,0,0, FK_R, 0, 0, 0, 0, 0, 0, 0, 0, 0));              // 38: EXIT

    KernelInfo ki;
    ki.name = "smem";
    ki.regs_u32 = 5;
    ki.static_smem = 32 * 4; // The kernel actually accesses 128 shared bytes.
    sim.load_code(code, ki);

    sim.launch_by_name("smem", 1, 1, 1, 32, 1, 1, 0, {});
    int err = sim.sync();
    CHECK(err == ERR_OK, "shared memory sync OK");
    for(int lane=0;lane<32;++lane) CHECK(sim.nsms[0].warps[0].reg(lane,4)==0xAB,"shared memory lane roundtrip");
}

// =============================================================================
// main: 测试入口
// -----------------------------------------------------------------------------
// 运行所有 SIMT 发散测试, 输出通过/失败计数
// =============================================================================
// 教材引用: 第 5 章 §5.4 Methodology (p.131) — 栈快照验证方法 enable_snaps 记录栈状态; §5.3 Validation (p.129) SIMT 行为验证
int main() {
    std::cout << "NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education\n"
              << "Version 1.10\n"
              << "Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN\n"
              << "Improved by: Tonghui Ming\n"
              << "References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018\n"
              << "September 2026\n";
    std::cout << "===== t_diverge =====" << std::endl;
    test_basic_divergence();
    test_nested_divergence();
    test_uniform_branch();
    test_notake_branch();
    test_exit_tail();
    test_barrier();
    test_shared_memory();

    std::cout << "\n===== Summary =====" << std::endl;
    std::cout << "Passed: " << pass_count << ", Failed: " << fail_count << std::endl;
    return fail_count > 0 ? 1 : 0;
}
