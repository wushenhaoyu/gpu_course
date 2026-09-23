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
// Coon 式顺序记分牌模块（Scoreboard）
// scoreboard.sv
// -----------------------------------------------------------------------------
// 作用：每个 warp 持有一个本模块实例，用于追踪该 warp 已发射但尚未写回的
//       目的寄存器（最多 4 项），从而检测 RAW/WAW/WAR 依赖。指令在 D 级译码
//       时生成 depvec（依赖位向量），只有 depvec==0 的指令才能在 I 级发射。
//       写回（W 级）时释放对应项并清零相关 depvec 位。
//
// 对应规格册：依据 03 册《NSM SIMT 核心微结构》§6 Coon 式顺序记分牌。
//   - §6.1 依赖位向量生成（D 拍，指令入 ibuf 时）
//   - §6.2 分配（I 拍发射时）
//   - §6.3 释放（W 拍写回时）
//
// 端口概要：
//   - chk_valid/s0..s3：依赖检查输入（4 个源操作数 + 谓词守卫）
//   - can_issue：depvec==0 输出（指令可发射）
//   - alloc/dst_*：发射时为目的寄存器分配项
//   - alloc_ok/full：是否有空项
//   - rel_en/rel_*：写回时释放项
//
// 教学注记 - 为什么记分牌是 4 项/warp：
//   Coon 式记分牌的核心思想是"顺序追踪未完成目的寄存器"——它不记录所有
//   寄存器依赖图（那样表太大），而是只记录当前已发射但还没写回的"目的"
//   寄存器。每项代表一条在飞的指令。
//   选 4 项/warp 的依据是寄存器依赖窗口大小：
//     (1) SP 流水线 4 拍延迟，意味着同 warp 连续发射 4 条 ALU 指令时，
//         第 5 条若依赖第 1 条的结果就必须等待——4 项恰好覆盖一个 SP 延迟
//         窗口，使常规 RAW 依赖能被检测到；
//     (2) LSU 的访存指令延迟数十至数百拍，但同 warp 一次只能有 1 条
//         访存在飞（受 ibuf 2 槽 + 顺序发射约束），4 项已足够；
//     (3) 表项数与硬件代价线性相关，4 项是面积与性能的折中。
//   由于同 warp 指令按程序序经过 I→R→E→W（collector 强制同 warp 按序离开），
//   该简单结构同时防 RAW/WAW/WAR（03 §6.3 末）。
// =============================================================================

`include "NutShellGPU_defines.svh"

module scoreboard #(
    parameter NUM_ENTRIES = 4   // 每 warp 记分牌项数（03 §6：默认 4）
) (
    input  logic clk,           // GPU 时钟（上升沿采样）
    input  logic rst_n,         // 异步低有效复位

    // --- 依赖检查读口（D 级/I 级查 depvec == 0）---
    // 输入：待检查指令的源/目的操作数（最多 4 个源）
    // chk_valid=1 时执行依赖检查；can_issue=1 表示无依赖可发射
    input  logic        chk_valid,
    input  logic [7:0]  s0_reg,    input logic s0_is_pred, input logic s0_valid,
    input  logic [7:0]  s1_reg,    input logic s1_is_pred, input logic s1_valid,
    input  logic [7:0]  s2_reg,    input logic s2_is_pred, input logic s2_valid,
    input  logic [7:0]  s3_reg,    input logic s3_is_pred, input logic s3_valid,
    output logic        can_issue,  // depvec == 0，指令可发射

    // --- 分配口（I 级发射时为目的寄存器占用一项）---
    input  logic        alloc,       // 发射脉冲（I 级）
    input  logic        dst_is_pred,  // 目的是谓词还是通用寄存器
    input  logic [7:0]  dst_reg,      // 目的寄存器号
    output logic        alloc_ok,     // 有空闲项可分配

    // --- 释放口（W 级写回时清对应项）---
    input  logic        rel_en,       // 写回脉冲（W 级）
    input  logic        rel_is_pred,
    input  logic [7:0]  rel_reg,

    // --- 调试/状态 ---
    output logic        full          // 所有项都占用（不可再发射）
);

    import NutShellGPU_pkg::*;

    // 记分牌项数组：sb_entry_t = {valid, is_pred, reg_num}
    sb_entry_t entries [NUM_ENTRIES];

    // -------------------------------------------------------------------------
    // 依赖检查函数（依据 03 册 §6.1）
    // -------------------------------------------------------------------------
    // depvec[j]=1 iff entries[j].valid 且其 (class, 号) 在操作数集合中。
    // 即：若指令的某个源寄存器与某个未完成目的寄存器匹配，则该 j 位置 1。
    // 教学注记：通过把"目的-源"匹配转成位向量，可以用一个 4-bit OR 判断
    //   整条指令是否可发射（depvec==0）。这是 Coon 式记分牌简洁的关键。
    function automatic logic match_operand(input sb_entry_t e,
        input logic [7:0] reg_num, input logic is_pred, input logic valid);
        logic m;
        m = 1'b0;
        if (valid && e.valid && e.is_pred == is_pred && e.reg_num == reg_num)
            m = 1'b1;
        return m;
    endfunction

    // -------------------------------------------------------------------------
    // 依赖位向量计算（组合逻辑 always_comb）
    // -------------------------------------------------------------------------
    // 教学注记 - depvec 4 位分别对应 4 个 sb 项：
    //   对每个源操作数，扫描 4 个 sb 项；若任一匹配，对应 dep 位为 1。
    //   can_issue = !(dep0 | dep1 | dep2 | dep3) = (depvec == 0)。
    logic dep0, dep1, dep2, dep3;
    always_comb begin
        // 默认值：无依赖
        dep0 = 1'b0; dep1 = 1'b0; dep2 = 1'b0; dep3 = 1'b0;
        if (chk_valid) begin
            // Entry 0：检查 4 个源是否匹配 sb[0]
            if (s0_valid) dep0 |= match_operand(entries[0], s0_reg, s0_is_pred, 1'b1);
            if (s1_valid) dep0 |= match_operand(entries[0], s1_reg, s1_is_pred, 1'b1);
            if (s2_valid) dep0 |= match_operand(entries[0], s2_reg, s2_is_pred, 1'b1);
            if (s3_valid) dep0 |= match_operand(entries[0], s3_reg, s3_is_pred, 1'b1);
            // Entry 1：同上，检查 sb[1]
            if (s0_valid) dep1 |= match_operand(entries[1], s0_reg, s0_is_pred, 1'b1);
            if (s1_valid) dep1 |= match_operand(entries[1], s1_reg, s1_is_pred, 1'b1);
            if (s2_valid) dep1 |= match_operand(entries[1], s2_reg, s2_is_pred, 1'b1);
            if (s3_valid) dep1 |= match_operand(entries[1], s3_reg, s3_is_pred, 1'b1);
            // Entry 2
            if (s0_valid) dep2 |= match_operand(entries[2], s0_reg, s0_is_pred, 1'b1);
            if (s1_valid) dep2 |= match_operand(entries[2], s1_reg, s1_is_pred, 1'b1);
            if (s2_valid) dep2 |= match_operand(entries[2], s2_reg, s2_is_pred, 1'b1);
            if (s3_valid) dep2 |= match_operand(entries[2], s3_reg, s3_is_pred, 1'b1);
            // Entry 3
            if (s0_valid) dep3 |= match_operand(entries[3], s0_reg, s0_is_pred, 1'b1);
            if (s1_valid) dep3 |= match_operand(entries[3], s1_reg, s1_is_pred, 1'b1);
            if (s2_valid) dep3 |= match_operand(entries[3], s2_reg, s2_is_pred, 1'b1);
            if (s3_valid) dep3 |= match_operand(entries[3], s3_reg, s3_is_pred, 1'b1);
        end
    end

    // depvec == 0 时可发射
    assign can_issue = !(dep0 | dep1 | dep2 | dep3);

    // -------------------------------------------------------------------------
    // 空闲项查找（组合逻辑 always_comb）
    // -------------------------------------------------------------------------
    // 找第一个 invalid 的项索引，用于发射时分配。
    // 教学注记：扫描顺序固定（i 从 0 到 NUM_ENTRIES-1），平局取小编号——
    //   这是 NutShellGPU 的确定性约定（05 册 §2.3 末："仲裁所有平局用编号小者胜"）。
    logic [1:0] free_idx;
    logic       has_free;
    always_comb begin
        has_free = 1'b0;
        free_idx = 2'h0;
        for (int i = 0; i < NUM_ENTRIES; i++) begin
            if (!entries[i].valid && !has_free) begin
                free_idx = i[1:0];
                has_free = 1'b1;
            end
        end
    end
    assign alloc_ok = has_free;
    assign full = !has_free;

    // -------------------------------------------------------------------------
    // 状态更新（时序逻辑 always_ff）
    // -------------------------------------------------------------------------
    // 教学注记 - 释放优先于分配：
    //   写回（释放）与发射（分配）可能同周期发生。本实现释放优先：先清
    //   valid，再分配新项，这样同拍写回腾出的项可以立即被新发射占用。
    //   03 册 §6.3 末保证：同 warp 指令按程序序经过 I→R→E→W，因此释放
    //   与分配的 (类,号) 不会冲突，该简单结构同时防 RAW/WAW/WAR。
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 复位：所有项清零
            for (int i = 0; i < NUM_ENTRIES; i++) begin
                entries[i] <= '{valid: 1'b0, is_pred: 1'b0, reg_num: 8'h0};
            end
        end else begin
            // 释放优先：写回时清匹配项（W 级）
            if (rel_en) begin
                for (int i = 0; i < NUM_ENTRIES; i++) begin
                    if (entries[i].valid && entries[i].is_pred == rel_is_pred
                        && entries[i].reg_num == rel_reg) begin
                        entries[i].valid <= 1'b0;
                    end
                end
            end
            // 分配：发射时占用空闲项（I 级）
            if (alloc && has_free) begin
                entries[free_idx].valid   <= 1'b1;
                entries[free_idx].is_pred <= dst_is_pred;
                entries[free_idx].reg_num <= dst_reg;
            end
        end
    end

endmodule
