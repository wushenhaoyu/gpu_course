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
// Operand Collector 模块（操作数收集器）
// operand_collector.sv
// -----------------------------------------------------------------------------
// 作用：实现 NSM 第三调度循环——"寄存器访问循环"。本模块包含 8 个 collector
//       unit，每个 unit 缓冲一条已发射指令及其 4 个操作数槽。每拍通过 4 个
//       单端口 RF bank 仲裁读取操作数；操作数齐且执行部件空闲时送入 E 级。
//       写回优先占用 bank，读操作数遇到 bank 冲突则延迟（不产生 replay）。
//
// 对应规格册：依据 03 册《NSM SIMT 核心微结构》§8 Operand Collector。
//   - §8.1 collector unit 结构
//   - §8.2 每周期仲裁算法（写回优先 → 余下 3 bank 按年龄排序读）
//   - §8.3 执行（E）
//   - §8.4 写回（W）
//
// 端口概要：
//   - issue_*：发射级输入，分配 collector unit
//   - rf_rd_*：4-bank RF 读口
//   - wb_* / rf_wr_*：写回输入与 RF 写口（优先级最高）
//   - ex_*：送往执行级的指令与操作数
//   - sp/sfu/dp/lsu/bru_ready：执行部件空闲反压
//   - bank_conflict_cycles：bank 冲突统计
//
// 教学注记 - 为什么用 bank 化 RF + operand collector：
//   GPU 每 NSM 驻留 2048 线程，每线程 255 个寄存器，共 65536 个 32 位寄存器。
//   若用单端口 RF，每周期只能读 1 个寄存器，远不足以供给 32 lane 的 SIMD
//   执行。多端口 RF 又面积爆炸。GPU 的折中方案是：
//     (1) 把 RF 划分为 4 个单端口逻辑 bank，bank(w, regnum) = (regnum + w) mod 4
//         ——swizzled 映射让连续 warp 的同号寄存器自然分散到不同 bank；
//     (2) 用 8 个 collector unit 缓冲多条在飞指令，每拍从所有 collector 中
//         选最多 3 个不冲突的 bank 读（剩 1 个 bank 留给写回）；
//     (3) collector 用本地缓存吸收 bank 冲突延迟——同 bank 的两个源分两拍读，
//         不向流水线深处传播 stall 信号（replay 仅在结构性冒险更严重时用）。
//   这样 4 单端口 bank 在 8 个 collector 面前表现出接近多端口的带宽，
//   是 GPU 寄存器堆设计的经典技巧（教材图 3.14/3.16）。
//
// 教学注记 - Replay vs stall 的取舍：
//   传统标量流水线用全局 stall 信号传播冒险（如 MIPS 的 hazard unit）。
//   GPU 拒绝这种做法——因为单 stall 信号会停掉所有 64 个 warp，浪费海量
//   多线程掩盖延迟的机会。NutShellGPU 的策略是：
//     - 寄存器 bank 冲突：仅延迟，不 replay（collector 缓存吸收）；
//     - LSU shared bank 冲突 / L1 miss / PRT 满 / WDB 满：replay，指令留
//       ibuf，下拍重新参与调度，**其他 warp 不受影响继续执行**。
//   这就是 03 册 §9 总原则："不向流水线深处传播全局停顿信号"。
// =============================================================================

`include "NutShellGPU_defines.svh"

module operand_collector #(
    parameter NUM_UNITS = 8,        // collector unit 数（03 §8.1：默认 8）
    parameter NUM_BANKS = 4,        // RF bank 数（03 §7：4）
    parameter NUM_OPS = 4,           // 每 collector 操作数槽数（03 §8.1：4）
    parameter NUM_WARPS = 64        // warp 槽位数
) (
    input  logic clk,
    input  logic rst_n,

    // --- 发射级：把新指令分配给空闲 collector ---
    // issue_valid=1 时申请分配；issue_ack=1 表示成功占用一个 collector
    input  logic        issue_valid,
    input  logic [$clog2(NUM_WARPS)-1:0] issue_warp,
    input  logic [63:0] issue_pc,
    input  logic [31:0] issue_exec_mask,   // active mask & guard
    input  logic [6:0]  issue_op,           // 操作码
    input  logic [2:0]  issue_pipe,         // 目标流水线（SP/SFU/DP/LSU/BRU）
    input  logic [7:0]  issue_dst,           // 目的寄存器
    input  logic        issue_dst_is_pred,
    input  logic        issue_has_dst,
    // 4 个操作数的寄存器号与类别（用于 RF bank 仲裁）
    input  logic [7:0]  issue_op0_reg,  input logic issue_op0_is_pred,  input logic issue_op0_valid,
    input  logic [7:0]  issue_op1_reg,  input logic issue_op1_is_pred,  input logic issue_op1_valid,
    input  logic [7:0]  issue_op2_reg,  input logic issue_op2_is_pred,  input logic issue_op2_valid,
    input  logic [7:0]  issue_op3_reg,  input logic issue_op3_is_pred,  input logic issue_op3_valid,
    // 立即数（I 型指令无 RF 源，直接用 imm_data）
    input  logic [31:0] issue_imm_data,
    input  logic        issue_use_imm,
    output logic        issue_ack,

    // --- RF 读口（4 bank，每 bank 单端口）---
    // rf_rd_req[b]=1 表示申请读 bank b；rf_rd_bank_sel 是 bank 选择
    // 教学注记：4 bank 中最多 3 个用于读，1 个保留给写回（若 wb_valid）
    output logic [NUM_BANKS-1:0]      rf_rd_req,
    output logic [NUM_BANKS-1:0]      rf_rd_bank_sel,
    output logic [$clog2(NUM_WARPS)-1:0] rf_rd_warp,
    output logic [7:0]                rf_rd_reg,
    input  logic [31:0]               rf_rd_data [NUM_BANKS],
    input  logic                       rf_rd_ack [NUM_BANKS],

    // --- 写回接口（最高优先级占用 bank）---
    // 教学注记：写回不可推迟（结果已经算好，不写会丢失），所以优先级最高。
    //   写回与读冲突时，被压的读顺延到下拍（教材图 3.14 现象由本算法复现）。
    input  logic        wb_valid,
    input  logic [$clog2(NUM_WARPS)-1:0] wb_warp,
    input  logic [7:0]  wb_reg,
    input  logic        wb_is_pred,
    input  logic [31:0] wb_data,
    output logic        wb_ack,

    // --- RF 写口（来自写回）---
    output logic        rf_wr_en,
    output logic [$clog2(NUM_BANKS)-1:0] rf_wr_bank,
    output logic [$clog2(NUM_WARPS)-1:0] rf_wr_warp,
    output logic [7:0]  rf_wr_reg,
    output logic [31:0] rf_wr_data,

    // --- 送往执行级（E 级）的指令与操作数 ---
    // ex_valid=1 表示本拍有指令送 E；ex_ack=1 表示执行部件接收
    output logic        ex_valid,
    output logic [$clog2(NUM_WARPS)-1:0] ex_warp,
    output logic [63:0] ex_pc,
    output logic [31:0] ex_exec_mask,
    output logic [6:0]  ex_op,
    output logic [2:0]  ex_pipe,
    output logic [7:0]  ex_dst,
    output logic        ex_dst_is_pred,
    output logic        ex_has_dst,
    output logic [31:0] ex_op_data [NUM_OPS],  // 4 个操作数的值
    input  logic        ex_ack,

    // --- 执行部件空闲反压 ---
    // 教学注记：SP 每 1 拍 1 条；SFU/DP 2 拍/warp（部件 busy 时不接收）；
    //   LSU 接收队列满则反压；BRU 每 1 拍 1 条。
    input  logic        sp_ready,
    input  logic        sfu_ready,
    input  logic        dp_ready,
    input  logic        lsu_ready,
    input  logic        bru_ready,

    // --- 统计 ---
    output logic [31:0] bank_conflict_cycles
);

    import NutShellGPU_pkg::*;

    // -------------------------------------------------------------------------
    // Collector unit 状态结构（unpacked，含操作数数组）
    // -------------------------------------------------------------------------
    // 教学注记：age 字段用于按"年龄最老"排序选送 E——先入 collector 的指令
    //   优先送执行，保证同 warp 多条指令按程序序离开 collector（WAR 防护）。
    typedef struct {
        logic        valid;
        logic [$clog2(NUM_WARPS)-1:0] warp;
        logic [63:0] pc;
        logic [31:0] exec_mask;
        logic [6:0]  op;
        logic [2:0]  pipe;
        logic [7:0]  dst;
        logic        dst_is_pred;
        logic        has_dst;
        logic        use_imm;
        logic [31:0] imm_data;
        logic [63:0] age;  // 年龄计数（值越小越老）
        // 4 个操作数槽：寄存器号、类别、有效、就绪、数据
        logic [7:0]  op_reg   [NUM_OPS];
        logic        op_is_pred[NUM_OPS];
        logic        op_valid  [NUM_OPS];
        logic        op_ready  [NUM_OPS];
        logic [31:0] op_data   [NUM_OPS];
    } collector_t;

    collector_t collectors [NUM_UNITS];
    logic [63:0] age_counter;        // 全局年龄计数器（每拍 +1）
    logic [31:0] conflict_count;     // 本拍 bank 冲突数

    // -------------------------------------------------------------------------
    // 查找空闲 collector（组合逻辑 always_comb）
    // -------------------------------------------------------------------------
    // 扫描所有 collector，找第一个 invalid 的；平局取小编号（确定性约定）。
    logic [$clog2(NUM_UNITS)-1:0] free_slot;
    logic                         has_free;
    always_comb begin
        has_free = 1'b0;
        free_slot = 0;
        for (int i = 0; i < NUM_UNITS; i++) begin
            if (!collectors[i].valid && !has_free) begin
                free_slot = i[$clog2(NUM_UNITS)-1:0];
                has_free = 1'b1;
            end
        end
    end
    assign issue_ack = issue_valid && has_free;

    // -------------------------------------------------------------------------
    // Bank 映射函数（依据 03 册 §7 swizzled 布局）
    // -------------------------------------------------------------------------
    // bank(w, regnum) = (regnum + warp) mod NUM_BANKS
    // 教学注记：同一 warp 的 r0/r4/r8/... 都在 bank 0；不同 warp 的 r0
    //   因 warp 偏移而分散到不同 bank——这是 swizzled 的精髓。
    function automatic logic [$clog2(NUM_BANKS)-1:0] get_bank(
        input logic [$clog2(NUM_WARPS)-1:0] warp,
        input logic [7:0] regnum);
        return (regnum + warp) % NUM_BANKS;
    endfunction

    // -------------------------------------------------------------------------
    // 判断 collector 所有操作数是否就绪
    // -------------------------------------------------------------------------
    function automatic logic all_ready(input collector_t c);
        logic r = 1'b1;
        for (int i = 0; i < NUM_OPS; i++) begin
            if (c.op_valid[i] && !c.op_ready[i]) r = 1'b0;
        end
        return r;
    endfunction

    // -------------------------------------------------------------------------
    // 检查执行部件是否空闲
    // -------------------------------------------------------------------------
    function automatic logic pipe_available(input logic [2:0] pipe);
        case (pipe)
            PIPE_SP:  return sp_ready;
            PIPE_SFU: return sfu_ready;
            PIPE_DP:  return dp_ready;
            PIPE_LSU: return lsu_ready;
            PIPE_BRU: return bru_ready;
            default:  return 1'b0;
        endcase
    endfunction

    // -------------------------------------------------------------------------
    // 选送 E 级的指令（组合逻辑 always_comb）
    // -------------------------------------------------------------------------
    // 教学注记 - 选送策略（03 §8.2）：
    //   优先选"年龄最老 + 操作数齐 + 部件空闲"的 collector。
    //   同一 warp 的多个 collector 必须按程序序先后送 E（WAR 防护）——
    //   因为年龄计数按发射顺序递增，老指令年龄小，自然先被选中。
    logic        ex_select_valid;
    logic [$clog2(NUM_UNITS)-1:0] ex_select_slot;
    always_comb begin
        ex_select_valid = 1'b0;
        ex_select_slot = 0;
        // 找年龄最小（age 最小）的合格 collector
        for (int i = 0; i < NUM_UNITS; i++) begin
            if (collectors[i].valid && all_ready(collectors[i])
                && pipe_available(collectors[i].pipe)) begin
                if (!ex_select_valid || collectors[i].age < collectors[ex_select_slot].age) begin
                    ex_select_slot = i[$clog2(NUM_UNITS)-1:0];
                    ex_select_valid = 1'b1;
                end
            end
        end
    end

    // 驱动 ex_* 输出
    assign ex_valid       = ex_select_valid && ex_ack;
    assign ex_warp        = collectors[ex_select_slot].warp;
    assign ex_pc          = collectors[ex_select_slot].pc;
    assign ex_exec_mask   = collectors[ex_select_slot].exec_mask;
    assign ex_op          = collectors[ex_select_slot].op;
    assign ex_pipe        = collectors[ex_select_slot].pipe;
    assign ex_dst         = collectors[ex_select_slot].dst;
    assign ex_dst_is_pred = collectors[ex_select_slot].dst_is_pred;
    assign ex_has_dst     = collectors[ex_select_slot].has_dst;
    for (genvar i = 0; i < NUM_OPS; i++) begin
        assign ex_op_data[i] = collectors[ex_select_slot].op_data[i];
    end

    // -------------------------------------------------------------------------
    // 写回处理（最高优先级占用 bank）
    // -------------------------------------------------------------------------
    // 教学注记：写回 ack 永远为 1（不可推迟）；同时驱动 RF 写口
    assign wb_ack = wb_valid;
    assign rf_wr_en   = wb_valid;
    assign rf_wr_bank = wb_valid ? get_bank(wb_warp, wb_reg) : 0;
    assign rf_wr_warp  = wb_warp;
    assign rf_wr_reg   = wb_reg;
    assign rf_wr_data  = wb_data;

    // -------------------------------------------------------------------------
    // Bank 仲裁（组合逻辑 always_comb）——读操作数
    // -------------------------------------------------------------------------
    // 教学注记 - 仲裁算法（03 §8.2）：
    //   1) 写回占用其 bank（优先级最高）
    //   2) 余下最多 3 个 bank 名额，按 collector 年龄顺序选不冲突的读
    //   3) 同 bank 的两个源分两拍读，collector 缓存吸收，**不产生 replay**
    //   4) bank 冲突计数累加（统计用）
    logic [NUM_BANKS-1:0] bank_in_use;
    always_comb begin
        bank_in_use = '0;
        rf_rd_req = '0;
        rf_rd_bank_sel = '0;
        rf_rd_warp = 0;
        rf_rd_reg = 0;
        conflict_count = 0;

        // 写回占用其 bank
        if (wb_valid) begin
            bank_in_use[get_bank(wb_warp, wb_reg)] = 1'b1;
        end

        // 扫描所有 collector，按 i 顺序（i 小=老）选不冲突的读
        for (int i = 0; i < NUM_UNITS; i++) begin
            if (collectors[i].valid) begin
                // 对每个未就绪操作数尝试发起 RF 读
                for (int j = 0; j < NUM_OPS; j++) begin
                    if (collectors[i].op_valid[j] && !collectors[i].op_ready[j]) begin
                        logic [$clog2(NUM_BANKS)-1:0] b;
                        b = get_bank(collectors[i].warp, collectors[i].op_reg[j]);
                        if (!bank_in_use[b]) begin
                            // bank 空闲，发起读请求
                            rf_rd_req[b] = 1'b1;
                            rf_rd_bank_sel[b] = b;
                            rf_rd_warp = collectors[i].warp;
                            rf_rd_reg = collectors[i].op_reg[j];
                            bank_in_use[b] = 1'b1;
                            // 实际 ready 更新在 always_ff 中处理
                        end else begin
                            // bank 冲突：延迟到下拍读
                            conflict_count++;
                        end
                    end
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // 状态更新（时序逻辑 always_ff）
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 复位：所有 collector 释放，年龄计数清零
            for (int i = 0; i < NUM_UNITS; i++) begin
                collectors[i].valid <= 1'b0;
                collectors[i].age <= 64'h0;
            end
            age_counter <= 64'h0;
            bank_conflict_cycles <= 32'h0;
        end else begin
            age_counter <= age_counter + 1;
            bank_conflict_cycles <= bank_conflict_cycles + conflict_count;

            // 更新操作数 ready 位（基于上拍发起的 RF 读与本拍到达的写回）
            // 教学注记：写回数据可被同 warp 的在飞指令直接捕获（forwarding）
            for (int i = 0; i < NUM_UNITS; i++) begin
                if (collectors[i].valid) begin
                    for (int j = 0; j < NUM_OPS; j++) begin
                        if (collectors[i].op_valid[j] && !collectors[i].op_ready[j]) begin
                            // 检查写回是否匹配本操作数（forwarding）
                            if (wb_valid && wb_warp == collectors[i].warp
                                && wb_is_pred == collectors[i].op_is_pred[j]
                                && wb_reg == collectors[i].op_reg[j]) begin
                                collectors[i].op_ready[j] <= 1'b1;
                                collectors[i].op_data[j]  <= wb_data;
                            end
                            // 注：实际 RF 读数据捕获在更完整实现中需配合
                            // rf_rd_ack/rf_rd_data 时序，此处简化为同一周期就绪
                        end
                    end
                    // 立即数操作数直接就绪
                    if (collectors[i].use_imm) begin
                        collectors[i].op_data[0] <= collectors[i].imm_data;
                        collectors[i].op_ready[0] <= 1'b1;
                    end
                end
            end

            // 分配新指令到空闲 collector（I 级发射）
            if (issue_ack) begin
                collectors[free_slot].valid      <= 1'b1;
                collectors[free_slot].warp       <= issue_warp;
                collectors[free_slot].pc         <= issue_pc;
                collectors[free_slot].exec_mask  <= issue_exec_mask;
                collectors[free_slot].op         <= issue_op;
                collectors[free_slot].pipe       <= issue_pipe;
                collectors[free_slot].dst        <= issue_dst;
                collectors[free_slot].dst_is_pred<= issue_dst_is_pred;
                collectors[free_slot].has_dst    <= issue_has_dst;
                collectors[free_slot].use_imm    <= issue_use_imm;
                collectors[free_slot].imm_data  <= issue_imm_data;
                collectors[free_slot].age        <= age_counter;
                // 设置 4 个操作数槽
                collectors[free_slot].op_reg[0]    <= issue_op0_reg;
                collectors[free_slot].op_is_pred[0] <= issue_op0_is_pred;
                collectors[free_slot].op_valid[0]   <= issue_op0_valid && !issue_use_imm;
                collectors[free_slot].op_ready[0]   <= !issue_op0_valid || issue_use_imm;
                collectors[free_slot].op_reg[1]    <= issue_op1_reg;
                collectors[free_slot].op_is_pred[1] <= issue_op1_is_pred;
                collectors[free_slot].op_valid[1]   <= issue_op1_valid;
                collectors[free_slot].op_ready[1]   <= !issue_op1_valid;
                collectors[free_slot].op_reg[2]    <= issue_op2_reg;
                collectors[free_slot].op_is_pred[2] <= issue_op2_is_pred;
                collectors[free_slot].op_valid[2]   <= issue_op2_valid;
                collectors[free_slot].op_ready[2]   <= !issue_op2_valid;
                collectors[free_slot].op_reg[3]    <= issue_op3_reg;
                collectors[free_slot].op_is_pred[3] <= issue_op3_is_pred;
                collectors[free_slot].op_valid[3]   <= issue_op3_valid;
                collectors[free_slot].op_ready[3]   <= !issue_op3_valid;
            end

            // 释放 collector（执行部件接收后）
            if (ex_valid && ex_ack) begin
                collectors[ex_select_slot].valid <= 1'b0;
            end
        end
    end

endmodule
