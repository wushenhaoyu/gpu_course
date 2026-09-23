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
// NSM（Newton Streaming Multiprocessor，牛顿流式多处理器）顶层模块
// nsm.sv
// -----------------------------------------------------------------------------
// 作用：NSM 是 NutShellGPU 的核心计算单元。一个 NSM 内含：
//     (1) 指令存储器 IMEM（8 KiB，1024 条 64-bit 指令）；
//     (2) 寄存器堆 RF——2048 线程 × 256 寄存器（R0..R255），加 7 位谓词
//         寄存器 P0..P6；
//     (3) SIMT 栈——处理控制流分歧与再收敛；
//     (4) 记分牌——每 warp 4 项，检测 RAW/WAW/WAR 依赖；
//     (5) Operand Collector——8 个 collector 单元 + 4-bank RF 仲裁；
//     (6) L1D Cache（本测试程序未使用，仅例化不驱动）；
//     (7) 五级流水线控制器 F → D → I → R → E → W。
//
// 对应规格册：依据 03 册《NSM SIMT 核微结构》。
//   - §2 顶层组成与五级流水线
//   - §3 warp 调度（round-robin 选 ready warp）
//   - §4 取指/I-Cache 行为（本实现简化为单周期命中）
//   - §5 记分牌与依赖检测
//   - §6 Operand Collector 寄存器读
//   - §7 各流水线 (SP/INT/SFU/MEM/BRU) 时延
//   - §8 SIMT 栈与控制流（分歧/再收敛/SSY/BRA）
//   - §9 EXIT 处理与 warp 退出
//
// 端口概要：
//   - imem_load/imem_addr/imem_data：指令加载口（仿真期由 tb 灌入指令）
//   - warp_init*：warp 启动口（PC + active mask）
//   - rf_rd_reg/rf_rd_data：RF 旁路读口（仅供 tb 校验结果）
//   - warp_done/inst_count/cycle_count/cur_pc_out：状态输出
//
// 关键参数：
//   - NUM_WARPS = 64：每 NSM 最多 64 个 warp 插槽
//   - NUM_LANES = 32：每 warp 32 个 lane（SIMT 锁步宽度）
//   - IMEM_SIZE = 1024：8 KiB 指令存储（1024 条 64-bit 指令）
//   - RF_THREADS = NUM_WARPS * NUM_LANES = 2048：物理线程总数
//
// 教学注记 - 为什么 NSM 是 GPU 的核心：
//   GPU "大核" 的等价物就是 NSM（NVIDIA 称 SM）。一个 NutShellGPU 芯片有多个
//   NSM（具体数量见 01 册 §2 配置），每个 NSM 独立取指、调度、执行，靠
//   片上互连 NoC 访问共享的 L2/DRAM。NSM 内部通过 SIMT 模型让 32 个 lane
//   锁步执行同一条指令，配合 SIMT 栈实现控制流分歧与再收敛——这是 GPU
//   区别于 CPU 的核心特征。
//
// 教学注记 - 五级流水线各阶段职责（03 §2）：
//   PS_FETCH     取指：从 IMEM 读 64-bit 指令字，decode_inst 解码为 inst_t
//   PS_DECODE    译码：指令已解码；此拍设置记分牌源操作数（组合逻辑）
//   PS_ISSUE     发射：查记分牌 can_issue；通过则分配 sb 项、推进 inst_cnt
//   PS_READ      读寄存器：Operand Collector 收集源操作数（本实现简化为立即就绪）
//   PS_EXECUTE   执行：按 opcode/pipe 跑对应延迟（SP=4, INT=4, SFU=20, BRU=1）
//   PS_WRITEBACK 写回：RF 写入 exec_result；SIMT 栈推进 PC（advance/bra/ssy/exit）
//
// 教学注记 - 与真实硬件的简化差异：
//   本实现是教学简化版——
//     (1) 单 warp 串行处理：每个时钟只服务 1 个 warp，没有 warp 调度器
//         跨 warp round-robin 切换；
//     (2) RF 单端口读：E 阶段直接读 rf[cur_warp*32][reg]，未走 Operand
//         Collector 的 4-bank 仲裁路径（collector 已例化但接口未充分驱动）；
//     (3) I-Cache 模型为单周期命中：未建模 miss；
//     (4) 分支简化为统一跳转：BRA 当作不分歧处理（taken_mask=全 1）；
//     (5) L1D 接口未驱动：仅例化保持结构完整性，测试程序不访存。
//   这些简化保留了流水线/记分牌/SIMT 栈的核心交互，便于教学。
// =============================================================================

`include "NutShellGPU_defines.svh"

module nsm #(
    parameter NSM_ID = 0,            // NSM 实例号（多 NSM 系统中用于路由/统计）
    parameter NUM_WARPS = 64,        // 每 NSM warp 插槽数（03 §2，最大 64）
    parameter NUM_LANES = 32,        // 每 warp lane 数（SIMT 宽度，固定 32）
    parameter IMEM_SIZE = 1024      // 指令存储容量（8 KiB = 1024 条 64-bit 指令）
) (
    input  logic clk,                // 主时钟（1 GHz，单边沿触发）
    input  logic rst_n,              // 异步低有效复位

    // --- 指令存储加载口 ---
    // 教学注记：仿真时由 tb 逐字灌入指令；真实硬件由驱动加载
    input  logic        imem_load,           // 加载使能
    input  logic [10:0] imem_addr,           // 字地址（0..1023，每字 64-bit = 8 字节）
    input  logic [63:0] imem_data,            // 待写入的 64-bit 指令字

    // --- warp 启动口 ---
    // 教学注记：CPU 端通过此口启动一个 warp——给定入口 PC 和 active mask
    //   （mask 标识 32 lane 中哪些有效，通常全 1）。SIMT 栈在 IDLE 状态被
    //   op_init 信号初始化（推入一个 TOS 项，nextpc=entry_pc, mask=init_mask）
    input  logic        warp_init,                       // 启动使能
    input  logic [$clog2(NUM_WARPS)-1:0] warp_init_id,  // 待启动 warp 的 ID
    input  logic [63:0] warp_init_entry_pc,             // 入口 PC（字节地址）
    input  logic [31:0] warp_init_mask,                 // active lane mask

    // --- RF 旁路读口（仅供 tb 校验结果） ---
    // 教学注记：仿真结束后 tb 通过此口读 R0 校验计算结果。固定读
    //   warp 0 / lane 0 的寄存器。真实硬件无此口。
    input  logic [7:0]  rf_rd_reg,    // 待读寄存器号 R0..R255
    output logic [31:0] rf_rd_data,   // 读出的 32-bit 值

    // --- 状态输出 ---
    output logic        warp_done,     // warp 0 已执行 EXIT（tb 用作结束信号）
    output logic [31:0] inst_count,   // 已发射指令数（IPC 计算用）
    output logic [31:0] cycle_count,  // 已运行周期数（IPC 计算用）
    output logic [63:0] cur_pc_out     // 当前 PC（调试观察用）
);

    import NutShellGPU_pkg::*;

    // =========================================================================
    // 指令存储器 IMEM
    // =========================================================================
    // 教学注记：每条 NTAS 指令编码为 64-bit（见 02 册 ABI + 03 册 NTAS 手册）。
    //   IMEM 容量 = 1024 字 = 8 KiB。地址 imem_addr 是字地址（0..1023），
    //   实际字节 PC = word_addr * 8。取指时用 cur_pc[10:3] 作为字地址索引。
    //   本实现未建模 I-Cache miss，假设每拍命中。
    // =========================================================================
    logic [63:0] imem [0:IMEM_SIZE-1];

    // =========================================================================
    // 寄存器堆 RF
    // =========================================================================
    // 教学注记 - RF 组织（03 §6）：
    //   RF 物理上按 thread 平铺：rf[thread][reg]，thread = warp * 32 + lane。
    //   每个线程拥有 256 个 32-bit 通用寄存器（R0..R255，R0 通常是 zero
    //   register，但本实现未硬连线为 0——由编译器/ISA 约定）。
    //   总大小 = 2048 × 256 × 4B = 2 MiB——真实硬件中按 bank 切分以避免
    //   端口爆炸（见 operand_collector.sv 的 4-bank swizzled 映射）。
    //
    // 教学注记 - 谓词寄存器 pred：
    //   每线程 7 位谓词寄存器 P0..P6（03 §6）。谓词由 SETP 指令写入，
    //   用于 guarded execution——指令的 guard_en+guard_pred 字段决定哪些
    //   lane 真正执行（mask & predicate）。本测试程序不使用谓词。
    // =========================================================================
    localparam RF_THREADS = NUM_WARPS * NUM_LANES;  // 2048：物理线程总数
    logic [31:0] rf [0:RF_THREADS-1][0:255];

    // 谓词寄存器：pred[thread][0..6]，每位一个谓词
    logic pred [0:RF_THREADS-1][0:6];

    // RF 旁路读口：固定读 warp 0 / lane 0 的 rf[0][rf_rd_reg]
    // 教学注记：组合逻辑直读，无 bank 仲裁；仅 tb 用作结果校验
    assign rf_rd_data = rf[0][rf_rd_reg];

    // =========================================================
    // 每 warp 状态
    // =========================================================
    // 教学注记：每个 warp 插槽维护两个状态位：
    //   warp_valid - warp 已被初始化且未退出（参与调度）
    //   warp_exit  - warp 已执行 EXIT，不再参与调度（终态）
    //   本简化实现只串行处理 cur_warp 一个 warp，真实硬件会 round-robin
    //   遍历所有 warp_valid 的 warp（03 §3）。
    logic        warp_valid    [NUM_WARPS];
    logic        warp_exit     [NUM_WARPS];

    // =========================================================================
    // 流水线状态机枚举（03 §2 五级流水线）
    // =========================================================================
    // 教学注记 - 各状态语义与转移：
    //   PS_IDLE      复位后空闲；收到 warp_init 后置 valid 并跳 PS_FETCH
    //   PS_FETCH     取指：从 IMEM 读 simt_tos_nextpc 处的指令，decode 入 cur_inst
    //   PS_DECODE    译码：指令已在 FETCH 解码；此拍组合逻辑设置记分牌源操作数
    //   PS_ISSUE     发射：查 sb_can_issue；通过则分配 sb 项、++inst_cnt，进 READ
    //   PS_READ      读 RF：Operand Collector 收集源（本实现简化为立即就绪）
    //   PS_EXECUTE   执行：按 opcode 跑对应延迟（SP/INT=4, SFU=20, BRU=1）；
    //                  exec_done 拉高后进 WRITEBACK
    //   PS_WRITEBACK 写回：RF 写入 exec_result；SIMT 栈推进 PC（advance/bra/ssy/exit）
    //   PS_SIMT_WAIT SIMT 栈等待（本实现未使用，保留状态）
    // =========================================================================
    typedef enum logic [3:0] {
        PS_IDLE     = 4'd0,   // 空闲：等待 warp_init
        PS_FETCH    = 4'd1,   // 取指
        PS_DECODE   = 4'd2,   // 译码（设置记分牌源）
        PS_ISSUE    = 4'd3,   // 发射（查记分牌）
        PS_READ     = 4'd4,   // 读寄存器
        PS_EXECUTE  = 4'd5,   // 执行（按 pipe 跑延迟）
        PS_WRITEBACK= 4'd6,   // 写回（RF 写 + SIMT 栈推进）
        PS_SIMT_WAIT= 4'd7   // SIMT 等待（保留）
    } pipe_state_e;

    pipe_state_e pipe_state;   // 当前流水线状态

    // --- 当前正在处理的指令的上下文 ---
    // 教学注记：本简化实现用一组寄存器贯穿整条流水线，而非每级单独
    //   维护流水线寄存器（真实硬件每级有独立的 pipeline register）。
    inst_t       cur_inst;              // 当前解码后的指令结构体
    logic [31:0] cur_mask;              // 当前 active lane mask（32 lane 哪些有效）
    logic [$clog2(NUM_WARPS)-1:0] cur_warp;  // 当前 warp ID
    logic [63:0] cur_pc;                // 当前 PC（字节地址）
    logic [7:0]  exec_latency_cnt;     // 执行阶段已等待周期数（倒数 LAT）
    logic [31:0] exec_result;           // 执行结果（写回 RF 用）
    logic        exec_done;             // 执行完成标志（达到延迟后拉高）

    // 统计计数器
    logic [31:0] inst_cnt, cyc_cnt;

    // =========================================================================
    // SIMT 栈例化（仅服务当前活跃 warp）
    // =========================================================================
    // 教学注记 - SIMT 栈的作用（03 §8 + simt_stack.sv）：
    //   SIMT 栈是 GPU 处理控制流分歧/再收敛的核心数据结构。它本质是一个
    //   栈，每项记录 (next_pc, mask, reconv_pc, div_flag)。当 warp 内部分
    //   lane 走 if 分支、另一部分走 else 分支时（BRA DIV），栈推入两个项
    //   ——先执行的分支项放栈顶，后执行的分支项放其下，并标记再收敛点
    //   (RPC)。两条分支都执行完后再收敛到 RPC，恢复全 mask 锁步执行。
    //
    //   栈顶 TOS (top-of-stack) 的 nextpc 就是下一条要取指的 PC——
    //   PS_FETCH 阶段从 simt_tos_nextpc 取指。
    //
    // 本顶层驱动 SIMT 栈的 6 类操作（在 PS_WRITEBACK 阶段组合逻辑产生）：
    //   op_init      初始化：推入一个初始 TOS（nextpc=entry_pc, mask=init_mask）
    //   op_ssy       设置再收敛点：记录 SSY 指令后的 RPC
    //   op_bra_uni   统一分支：全 warp 跳转（不分歧）
    //   op_bra_div   分歧分支：推入 taken/fall 两个分支项
    //   op_advance   顺序推进：PC = PC + 8
    //   op_exit      warp 退出：从 mask 中清除 EXIT 的 lane
    // =========================================================================

    // --- SIMT 栈 TOS（栈顶）输出 ---
    logic [63:0] simt_tos_nextpc;   // 栈顶 next_pc：PS_FETCH 用的取指地址
    logic [31:0] simt_tos_mask;      // 栈顶 active mask：cur_mask 来源
    logic [63:0] simt_tos_rpc;      // 栈顶再收敛点（IPDOM）
    logic        simt_tos_dflag;     // 栈顶 div flag：标识该 TOS 项是否来自分歧
    logic        simt_stack_empty;   // 栈空标志
    logic        simt_warp_exit_done;// warp 全部 lane 已 EXIT，warp 可结束
    logic        simt_err_no_ipdom;  // 错误：BRA DIV 前未 SSY 设置再收敛点
    logic        simt_err_stack_overflow; // 错误：栈溢出（超过 STACK_DEPTH）

    // --- SIMT 栈操作输入（组合逻辑驱动） ---
    logic        simt_op_init;       // 初始化栈
    logic        simt_op_ssy;        // 记录再收敛点
    logic [63:0] simt_ssy_target;    // SSY 的 RPC 目标地址
    logic        simt_op_bra_uni;    // 统一分支
    logic        simt_bra_taken;      // 统一分支是否 taken
    logic [63:0] simt_bra_target;    // 统一分支目标
    logic        simt_op_bra_div;    // 分歧分支
    logic [63:0] simt_div_target;    // 分歧分支目标
    logic [31:0] simt_taken_mask;    // taken 路径的 mask
    logic [31:0] simt_fall_mask;     // fall-through 路径的 mask
    logic        simt_op_advance;    // 顺序推进 PC
    logic [63:0] simt_advance_newpc;// 推进后的新 PC
    logic        simt_op_exit;       // warp 退出处理
    logic [31:0] simt_exit_mask;    // EXIT 时的 lane mask

    simt_stack #(.STACK_DEPTH(SIMT_STACK_DEPTH)) u_simt_stack (
        .clk(clk), .rst_n(rst_n),
        .tos_nextpc(simt_tos_nextpc),
        .tos_mask(simt_tos_mask),
        .tos_rpc(simt_tos_rpc),
        .tos_dflag(simt_tos_dflag),
        .tos_ptr(),
        .stack_empty(simt_stack_empty),
        .warp_exit_done(simt_warp_exit_done),
        .op_init(simt_op_init),
        .init_entry_pc(warp_init_entry_pc),
        .init_mask(warp_init_mask),
        .op_ssy(simt_op_ssy),
        .ssy_target(simt_ssy_target),
        .cur_pc(cur_pc),
        .op_bra_uni(simt_op_bra_uni),
        .bra_taken(simt_bra_taken),
        .bra_target(simt_bra_target),
        .op_bra_div(simt_op_bra_div),
        .div_target(simt_div_target),
        .taken_mask(simt_taken_mask),
        .fall_mask(simt_fall_mask),
        .op_advance(simt_op_advance),
        .advance_newpc(simt_advance_newpc),
        .op_exit(simt_op_exit),
        .exit_lane_mask(simt_exit_mask),
        .err_no_ipdom(simt_err_no_ipdom),
        .err_stack_overflow(simt_err_stack_overflow)
    );

    // =========================================================================
    // 记分牌例化（仅服务当前活跃 warp）
    // =========================================================================
    // 教学注记 - 记分牌的作用（03 §5 + scoreboard.sv）：
    //   记分牌（scoreboard）用于解决流水线内的数据依赖——当一个未完成的
    //   指令正在写某寄存器时，后续读该寄存器的指令必须等待（stall），否则
    //   会读到旧值。本实现采用 Coon 式记分牌，每 warp 4 项，每项记录一个
    //   未完成的目的寄存器号。
    //
    //   两阶段使用：
    //     (1) PS_DECODE/PS_ISSUE 阶段（chk_valid=1）：检查源操作数 s0..s3 是否
    //         命中记分牌——若命中则 sb_can_issue=0，指令 stall；
    //     (2) PS_ISSUE 阶段（alloc=1）：发射通过的指令，分配一个记分牌项
    //         记录其目的寄存器，防止后续指令读过期值；
    //     (3) PS_WRITEBACK 阶段（release=1）：写回完成，释放该记分牌项。
    // =========================================================================

    // --- 记分牌输出 ---
    logic        sb_can_issue;   // 源操作数无依赖，允许发射
    logic        sb_alloc;       // 分配项请求（PS_ISSUE 拍产生脉冲）
    logic        sb_alloc_ok;    // 分配成功（有空闲项）
    logic        sb_release;     // 释放项请求（PS_WRITEBACK 拍产生脉冲）
    logic        sb_full;        // 记分牌满（4 项都用尽）

    // --- 记分牌检查输入（源操作数） ---
    logic        sb_chk_valid;   // 检查使能：PS_DECODE 或 PS_ISSUE 时为 1
    logic [7:0]  sb_s0_reg, sb_s1_reg, sb_s2_reg, sb_s3_reg;  // 4 个源寄存器号
    logic        sb_s0_is_pred, sb_s1_is_pred, sb_s2_is_pred, sb_s3_is_pred; // 是否谓词源
    logic        sb_s0_valid, sb_s1_valid, sb_s2_valid, sb_s3_valid;  // 各源是否有效

    // --- 记分牌分配/释放输入（目的寄存器） ---
    logic        sb_dst_is_pred; // 目的是否谓词寄存器
    logic [7:0]  sb_dst_reg;     // 目的寄存器号
    logic        sb_rel_is_pred; // 释放项是否谓词
    logic [7:0]  sb_rel_reg;     // 释放项的寄存器号

    scoreboard #(.NUM_ENTRIES(SCOREBOARD_ENTRIES_PER_WARP)) u_scoreboard (
        .clk(clk), .rst_n(rst_n),
        .chk_valid(sb_chk_valid),
        .s0_reg(sb_s0_reg), .s0_is_pred(sb_s0_is_pred), .s0_valid(sb_s0_valid),
        .s1_reg(sb_s1_reg), .s1_is_pred(sb_s1_is_pred), .s1_valid(sb_s1_valid),
        .s2_reg(sb_s2_reg), .s2_is_pred(sb_s2_is_pred), .s2_valid(sb_s2_valid),
        .s3_reg(sb_s3_reg), .s3_is_pred(sb_s3_is_pred), .s3_valid(sb_s3_valid),
        .can_issue(sb_can_issue),
        .alloc(sb_alloc),
        .dst_is_pred(sb_dst_is_pred),
        .dst_reg(sb_dst_reg),
        .alloc_ok(sb_alloc_ok),
        .rel_en(sb_release),
        .rel_is_pred(sb_rel_is_pred),
        .rel_reg(sb_rel_reg),
        .full(sb_full)
    );

    // =========================================================================
    // Operand Collector 例化（8 个 collector 单元，4-bank RF 仲裁）
    // =========================================================================
    // 教学注记 - Operand Collector 的作用（03 §6 + operand_collector.sv）：
    //   GPU 的 RF 巨大（2048 线程 × 256 寄存器），若给每条指令都配独立读
    //   端口会面积爆炸。Operand Collector 采用"收集"模式：每周期最多 8
    //   个 collector 单元发起 RF 读请求，仲裁到 4 个物理 bank 上，几个周期
    //   内把指令所需的源操作数凑齐，再送入执行流水线。这样大幅减少 RF
    //   读端口数量（每 bank 一个端口，共 4 个）。
    //
    //   4 个 bank 用 swizzled 映射（见 operand_collector.sv），让同一 warp
    //   的相邻寄存器落到不同 bank，降低 bank 冲突概率。冲突时排队，故有
    //   bank_conflict_cycles 统计。
    //
    // 教学注记 - 本顶层与 collector 的关系：
    //   完整实现中，PS_ISSUE→PS_READ 的过渡就是 collector 在收集操作数，
    //   collector 凑齐操作数后通过 ex_* 接口把指令推进到 EX 阶段。本简化
    //   实现未充分驱动 collector 的 RF 读写接口（rf_rd_*、rf_wr_* 留空），
    //   而是 E 阶段直接读 rf[cur_warp*32][reg]——保留 collector 例化是
    //   为了结构完整和后续扩展。
    // =========================================================================

    // --- Operand Collector 发射接口（PS_ISSUE 阶段送入） ---
    logic        oc_issue_valid;          // 发射有效
    logic [$clog2(NUM_WARPS)-1:0] oc_issue_warp;   // 目标 warp
    logic [63:0] oc_issue_pc;              // 指令 PC
    logic [31:0] oc_issue_exec_mask;      // active mask
    logic [6:0]  oc_issue_op;              // 操作码
    logic [2:0]  oc_issue_pipe;            // 流水线类型 (SP/INT/SFU/MEM/BRU)
    logic [7:0]  oc_issue_dst;             // 目的寄存器号
    logic        oc_issue_dst_is_pred;     // 目的是否谓词
    logic        oc_issue_has_dst;         // 是否有目的寄存器
    logic [7:0]  oc_op0_reg, oc_op1_reg, oc_op2_reg, oc_op3_reg;  // 4 个源寄存器号
    logic        oc_op0_is_pred, oc_op1_is_pred, oc_op2_is_pred, oc_op3_is_pred;
    logic        oc_op0_valid, oc_op1_valid, oc_op2_valid, oc_op3_valid;
    logic [31:0] oc_issue_imm_data;       // 立即数
    logic        oc_issue_use_imm;         // 是否用立即数替代某源
    logic        oc_issue_ack;             // collector 接受发射

    // --- Operand Collector 执行输出（操作数收集完毕后送 EX） ---
    logic        oc_ex_valid;
    logic [$clog2(NUM_WARPS)-1:0] oc_ex_warp;
    logic [63:0] oc_ex_pc;
    logic [31:0] oc_ex_exec_mask;
    logic [6:0]  oc_ex_op;
    logic [2:0]  oc_ex_pipe;
    logic [7:0]  oc_ex_dst;
    logic        oc_ex_dst_is_pred;
    logic        oc_ex_has_dst;
    logic [31:0] oc_ex_op_data [4];       // 4 个收集好的操作数值
    logic        oc_ex_ack;

    // --- Operand Collector 写回接口（EX 完成后写 RF） ---
    logic        oc_wb_valid;
    logic [$clog2(NUM_WARPS)-1:0] oc_wb_warp;
    logic [7:0]  oc_wb_reg;
    logic        oc_wb_is_pred;
    logic [31:0] oc_wb_data;
    logic        oc_wb_ack;
    logic [31:0] oc_bank_conflict_cycles; // RF bank 冲突周期统计

    operand_collector #(
        .NUM_UNITS(NUM_COLLECTORS),              // 8 个 collector 单元
        .NUM_BANKS(RF_BANKS),                    // 4 个 RF bank
        .NUM_OPS(OPERAND_SLOTS_PER_COLLECTOR),  // 每 collector 4 个操作数槽
        .NUM_WARPS(NUM_WARPS)
    ) u_operand_collector (
        .clk(clk), .rst_n(rst_n),
        .issue_valid(oc_issue_valid),
        .issue_warp(oc_issue_warp),
        .issue_pc(oc_issue_pc),
        .issue_exec_mask(oc_issue_exec_mask),
        .issue_op(oc_issue_op),
        .issue_pipe(oc_issue_pipe),
        .issue_dst(oc_issue_dst),
        .issue_dst_is_pred(oc_issue_dst_is_pred),
        .issue_has_dst(oc_issue_has_dst),
        .issue_op0_reg(oc_op0_reg), .issue_op0_is_pred(oc_op0_is_pred), .issue_op0_valid(oc_op0_valid),
        .issue_op1_reg(oc_op1_reg), .issue_op1_is_pred(oc_op1_is_pred), .issue_op1_valid(oc_op1_valid),
        .issue_op2_reg(oc_op2_reg), .issue_op2_is_pred(oc_op2_is_pred), .issue_op2_valid(oc_op2_valid),
        .issue_op3_reg(oc_op3_reg), .issue_op3_is_pred(oc_op3_is_pred), .issue_op3_valid(oc_op3_valid),
        .issue_imm_data(oc_issue_imm_data),
        .issue_use_imm(oc_issue_use_imm),
        .issue_ack(oc_issue_ack),
        .rf_rd_req(),
        .rf_rd_bank_sel(),
        .rf_rd_warp(),
        .rf_rd_reg(),
        .rf_rd_data('{default: 32'h0}),
        .rf_rd_ack('{default: 1'b1}),
        .wb_valid(oc_wb_valid),
        .wb_warp(oc_wb_warp),
        .wb_reg(oc_wb_reg),
        .wb_is_pred(oc_wb_is_pred),
        .wb_data(oc_wb_data),
        .wb_ack(oc_wb_ack),
        .rf_wr_en(),
        .rf_wr_bank(),
        .rf_wr_warp(),
        .rf_wr_reg(),
        .rf_wr_data(),
        .ex_valid(oc_ex_valid),
        .ex_warp(oc_ex_warp),
        .ex_pc(oc_ex_pc),
        .ex_exec_mask(oc_ex_exec_mask),
        .ex_op(oc_ex_op),
        .ex_pipe(oc_ex_pipe),
        .ex_dst(oc_ex_dst),
        .ex_dst_is_pred(oc_ex_dst_is_pred),
        .ex_has_dst(oc_ex_has_dst),
        .ex_op_data(oc_ex_op_data),
        .ex_ack(oc_ex_ack),
        .sp_ready(1'b1),
        .sfu_ready(1'b1),
        .dp_ready(1'b1),
        .lsu_ready(1'b1),
        .bru_ready(1'b1),
        .bank_conflict_cycles(oc_bank_conflict_cycles)
    );

    // =========================================================
    // L1D Cache 例化（本测试程序未使用，仅保留结构完整）
    // =========================================================
    // 教学注记：L1D 是每 NSM 私有的数据 cache（VI/VT，04 §6），服务
    //   LD/ST/ATOM 指令。本测试程序（MOV32I + EXIT）不访存，故所有
    //   L1D 输入接 0、输出悬空、下游 ack 接常 1。保留例化是为了让
    //   仿真器能综合模块层级，便于后续扩展访存测试程序。
    //   详见 l1d.sv 的实现说明。
    // =========================================================
    l1d #(.NUM_WARPS(NUM_WARPS)) u_l1d (
        .clk(clk), .rst_n(rst_n),
        .req_valid(1'b0),
        .req_is_write(1'b0),
        .req_addr(64'h0),
        .req_sectors(4'h0),
        .req_wdata(32'h0),
        .req_be(4'h0),
        // 教学注记 - enum 常量直连（02 §2.2 / NutShellGPU_pkg.sv）：
        //   SP_GLOBAL/W_U32/HINT_CA 是 NutShellGPU_pkg.sv 中 enum 的元素，
        //   其位宽与 l1d.sv 端口声明严格匹配（3/4/3 位）。
        //   若 l1d.sv 把这些端口写成 [7:0]，此处直连会触发 vopt-2241
        //   "Connection width does not match" 警告（3→8 位零扩展在
        //   仿真中高位变 X，导致 L1D 内部 `req_space == SP_GLOBAL`
        //   比较失败）。保持 enum 位宽 = 端口位宽即可消除。
        .req_space(SP_GLOBAL),   // 3 位，space_e::GLOBAL
        .req_width(W_U32),       // 4 位，width_e::U32
        .req_hint(HINT_CA),      // 3 位，hint_e::CA（逐级缓存）
        // 教学注记 - 字面量必须显式标宽：
        //   裸字面量 0 在 SystemVerilog 中默认 32 位整数，而 req_warp
        //   端口是 $clog2(NUM_WARPS)-1:0 = [5:0]（6 位）。32→6 的
        //   截断会触发 vopt-2241 警告。用 6'd0 显式标 6 位即可消除。
        //   注意：若 NUM_WARPS 参数被覆盖为其他值，此处宽度也需
        //   同步修改（$clog2 的结果会变）。
        .req_warp(6'd0),
        .req_dst_reg(8'h0),
        .req_pc(64'h0),
        .req_subid(16'h0),
        .req_accept(),
        .req_replay(),
        .req_miss(),
        .fill_valid(1'b0),
        .fill_addr(64'h0),
        .fill_data(128'h0),
        .fill_sectors(4'h0),
        .fill_subid(16'h0),
        .fill_ack(),
        .noc_req_valid(),
        .noc_req_addr(),
        .noc_req_is_write(),
        .noc_req_sectors(),
        .noc_req_wdata(),
        .noc_req_subid(),
        .noc_req_ack(1'b1),
        .ld_resp_valid(),
        .ld_resp_warp(),
        .ld_resp_dst_reg(),
        .ld_resp_data(),
        .ld_resp_pc(),
        .stat_access_lines(),
        .stat_miss_sectors(),
        .stat_prt_full(),
        .stat_assoc_stall(),
        .stat_wdb_full()
    );

    // =========================================================================
    // 组合逻辑：SIMT 栈操作触发（PS_WRITEBACK 阶段产生，下拍处理）
    // =========================================================================
    // 教学注记 - 为什么 SIMT 操作在 WRITEBACK 阶段触发：
    //   指令执行完毕、结果写回 RF 后，才确定下一步 PC 走向：
    //     - 普通指令：顺序推进 PC + 8（op_advance）
    //     - SSY 指令：登记再收敛点（op_ssy）
    //     - BRA 指令：分支跳转（op_bra_uni 或 op_bra_div）
    //     - EXIT 指令：从 active mask 中移除已退出 lane（op_exit）
    //   这些 op_* 信号是组合逻辑产生的脉冲，下个时钟沿被 SIMT 栈采样，
    //   栈内部状态机处理后更新 TOS（影响下一拍 PS_FETCH 的取指地址）。
    //
    // 教学注记 - branch_target() 函数（NutShellGPU_pkg.sv）：
    //   计算分支目标地址 = cur_pc + sign_extend(imm26 << 3) + x_offset。
    //   imm26 是 26 位偏移，左移 3 位变字节偏移，再加 PC 得绝对目标。
    // =========================================================================
    always_comb begin
        // 默认：所有 SIMT 操作无效（避免锁存）
        // 注：simt_op_init 在另一组合块中驱动
        simt_op_ssy      = 1'b0;
        simt_ssy_target  = 64'h0;
        simt_op_bra_uni  = 1'b0;
        simt_bra_taken   = 1'b0;
        simt_bra_target  = 64'h0;
        simt_op_bra_div  = 1'b0;
        simt_div_target  = 64'h0;
        simt_taken_mask  = 32'h0;
        simt_fall_mask   = 32'h0;
        simt_op_advance  = 1'b0;
        simt_advance_newpc = 64'h0;
        simt_op_exit     = 1'b0;
        simt_exit_mask   = 32'h0;

        // 仅在 WRITEBACK 阶段触发 SIMT 操作
        if (pipe_state == PS_WRITEBACK) begin
            case (cur_inst.op)
                OP_SSY: begin
                    // SSY：登记再收敛点（IPDOM），为后续 BRA DIV 准备
                    simt_op_ssy = 1'b1;
                    simt_ssy_target = branch_target(cur_pc, cur_inst.imm26, cur_inst.x[0]);
                end
                OP_BRA: begin
                    // BRA：本简化实现按统一分支处理（全 warp 跳转）
                    // 真实分歧分支需根据 lane 谓词计算 taken_mask / fall_mask，
                    //   走 op_bra_div 路径（见 03 §8）
                    simt_op_bra_uni = 1'b1;
                    simt_bra_taken = 1'b1;
                    simt_bra_target = branch_target(cur_pc, cur_inst.imm26, cur_inst.x[0]);
                end
                OP_EXIT: begin
                    // EXIT：从 active mask 中移除当前 cur_mask 范围内的 lane
                    //   当 mask 清空后 simt_warp_exit_done 拉高，warp 终结
                    simt_op_exit = 1'b1;
                    simt_exit_mask = cur_mask;
                end
                default: begin
                    // 普通指令：顺序推进 PC = PC + 8（每条指令 8 字节）
                    simt_op_advance = 1'b1;
                    simt_advance_newpc = cur_pc + 64'd8;
                end
            endcase
        end
    end

    // =========================================================================
    // 组合逻辑：记分牌信号（chk / alloc / release）
    // =========================================================================
    // 教学注记 - 本块三个职责：
    //   (1) 设置源操作数 s0..s3（供记分牌检查 RAW 依赖）；
    //   (2) 产生 sb_alloc 脉冲（PS_ISSUE 拍分配记分牌项）；
    //   (3) 产生 sb_release 脉冲（PS_WRITEBACK 拍释放记分牌项）。
    //
    // 教学注记 - 源操作数如何决定（02 册指令格式 + 03 §5）：
    //   不同指令格式（FMT_R/I/M/MI）的源操作数寄存器字段位置不同。本块
    //   按 cur_inst.fmt 解析出最多 4 个源（a/c/d 字段或其变体）。
    //   另外若指令带 guard 谓词（guard_en=1），谓词寄存器也作为源 s3 检查。
    //   RZ（寄存器 255）被定义为 zero constant，读它不需等依赖——故
    //   valid=0 跳过检查。
    // =========================================================================
    always_comb begin
        // 检查使能：DECODE 或 ISSUE 阶段都需要查依赖
        sb_chk_valid = (pipe_state == PS_DECODE) || (pipe_state == PS_ISSUE);

        // 源操作数默认清零（避免锁存）
        sb_s0_reg = 8'h0; sb_s0_is_pred = 1'b0; sb_s0_valid = 1'b0;
        sb_s1_reg = 8'h0; sb_s1_is_pred = 1'b0; sb_s1_valid = 1'b0;
        sb_s2_reg = 8'h0; sb_s2_is_pred = 1'b0; sb_s2_valid = 1'b0;
        sb_s3_reg = 8'h0; sb_s3_is_pred = 1'b0; sb_s3_valid = 1'b0;

        // 若指令带 guard 谓词，则把谓词寄存器作为 s3 源参与检查
        if (cur_inst.guard_en) begin
            sb_s3_reg = {5'b0, cur_inst.guard_pred};
            sb_s3_is_pred = 1'b1;
            sb_s3_valid = 1'b1;
        end

        // 按指令格式解析源操作数（FMT_R/I/M/MI，02 册 ABI）
        case (cur_inst.fmt)
            FMT_R: begin
                // R 型：源 a、源 c
                sb_s0_reg = cur_inst.a;
                sb_s0_is_pred = 1'b0;
                sb_s0_valid = (cur_inst.a != RZ);  // RZ 是 zero，不查依赖
                sb_s1_reg = cur_inst.c;
                sb_s1_is_pred = 1'b0;
                sb_s1_valid = (cur_inst.c != RZ);
                // 位插入/提取 BFI/BFE 还需读 d 的旧值（read-modify-write）
                if (cur_inst.op == OP_BFI || cur_inst.op == OP_BFE) begin
                    sb_s2_reg = cur_inst.d;
                    sb_s2_valid = (cur_inst.d != RZ);
                end
            end
            FMT_I: begin
                // I 型：LOP3 用 a、c 作源；MOV32I/SETPI 无寄存器源（仅立即数）
                if (cur_inst.op == OP_LOP3) begin
                    sb_s0_reg = cur_inst.a;
                    sb_s0_valid = (cur_inst.a != RZ);
                    sb_s1_reg = cur_inst.c;
                    sb_s1_valid = (cur_inst.c != RZ);
                end
                // MOV32I/SETPI: 无寄存器源
            end
            FMT_M, FMT_MI: begin
                // M/MI 型：访存指令。源 a 是基地址寄存器
                sb_s0_reg = cur_inst.a;
                sb_s0_valid = (cur_inst.a != RZ);
                if (cur_inst.fmt == FMT_M) begin
                    // FMT_M 还需源 c 作偏移索引
                    sb_s1_reg = cur_inst.c;
                    sb_s1_valid = (cur_inst.c != RZ);
                end
                if (cur_inst.op == OP_ST || cur_inst.op == OP_ST128) begin
                    // ST/ST128：源 d 是待写数据
                    sb_s2_reg = cur_inst.d;
                    sb_s2_valid = (cur_inst.d != RZ);
                end
                if (cur_inst.op == OP_ATOM) begin
                    // ATOM：源 c 是待写数据（read-modify-write）
                    sb_s2_reg = cur_inst.c;
                    sb_s2_valid = (cur_inst.c != RZ);
                end
            end
            default: ;
        endcase

        // 分配脉冲：PS_ISSUE 拍且 can_issue 且有目的寄存器且 alloc_ok
        sb_alloc = (pipe_state == PS_ISSUE) && sb_can_issue &&
                   has_destination(cur_inst.op) && sb_alloc_ok;
        sb_dst_reg = cur_inst.d;
        sb_dst_is_pred = (cur_inst.op == OP_SETP);  // SETP 写谓词寄存器

        // 释放脉冲：PS_WRITEBACK 拍且有目的寄存器
        sb_release = (pipe_state == PS_WRITEBACK) &&
                     has_destination(cur_inst.op) && (cur_inst.d != RZ);
        sb_rel_reg = cur_inst.d;
        sb_rel_is_pred = (cur_inst.op == OP_SETP);
    end

    // =========================================================
    // 指令存储器加载（时序逻辑）
    // =========================================================
    // 教学注记：仿真时由 tb 逐字灌入指令。复位时全清零，imem_load=1 时
    //   按 imem_addr 写入 imem_data。真实硬件由驱动经 I-Cache 路径加载。
    // =========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < IMEM_SIZE; i++) begin
                imem[i] <= 64'h0;
            end
        end else if (imem_load) begin
            imem[imem_addr] <= imem_data;
        end
    end

    // =========================================================================
    // 流水线主状态机（main controller）
    // =========================================================================
    // 教学注记 - 整体时序约定：
    //   复位异步（negedge rst_n）；每拍 cyc_cnt++ 计周期数。
    //   状态机从 PS_IDLE 出发，经 PS_FETCH→PS_DECODE→PS_ISSUE→PS_READ→
    //   PS_EXECUTE→PS_WRITEBACK 一条直线走完一条指令，再回 PS_FETCH 取下一条。
    //   这是简化的"单条指令贯穿"模型——真实 GPU 是流水化的，多条指令同时
    //   在不同级中推进（详见 NutShellGPU_pkg.sv 的 LAT_* 延迟常量）。
    //
    // 教学注记 - 复位清零范围：
    //   复位时清零所有 warp 状态、cur_* 上下文、统计计数器，以及整个 RF（2048
    //   × 256 × 4B = 2 MiB）和谓词寄存器。真实硬件不会复位整个 RF（面积代价
    //   太大），而是用 valid 位或依赖 ISA 保证——这里为教学简化全清零。
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_state <= PS_IDLE;
            for (int w = 0; w < NUM_WARPS; w++) begin
                warp_valid[w] <= 1'b0;
                warp_exit[w]  <= 1'b0;
            end
            cur_pc <= 64'h0;
            cur_inst <= '{default: 0};
            cur_mask <= 32'h0;
            cur_warp <= 0;
            exec_latency_cnt <= 0;
            exec_result <= 32'h0;
            exec_done <= 1'b0;
            inst_cnt <= 0;
            cyc_cnt <= 0;
            // 复位整个 RF（教学简化；真实硬件不会这样做）
            for (int t = 0; t < RF_THREADS; t++) begin
                for (int r = 0; r < 256; r++) begin
                    rf[t][r] <= 32'h0;
                end
                for (int p = 0; p < 7; p++) begin
                    pred[t][p] <= 1'b0;
                end
            end
        end else begin
            cyc_cnt <= cyc_cnt + 1;

            case (pipe_state)
                // ---------------------------------------------
                // PS_IDLE：等待 warp_init 信号启动一个 warp
                // ---------------------------------------------
                PS_IDLE: begin
                    if (warp_init) begin
                        // 注：simt_op_init 在下方的 always_comb 中产生
                        //   = (PS_IDLE) && warp_init
                        //   SIMT 栈在 IDLE→FETCH 边沿被初始化（推入 entry TOS）
                        warp_valid[warp_init_id] <= 1'b1;
                        warp_exit[warp_init_id]  <= 1'b0;
                        cur_warp <= warp_init_id;
                        pipe_state <= PS_FETCH;
                    end
                end

                // ---------------------------------------------
                // PS_FETCH：从 IMEM 取指令，decode 入 cur_inst
                // ---------------------------------------------
                PS_FETCH: begin
                    // 检查 warp 是否已 EXIT 完成（SIMT 栈处理后 mask 清空）
                    if (simt_warp_exit_done) begin
                        warp_exit[cur_warp] <= 1'b1;
                        warp_valid[cur_warp] <= 1'b0;
                        pipe_state <= PS_IDLE;
                    end else if (warp_valid[cur_warp] && !warp_exit[cur_warp]) begin
                        // 从 SIMT 栈 TOS 取下一条 PC
                        // I-cache 简化为单周期命中（无 miss 建模）
                        // 字地址 = nextpc[10:3]（nextpc 是字节地址，右移 3 位）
                        cur_pc <= simt_tos_nextpc;
                        cur_mask <= simt_tos_mask;
                        cur_inst <= decode_inst(imem[simt_tos_nextpc[10:3]]);
                        pipe_state <= PS_DECODE;
                    end
                end

                // ---------------------------------------------
                // PS_DECODE：指令已解码；此拍组合逻辑设置记分牌源
                // ---------------------------------------------
                // 教学注记：本简化实现 decode 在 FETCH 拍完成；DECODE 拍仅
                //   让 always_comb 的记分牌源操作数稳定一拍，随后进 ISSUE。
                //   真实硬件的译码逻辑更复杂（多周期、解码 ROM 等）。
                PS_DECODE: begin
                    pipe_state <= PS_ISSUE;
                end

                // ---------------------------------------------
                // PS_ISSUE：查记分牌 can_issue；通过则发射
                // ---------------------------------------------
                PS_ISSUE: begin
                    if (sb_can_issue) begin
                        // 发射：组合逻辑同时产生 sb_alloc 脉冲分配记分牌项
                        //   （防止后续指令读目的寄存器读到旧值）
                        inst_cnt <= inst_cnt + 1;
                        pipe_state <= PS_READ;
                    end
                    // 不能发射则停留 PS_ISSUE（stall，等依赖解除）
                end

                // ---------------------------------------------
                // PS_READ：Operand Collector 收集源操作数
                // ---------------------------------------------
                // 教学注记：本简化实现假设操作数立即就绪——MOV32I 的立即数
                //   直接来自 cur_inst.imm32，EXIT 无需操作数。真实硬件此阶段
                //   经 4-bank RF 仲裁收集（见 operand_collector.sv），可能
                //   因 bank 冲突而多拍。
                PS_READ: begin
                    exec_latency_cnt <= 0;
                    exec_done <= 1'b0;
                    pipe_state <= PS_EXECUTE;
                end

                // ---------------------------------------------
                // PS_EXECUTE：按 opcode/pipe 跑对应延迟
                // ---------------------------------------------
                // 教学注记 - 执行模型：
                //   每个操作码有固定延迟 LAT_*（见 NutShellGPU_pkg.sv）：
                //     SP/INT 流水线 (MOV/IADD/ISUB/LOP/MOV32I) = 4 周期
                //     SFU（特殊函数单元）= 20 周期
                //     BRU（分支单元, EXIT/SSY/BRA/YIELD/BRKPT）= 1 周期
                //   本实现用 exec_latency_cnt 倒数：第 0 拍算出 exec_result，
                //   等到 cnt >= LAT-1 时 exec_done=1，进 WRITEBACK。
                //   真实硬件是流水化输出（每拍吐一条），延迟是吞吐延迟。
                //   另外本实现简化为只读 lane 0 的寄存器（rf[cur_warp*32]），
                //   真实硬件是 32 lane 并行。
                PS_EXECUTE: begin
                    case (cur_inst.op)
                        OP_MOV32I: begin
                            // MOV32I：d ← imm32（SP，4 周期）
                            if (exec_latency_cnt == 0) begin
                                exec_result <= cur_inst.imm32;
                            end
                            if (exec_latency_cnt >= LAT_SP - 1) begin
                                exec_done <= 1'b1;
                            end
                        end
                        OP_MOV: begin
                            // MOV：d ← a（SP，4 周期）
                            if (exec_latency_cnt == 0) begin
                                // 简化：只读 lane 0 的 RF
                                exec_result <= rf[cur_warp * NUM_LANES][cur_inst.a];
                            end
                            if (exec_latency_cnt >= LAT_SP - 1) begin
                                exec_done <= 1'b1;
                            end
                        end
                        OP_IADD: begin
                            // IADD：d ← a + c（SP，4 周期）
                            if (exec_latency_cnt == 0) begin
                                exec_result <= rf[cur_warp * NUM_LANES][cur_inst.a]
                                             + rf[cur_warp * NUM_LANES][cur_inst.c];
                            end
                            if (exec_latency_cnt >= LAT_SP - 1) begin
                                exec_done <= 1'b1;
                            end
                        end
                        OP_ISUB: begin
                            // ISUB：d ← a - c（SP，4 周期）
                            if (exec_latency_cnt == 0) begin
                                exec_result <= rf[cur_warp * NUM_LANES][cur_inst.a]
                                             - rf[cur_warp * NUM_LANES][cur_inst.c];
                            end
                            if (exec_latency_cnt >= LAT_SP - 1) begin
                                exec_done <= 1'b1;
                            end
                        end
                        OP_LOP: begin
                            // LOP：逻辑运算 d ← a OP c，由 x[1:0] 选 OP（SP，4 周期）
                            if (exec_latency_cnt == 0) begin
                                case (cur_inst.x[1:0])
                                    2'd0: exec_result <= rf[cur_warp * NUM_LANES][cur_inst.a] & rf[cur_warp * NUM_LANES][cur_inst.c];   // AND
                                    2'd1: exec_result <= rf[cur_warp * NUM_LANES][cur_inst.a] | rf[cur_warp * NUM_LANES][cur_inst.c];   // OR
                                    2'd2: exec_result <= rf[cur_warp * NUM_LANES][cur_inst.a] ^ rf[cur_warp * NUM_LANES][cur_inst.c];   // XOR
                                    2'd3: exec_result <= rf[cur_warp * NUM_LANES][cur_inst.a] & ~rf[cur_warp * NUM_LANES][cur_inst.c];  // ANDN
                                endcase
                            end
                            if (exec_latency_cnt >= LAT_SP - 1) begin
                                exec_done <= 1'b1;
                            end
                        end
                        OP_EXIT, OP_NOP, OP_SSY, OP_BRA, OP_YIELD, OP_BRKPT: begin
                            // BRU 类指令：1 周期（控制流，无算术结果）
                            exec_done <= 1'b1;
                        end
                        default: begin
                            // 未实现的指令：默认 1 周期完成（占位）
                            exec_done <= 1'b1;
                        end
                    endcase

                    // 延迟未到则继续等待，到则进 WRITEBACK
                    if (!exec_done) begin
                        exec_latency_cnt <= exec_latency_cnt + 1;
                    end else begin
                        pipe_state <= PS_WRITEBACK;
                    end
                end

                // ---------------------------------------------
                // PS_WRITEBACK：写回 RF + 推进 SIMT 栈
                // ---------------------------------------------
                // 教学注记 - W 阶段两件事：
                //   (1) RF 写回：对有目的寄存器的指令，把 exec_result 写入
                //       所有 active lane（cur_mask[l]=1）的 rf[warp*32+l][d]。
                //       SIMT 模型的核心——同一条指令写多个 lane 的对应寄存器。
                //   (2) SIMT 栈推进：组合逻辑的 op_* 信号在本拍产生，下拍
                //       被 SIMT 栈采样更新 TOS——下一拍 PS_FETCH 用新 TOS 取指。
                //       状态机本身直接回 PS_FETCH（取下一条指令）。
                PS_WRITEBACK: begin
                    // 写回 RF（仅对有目的寄存器的指令，且目的非 RZ）
                    if (has_destination(cur_inst.op) && cur_inst.d != RZ) begin
                        // 遍历 32 lane，仅对 cur_mask[l]=1 的 lane 写入
                        for (int l = 0; l < NUM_LANES; l++) begin
                            if (cur_mask[l]) begin
                                rf[cur_warp * NUM_LANES + l][cur_inst.d] <= exec_result;
                            end
                        end
                    end

                    // SIMT 栈操作已在上方 always_comb 组合块产生（op_advance/
                    //   op_exit/op_bra/op_ssy），下拍被 SIMT 栈采样处理
                    // 回 FETCH 取下一条指令
                    pipe_state <= PS_FETCH;
                end

                // ---------------------------------------------
                default: pipe_state <= PS_IDLE;  // 防御性：未知状态回 IDLE
            endcase
        end
    end

    // =========================================================================
    // 组合逻辑：SIMT 栈 init 操作触发
    // =========================================================================
    // 教学注记：simt_op_init 在 PS_IDLE 拍 && warp_init 时拉高，触发 SIMT 栈
    //   推入一个初始 TOS 项（nextpc=warp_init_entry_pc, mask=warp_init_mask）。
    //   与其他 op_*（ssy/bra/advance/exit）不同——那些在 WRITEBACK 拍产生，
    //   而 init 在 IDLE 拍产生（因为 warp 启动发生在 IDLE 状态）。
    // =========================================================================
    always_comb begin
        simt_op_init = (pipe_state == PS_IDLE) && warp_init;
    end

    // =========================================================================
    // 输出赋值
    // =========================================================================
    // 教学注记：warp_done 固定监视 warp 0 的退出状态（tb 用的就是 warp 0）；
    //   多 warp 系统应聚合所有 warp 的 exit 位。
    assign warp_done = warp_exit[0];   // warp 0 已 EXIT
    assign inst_count = inst_cnt;       // 已发射指令数
    assign cycle_count = cyc_cnt;       // 已运行周期数
    assign cur_pc_out = cur_pc;         // 当前 PC（调试观察）

endmodule
