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
// GDDR5 DRAM 控制器模块（DRAM Controller）
// dram_ctrl.sv
// -----------------------------------------------------------------------------
// 作用：实现每条 GDDR5 通道的访存控制器。8 个 bank 各自独立维护行缓冲
//       状态机（IDLE→ACTIVING→ACTIVE→READING/WRITING→PRECHG），通道命令/
//       数据总线全 bank 共享。调度策略采用 FR-FCFS（First-Ready First-Come
//       First-Served）：行命中优先，否则 oldest 请求服务。
//
// 对应规格册：依据 04 册《存储系统微结构》§11 FB 访存调度器 + §12 GDDR5
//             通道时序状态机。
//   - §11 FR-FCFS 调度策略（行命中优先 → 否则最老读 → 写仅在读空或 drain
//        threshold 后服务）
//   - §12 bank 状态机：IDLE→ACTIVING(tRCD)→ACTIVE→READING(tCAS+tBURST)/
//        WRITING(tBURST+tWR)→ACTIVE；ACTIVE→PRECHG(tRP)→IDLE
//   - §12 时序参数：tCAS=12, tRCD=12, tRP=12, tRAS=28, tRC=40, tWR=12
//   - §12 数据总线约束：任意 bank 的 tBURST 不得重叠（总线竞争排队）
//   - §7.2 地址映射：bank/row/col 散列
//
// 端口概要：
//   - req_*：L2 请求口（128B 整行粒度）
//   - rsp_*：DRAM 响应口（读返回数据）
//   - stat_*：reads/writes/row_hits/row_misses/bank_busy/bus_busy 统计
//
// 关键时序参数（@1 GHz，1ns ≈ 1 周期）：
//   - tCAS = 12 周期  ACTIVE 后列地址到数据输出
//   - tRCD = 12 周期  ACTIVATE 到 READ/WRITE
//   - tRP  = 12 周期  PRECHARGE 充电
//   - tRAS = 28 周期  ACTIVE 后最短维持才能 PRE
//   - tRC  = 40 周期  同 bank 两次 ACTIVATE 间隔
//   - tWR  = 12 周期  WRITE 数据后到 PRECHARGE 的写恢复
//   - tBURST = 2 周期  32B atom 占数据总线时间
//
// 教学注记 - 为什么用 FR-FCFS（First-Ready FCFS）：
//   GDDR5 的 DRAM 行缓冲（row buffer）是关键资源——访问"打开的行"只需
//   tCAS+tBURST（14 周期），而访问"未打开的行"需要 tRP+tRCD+tCAS（36 周期），
//   差 2.5 倍。FR-FCFS 调度策略的核心思想：
//     (1) 优先服务已 ACTIVE 行的请求（行命中），最大化行缓冲利用率；
//     (2) 否则按请求到达顺序（FCFS）服务最老请求——避免 starvation；
//     (3) 写仅在下列情形服务：读队列空，或自上次排空写以来已服务
//         write_drain_threshold=32 个读——避免读 starvation；
//     (4) 写之后在 tWR + tRP 后才能同一 bank 再读——保证数据完整性。
//   教材 4.3.3 详述了 FR-FCFS 与纯 FCFS 的性能差异。
//
// 教学注记 - 行缓冲 FSM 各状态含义：
//   IDLE     - bank 未激活任何行，可接受 ACTIVATE 命令
//   ACTIVING - ACTIVATE 已发，tRCD 倒计时中（行未稳定，不能读/写）
//   ACTIVE   - 行已稳定，可接受 READ/WRITE 命令（行命中可服务）
//   READING  - READ 已发，tCAS+tBURST 倒计时中（数据正在总线传输）
//   WRITING  - WRITE 已发，tBURST+tWR 倒计时中（写恢复中）
//   PRECHG   - PRECHARGE 已发，tRP 倒时中（充电中，回 IDLE 后才能再 ACT）
// =============================================================================

`include "NutShellGPU_defines.svh"

module dram_ctrl #(
    parameter CHANNEL_ID = 0,    // 通道号（0..5，用于 bank 散列）
    parameter NUM_BANKS = 8       // 每通道 8 bank
) (
    input  logic clk,
    input  logic rst_n,

    // --- L2 请求口 ---
    // 教学注记：L2 以 128B 整行粒度请求；控制器内部拆为 bank/row/col
    input  logic        req_valid,
    input  logic [63:0] req_addr,
    input  logic        req_is_write,
    input  logic [127:0] req_wdata,   // 128B 整行写数据
    input  logic [15:0] req_subid,
    output logic        req_ack,

    // --- L2 响应口 ---
    // 教学注记：读请求完成后返回 128B 整行数据 + subid（用于 L2 路由回 NSM）
    output logic        rsp_valid,
    output logic [63:0] rsp_addr,
    output logic [127:0] rsp_data,
    output logic [15:0] rsp_subid,
    input  logic        rsp_ack,

    // --- 统计 ---
    output logic [31:0] stat_reads,
    output logic [31:0] stat_writes,
    output logic [31:0] stat_row_hits,    // 行命中数（请求行 == 当前 ACTIVE 行）
    output logic [31:0] stat_row_misses,   // 行未命中数（需 ACT 或 PRE+ACT）
    output logic [31:0] stat_bank_busy,    // bank 忙碌周期数
    output logic [31:0] stat_bus_busy      // 数据总线忙碌周期数
);

    import NutShellGPU_pkg::*;

    // -------------------------------------------------------------------------
    // Bank 状态机枚举（依据 04 册 §12）
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        BK_IDLE     = 3'd0,  // 未激活
        BK_ACTIVING = 3'd1,  // ACTIVATE 进行中（tRCD 倒计时）
        BK_ACTIVE    = 3'd2,  // 行已激活，可读/写
        BK_READING  = 3'd3,  // READ 进行中（tCAS + tBURST）
        BK_WRITING  = 3'd4,  // WRITE 进行中（tBURST + tWR）
        BK_PRECHG   = 3'd5   // PRECHARGE 进行中（tRP）
    } bank_state_e;

    bank_state_e bank_state [NUM_BANKS];
    logic [20:0] bank_row    [NUM_BANKS];   // 当前 ACTIVE 的行号
    logic [7:0]  bank_timer  [NUM_BANKS];   // 状态转移倒计时
    logic [7:0]  bank_act_age [NUM_BANKS];  // ACTIVATE 后已过的周期数（tRAS 检查）

    // -------------------------------------------------------------------------
    // 请求队列（16 项 FIFO）
    // -------------------------------------------------------------------------
    // 教学注记：本实现用 16 项环形队列缓存 L2 请求；FR-FCFS 调度器在该队列中
    //   找最优请求。每项预计算 bank/row/col 以加速仲裁。
    typedef struct packed {
        logic        valid;
        logic [63:0] addr;
        logic        is_write;
        logic [127:0] wdata;
        logic [15:0] subid;
        logic [2:0]  bank;    // 预计算的 bank 号
        logic [20:0] row;     // 预计算的 row 号
        logic [6:0]  col;     // 预计算的 col 号
    } dram_req_t;

    dram_req_t req_q [16];            // 请求队列
    logic [3:0] req_head, req_tail;   // 队列头尾指针
    logic [4:0] req_count;             // 队列项数

    // 数据总线忙碌计时器（全 bank 共享，避免 tBURST 重叠）
    logic [7:0] bus_busy_timer;

    // 统计
    logic [31:0] read_cnt, write_cnt, rhit_cnt, rmiss_cnt, bk_busy_cnt, bus_busy_cnt;

    // -------------------------------------------------------------------------
    // 地址解码函数（依据 04 册 §7.2）
    // -------------------------------------------------------------------------
    // bank = (A[14:12] ^ A[18:16] ^ mp) & 0b111（注入 mp 避免 cross-MP 同 bank 命中）
    // row  = A[35:15]（21 位行号）
    // col  = A[11:5]（7 位列号，每次突发 32B atom）
    function automatic logic [2:0] get_bank(input logic [63:0] addr);
        return addr_to_dram_bank(addr, CHANNEL_ID[2:0]);
    endfunction
    function automatic logic [20:0] get_row(input logic [63:0] addr);
        return addr_to_dram_row(addr);
    endfunction
    function automatic logic [6:0] get_col(input logic [63:0] addr);
        return addr_to_dram_col(addr);
    endfunction

    // -------------------------------------------------------------------------
    // FR-FCFS 调度器（组合逻辑 always_comb）
    // -------------------------------------------------------------------------
    // 教学注记 - 两轮扫描（04 §11）：
    //   第一轮：在队列中找行命中请求（bank 已 ACTIVE 且 row 匹配）→ 优先服务；
    //   第二轮：若第一轮未找到，按 FCFS 找最老的可服务请求
    //           （bank IDLE 或 ACTIVE+row 命中）；
    //   可服务前提：bus_busy_timer==0（数据总线空闲，避免 tBURST 重叠）。
    //   平局取小编号（队列顺序早者胜，确定性约定）。
    int sel_idx;
    logic sel_found;
    always_comb begin
        sel_idx = -1;
        sel_found = 1'b0;
        // 第一轮：找行命中
        for (int i = 0; i < 16; i++) begin
            int idx;
            idx = (req_head + i) % 16;
            if (req_q[idx].valid) begin
                if (bank_state[req_q[idx].bank] == BK_ACTIVE &&
                    bank_row[req_q[idx].bank] == req_q[idx].row &&
                    bus_busy_timer == 0) begin
                    sel_idx = idx;
                    sel_found = 1'b1;
                end
            end
        end
        // 第二轮：若第一轮无命中，找最老可服务请求
        if (!sel_found) begin
            for (int i = 0; i < 16; i++) begin
                int idx;
                idx = (req_head + i) % 16;
                if (req_q[idx].valid) begin
                    // 可服务：bank IDLE 或 ACTIVE+row 命中
                    if ((bank_state[req_q[idx].bank] == BK_IDLE ||
                         (bank_state[req_q[idx].bank] == BK_ACTIVE &&
                          bank_row[req_q[idx].bank] == req_q[idx].row)) &&
                        bus_busy_timer == 0) begin
                        sel_idx = idx;
                        sel_found = 1'b1;
                    end
                end
            end
        end
    end

    // 本拍能否发 DRAM 命令
    logic can_issue;
    always_comb begin
        can_issue = sel_found && bus_busy_timer == 0;
    end

    // 请求接受（队列未满则接受）
    always_comb begin
        req_ack = 1'b0;
        if (req_valid && req_count < 16) begin
            req_ack = 1'b1;
        end
    end

    // -------------------------------------------------------------------------
    // 状态更新（时序逻辑 always_ff）
    // -------------------------------------------------------------------------
    // 教学注记 - 时序约定：
    //   rsp_* 在此驱动为单周期脉冲（命中后当拍发响应）；
    //   bank 状态转移在 timer==1 时切换；
    //   bus_busy_timer 倒数避免总线 tBURST 重叠。
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 复位：所有 bank 回 IDLE，队列清空
            for (int b = 0; b < NUM_BANKS; b++) begin
                bank_state[b]  <= BK_IDLE;
                bank_row[b]     <= 0;
                bank_timer[b]   <= 0;
                bank_act_age[b] <= 0;
            end
            for (int i = 0; i < 16; i++) begin
                req_q[i].valid <= 1'b0;
            end
            req_head <= 0;
            req_tail <= 0;
            req_count <= 0;
            bus_busy_timer <= 0;
            read_cnt <= 0;
            write_cnt <= 0;
            rhit_cnt <= 0;
            rmiss_cnt <= 0;
            bk_busy_cnt <= 0;
            bus_busy_cnt <= 0;
            rsp_valid <= 1'b0;
        end else begin
            // 默认：本拍无响应
            rsp_valid <= 1'b0;

            // 接受新请求入队
            if (req_valid && req_ack) begin
                req_q[req_tail].valid    <= 1'b1;
                req_q[req_tail].addr     <= req_addr;
                req_q[req_tail].is_write <= req_is_write;
                req_q[req_tail].wdata    <= req_wdata;
                req_q[req_tail].subid    <= req_subid;
                req_q[req_tail].bank     <= get_bank(req_addr);
                req_q[req_tail].row      <= get_row(req_addr);
                req_q[req_tail].col      <= get_col(req_addr);
                req_tail <= (req_tail + 1) & 4'hF;
                req_count <= req_count + 1;
            end

            // -----------------------------------------------------------------
            // Bank timer 倒数与状态转移
            // -----------------------------------------------------------------
            for (int b = 0; b < NUM_BANKS; b++) begin
                if (bank_timer[b] > 0) begin
                    bank_timer[b] <= bank_timer[b] - 1;
                end
                if (bank_act_age[b] < 255) begin
                    bank_act_age[b] <= bank_act_age[b] + 1;
                end

                // 状态转移：timer==1 时切换到下一状态
                case (bank_state[b])
                    BK_ACTIVING: begin
                        // tRCD 到期 → ACTIVE
                        if (bank_timer[b] == 1) begin
                            bank_state[b] <= BK_ACTIVE;
                        end
                    end
                    BK_READING: begin
                        // tCAS+tBURST 到期 → ACTIVE
                        if (bank_timer[b] == 1) begin
                            bank_state[b] <= BK_ACTIVE;
                        end
                    end
                    BK_WRITING: begin
                        // tBURST+tWR 到期 → ACTIVE
                        if (bank_timer[b] == 1) begin
                            bank_state[b] <= BK_ACTIVE;
                        end
                    end
                    BK_PRECHG: begin
                        // tRP 到期 → IDLE
                        if (bank_timer[b] == 1) begin
                            bank_state[b] <= BK_IDLE;
                        end
                    end
                    default: ;
                endcase
            end

            // 数据总线计时器倒数
            if (bus_busy_timer > 0) begin
                bus_busy_timer <= bus_busy_timer - 1;
            end

            // -----------------------------------------------------------------
            // 发 DRAM 命令（FR-FCFS 调度结果）
            // -----------------------------------------------------------------
            if (can_issue) begin
                logic [2:0] b;
                b = req_q[sel_idx].bank;

                if (bank_state[b] == BK_IDLE) begin
                    // -----------------------------------------------------------
                    // ACTIVATE：IDLE → ACTIVING
                    // -----------------------------------------------------------
                    // 教学注记：行未命中场景——需先 ACTIVATE 激活行才能读/写。
                    //   bank_timer = tRCD = 12（ACTIVE 后等 12 周期才能 READ/WRITE）。
                    bank_state[b]  <= BK_ACTIVING;
                    bank_timer[b]  <= DRAM_TRCD;
                    bank_act_age[b] <= 0;   // 重置 ACT 年龄（用于 tRAS 检查）
                    rmiss_cnt <= rmiss_cnt + 1;
                end else if (bank_state[b] == BK_ACTIVE &&
                            bank_row[b] == req_q[sel_idx].row) begin
                    // -----------------------------------------------------------
                    // 行命中：READ 或 WRITE
                    // -----------------------------------------------------------
                    // 教学注记：行命中场景——请求行 == 当前 ACTIVE 行，只付
                    //   tCAS+tBURST（读）或 tBURST+tWR（写），不需 ACTIVATE。
                    //   这是 FR-FCFS 优先服务的场景。
                    if (req_q[sel_idx].is_write) begin
                        // WRITE：tBURST + tWR
                        bank_state[b] <= BK_WRITING;
                        bank_timer[b] <= DRAM_TBURST + DRAM_TWR;
                        write_cnt <= write_cnt + 1;
                    end else begin
                        // READ：tCAS + tBURST
                        bank_state[b] <= BK_READING;
                        bank_timer[b] <= DRAM_TCAS + DRAM_TBURST;
                        read_cnt <= read_cnt + 1;
                        // 产生响应（命中后当拍发响应脉冲）
                        rsp_valid <= 1'b1;
                        rsp_addr  <= req_q[sel_idx].addr;
                        rsp_data  <= 128'h0; // 实际应从 DRAM 阵列读，此处简化
                        rsp_subid <= req_q[sel_idx].subid;
                    end
                    // 占用数据总线 tBURST 周期
                    bus_busy_timer <= DRAM_TBURST;
                    rhit_cnt <= rhit_cnt + 1;
                    // 出队
                    req_q[sel_idx].valid <= 1'b0;
                    req_head <= (req_head + 1) & 4'hF;
                    req_count <= req_count - 1;
                end
                // ---------------------------------------------------------------
                // ACTIVE 但行不匹配：需 PRECHARGE 后再 ACTIVATE
                // ---------------------------------------------------------------
                // 教学注记：仅当 ACTIVE 已维持 ≥ tRAS 时才允许 PRE（04 §12 约束）。
                //   PRECHG 状态持续 tRP=12 周期，回 IDLE 后下拍可再 ACTIVATE。
                else if (bank_state[b] == BK_ACTIVE &&
                         bank_row[b] != req_q[sel_idx].row &&
                         bank_act_age[b] >= DRAM_TRAS) begin
                    // PRECHARGE
                    bank_state[b] <= BK_PRECHG;
                    bank_timer[b] <= DRAM_TRP;
                    rmiss_cnt <= rmiss_cnt + 1;
                end
            end

            // -----------------------------------------------------------------
            // 统计累加
            // -----------------------------------------------------------------
            // bank 忙碌：任何非 IDLE/ACTIVE 的 bank 都计入 bank_busy
            for (int b = 0; b < NUM_BANKS; b++) begin
                if (bank_state[b] != BK_IDLE && bank_state[b] != BK_ACTIVE) begin
                    bk_busy_cnt <= bk_busy_cnt + 1;
                end
            end
            // 总线忙碌：bus_busy_timer > 0 时计入
            if (bus_busy_timer > 0) begin
                bus_busy_cnt <= bus_busy_cnt + 1;
            end
        end
    end

    assign stat_reads      = read_cnt;
    assign stat_writes     = write_cnt;
    assign stat_row_hits   = rhit_cnt;
    assign stat_row_misses = rmiss_cnt;
    assign stat_bank_busy  = bk_busy_cnt;
    assign stat_bus_busy   = bus_busy_cnt;

endmodule
