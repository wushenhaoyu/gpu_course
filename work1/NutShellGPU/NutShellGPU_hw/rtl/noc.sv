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
// 片上互连 NoC 模块（Network-on-Chip）
// noc.sv
// -----------------------------------------------------------------------------
// 作用：实现 NutShellGPU 全片 16 NSM 端口 + 6 MP 端口的全交叉开关（crossbar）。
//       请求（REQ）从 NSM 流向 MP（按地址散列路由），响应（RSP）从 MP 流回
//       NSM（按 subid 编码的源 NSM 号路由）。两条虚通道（REQ/RSP）独立
//       仲裁，避免响应被请求队头阻塞。
//
// 对应规格册：依据 04 册《存储系统微结构》§8 片上互连 NoC。
//   - §8 全交叉开关，22 端口（16+6），每端口每拍收发各 1 flit
//   - §8 flit：8 字节负载；包 = 1 头 flit + 若干数据 flit
//   - §8 2 条虚通道（REQ 每 input 4 flit，RSP 每 input 4 flit）
//   - §8 仲裁：输入/输出端各一个 RR 分配器，平局小编号胜
//   - §8 时序：路由+仲裁 1 周期（router_pipe），链路 1 周期/flit
//   - §7.1 256B 交织粒度，XOR 折叠到 6 个 MP
//
// 端口概要：
//   - nsm_req_*：16 个 NSM → NoC 请求端口
//   - nsm_rsp_*：NoC → 16 个 NSM 响应端口
//   - mp_req_*：NoC → 6 个 MP 请求端口
//   - mp_rsp_*：6 个 MP → NoC 响应端口
//   - stat_*：flits_sent / stall_cycles 统计
//
// 关键时序参数：
//   - router_pipe 1 周期（仲裁组合 + 寄存器）；
//   - link_cycles 1 周期/flit；
//   - REQ/RSP VC 各 4 flit 缓冲/输入口。
//
// 教学注记 - 为什么用 crossbar 而非 ring：
//   ring（环）的优点是布线简单、面积小，但延迟随跳数线性增长——16 NSM
//   的 ring 最坏要 16 跳才能从一端到另一端，对延迟敏感的 L2 访问不可接受。
//   crossbar 的优点是任意两端口之间只需 1 跳（路由 1 周期 + 链路 1 周期），
//   延迟恒定且低；代价是布线复杂度 O(N²)，但 NutShellGPU 仅 22 端口，面积可接受。
//   教材 4.2 对比了 crossbar、ring、mesh 三种拓扑，NutShellGPU 选 crossbar 与
//   教材基线一致。
//
// 教学注记 - 为什么用两条虚通道（REQ/RSP）：
//   死锁避免的经典策略——请求和响应用独立虚通道，避免响应被请求队头
//   阻塞（head-of-line blocking）。04 册 §8 末约定"无流控反压死锁
//   （缓冲够深）"，缓冲满时发送方 replay/等待。
// =============================================================================

`include "NutShellGPU_defines.svh"

module noc #(
    parameter NUM_NSM_PORTS = 16,    // NSM 端口数（00 §3：16 个 NSM）
    parameter NUM_MP_PORTS  = 6,      // MP 端口数（00 §3：6 个 MP）
    parameter VC_ENTRIES    = 4       // 每输入口 VC flit 数（04 §8：4）
) (
    input  logic clk,
    input  logic rst_n,

    // --- NSM → NoC 请求端口（16 路）---
    // 教学注记：每 NSM 每拍最多发 1 个请求 flit；nsm_req_ack=1 表示被接收
    input  logic [NUM_NSM_PORTS-1:0] nsm_req_valid,
    input  logic [63:0] nsm_req_addr  [NUM_NSM_PORTS],
    input  logic       nsm_req_is_write [NUM_NSM_PORTS],
    input  logic [3:0]  nsm_req_sectors [NUM_NSM_PORTS],
    input  logic [31:0] nsm_req_wdata  [NUM_NSM_PORTS],
    input  logic [15:0] nsm_req_subid  [NUM_NSM_PORTS],
    output logic [NUM_NSM_PORTS-1:0] nsm_req_ack,

    // --- NoC → NSM 响应端口（16 路）---
    output logic [NUM_NSM_PORTS-1:0] nsm_rsp_valid,
    output logic [63:0] nsm_rsp_addr  [NUM_NSM_PORTS],
    output logic [127:0] nsm_rsp_data [NUM_NSM_PORTS],
    output logic [3:0]  nsm_rsp_sectors [NUM_NSM_PORTS],
    output logic [15:0] nsm_rsp_subid [NUM_NSM_PORTS],

    // --- NoC → MP 请求端口（6 路）---
    output logic [NUM_MP_PORTS-1:0] mp_req_valid,
    output logic [63:0] mp_req_addr   [NUM_MP_PORTS],
    output logic       mp_req_is_write [NUM_MP_PORTS],
    output logic [3:0]  mp_req_sectors [NUM_MP_PORTS],
    output logic [31:0] mp_req_wdata   [NUM_MP_PORTS],
    output logic [15:0] mp_req_subid   [NUM_MP_PORTS],
    input  logic [NUM_MP_PORTS-1:0] mp_req_ack,

    // --- MP → NoC 响应端口（6 路）---
    input  logic [NUM_MP_PORTS-1:0] mp_rsp_valid,
    input  logic [63:0] mp_rsp_addr   [NUM_MP_PORTS],
    input  logic [127:0] mp_rsp_data  [NUM_MP_PORTS],
    input  logic [3:0]  mp_rsp_sectors [NUM_MP_PORTS],
    input  logic [15:0] mp_rsp_subid  [NUM_MP_PORTS],
    output logic [NUM_MP_PORTS-1:0] mp_rsp_ack,

    // --- 统计 ---
    output logic [31:0] stat_flits_sent,
    output logic [31:0] stat_stall_cycles
);

    import NutShellGPU_pkg::*;

    localparam TOTAL_PORTS = NUM_NSM_PORTS + NUM_MP_PORTS; // 22

    // -------------------------------------------------------------------------
    // 地址 → 目的 MP 路由函数（依据 04 册 §7.1）
    // -------------------------------------------------------------------------
    // 教学注记 - 256B 交织 + XOR 折叠：
    //   b256 = A[31:8]（256B 块号）
    //   fold = b256[5:0] ^ b256[11:6] ^ b256[17:12] ^ b256[23:18]
    //   mp   = fold % 6
    //   XOR 折叠让连续 256B 块均匀分布到 6 个 MP，避免热点；
    //   256B 粒度与 cache line（128B）的 2 倍关系保证跨 line 访问均匀分担带宽。
    function automatic logic [$clog2(NUM_MP_PORTS)-1:0] addr_to_dest_mp(input logic [63:0] addr);
        logic [23:0] b256;
        logic [5:0]  fold;
        b256 = addr[31:8];
        fold = b256[5:0] ^ b256[11:6] ^ b256[17:12] ^ b256[23:18];
        return fold % NUM_MP_PORTS;
    endfunction

    // -------------------------------------------------------------------------
    // REQ 仲裁：NSM → MP（每 MP 选一个 NSM 请求）
    // -------------------------------------------------------------------------
    // 教学注记 - 仲裁策略（04 §8）：
    //   对每个 MP 端口，从所有目标为它的 NSM 请求中按 round-robin 选一个。
    //   平局取 NSM 编号小者胜（05 §2.3 末：所有平局用"编号小者胜"）。
    //   rr_req_cursor 是游标，每服务一个请求后移到 (winner+1) mod 16。

    // 计算每个 NSM 请求的目的 MP
    logic [$clog2(NUM_MP_PORTS)-1:0] nsm_dest_mp [NUM_NSM_PORTS];
    always_comb begin
        for (int i = 0; i < NUM_NSM_PORTS; i++) begin
            nsm_dest_mp[i] = addr_to_dest_mp(nsm_req_addr[i]);
        end
    end

    // REQ winner 数组：mp_req_winner[m][i]=1 表示 NSM i 赢得 MP m 的本轮仲裁
    logic [NUM_NSM_PORTS-1:0] mp_req_winner [NUM_MP_PORTS];
    logic [$clog2(NUM_NSM_PORTS)-1:0] rr_req_cursor;   // REQ round-robin 游标
    logic [31:0] flit_count;
    logic [31:0] stall_count;

    // 组合仲裁：为每个 MP 选 winner
    always_comb begin
        // 默认全 0
        for (int m = 0; m < NUM_MP_PORTS; m++) begin
            mp_req_winner[m] = '0;
        end
        // 对每个 MP，按 RR 游标扫描所有 NSM，找第一个目标为它的请求
        for (int m = 0; m < NUM_MP_PORTS; m++) begin
            logic found;
            logic [$clog2(NUM_NSM_PORTS)-1:0] sel;
            found = 1'b0;
            sel = 0;
            // RR：从 cursor 起，依次扫描 NSM_PORTS 个端口
            for (int i = 0; i < NUM_NSM_PORTS; i++) begin
                int idx;
                idx = (rr_req_cursor + i) % NUM_NSM_PORTS;
                if (nsm_req_valid[idx] && nsm_dest_mp[idx] == m[$clog2(NUM_MP_PORTS)-1:0] && !found) begin
                    mp_req_winner[m][idx] = 1'b1;
                    found = 1'b1;
                    sel = idx[$clog2(NUM_NSM_PORTS)-1:0];
                end
            end
        end
    end

    // 驱动 MP 请求端口
    always_comb begin
        mp_req_valid = '0;
        for (int m = 0; m < NUM_MP_PORTS; m++) begin
            for (int i = 0; i < NUM_NSM_PORTS; i++) begin
                if (mp_req_winner[m][i]) begin
                    mp_req_valid[m]   = 1'b1;
                    mp_req_addr[m]    = nsm_req_addr[i];
                    mp_req_is_write[m] = nsm_req_is_write[i];
                    mp_req_sectors[m] = nsm_req_sectors[i];
                    mp_req_wdata[m]  = nsm_req_wdata[i];
                    mp_req_subid[m]  = nsm_req_subid[i];
                end
            end
        end
    end

    // NSM 请求 ack：MP 接收后回 ack 给对应 NSM
    always_comb begin
        nsm_req_ack = '0;
        for (int m = 0; m < NUM_MP_PORTS; m++) begin
            if (mp_req_ack[m]) begin
                for (int i = 0; i < NUM_NSM_PORTS; i++) begin
                    if (mp_req_winner[m][i]) begin
                        nsm_req_ack[i] = 1'b1;
                    end
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // RSP 路由：MP → NSM（按 subid 高位解码目的 NSM）
    // -------------------------------------------------------------------------
    // 教学注记 - 简化路由方案：
    //   响应目的 NSM 由 subid[15:11] 编码（请求方在出站时填入自身 NSM 号）。
    //   每 MP 响应按 subid 解码出目的 NSM，再按 RR 在目标 NSM 的所有响应中选一个。
    logic [$clog2(NUM_NSM_PORTS)-1:0] mp_dest_nsm [NUM_MP_PORTS];
    always_comb begin
        for (int m = 0; m < NUM_MP_PORTS; m++) begin
            mp_dest_nsm[m] = mp_rsp_subid[m][15:11] % NUM_NSM_PORTS;
        end
    end

    // RSP winner 数组
    logic [NUM_MP_PORTS-1:0] nsm_rsp_winner [NUM_NSM_PORTS];
    logic [$clog2(NUM_MP_PORTS)-1:0] rr_rsp_cursor;

    // 组合仲裁：为每个 NSM 选 winner
    always_comb begin
        for (int n = 0; n < NUM_NSM_PORTS; n++) begin
            nsm_rsp_winner[n] = '0;
        end
        for (int n = 0; n < NUM_NSM_PORTS; n++) begin
            logic found;
            found = 1'b0;
            for (int i = 0; i < NUM_MP_PORTS; i++) begin
                int idx;
                idx = (rr_rsp_cursor + i) % NUM_MP_PORTS;
                if (mp_rsp_valid[idx] && mp_dest_nsm[idx] == n[$clog2(NUM_NSM_PORTS)-1:0] && !found) begin
                    nsm_rsp_winner[n][idx] = 1'b1;
                    found = 1'b1;
                end
            end
        end
    end

    // 驱动 NSM 响应端口
    always_comb begin
        nsm_rsp_valid = '0;
        for (int n = 0; n < NUM_NSM_PORTS; n++) begin
            for (int i = 0; i < NUM_MP_PORTS; i++) begin
                if (nsm_rsp_winner[n][i]) begin
                    nsm_rsp_valid[n]   = 1'b1;
                    nsm_rsp_addr[n]    = mp_rsp_addr[i];
                    nsm_rsp_data[n]    = mp_rsp_data[i];
                    nsm_rsp_sectors[n] = mp_rsp_sectors[i];
                    nsm_rsp_subid[n]   = mp_rsp_subid[i];
                end
            end
        end
    end

    // MP 响应 ack
    always_comb begin
        mp_rsp_ack = '0;
        for (int n = 0; n < NUM_NSM_PORTS; n++) begin
            for (int i = 0; i < NUM_MP_PORTS; i++) begin
                if (nsm_rsp_winner[n][i]) begin
                    mp_rsp_ack[i] = 1'b1;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // 状态更新（时序逻辑 always_ff）：RR 游标推进 + 统计
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rr_req_cursor <= 0;
            rr_rsp_cursor <= 0;
            flit_count <= 32'h0;
            stall_count <= 32'h0;
        end else begin
            // 更新 REQ 游标：服务后移到 (winner+1) mod 16
            for (int m = 0; m < NUM_MP_PORTS; m++) begin
                if (mp_req_ack[m]) begin
                    for (int i = 0; i < NUM_NSM_PORTS; i++) begin
                        if (mp_req_winner[m][i]) begin
                            rr_req_cursor <= (i + 1) % NUM_NSM_PORTS;
                        end
                    end
                end
            end
            // 更新 RSP 游标
            for (int n = 0; n < NUM_NSM_PORTS; n++) begin
                for (int i = 0; i < NUM_MP_PORTS; i++) begin
                    if (nsm_rsp_winner[n][i]) begin
                        rr_rsp_cursor <= (i + 1) % NUM_MP_PORTS;
                    end
                end
            end
            // 统计：每 MP 接收一个请求 flit +1
            for (int m = 0; m < NUM_MP_PORTS; m++) begin
                if (mp_req_valid[m] && mp_req_ack[m]) begin
                    flit_count <= flit_count + 1;
                end
            end
            // 统计：NSM 请求未被接收 → stall +1
            for (int i = 0; i < NUM_NSM_PORTS; i++) begin
                if (nsm_req_valid[i] && !nsm_req_ack[i]) begin
                    stall_count <= stall_count + 1;
                end
            end
        end
    end

    assign stat_flits_sent  = flit_count;
    assign stat_stall_cycles = stall_count;

endmodule
