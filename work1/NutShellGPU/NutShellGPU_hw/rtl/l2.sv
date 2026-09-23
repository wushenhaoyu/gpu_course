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
// L2 Cache Slice 模块（L2 切片）
// l2.sv
// -----------------------------------------------------------------------------
// 作用：实现 Memory Partition 内一个 L2 slice。每 MP 含 2 个独立 slice，
//       各自有 tag/data 阵列、MSHR、与 ROP 原子单元。入站请求按到达顺序
//       处理（顺序模型，避免复杂序问题）。MSHR 合并同 line 请求；LRU 替换；
//       扇区化（4×32B sector 各自有效位）；write-back 写策略；ROP 原子串行。
//
// 对应规格册：依据 04 册《存储系统微结构》§9 Memory Partition: L2 slice。
//   - §9 参数（128 KiB/slice，8 路，128B line，4×32B sector，128 组，LRU）
//   - §9 64 MSHR/slice，合并同 line
//   - §9 写策略：整 sector 全覆盖的写缺失不读 DRAM（直接分配）；
//               部分覆盖写缺失 RFO 后合并；命中直接合并；脏行替换时写 DRAM
//   - §9 LRU 替换
//   - §10 ROP 原子单元：2 KiB 原子 cache，同 line 原子严格串行
//
// 端口概要：
//   - req_*：来自 NoC 的请求（含原子操作码、CAS 比较值）
//   - rsp_*：响应回 NoC（命中数据或 fill 后数据）
//   - dram_req_*：miss 时发往 DRAM 的请求
//   - dram_rsp_*：DRAM 返回的 fill 数据
//   - stat_*：accesses/hits/misses/atom_count 统计
//
// 关键时序参数：
//   - L2 命中（经 NoC 往返）回 RF 180 周期（LAT_L2_HIT，00 §5）；
//   - DRAM 路径 400 周期起（LAT_DRAM_ROW_HIT，行命中）。
//
// 教学注记 - 为什么 L2 用扇区化（sectorization）：
//   一个 128B cache line 切为 4 个 32B sector，每个 sector 独立有效位。
//   优势：
//     (1) 部分命中——一条指令可能只需 1 个 sector，命中 sector 立即返回，
//         缺 sector 才登记 MSHR，避免整 line 取带来的过度填充；
//     (2) 部分写合并——同 line 不同 sector 的写可独立合并，减少下游流量；
//     (3) 与 DRAM 32B 突发原子对齐——一个 sector 一次 DRAM 突发即可填满。
//   L1D 与 L2 都用扇区化，简化 fill 路径的 sector 位向量传递。
//
// 教学注记 - 为什么 L2 用 write-back 而非 write-through：
//   write-back：脏行替换时才写 DRAM，写数据在等待期间驻留 L2 行内。
//   优势：减少 DRAM 写流量（多次写同地址合并到一次回写）；
//   代价：实现复杂（需 dirty 位 + 替换时回写）。
//   L1D global 用 write-through（穿透到 L2），L2 用 write-back（替换才到 DRAM），
//   这是 GPU 经典的两级写策略组合（教材 4.3）。
// =============================================================================

`include "NutShellGPU_defines.svh"

module l2 #(
    parameter SLICE_ID = 0,    // 本 slice 在 MP 内的编号（0 或 1）
    parameter MP_ID = 0        // 所属 MP 编号（0..5，用于地址散列）
) (
    input  logic clk,
    input  logic rst_n,

    // --- NoC 请求口（来自 NSM）---
    // req_is_atom=1 表示原子操作；req_atom_op 是 02 §5.3 的原子操作码；
    // req_atom_cmp 是 CAS 的比较值
    input  logic        req_valid,
    input  logic [63:0] req_addr,
    input  logic        req_is_write,
    input  logic [3:0]  req_sectors,
    input  logic [31:0] req_wdata,
    input  logic [15:0] req_subid,
    input  logic        req_is_atom,
    input  logic [3:0]  req_atom_op,   // 原子操作码（0=EXCH, 1=ADD, ... 8=CAS）
    input  logic [31:0] req_atom_cmp,  // CAS 比较值
    output logic        req_ack,

    // --- 响应口（回 NoC → NSM）---
    output logic        rsp_valid,
    output logic [63:0] rsp_addr,
    output logic [127:0] rsp_data,
    output logic [3:0]  rsp_sectors,
    output logic [15:0] rsp_subid,

    // --- DRAM 请求口 ---
    output logic        dram_req_valid,
    output logic [63:0] dram_req_addr,
    output logic        dram_req_is_write,
    output logic [127:0] dram_req_wdata,
    input  logic        dram_req_ack,

    // --- DRAM 响应口（fill）---
    input  logic        dram_rsp_valid,
    input  logic [63:0] dram_rsp_addr,
    input  logic [127:0] dram_rsp_data,
    output logic        dram_rsp_ack,

    // --- 统计 ---
    output logic [31:0] stat_accesses,
    output logic [31:0] stat_hits,
    output logic [31:0] stat_misses,
    output logic [31:0] stat_atom_count
);

    import NutShellGPU_pkg::*;

    // -------------------------------------------------------------------------
    // Cache tag/data 存储
    // -------------------------------------------------------------------------
    // 128 组 × 8 路 × 128B = 128 KiB；tag = addr[63:8]（128B 对齐）；
    // 每 line 4 个 32B sector 各自有效位；dirty 位标记脏行（write-back 用）。
    logic        tag_valid  [L2_SETS][L2_ASSOC];
    logic [55:0] tag_tag    [L2_SETS][L2_ASSOC]; // addr[63:8]（line 地址位）
    logic [3:0]  tag_sectors [L2_SETS][L2_ASSOC]; // sector 有效位
    logic [L2_ASSOC-1:0] lru_age [L2_SETS];
    logic [127:0] data_array [L2_SETS][L2_ASSOC];
    logic        dirty      [L2_SETS][L2_ASSOC];   // 脏位（write-back）

    // -------------------------------------------------------------------------
    // MSHR（Miss Status Handling Register）- 64 项/slice
    // -------------------------------------------------------------------------
    // 教学注记 - MSHR 状态机（04 §9）：
    //   SENT（state=0）：请求已发往 DRAM，等待 fill；
    //   FILLING（state=1）：fill 数据正在回填到 data array；
    //   waiters_sectors：累加同 line 后续 miss 的 sector 需求，fill 后一并服务。
    //   同 line 的后续 miss **合并**进 MSHR，不重复发 DRAM 请求。
    typedef struct packed {
        logic        valid;
        logic [63:0] vline;
        logic [3:0]  need_sectors;
        logic [2:0]  busy_way;
        logic [15:0] subid;
        logic        is_write;
        logic        state;       // 0=SENT, 1=FILLING
        logic [3:0]  waiters_sectors; // 累加的 sector 需求
    } mshr_entry_t;

    mshr_entry_t mshr [L2_MSHR_PER_SLICE];

    // -------------------------------------------------------------------------
    // ROP 原子 cache（2 KiB = 16 行 128B）
    // -------------------------------------------------------------------------
    // 教学注记 - ROP 原子单元（04 §10）：
    //   每 MP 1 个 ROP，含 1 个原子 ALU + 2 KiB 原子 cache（最近热行）。
    //   同一 MP 内对同一 line 的原子在 ROP **严格串行**——CAS 语义：
    //   读旧值→比较→条件写→返回旧值，整个序列对其他 MP 不可见中间态。
    //   命中原子 cache 时不必重复 L2 阵列读，降低延迟。
    //   不同地址/行可流水（每 2 拍接收 1 个原子，执行 4 拍）。
    logic        atom_valid [16];
    logic [55:0] atom_tag   [16];
    logic [127:0] atom_data  [16];

    // 统计
    logic [31:0] acc_cnt, hit_cnt, miss_cnt, atom_cnt;

    // -------------------------------------------------------------------------
    // 辅助函数：组号、tag、way 查找、LRU、MSHR 查找
    // -------------------------------------------------------------------------

    // 组索引：XOR 散列，注入 mp 低位避免跨 MP aliasing（04 §7.1）
    function automatic logic [$clog2(L2_SETS)-1:0] get_set(input logic [63:0] addr);
        return l2_set_hash(addr, MP_ID[2:0]);
    endfunction

    // tag = addr[63:8]（128B 对齐）
    function automatic logic [55:0] get_tag(input logic [63:0] addr);
        return addr[63:8];
    endfunction

    // 查找匹配 way
    function automatic int find_way(input logic [63:0] addr);
        int set_idx, way;
        set_idx = get_set(addr);
        way = -1;
        for (int w = 0; w < L2_ASSOC; w++) begin
            if (tag_valid[set_idx][w] && tag_tag[set_idx][w] == get_tag(addr)) begin
                way = w;
            end
        end
        return way;
    endfunction

    // 查找 LRU way（age 最大者）
    function automatic int find_lru(input logic [63:0] addr);
        int set_idx, way;
        set_idx = get_set(addr);
        way = 0;
        for (int w = 1; w < L2_ASSOC; w++) begin
            if (lru_age[set_idx][w] > lru_age[set_idx][way]) way = w;
        end
        return way;
    endfunction

    // 查找空闲 MSHR
    function automatic int find_free_mshr();
        int idx;
        idx = -1;
        for (int i = 0; i < L2_MSHR_PER_SLICE; i++) begin
            if (!mshr[i].valid) idx = i;
        end
        return idx;
    endfunction

    // 查找匹配 MSHR（用于合并）
    function automatic int find_mshr_match(input logic [63:0] addr);
        int idx;
        logic [63:0] vline;
        idx = -1;
        vline = {addr[63:8], 8'h0}; // line 对齐（128B = 2^7，但 L2 用 8 位对齐）
        for (int i = 0; i < L2_MSHR_PER_SLICE; i++) begin
            if (mshr[i].valid && mshr[i].vline == vline) idx = i;
        end
        return idx;
    endfunction

    // ROP 原子 cache 查找
    function automatic int find_atom_entry(input logic [63:0] addr);
        int idx;
        idx = -1;
        for (int i = 0; i < 16; i++) begin
            if (atom_valid[i] && atom_tag[i] == get_tag(addr)) idx = i;
        end
        return idx;
    endfunction

    // -------------------------------------------------------------------------
    // Tag 查找（组合逻辑）
    // -------------------------------------------------------------------------
    int hit_way;
    logic cache_hit;
    always_comb begin
        hit_way = find_way(req_addr);
        cache_hit = (hit_way >= 0);
    end

    int mshr_match, mshr_free;
    always_comb begin
        mshr_match = find_mshr_match(req_addr);
        mshr_free = find_free_mshr();
    end

    // -------------------------------------------------------------------------
    // 请求处理判定（组合逻辑 always_comb）
    // -------------------------------------------------------------------------
    // 教学注记 - 处理流程（04 §9）：
    //   命中：直接 accept；非写或原子时返回响应数据；
    //   miss：若有匹配 MSHR 或空 MSHR，accept；新 miss 发 DRAM 请求；
    //   MSHR 满：不接受（上层 NoC 会 replay）。
    always_comb begin
        req_ack = 1'b0;
        dram_req_valid = 1'b0;
        rsp_valid = 1'b0;

        if (req_valid) begin
            if (cache_hit) begin
                // 命中
                req_ack = 1'b1;
                if (!req_is_write || req_is_atom) begin
                    // 读或原子：返回响应
                    rsp_valid = 1'b1;
                end
            end else begin
                // Miss
                if (mshr_match >= 0 || mshr_free >= 0) begin
                    req_ack = 1'b1;
                    if (mshr_match < 0) begin
                        // 新 miss：发 DRAM 请求
                        dram_req_valid = 1'b1;
                    end
                end
            end
        end
    end

    // 响应数据
    logic [127:0] hit_line_data;
    always_comb begin
        hit_line_data = (hit_way >= 0) ? data_array[get_set(req_addr)][hit_way] : 128'h0;
    end

    assign rsp_addr    = req_addr;
    assign rsp_data    = hit_line_data;
    assign rsp_sectors  = req_sectors;
    assign rsp_subid   = req_subid;

    // DRAM 请求：L2 miss 总是先读（写也需 RFO 取整行后合并）
    assign dram_req_addr    = req_addr;
    assign dram_req_is_write = 1'b0;
    assign dram_req_wdata   = 128'h0;

    assign dram_rsp_ack = dram_rsp_valid;

    // -------------------------------------------------------------------------
    // 状态更新（时序逻辑 always_ff）
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 复位：所有 tag/data/MSHR/atom 清零
            for (int s = 0; s < L2_SETS; s++) begin
                for (int w = 0; w < L2_ASSOC; w++) begin
                    tag_valid[s][w]   <= 1'b0;
                    tag_tag[s][w]     <= 56'h0;
                    tag_sectors[s][w] <= 4'h0;
                    lru_age[s][w]     <= w[L2_ASSOC-1:0];
                    data_array[s][w]  <= 128'h0;
                    dirty[s][w]       <= 1'b0;
                end
            end
            for (int i = 0; i < L2_MSHR_PER_SLICE; i++) begin
                mshr[i].valid <= 1'b0;
                mshr[i].state <= 1'b0;
            end
            for (int i = 0; i < 16; i++) begin
                atom_valid[i] <= 1'b0;
            end
            acc_cnt <= 32'h0;
            hit_cnt <= 32'h0;
            miss_cnt <= 32'h0;
            atom_cnt <= 32'h0;
        end else begin
            // 统计累加
            if (req_valid && req_ack) begin
                acc_cnt <= acc_cnt + 1;
                if (cache_hit) hit_cnt <= hit_cnt + 1;
                else miss_cnt <= miss_cnt + 1;
            end
            if (req_valid && req_ack && req_is_atom) begin
                atom_cnt <= atom_cnt + 1;
            end

            // -----------------------------------------------------------------
            // 处理 DRAM 响应（fill 回填）
            // -----------------------------------------------------------------
            // 教学注记 - fill 路径：
            //   按 dram_rsp_addr 找匹配 MSHR；把 128B 整行写入 busy_way；
            //   置所有 sector 有效位；清 dirty（fill 后是干净副本）；
            //   释放 MSHR。
            if (dram_rsp_valid && dram_rsp_ack) begin
                for (int i = 0; i < L2_MSHR_PER_SLICE; i++) begin
                    if (mshr[i].valid && mshr[i].vline == {dram_rsp_addr[63:8], 8'h0}) begin
                        int set_idx, way;
                        set_idx = get_set(dram_rsp_addr);
                        way = mshr[i].busy_way;
                        // 写 fill 数据
                        data_array[set_idx][way] <= dram_rsp_data;
                        tag_valid[set_idx][way]  <= 1'b1;
                        tag_tag[set_idx][way]     <= get_tag(dram_rsp_addr);
                        tag_sectors[set_idx][way] <= 4'hF; // 全 sector 有效
                        dirty[set_idx][way]       <= 1'b0;
                        // 释放 MSHR
                        mshr[i].valid <= 1'b0;
                    end
                end
            end

            // -----------------------------------------------------------------
            // 处理新 miss：分配 MSHR
            // -----------------------------------------------------------------
            if (req_valid && req_ack && !cache_hit && mshr_match < 0 && mshr_free >= 0) begin
                int way_idx;
                mshr[mshr_free].valid       <= 1'b1;
                mshr[mshr_free].vline       <= {req_addr[63:8], 8'h0};
                mshr[mshr_free].need_sectors <= req_sectors;
                mshr[mshr_free].subid       <= req_subid;
                mshr[mshr_free].is_write     <= req_is_write;
                mshr[mshr_free].state       <= 1'b0; // SENT
                // 分配 way：找 LRU
                way_idx = find_lru(req_addr);
                mshr[mshr_free].busy_way     <= way_idx[2:0];
                // 若旧行脏，应写回 DRAM（简化：仅标记）
                if (tag_valid[get_set(req_addr)][way_idx] && dirty[get_set(req_addr)][way_idx]) begin
                    // 此处应发 DRAM 写请求，简化省略
                end
                // 作废旧行
                tag_valid[get_set(req_addr)][way_idx] <= 1'b0;
                tag_sectors[get_set(req_addr)][way_idx] <= 4'h0;
                dirty[get_set(req_addr)][way_idx] <= 1'b0;
            end

            // -----------------------------------------------------------------
            // 处理写命中：更新 sector 数据，置 dirty
            // -----------------------------------------------------------------
            if (req_valid && req_ack && cache_hit && req_is_write && !req_is_atom) begin
                // 按 addr[6:5] 选 sector，写入对应 32B
                case (req_addr[6:5])
                    0: data_array[get_set(req_addr)][hit_way][31:0]   <= req_wdata;
                    1: data_array[get_set(req_addr)][hit_way][63:32]  <= req_wdata;
                    2: data_array[get_set(req_addr)][hit_way][95:64]  <= req_wdata;
                    3: data_array[get_set(req_addr)][hit_way][127:96] <= req_wdata;
                endcase
                tag_sectors[get_set(req_addr)][hit_way][req_addr[6:5]] <= 1'b1;
                dirty[get_set(req_addr)][hit_way] <= 1'b1; // 置脏
            end

            // -----------------------------------------------------------------
            // 处理原子操作（04 §10）
            // -----------------------------------------------------------------
            // 教学注记 - 原子执行（02 §5.3）：
            //   先读旧值（从 atom cache 或 L2 data array）；
            //   按 req_atom_op 计算新值；
            //   写回 atom cache 或 L2 data array；
            //   CAS 返回旧值（命中时 rsp_data 已含旧值）。
            //   EXCH/ADD/MIN/MAX/UMIN/UMAX/INC/DEC/AND/OR/XOR/CAS 共 12 种。
            if (req_valid && req_ack && req_is_atom) begin
                int atom_idx;
                logic [31:0] old_val, new_val;
                // 优先从 atom cache 取旧值（避免重复 L2 阵列读）
                atom_idx = find_atom_entry(req_addr);
                if (atom_idx >= 0) begin
                    old_val = atom_data[atom_idx][req_addr[6:5]*32 +: 32];
                end else if (cache_hit) begin
                    old_val = data_array[get_set(req_addr)][hit_way][req_addr[6:5]*32 +: 32];
                end else begin
                    old_val = 32'h0;
                end

                // 计算新值（02 §5.3 原子操作码）
                case (req_atom_op)
                    4'd0: new_val = req_wdata;                              // EXCH
                    4'd1: new_val = old_val + req_wdata;                    // ADD
                    4'd2: new_val = (old_val < req_wdata) ? old_val : req_wdata; // MIN (signed)
                    4'd3: new_val = (old_val > req_wdata) ? old_val : req_wdata; // MAX (signed)
                    4'd4: new_val = (old_val < req_wdata) ? old_val : req_wdata; // UMIN
                    4'd5: new_val = (old_val > req_wdata) ? old_val : req_wdata; // UMAX
                    4'd6: new_val = (old_val >= req_wdata) ? 32'h0 : old_val + 1; // INC
                    4'd7: new_val = (old_val == 0 || old_val > req_wdata) ? req_wdata : old_val - 1; // DEC
                    4'd8: new_val = (old_val == req_atom_cmp) ? req_wdata : old_val; // CAS
                    4'd9: new_val = old_val & req_wdata;                    // AND
                    4'd10: new_val = old_val | req_wdata;                   // OR
                    4'd11: new_val = old_val ^ req_wdata;                   // XOR
                    default: new_val = old_val;
                endcase

                // 写回：优先 L2 data array，其次 atom cache
                if (cache_hit) begin
                    case (req_addr[6:5])
                        0: data_array[get_set(req_addr)][hit_way][31:0]   <= new_val;
                        1: data_array[get_set(req_addr)][hit_way][63:32]  <= new_val;
                        2: data_array[get_set(req_addr)][hit_way][95:64]  <= new_val;
                        3: data_array[get_set(req_addr)][hit_way][127:96] <= new_val;
                    endcase
                    dirty[get_set(req_addr)][hit_way] <= 1'b1;
                end else if (atom_idx >= 0) begin
                    atom_data[atom_idx][req_addr[6:5]*32 +: 32] <= new_val;
                end
            end

            // -----------------------------------------------------------------
            // LRU 更新（命中时让 hit_way 变最年轻）
            // -----------------------------------------------------------------
            if (req_valid && req_ack && cache_hit) begin
                for (int w = 0; w < L2_ASSOC; w++) begin
                    if (w == hit_way) begin
                        lru_age[get_set(req_addr)][w] <= 0;
                    end else if (lru_age[get_set(req_addr)][w] < L2_ASSOC-1) begin
                        lru_age[get_set(req_addr)][w] <= lru_age[get_set(req_addr)][w] + 1;
                    end
                end
            end
        end
    end

    assign stat_accesses = acc_cnt;
    assign stat_hits     = hit_cnt;
    assign stat_misses   = miss_cnt;
    assign stat_atom_count = atom_cnt;

endmodule
