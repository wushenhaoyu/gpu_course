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
// L1 数据 Cache 模块（L1 Data Cache）
// l1d.sv
// -----------------------------------------------------------------------------
// 作用：实现 NSM 内部 16 KiB L1D cache，支持 global/local 读写、PRT miss
//       合并、WDB 写缓冲、VI/VT（虚拟索引虚拟标识）。与 LSU 之间是非停滞
//       接口——资源不足时返回"拒绝"，由 LSU replay，不向流水线深处传播
//       stall 信号。
//
// 对应规格册：依据 04 册《存储系统微结构》§4 L1D 数据 cache。
//   - §4.1 参数（16 KiB，4 路，128B line，4×32B sector，VI/VT，32 组，LRU）
//   - §4.2 读路径（global/local LD；hits under miss 非阻塞）
//   - §4.3 写路径（ST；global write-through + no-write-allocate；
//                  local write-back + write-allocate；WDB 满 → replay）
//   - §4.4 Pending Request Table（PRT/MSHR，32 项，subid 路由回 fill）
//   - §4.5 一致性（无硬件协议，L1D 之间不互保一致）
//
// 端口概要：
//   - req_*：LSU 请求口（accept/replay/miss 三态响应）
//   - fill_*：从 NoC/L2 返回的 line 填充
//   - noc_req_*：miss 时发往 NoC/L2 的请求
//   - ld_resp_*：load 命中时的数据响应（回 RF）
//   - stat_*：统计计数
//
// 关键时序参数：
//   - L1D 命中延迟 30 周期（LAT_L1_HIT，00 §5）；
//   - cache 参数：32 组 × 4 路 × 128B = 16 KiB；每 line 4 个 32B sector；
//   - PRT 32 项；WDB 16 项。
//
// 教学注记 - 为什么 L1D 用 VI/VT（虚拟索引虚拟标识）：
//   VI/VT 指用虚拟地址直接索引和标识 cache，避免每次访问都经过 MMU 翻译。
//   优势：
//     (1) 命中延迟低——省去 TLB 查找时间；
//     (2) 简化硬件——无需 TLB 流水化或物理 tagged cache 的旁路。
//   代价：
//     (1) 同义别名（synonym alias）：不同虚拟地址映射同一物理地址时，
//         可能在 cache 中出现两份副本。NutShellGPU 通过基线恒等页映射
//         （00 §7）规避此问题——虚拟地址 == 物理地址，无别名。
//     (2) 跨 NSM 不一致：L1D 之间不互保一致（04 §4.5）。
//         NutShellGPU 通过软件策略规避：L1D 默认只缓存只读 global 与 local，
//         可写共享数据必须走 L2 + MEMBAR.GL/原子（程序员责任）。
// =============================================================================

`include "NutShellGPU_defines.svh"

module l1d #(
    parameter NUM_WARPS = 64
) (
    input  logic clk,
    input  logic rst_n,

    // --- LSU 请求口 ---
    // 教学注记 - 三态响应（accept/replay/miss）：
    //   req_accept=1：本拍接受请求（命中或可分配 PRT/WDB）；
    //   req_replay=1：资源不足（PRT 满/WDB 满/set 全 busy），LSU 应 replay；
    //   req_miss=1：cache miss，已在 PRT 登记，指令留 ibuf 等 fill。
    //   三者并非互斥——accept 与 miss 可同时为 1（表示接受但需等 fill）。
    input  logic        req_valid,
    input  logic        req_is_write,
    input  logic [63:0] req_addr,
    input  logic [3:0]  req_sectors,    // 128B line 内 4 sector 的有效位向量
    input  logic [31:0] req_wdata,
    input  logic [3:0]  req_be,         // 每 4B word 的字节使能（合并写用）
    // 教学注记 - 端口位宽与 enum 对齐（02 §2.2 / NutShellGPU_pkg.sv）：
    //   这三个字段的编码在 NTAS1 指令的 X/Y 子字段中定义：
    //     - space_e 占 3 位（X[2:0]）：GLOBAL/SHARED/LOCAL/CONST/FLAT
    //     - width_e 占 4 位（X[6:3]）：U8/U16/U32/U64/.../F16/U128
    //     - hint_e  占 3 位（Y[2:0]）：CA/CG/CS/CV/LU
    //   故端口声明必须与 enum 实际位宽严格对齐（3/4/3），否则例化时
    //   传入 SP_GLOBAL（3 位）/ W_U32（4 位）/ HINT_CA（3 位）会触发
    //   vopt-2241 "Connection width does not match" 警告。
    //   不要图省事统一写 [7:0]——SystemVerilog 不会自动截位，且会
    //   在 L1D 内部 `req_space == SP_GLOBAL` 比较时引入高位 X，导致
    //   仿真结果不确定。
    input  logic [2:0]  req_space,      // space_e（GLOBAL/SHARED/LOCAL/...）3 位
    input  logic [3:0]  req_width,       // width_e（U8/U16/U32/U64/...）4 位
    input  logic [2:0]  req_hint,       // hint_e（CA/CG/CS/CV/LU）3 位
    // 教学注记 - req_warp 宽度：$clog2(NUM_WARPS)-1:0
    //   NUM_WARPS=64 → $clog2(64)=6 → [5:0]，即 6 位。
    //   例化端必须用 6'd0 而非裸字面量 0（后者默认 32 位整数，
    //   会触发 vopt-2241 截断警告）。若 NUM_WARPS 参数被覆盖为
    //   其他值（如 32），此宽度会自动适配，例化端也需同步改。
    input  logic [$clog2(NUM_WARPS)-1:0] req_warp,
    input  logic [7:0]  req_dst_reg,
    input  logic [63:0] req_pc,
    input  logic [15:0] req_subid,       // 本 NSM 内出站请求序号
    output logic        req_accept,
    output logic        req_replay,
    output logic        req_miss,

    // --- Fill 口（从 NoC/L2 返回的 line 填充）---
    // 教学注记：fill 由 fill unit 路由——按 subid 查 PRT 表，找到对应项后
    //   把数据写入 busy_way 的指定 sector，然后通知 Arbiter 重放 waiters。
    input  logic        fill_valid,
    input  logic [63:0] fill_addr,
    input  logic [127:0] fill_data,      // 128B 整行数据
    input  logic [3:0]  fill_sectors,    // 哪些 sector 有效（部分填充也支持）
    input  logic [15:0] fill_subid,
    output logic        fill_ack,

    // --- NoC 请求口（miss → L2）---
    output logic        noc_req_valid,
    output logic [63:0] noc_req_addr,
    output logic        noc_req_is_write,
    output logic [3:0]  noc_req_sectors,
    output logic [31:0] noc_req_wdata,
    output logic [15:0] noc_req_subid,
    input  logic        noc_req_ack,

    // --- Load 响应口（命中时回 RF）---
    output logic        ld_resp_valid,
    output logic [$clog2(NUM_WARPS)-1:0] ld_resp_warp,
    output logic [7:0]  ld_resp_dst_reg,
    output logic [31:0] ld_resp_data,
    output logic [63:0] ld_resp_pc,

    // --- 统计 ---
    output logic [31:0] stat_access_lines,
    output logic [31:0] stat_miss_sectors,
    output logic [31:0] stat_prt_full,
    output logic [31:0] stat_assoc_stall,
    output logic [31:0] stat_wdb_full
);

    import NutShellGPU_pkg::*;

    // -------------------------------------------------------------------------
    // Cache tag 存储
    // -------------------------------------------------------------------------
    // 每组 4 路，每路有：valid 位、tag（虚拟 line 地址高位）、sector 有效位、
    // LRU 年龄位。tag = addr[63:12]（line 地址高位，因 32 组 × 128B = 4 KiB
    // 索引位 = 12 位）。
    logic        tag_valid  [L1D_SETS][L1D_ASSOC];
    logic [51:0] tag_tag    [L1D_SETS][L1D_ASSOC]; // addr[63:12]（line 地址位）
    logic [3:0]  tag_sectors [L1D_SETS][L1D_ASSOC]; // 4 sector 各自有效位
    logic [L1D_ASSOC-1:0] lru_age [L1D_SETS];      // LRU 跟踪

    // 数据阵列：每 line 128B，简化为整行存储
    logic [127:0] data_array [L1D_SETS][L1D_ASSOC];

    // -------------------------------------------------------------------------
    // PRT（Pending Request Table）- 32 项 MSHR
    // -------------------------------------------------------------------------
    // 教学注记 - PRT 的作用（04 §4.4）：
    //   PRT 是 L1D 的 MSHR（Miss Status Handling Register）表，追踪所有已发出
    //   但未完成的 miss 请求。每项记录：
    //     - vline：虚拟 line 地址（用于合并同 line 的后续 miss）
    //     - need_sectors：尚缺的 4 位 sector 向量
    //     - busy_way：占用中的 way（行锁，避免替换冲突）
    //     - subid：出站请求序号（fill 路由回 NSM 用）
    //     - state：SENT（已发往 L2）/ FILLING（数据正在回填）
    //   到同一 line 的后续 miss **合并**进 waiters，不重复发下游——
    //   这就是 "hits under miss" 非阻塞的关键。
    typedef struct packed {
        logic        valid;
        logic [63:0] vline;       // 虚拟 line 地址（128B 对齐）
        logic [3:0]  need_sectors; // 尚缺的 sector 向量
        logic [2:0]  busy_way;     // 占用中的 way（行锁）
        logic [15:0] subid;
        logic        state;        // 0=SENT, 1=FILLING
    } prt_entry_t;

    prt_entry_t prt [L1D_MSHR_ENTRIES];

    // -------------------------------------------------------------------------
    // WDB（Write Data Buffer）- 16 项
    // -------------------------------------------------------------------------
    // 教学注记 - WDB 的作用（04 §4.3）：
    //   global ST 用 write-through + no-write-allocate——数据直接送 L2，
    //   L1 中若有该 line 副本则使对应 sector tag 失效（防旧值）。
    //   WDB 缓冲这些写数据，合并同 line 的多个写以减少下游流量。
    //   WDB 满 → ST 指令 replay（replay_wdb_full）。
    typedef struct packed {
        logic        valid;
        logic [63:0] addr;
        logic [127:0] data;
        logic [3:0]  be;           // sector 字节使能
    } wdb_entry_t;

    wdb_entry_t wdb [L1D_WB_ENTRIES];

    // 统计计数器
    logic [31:0] access_lines, miss_sectors, prt_full_cnt, assoc_stall_cnt, wdb_full_cnt;

    // -------------------------------------------------------------------------
    // 辅助函数：组号、tag、way 查找、LRU、PRT/WDB 空闲项查找
    // -------------------------------------------------------------------------

    // 组索引：用 XOR 散列（04 §7.1）削弱 warp 内连续 sector 访问的组冲突
    function automatic logic [$clog2(L1D_SETS)-1:0] get_set(input logic [63:0] addr);
        return l1d_set_hash(addr);
    endfunction

    // tag = addr[63:12]（line 地址高位）
    function automatic logic [51:0] get_tag(input logic [63:0] addr);
        return addr[63:12];
    endfunction

    // 在指定组中查找 tag 匹配的 way（-1 表示未命中）
    function automatic int find_way(input logic [63:0] addr);
        int set_idx, way;
        set_idx = get_set(addr);
        way = -1;
        for (int w = 0; w < L1D_ASSOC; w++) begin
            if (tag_valid[set_idx][w] && tag_tag[set_idx][w] == get_tag(addr)) begin
                way = w;
            end
        end
        return way;
    endfunction

    // 查找 LRU way（用于替换）
    // 教学注记：lru_age 数值越大表示越久未用，find_lru 返回 age 最大的 way
    function automatic int find_lru(input logic [63:0] addr);
        int set_idx, way;
        set_idx = get_set(addr);
        way = 0;
        for (int w = 1; w < L1D_ASSOC; w++) begin
            if (lru_age[set_idx][w] > lru_age[set_idx][way]) begin
                way = w;
            end
        end
        return way;
    endfunction

    // 查找空闲 PRT 项（-1 表示全满）
    function automatic int find_free_prt();
        int idx;
        idx = -1;
        for (int i = 0; i < L1D_MSHR_ENTRIES; i++) begin
            if (!prt[i].valid) begin
                idx = i;
            end
        end
        return idx;
    endfunction

    // 查找匹配的 PRT 项（用于合并同 line 的 miss）
    function automatic int find_prt_match(input logic [63:0] addr);
        int idx;
        logic [63:0] vline;
        idx = -1;
        vline = {addr[63:7], 7'h0}; // line 对齐（128B = 2^7）
        for (int i = 0; i < L1D_MSHR_ENTRIES; i++) begin
            if (prt[i].valid && prt[i].vline == vline) begin
                idx = i;
            end
        end
        return idx;
    endfunction

    // 查找空闲 WDB 项
    function automatic int find_free_wdb();
        int idx;
        idx = -1;
        for (int i = 0; i < L1D_WB_ENTRIES; i++) begin
            if (!wdb[i].valid) begin
                idx = i;
            end
        end
        return idx;
    endfunction

    // -------------------------------------------------------------------------
    // Tag 查找（组合逻辑 always_comb）
    // -------------------------------------------------------------------------
    int hit_way;
    logic cache_hit;
    always_comb begin
        hit_way = find_way(req_addr);
        cache_hit = (hit_way >= 0) && tag_valid[get_set(req_addr)][hit_way];
    end

    // PRT 查找（合并 miss 检查）
    int prt_match;
    int prt_free;
    always_comb begin
        prt_match = find_prt_match(req_addr);
        prt_free = find_free_prt();
    end

    // WDB 空闲查找
    int wdb_free;
    always_comb begin
        wdb_free = find_free_wdb();
    end

    // -------------------------------------------------------------------------
    // 接受/拒绝/miss 判定（组合逻辑 always_comb）
    // -------------------------------------------------------------------------
    // 教学注记 - 判定优先级（04 §4.2/§4.3）：
    //   命中 → 直接 accept；
    //   写 miss + global → write-through，不分配 cache 行，进 WDB 送 L2；
    //   写 miss + local → write-back + write-allocate，分配 PRT 做 RFO；
    //   读 miss → 若有同 line PRT 项则合并，否则分配新 PRT 项发往 L2；
    //   PRT 满 / WDB 满 → replay（LSU 重试）。
    always_comb begin
        req_accept = 1'b0;
        req_replay = 1'b0;
        req_miss  = 1'b0;
        noc_req_valid = 1'b0;

        if (req_valid) begin
            if (cache_hit) begin
                // 命中：接受
                req_accept = 1'b1;
            end else begin
                // Miss
                if (req_is_write) begin
                    // 写 miss
                    if (req_space == SP_GLOBAL) begin
                        // Global：write-through + no-write-allocate → 送 L2
                        // 教学注记：global 写绕过 L1 数据阵列，仅可能使
                        //   L1 中已有副本失效（防旧值）。整 128B 被写满且
                        //   全缓存行无效时直接绕过 L1 tag 阵列（04 §4.3）。
                        if (wdb_free >= 0) begin
                            req_accept = 1'b1;
                            noc_req_valid = 1'b1; // write-through 到 L2
                        end else begin
                            req_replay = 1'b1;     // WDB 满 → replay
                        end
                    end else begin
                        // Local：write-back + write-allocate → RFO（Read-For-Own）
                        // miss 时先取 line，合并后进 cache，脏位置位
                        if (prt_free >= 0) begin
                            req_accept = 1'b1;
                            req_miss = 1'b1;
                            noc_req_valid = 1'b1;
                        end else begin
                            req_replay = 1'b1;     // PRT 满 → replay
                        end
                    end
                end else begin
                    // 读 miss
                    // 教学注记 - hits under miss：
                    //   若已有同 line 的 PRT 项，本请求合并进 waiters，
                    //   不重复发下游——非阻塞的关键。
                    if (prt_match >= 0) begin
                        // 合并进已有 PRT 项
                        req_accept = 1'b1;
                        req_miss = 1'b1;
                    end else if (prt_free >= 0) begin
                        // 新 PRT 项，发往 L2
                        req_accept = 1'b1;
                        req_miss = 1'b1;
                        noc_req_valid = 1'b1;
                    end else begin
                        req_replay = 1'b1;         // PRT 满 → replay
                    end
                end
            end
        end
    end

    // NoC 请求信号
    assign noc_req_addr     = req_addr;
    assign noc_req_is_write = req_is_write && (req_space == SP_GLOBAL);
    assign noc_req_sectors  = req_sectors;
    assign noc_req_wdata    = req_wdata;
    assign noc_req_subid    = req_subid;

    // -------------------------------------------------------------------------
    // Load 响应数据（命中时）——按地址 bits[6:5] 选 32B sector
    // -------------------------------------------------------------------------
    logic [127:0] hit_line_data;
    logic [31:0]  hit_sector_data;
    always_comb begin
        hit_line_data = (hit_way >= 0) ? data_array[get_set(req_addr)][hit_way] : 128'h0;
        // sector 选择：128B line 切 4 个 32B sector，由 addr[6:5] 选
        case (req_addr[6:5])
            2'b00: hit_sector_data = hit_line_data[31:0];
            2'b01: hit_sector_data = hit_line_data[63:32];
            2'b10: hit_sector_data = hit_line_data[95:64];
            2'b11: hit_sector_data = hit_line_data[127:96];
        endcase
    end

    assign ld_resp_valid   = req_valid && cache_hit && !req_is_write;
    assign ld_resp_warp    = req_warp;
    assign ld_resp_dst_reg = req_dst_reg;
    assign ld_resp_data    = hit_sector_data;
    assign ld_resp_pc     = req_pc;

    // fill 接受
    assign fill_ack = fill_valid;

    // -------------------------------------------------------------------------
    // 状态更新（时序逻辑 always_ff）
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 复位：所有 tag/data/PRT/WDB 清零，LRU 初始化为 way 编号
            for (int s = 0; s < L1D_SETS; s++) begin
                for (int w = 0; w < L1D_ASSOC; w++) begin
                    tag_valid[s][w]   <= 1'b0;
                    tag_tag[s][w]     <= 52'h0;
                    tag_sectors[s][w] <= 4'h0;
                    lru_age[s][w]     <= w[L1D_ASSOC-1:0];
                    data_array[s][w]  <= 128'h0;
                end
            end
            for (int i = 0; i < L1D_MSHR_ENTRIES; i++) begin
                prt[i].valid <= 1'b0;
                prt[i].state <= 1'b0;
            end
            for (int i = 0; i < L1D_WB_ENTRIES; i++) begin
                wdb[i].valid <= 1'b0;
            end
            access_lines  <= 32'h0;
            miss_sectors  <= 32'h0;
            prt_full_cnt  <= 32'h0;
            assoc_stall_cnt <= 32'h0;
            wdb_full_cnt  <= 32'h0;
        end else begin
            // 统计累加
            if (req_valid && req_accept && !req_miss) begin
                access_lines <= access_lines + 1;
            end
            if (req_miss) begin
                miss_sectors <= miss_sectors + 1;
            end
            if (req_replay && prt_free < 0) begin
                prt_full_cnt <= prt_full_cnt + 1;
            end
            if (req_replay && wdb_free < 0) begin
                wdb_full_cnt <= wdb_full_cnt + 1;
            end

            // -----------------------------------------------------------------
            // 处理 fill 返回（来自 NoC/L2）
            // -----------------------------------------------------------------
            // 教学注记 - fill 路径（04 §4.4）：
            //   按 subid 查 PRT，找到对应项后把数据写入 busy_way 的指定
            //   sector；若所有缺 sector 都到齐，则完成 PRT 项、置 tag_valid；
            //   否则置 state=FILLING 等剩余 sector。然后通知 Arbiter 重放
            //   waiters——重放访问必命中（line 锁定）。
            if (fill_valid && fill_ack) begin
                for (int i = 0; i < L1D_MSHR_ENTRIES; i++) begin
                    if (prt[i].valid && prt[i].subid == fill_subid) begin
                        int set_idx, way;
                        set_idx = get_set(fill_addr);
                        way = prt[i].busy_way;
                        // 把 fill_data 的对应 sector 写入数据阵列
                        for (int s = 0; s < 4; s++) begin
                            if (fill_sectors[s]) begin
                                case (s)
                                    0: data_array[set_idx][way][31:0]    <= fill_data[31:0];
                                    1: data_array[set_idx][way][63:32]   <= fill_data[63:32];
                                    2: data_array[set_idx][way][95:64]   <= fill_data[95:64];
                                    3: data_array[set_idx][way][127:96]  <= fill_data[127:96];
                                endcase
                                tag_sectors[set_idx][way][s] <= 1'b1;
                                prt[i].need_sectors[s] <= 1'b0;
                            end
                        end
                        // 若所有缺 sector 到齐，完成 PRT 项
                        if ((prt[i].need_sectors & ~fill_sectors) == 4'h0) begin
                            prt[i].valid <= 1'b0;
                            tag_valid[set_idx][way] <= 1'b1;
                            tag_tag[set_idx][way]   <= get_tag(fill_addr);
                        end else begin
                            prt[i].state <= 1'b1; // FILLING
                        end
                    end
                end
            end

            // -----------------------------------------------------------------
            // 处理新 miss：分配 PRT 项
            // -----------------------------------------------------------------
            if (req_valid && req_accept && req_miss && !req_is_write) begin
                int way_idx;
                if (prt_match < 0 && prt_free >= 0) begin
                    // 新 PRT 项
                    prt[prt_free].valid       <= 1'b1;
                    prt[prt_free].vline       <= {req_addr[63:7], 7'h0};
                    prt[prt_free].need_sectors <= req_sectors;
                    prt[prt_free].subid       <= req_subid;
                    prt[prt_free].state       <= 1'b0; // SENT
                    // 分配 way：找 LRU（行锁，避免替换冲突）
                    way_idx = find_lru(req_addr);
                    prt[prt_free].busy_way   <= way_idx[2:0];
                    // 若旧行有效，则作废（替换）
                    // 教学注记：local 分区脏行应写回 L2，本实现简化为直接作废
                    if (tag_valid[get_set(req_addr)][way_idx]) begin
                        tag_valid[get_set(req_addr)][way_idx] <= 1'b0;
                        tag_sectors[get_set(req_addr)][way_idx] <= 4'h0;
                    end
                end
            end

            // -----------------------------------------------------------------
            // 处理写命中：更新数据
            // -----------------------------------------------------------------
            if (req_valid && req_accept && cache_hit && req_is_write) begin
                // 更新命中的 sector
                case (req_addr[6:5])
                    0: data_array[get_set(req_addr)][hit_way][31:0]   <= req_wdata;
                    1: data_array[get_set(req_addr)][hit_way][63:32]  <= req_wdata;
                    2: data_array[get_set(req_addr)][hit_way][95:64]  <= req_wdata;
                    3: data_array[get_set(req_addr)][hit_way][127:96] <= req_wdata;
                endcase
                // global write-through：数据已送 L2（在 always_comb 中处理）
            end

            // -----------------------------------------------------------------
            // 处理 WDB 分配（global 写）
            // -----------------------------------------------------------------
            if (req_valid && req_accept && req_is_write && req_space == SP_GLOBAL) begin
                if (wdb_free >= 0) begin
                    wdb[wdb_free].valid <= 1'b1;
                    wdb[wdb_free].addr  <= req_addr;
                    // 按 sector 偏移写入对应 32B 段
                    case (req_addr[6:5])
                        0: wdb[wdb_free].data[31:0]   <= req_wdata;
                        1: wdb[wdb_free].data[63:32]  <= req_wdata;
                        2: wdb[wdb_free].data[95:64]  <= req_wdata;
                        3: wdb[wdb_free].data[127:96] <= req_wdata;
                    endcase
                    wdb[wdb_free].be[req_addr[6:5]] <= 1'b1;
                end
            end

            // -----------------------------------------------------------------
            // LRU 更新（命中时让 hit_way 变最年轻）
            // -----------------------------------------------------------------
            if (req_valid && req_accept && cache_hit) begin
                for (int w = 0; w < L1D_ASSOC; w++) begin
                    if (w == hit_way) begin
                        lru_age[get_set(req_addr)][w] <= 0;
                    end else if (lru_age[get_set(req_addr)][w] < L1D_ASSOC-1) begin
                        lru_age[get_set(req_addr)][w] <= lru_age[get_set(req_addr)][w] + 1;
                    end
                end
            end
        end
    end

    assign stat_access_lines = access_lines;
    assign stat_miss_sectors = miss_sectors;
    assign stat_prt_full     = prt_full_cnt;
    assign stat_assoc_stall  = assoc_stall_cnt;
    assign stat_wdb_full     = wdb_full_cnt;

endmodule
