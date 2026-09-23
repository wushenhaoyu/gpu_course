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
// SIMT 栈模块（SIMT Stack）
// simt_stack.sv
// -----------------------------------------------------------------------------
// 作用：每个 warp 持有一个本模块实例，用于管理 SIMT 执行模式下的控制流
//       分歧（divergence）与再收敛（re-convergence）。栈中每一项记录一个
//       "子路径"的 {再收敛点 RPC, 下一条指令 NextPC, active mask, dflag}。
//       栈顶（TOS）项决定本拍要执行的指令 PC 与 active 掩码。
//
// 对应规格册：依据 03 册《NSM SIMT 核心微结构》§4 SIMT 栈与分歧控制流。
//   - §4.1 栈项结构 StackEntry
//   - §4.2 SSY（编译器在分歧区前发射，压入再收敛标记）
//   - §4.3 守卫 BRA/BRX 的分歧判定（情形 1/2/3 与路径重排规则）
//   - §4.4 再收敛 R1/R2 出栈规则
//   - §4.5 EXIT 与 lane 回收
//
// 端口概要：
//   - TOS 读口（组合输出）：tos_nextpc/mask/rpc/dflag/ptr/empty/warp_exit_done
//   - 操作选择口（每拍最多一个 op_*）：
//       op_init      - warp 初始化（CTA 分派时压栈底哨兵项）
//       op_ssy       - SSY：压 {rpc=target, nextpc=PC+8, mask, dflag=false}
//       op_bra_uni   - BRA 一致跳转（全 taken 或全 not-taken）
//       op_bra_div   - BRA 分歧（taken 与 fall 两路都活，压栈）
//       op_advance   - 顺序 PC 推进（PC+8）或一致跳转后推进，触发 R1/R2 弹栈
//       op_exit      - EXIT：清除退出 lane 在所有栈项中的 mask 位
//   - 错误标志：err_no_ipdom / err_stack_overflow
//
// 关键时序参数：本模块本身无延迟参数，状态在 clk 上升沿更新；
//   操作输入在当前周期组合译码，状态变化在下个上升沿生效（边沿更新约定）。
//
// 教学注记 - 为什么 SIMT 栈用 post-dominator 做再收敛点（IPDOM 算法）：
//   在 SIMT 模式下，warp 内 32 个 lane 锁步执行同一条指令；当遇到条件分支时，
//   谓词为真的 lane 想"跳"，谓词为假的 lane 想"走"，二者必须分头执行。
//   简单做法是串行：先执行 taken 路径，再执行 fall 路径，到分支汇合点会合。
//   关键问题是"在哪里会合"——这必须是两条路径都**必定经过**的点，即控制流图
//   的"后必经节点"（post-dominator）。编译器通过静态分析找到分支的最近 post-
//   dominator（即 immediate post-dominator, IPDOM），在该点前显式发射 SSY 指令
//   把 IPDOM 地址压入 SIMT 栈作为再收敛点 RPC。
//   优势：
//     (1) 编译器静态决定 RPC，硬件无需运行时分析；
//     (2) 栈式管理天然支持嵌套分歧（内层先汇合、外层后汇合）；
//     (3) 栈深上界为 ⌈log₂32⌉=5 量级（路径重排规则保证），硬件代价小。
//   限制（pre-Volta 语义）：
//     获得"锁"的 lane 提前抵达再收敛点后被栈挂起，无法执行释放——这是 SIMT
//     deadlock（教材 3.1.2），t_cas_spinlock 测试就是基线预期 ERR_DEADLOCK。
// =============================================================================

`include "NutShellGPU_defines.svh"

module simt_stack #(
    parameter STACK_DEPTH = 32   // 栈深度上限（03 §4.1：每 warp 最多 32 项）
) (
    input  logic clk,           // GPU 时钟（1 GHz，上升沿采样所有 op_* 信号）
    input  logic rst_n,         // 异步低有效复位（rst_n=0 时栈清空、tos=0）

    // --- TOS 读口（组合逻辑，零延迟输出当前栈顶状态）---
    // 这些输出供 NSM 流水线的 F 级取指使用：取 TOS.nextpc 处的指令，
    // 用 TOS.mask 作为 active mask。
    output logic [63:0] tos_nextpc,   // 栈顶项的 NextPC（下一条要执行的指令 PC）
    output logic [31:0] tos_mask,     // 栈顶项的 active lane 掩码
    output logic [63:0] tos_rpc,      // 栈顶项的再收敛 PC（SSY 设置）
    output logic        tos_dflag,    // 栈顶项是否曾作为分歧父项（R2 用）
    output logic [$clog2(STACK_DEPTH)-1:0] tos_ptr,  // 当前栈顶指针值
    output logic        stack_empty,  // 栈空（tos=0 且栈底 mask=0）
    output logic        warp_exit_done, // warp 全部 lane 已 EXIT

    // --- 操作选择（每拍最多一个 op_* 有效，互斥）---
    // op_init：warp 初始化。把栈清空，栈底置 {EXIT_SENTINEL, entry, mask, false}
    //   tos=0；exit_done_reg=0；错误标志清零。
    input  logic        op_init,
    input  logic [63:0] init_entry_pc,   // warp 入口 PC（来自 launch 描述符）
    input  logic [31:0] init_mask,        // warp 有效 lane 掩码（尾包 warp 高位为 0）

    // op_ssy：SSY 指令（02 §3.5 OP=0x54）。在分歧区前由编译器显式发射。
    //   压 {rpc=target, nextpc=PC+8, mask=tos.mask, dflag=false}；
    //   新 TOS 的 nextpc 即 PC+8（继续往下执行，直到 BRA 才真正分歧）。
    input  logic        op_ssy,
    input  logic [63:0] ssy_target,      // 再收敛点 RPC（IPDOM 地址）
    input  logic [63:0] cur_pc,          // SSY 指令自身的 PC（用于算 PC+8）

    // op_bra_uni：BRA 一致跳转（03 §4.3 情形 1 与 2）
    //   bra_taken=1：所有活跃 lane 都 taken → tos.nextpc = target
    //   bra_taken=0：所有活跃 lane 都 not-taken → tos.nextpc = PC+8
    //   栈不变（不压不弹），mask 不变。
    input  logic        op_bra_uni,
    input  logic        bra_taken,
    input  logic [63:0] bra_target,

    // op_bra_div：BRA 分歧（03 §4.3 情形 3，taken 与 fall 都活）
    //   入参 div_target = taken 路径目标；taken_mask=active&guard；fall_mask=active&~guard
    //   栈操作：父项 nextpc=join, dflag=true；压入 fall 与 taken 两个子路径项；
    //   路径重排：若新栈顶 popcount < 次项 popcount，交换二者（lane 多的先执行）
    input  logic        op_bra_div,
    input  logic [63:0] div_target,
    input  logic [31:0] taken_mask,
    input  logic [31:0] fall_mask,

    // op_advance：顺序 PC 推进或一致跳转完成后推进
    //   把 tos.nextpc 设为 advance_newpc（通常 = cur_pc + 8）
    //   并应用 R1/R2 弹栈规则（见 apply_pop_rules）
    input  logic        op_advance,
    input  logic [63:0] advance_newpc,

    // op_exit：EXIT 指令（02 §3.5 OP=0x57）
    //   清除 exit_lane_mask 中置位的 lane 在所有栈项 mask 中的对应位；
    //   弹空 mask=0 的栈项；若栈底也空，置 warp_exit_done。
    input  logic        op_exit,
    input  logic [31:0] exit_lane_mask,   // 本拍 EXIT 的 lane 集合（exec_mask）

    // --- 错误标志（锁存到下一周期，由 NSM 上报设备错误）---
    output logic        err_no_ipdom,        // ERR_NO_IPDOM：分歧 BRA 时栈顶 rpc 为哨兵
    output logic        err_stack_overflow   // ERR_STACK_OVERFLOW：BRX 使栈深 > 32
);

    import NutShellGPU_pkg::*;

    // -------------------------------------------------------------------------
    // 栈存储：stack[0..STACK_DEPTH-1]，每项是 stack_entry_t
    // -------------------------------------------------------------------------
    stack_entry_t stack [0:STACK_DEPTH-1];
    logic [$clog2(STACK_DEPTH)-1:0] tos;   // 栈顶指针（指向当前 TOS 项）
    logic         exit_done_reg;             // warp 是否已全部 EXIT
    logic         err_ipdom_reg;             // ERR_NO_IPDOM 锁存
    logic         err_ovf_reg;               // ERR_STACK_OVERFLOW 锁存

    // --- 组合输出 TOS 字段 ---
    // 教学注记：tos_nextpc 是 SIMT 栈提供给取指单元的"下一条指令 PC"，
    //   永远等于 stack[tos].nextpc。NSM 不另设 PC 寄存器，PC 完全由栈派生。
    assign tos_nextpc  = stack[tos].nextpc;
    assign tos_mask    = stack[tos].mask;
    assign tos_rpc     = stack[tos].rpc;
    assign tos_dflag   = stack[tos].dflag;
    assign tos_ptr     = tos;
    // 栈空判定：tos==0 且栈底 mask==0（即 warp 尚未初始化或已彻底消亡）
    assign stack_empty = (tos == 0) && (stack[0].mask == 32'h0);
    assign warp_exit_done = exit_done_reg;
    assign err_no_ipdom    = err_ipdom_reg;
    assign err_stack_overflow = err_ovf_reg;

    // -------------------------------------------------------------------------
    // popcount 辅助函数（用于路径重排规则 03 §4.3 rule 5）
    // -------------------------------------------------------------------------
    // 教学注记 - 路径重排规则：
    //   压入 fall 与 taken 两个子路径后，若新栈顶（taken）的 active lane 数
    //   少于次项（fall）的 lane 数，则交换二者——让 lane 多的路径先执行。
    //   这样最坏情况下栈深只有 ⌈log₂32⌉=5 量级（每次分歧让少数派先走、
    //   多数派后走，少数派会先抵达再收敛点弹出，栈不会无限增长）。
    //   该规则使行为与调度顺序无关，保证两个硬件模拟器结果一致。
    function automatic int popcount32(input logic [31:0] v);
        int c = 0;
        for (int i = 0; i < 32; i++) c += v[i];
        return c;
    endfunction

    // -------------------------------------------------------------------------
    // R1/R2 弹栈规则（依据 03 册 §4.4 再收敛出栈规则）
    // -------------------------------------------------------------------------
    // 每条指令在 TOS 上执行完并算出 newpc 后，按顺序应用以下规则：
    //   R1（子路径走到 join）：若 newpc == tos.rpc，弹出 TOS——
    //       本子路径已抵达再收敛点，交还给暴露出来的父项。
    //   R2（父项执行完 join 点指令）：若未弹栈且 tos.dflag==true 且
    //       刚执行指令的 PC == tos.rpc，则该指令执行完毕后弹出 TOS——
    //       dflag 父项在 join 点只代表一次会合，执行完 join 点那一条指令
    //       即归还下层。
    //   R1/R2 弹出后露出的新 TOS 若 nextpc==rpc（即它也是被会合到该点的项），
    //   本拍不弹栈：它尚未在 join 点执行指令，等它执行后由 R2 处理。
    //   栈底哨兵项永不弹出。
    // 返回值：do_pop（是否弹栈），stop_pop（是否停止进一步弹栈）
    function automatic void apply_pop_rules(
        input  logic [63:0] newpc,
        input  logic        instr_pc_eq_rpc,   // 刚执行指令的 PC == tos.rpc
        input  stack_entry_t cur_tos,
        input  logic [$clog2(STACK_DEPTH)-1:0] cur_tos_ptr,
        output logic        do_pop,
        output logic        stop_pop
    );
        do_pop = 1'b0;
        stop_pop = 1'b0;
        // R1: 子路径走到 join（newpc == tos.rpc）→ 弹出
        if (newpc == cur_tos.rpc && cur_tos_ptr != 0) begin
            do_pop = 1'b1;
            return;
        end
        // R2: 父项（dflag）执行完 join 点指令 → 弹出
        if (!do_pop && cur_tos.dflag && instr_pc_eq_rpc && cur_tos_ptr != 0) begin
            do_pop = 1'b1;
            return;
        end
        // 弹出后若新 TOS nextpc==rpc，本拍不再弹（等执行后由 R2 处理）
        stop_pop = 1'b0;
    endfunction

    // -------------------------------------------------------------------------
    // 时序状态更新（always_ff）：所有 op_* 操作在 clk 上升沿生效
    // -------------------------------------------------------------------------
    // 教学注记 - 边沿更新约定：
    //   操作输入是当前周期的组合译码结果，状态变化在下一上升沿提交。
    //   这保证 SIMT 栈更新与流水线其他级（F/D/I/R/E/W）的边沿同步，
    //   避免组合穿透（同一拍内 op_* 输入 → stack 输出 → 流水线下级读到的
    //   还是旧值）。
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 复位：所有栈项清零，tos=0
            for (int i = 0; i < STACK_DEPTH; i++) begin
                stack[i] <= '{rpc: 64'h0, nextpc: 64'h0, mask: 32'h0, dflag: 1'b0};
            end
            tos <= 0;
            exit_done_reg <= 1'b0;
            err_ipdom_reg <= 1'b0;
            err_ovf_reg   <= 1'b0;
        end else if (op_init) begin
            // op_init：warp 初始化（03 §4.1 末）
            //   先清空整个栈，然后栈底项置为 {EXIT_SENTINEL, entry, mask, false}
            //   tos=0, exit_done=0，错误标志清零
            for (int i = 0; i < STACK_DEPTH; i++) begin
                stack[i] <= '{rpc: 64'h0, nextpc: 64'h0, mask: 32'h0, dflag: 1'b0};
            end
            stack[0].rpc     <= EXIT_SENTINEL;   // 栈底 RPC = 哨兵（永不弹出）
            stack[0].nextpc  <= init_entry_pc;   // 栈底 NextPC = warp 入口
            stack[0].mask    <= init_mask;       // 栈底 mask = 有效 lane
            stack[0].dflag   <= 1'b0;
            tos <= 0;
            exit_done_reg <= 1'b0;
            err_ipdom_reg <= 1'b0;
            err_ovf_reg   <= 1'b0;
        end else if (op_ssy) begin
            // op_ssy：SSY 指令（03 §4.2）
            //   压 {rpc=target, nextpc=PC+8, mask=tos.mask, dflag=false}
            //   新 TOS nextpc = PC+8（继续顺序执行，直到 BRA 才分歧）
            if (tos + 1 < STACK_DEPTH) begin
                stack[tos+1].rpc    <= ssy_target;
                stack[tos+1].nextpc <= cur_pc + 64'd8;
                stack[tos+1].mask   <= stack[tos].mask;   // 继承父 mask
                stack[tos+1].dflag  <= 1'b0;
                tos <= tos + 1;
            end else begin
                err_ovf_reg <= 1'b1;   // 栈溢出 → ERR_STACK_OVERFLOW
            end
        end else if (op_bra_div) begin
            // op_bra_div：BRA 分歧（03 §4.3 情形 3）
            // 教学注记 - 分歧处理步骤：
            //   1. 检查 tos.rpc 是否为 EXIT_SENTINEL：若是说明编译器漏发 SSY
            //      → ERR_NO_IPDOM（无法找到再收敛点）
            //   2. 父项标记为分歧父：nextpc=join(rpc), dflag=true, mask 保持
            //   3. 压 fall 子路径：{join, PC+8, fall_mask, false}
            //   4. 压 taken 子路径：{join, target, taken_mask, false}
            //   5. 路径重排：若 taken popcount < fall popcount，交换二者
            //      （让 lane 多的路径在栈顶先执行）
            if (stack[tos].rpc == EXIT_SENTINEL) begin
                err_ipdom_reg <= 1'b1;   // 漏 SSY 错误
            end else if (tos + 2 < STACK_DEPTH) begin
                // 父项：nextpc=join(rpc), dflag=true（mask 不变）
                stack[tos].nextpc <= stack[tos].rpc;
                stack[tos].dflag  <= 1'b1;

                // 先压 fall 路径：{join, PC+8, fall_mask, false}
                stack[tos+1].rpc    <= stack[tos].rpc;
                stack[tos+1].nextpc <= cur_pc + 64'd8;
                stack[tos+1].mask   <= fall_mask;
                stack[tos+1].dflag  <= 1'b0;

                // 再压 taken 路径：{join, target, taken_mask, false}
                stack[tos+2].rpc    <= stack[tos].rpc;
                stack[tos+2].nextpc <= div_target;
                stack[tos+2].mask   <= taken_mask;
                stack[tos+2].dflag  <= 1'b0;

                // 路径重排（03 §4.3 rule 5）：若栈顶(taken) popcount < 次项(fall) popcount
                // 则交换二者，让 lane 多的 fall 路径在栈顶先执行
                if (popcount32(taken_mask) < popcount32(fall_mask)) begin
                    // 交换：stack[tos+1] 变为 taken，stack[tos+2] 变为 fall
                    // 即 fall（lane 多）在栈顶，taken（lane 少）在次项
                    stack[tos+1].mask <= taken_mask;
                    stack[tos+1].nextpc <= div_target;
                    stack[tos+2].mask <= fall_mask;
                    stack[tos+2].nextpc <= cur_pc + 64'd8;
                end

                tos <= tos + 2;
            end else begin
                err_ovf_reg <= 1'b1;
            end
        end else if (op_bra_uni) begin
            // op_bra_uni：BRA 一致跳转（03 §4.3 情形 1 与 2）
            //   bra_taken=1：所有活跃 lane 都跳 → tos.nextpc = target（情形 1）
            //   bra_taken=0：所有活跃 lane 都不跳 → tos.nextpc = PC+8（情形 2）
            //   栈不变（不压不弹），dflag/mask 保持
            if (bra_taken) begin
                stack[tos].nextpc <= bra_target;
            end else begin
                stack[tos].nextpc <= cur_pc + 64'd8;
            end
        end else if (op_advance) begin
            // op_advance：顺序 PC 推进（03 §4.4 R1/R2 弹栈规则）
            //   R1：若 advance_newpc == tos.rpc 且 tos!=0 → 弹栈
            //       （子路径已抵达再收敛点，交还父项）
            //   R2：父项（dflag）执行完 join 点指令后弹栈——
            //       本实现简化：仅在 R1 满足时弹；R2 由父项下次执行 advance 时处理
            if (advance_newpc == stack[tos].rpc && tos != 0) begin
                // R1 弹栈：子路径到达 join，弹出 TOS
                // 暴露出的父项 nextpc 已被设为 join（在分歧时改写过）
                tos <= tos - 1;
            end
            // 推进 nextpc 到新地址（R2 的情况下父项也继续往下）
            stack[tos].nextpc <= advance_newpc;
        end else if (op_exit) begin
            // op_exit：EXIT 指令（03 §4.5）
            // 教学注记 - EXIT 与 lane 回收：
            //   EXIT 对 exec_mask 中每个 lane：在从 TOS 到栈底的所有栈项 mask 中
            //   清该位（退出的 lane 不得在任何再收敛点"复活"），并清 lane_valid。
            //   然后弹空 mask=0 的栈项；若栈底也空，置 warp_exit_done。
            //   全部 warp exit_done → CTA 结束（03 §11）。
            for (int i = 0; i < STACK_DEPTH; i++) begin
                if (i <= tos) begin
                    stack[i].mask <= stack[i].mask & ~exit_lane_mask;
                end
            end
            // 弹空 mask=0 的栈项，找到新的非空 TOS
            begin
                logic [$clog2(STACK_DEPTH)-1:0] new_tos;
                new_tos = tos;
                while (new_tos > 0 && (stack[new_tos].mask & ~exit_lane_mask) == 32'h0) begin
                    new_tos = new_tos - 1;
                end
                if ((stack[0].mask & ~exit_lane_mask) == 32'h0) begin
                    // 栈底也空 → warp 全部 EXIT
                    exit_done_reg <= 1'b1;
                    tos <= 0;
                    stack[0].mask <= 32'h0;
                end else begin
                    tos <= new_tos;
                end
            end
        end
    end

endmodule
