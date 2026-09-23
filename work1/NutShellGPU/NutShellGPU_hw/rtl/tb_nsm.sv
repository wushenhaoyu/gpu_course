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
// NSM 模块的测试平台（Testbench）
// tb_nsm.sv
// -----------------------------------------------------------------------------
// 作用：验证 nsm.sv 的基本功能。测试程序只有两条指令：
//       PC=0x00: MOV32I R0, 0x42   ; 把立即数 0x42 写入 R0
//       PC=0x08: EXIT              ; warp 退出
//   预期结果：warp 0 / lane 0 的 R0 == 0x42，warp_done 拉高。
//
// 对应规格册：验证 03 册 §2-§9 的 NSM 流水线、§9 EXIT 处理；02 册的
//   指令编码（MOV32I、EXIT 格式）。
//
// 测试流程（4 个 Phase）：
//   Phase 1：复位后逐字灌入 2 条指令到 IMEM
//   Phase 2：通过 warp_init 口启动 warp 0（entry_pc=0, mask=全 1）
//   Phase 3：等待 warp_done（warp 执行完 EXIT）
//   Phase 4：通过 RF 旁路读口读 R0，校验 == 0x42，打印 PASS/FAIL
//
// 教学注记 - 指令编码（详见 02 册 NTAS 手册 + 03 册）：
//   NTAS 指令固定 64-bit。Header 16 位 = OP(7) + G(1) + GN(1) + GP(3) + M(4)。
//   PC=0x00 MOV32I R0, 0x42：
//     OP=0000010(0x02, MOV32I), G=0, GN=0, GP=000, M=0000
//     Header = 0000010_0_0_000_0000 = 0x0400
//     Body  : d=0x00 (R0), imm32=0x00000042, X=0x00
//     64-bit = 0x0400_0000_0000_4200
//     注：OP 字段占 bit63:57，故 byte7 = OP<<1 = 0x02<<1 = 0x04（与 EXIT 一致：0x57<<1=0xAE）
//   PC=0x08 EXIT：
//     OP=1010111(0x57, EXIT), 其余字段 0
//     Header = 0xAE00
//     64-bit = 0xAE00_0000_0000_0000
//
// 教学注记 - 时钟：1 GHz = 周期 1ns，半周期 0.5ns。
//   与规格册 LAT_* 参数对齐（1 周期 ≈ 1ns）。
// =============================================================================

`timescale 1ns/1ps    // 时间精度：1ns 单位 / 1ps 分辨率

module tb_nsm;

    import NutShellGPU_pkg::*;    // 复用 ISA 类型与常量

    // --- 时钟与复位 ---
    logic clk = 0;          // 自由运行的时钟（initial 块中翻转）
    logic rst_n = 0;        // 复位（低有效）；初始保持复位状态

    // --- NSM 接口信号（与 nsm.sv 端口一一对应） ---
    logic        imem_load;
    logic [10:0] imem_addr;
    logic [63:0] imem_data;
    logic        warp_init;
    logic [5:0]  warp_init_id;          // 6 位宽（NUM_WARPS=64 时 $clog2=6）
    logic [63:0] warp_init_entry_pc;
    logic [31:0] warp_init_mask;
    logic [7:0]  rf_rd_reg;
    logic [31:0] rf_rd_data;
    logic        warp_done;
    logic [31:0] inst_count;
    logic [31:0] cycle_count;
    logic [63:0] cur_pc_out;

    // =========================================================================
    // 测试程序编码
    // =========================================================================
    // 教学注记：这两条 64-bit 字直接灌入 IMEM[0] 和 IMEM[1]。MOV32I 在地址
    //   0，EXIT 在地址 8（字节）。warp 0 从 PC=0 启动，顺序执行 MOV32I→EXIT。
    // =========================================================================

    logic [63:0] TEST_PROG [0:1];
    assign TEST_PROG[0] = 64'h0400_0000_0000_4200;  // MOV32I R0, 0x42 (OP<<1=0x04)
    assign TEST_PROG[1] = 64'hAE00_0000_0000_0000;  // EXIT

    // =========================================================================
    // 例化待测设计（DUT = Design Under Test）
    // =========================================================================
    nsm #(
        .NSM_ID(0),
        .NUM_WARPS(64),
        .NUM_LANES(32),
        .IMEM_SIZE(1024)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .imem_load(imem_load),
        .imem_addr(imem_addr),
        .imem_data(imem_data),
        .warp_init(warp_init),
        .warp_init_id(warp_init_id),
        .warp_init_entry_pc(warp_init_entry_pc),
        .warp_init_mask(warp_init_mask),
        .rf_rd_reg(rf_rd_reg),
        .rf_rd_data(rf_rd_data),
        .warp_done(warp_done),
        .inst_count(inst_count),
        .cycle_count(cycle_count),
        .cur_pc_out(cur_pc_out)
    );

    // =========================================================================
    // 时钟生成：1 GHz = 1ns 周期
    // =========================================================================
    // 教学注记：每 0.5ns 翻转一次 clk，故周期 = 1ns。这与规格册 LAT_* 参数
    //   对齐（1 周期 = 1ns = 1 GHz）。
    // =========================================================================
    always #0.5 clk = ~clk;

    // =========================================================================
    // 测试主流程（initial 块）
    // =========================================================================
    initial begin
        // --- 信号初始化（避免 X 态传播） ---
        imem_load = 0;
        imem_addr = 0;
        imem_data = 0;
        warp_init = 0;
        warp_init_id = 0;
        warp_init_entry_pc = 0;
        warp_init_mask = 32'hFFFFFFFF;   // 全 32 lane 有效
        rf_rd_reg = 0;

        // =================================================
        // 复位：保持 5 个时钟周期后释放
        // =================================================
        // 教学注记：rst_n=0 期间所有寄存器异步复位。多保持几拍确保
        //   时钟边沿已稳定、RF 完整清零。
        rst_n = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // =================================================
        // Phase 1：灌入指令存储
        // =================================================
        $display("[%0t] Loading instruction memory...", $time);

        // 写 MOV32I 到 IMEM[0]
        imem_load = 1;
        imem_addr = 0;
        imem_data = TEST_PROG[0];
        @(posedge clk);

        // 写 EXIT 到 IMEM[1]
        imem_addr = 1;
        imem_data = TEST_PROG[1];
        @(posedge clk);

        imem_load = 0;     // 关闭加载
        @(posedge clk);

        $display("[%0t] Instruction memory loaded:", $time);
        $display("  imem[0] = 0x%016h (MOV32I R0, 0x42)", TEST_PROG[0]);
        $display("  imem[1] = 0x%016h (EXIT)", TEST_PROG[1]);

        // =================================================
        // Phase 2：启动 warp 0
        // =================================================
        // 教学注记：warp_init 脉冲让 nsm 在 PS_IDLE 拍触发 simt_op_init，
        //   SIMT 栈推入初始 TOS（nextpc=0, mask=0xFFFFFFFF），进入 PS_FETCH。
        $display("[%0t] Initializing warp 0...", $time);
        warp_init = 1;
        warp_init_id = 0;
        warp_init_entry_pc = 64'h0000_0000_0000_0000;
        warp_init_mask = 32'hFFFFFFFF;   // 全 32 lane active
        @(posedge clk);
        warp_init = 0;     // 单拍脉冲

        // =================================================
        // Phase 3：等待 warp 完成
        // =================================================
        $display("[%0t] Running warp 0...", $time);
        wait (warp_done == 1'b1);    // 阻塞直到 warp 0 EXIT

        $display("[%0t] Warp 0 completed after %0d cycles, %0d instructions",
                 $time, cycle_count, inst_count);

        // =================================================
        // Phase 4：结果校验
        // =================================================
        // 通过 RF 旁路读口读 R0（固定读 warp 0 / lane 0）
        rf_rd_reg = 8'd0;    // R0
        @(posedge clk);       // 等一拍让组合读稳定

        $display("[%0t] Verification:", $time);
        $display("  R0 (warp 0, lane 0) = 0x%08h", rf_rd_data);

        if (rf_rd_data == 32'h0000_0042) begin
            $display("  [PASS] R0 == 0x42 (expected 0x42)");
            $display("\n========================================");
            $display("       TEST RESULT: PASS");
            $display("========================================");
        end else begin
            $display("  [FAIL] R0 == 0x%08h (expected 0x42)", rf_rd_data);
            $display("\n========================================");
            $display("       TEST RESULT: FAIL");
            $display("========================================");
        end

        // --- 流水线统计 ---
        $display("\n--- Pipeline Statistics ---");
        $display("  Total cycles:     %0d", cycle_count);
        $display("  Instructions:     %0d", inst_count);
        $display("  IPC:              %.3f", real'(inst_count) / real'(cycle_count));

        $finish;   // 结束仿真
    end

    // =========================================================================
    // 超时看门狗（Timeout watchdog）
    // =========================================================================
    // 教学注记：若 10000ns（≈10000 周期）内 warp 仍未完成，强制结束并报
    //   FAIL——防止仿真因 bug 死锁。打印关键状态辅助调试。
    // =========================================================================
    initial begin
        #10000;
        $display("\n[TIMEOUT] Simulation did not complete within 10000ns");
        $display("  warp_done = %b", warp_done);
        $display("  cur_pc = 0x%016h", cur_pc_out);
        $display("  cycle_count = %0d", cycle_count);
        $display("  inst_count = %0d", inst_count);
        $display("       TEST RESULT: FAIL (timeout)");
        $finish;
    end

endmodule
