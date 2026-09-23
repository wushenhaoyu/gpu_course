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
// NutShellGPU SystemVerilog 包定义文件
// NutShellGPU_pkg.sv
// -----------------------------------------------------------------------------
// 作用：本 package 集中声明 NutShellGPU 硬件模拟器（周期级 RTL）所有模块共享的：
//   (1) 系统级参数（parameter）
//   (2) 指令集/格式/存储空间/数据宽度/缓存策略等枚举（typedef enum）
//   (3) 解码后的指令/SIMT 栈项/记分牌项/操作数/collector 项/访存请求包/NoC flit
//       等结构体（typedef struct）
//   (4) NTAS1 64 位定长指令解码函数 decode_inst()
//   (5) 地址散列、寄存器堆 bank 映射、分支目标计算等辅助函数
//
// 对应规格册：
//   - 依据 00 册《总览与配置参数》§3 系统级规格总表、§5 延迟与吞吐参考表、
//     §8 标准配置文件 NutShellGPU.config 的全部数值；
//   - 依据 02 册《NTAS1 指令集手册》§2 指令字字段、§3 操作码表、§4 特殊寄存器
//     编号、§5 修饰与比较码；
//   - 依据 03 册《NSM SIMT 核心微结构》§4 SIMT 栈项结构、§6 记分牌项、
//     §7 寄存器堆物理组织、§8 collector；
//   - 依据 04 册《存储系统微结构》§4 L1D 参数、§7.1 地址散列、§8 NoC、
//     §9 L2、§12 DRAM 时序；
//   - 依据 05 册《模拟器实现契约》§2.1 Inst 结构与 decode() 签名、§6 错误码。
//
// 时序关键参数（@1 GHz GPU 时钟域，1ns ≈ 1 周期）：
//   - SP 流水线 4 拍，SFU/DP 8 拍，BRU 1 拍；
//   - L1D 命中 30 拍，L2 命中 180 拍，DRAM 行命中 400 拍；
//   - DRAM 行缓冲 FSM 时序：tCAS=12, tRCD=12, tRP=12, tRAS=28, tRC=40, tWR=12。
// =============================================================================

package NutShellGPU_pkg;

    // =========================================================================
    // 系统级参数（依据 00 册 §3 芯片级规格总表）
    // =========================================================================
    // 芯片有 16 个 NSM（Newton Streaming Multiprocessor，对标 NVIDIA SM），
    // 每 4 个 NSM 组成 1 个 GPC，共 4 个 GPC。compute 负载下 GPC 仅做时钟/电源
    // 域划分，功能上不影响。6 个 Memory Partition（MP）共享全片 L2 与 GDDR5 通道。
    parameter int NUM_NSM               = 16;     // 全片 NSM 数（00 §3）
    parameter int NUM_GPC               = 4;      // GPC 数
    parameter int NSM_PER_GPC           = 4;      // 每 GPC 内 NSM 数
    parameter int NUM_MP                = 6;      // Memory Partition 数（= GDDR5 通道数）
    parameter int L2_SLICES_PER_MP      = 2;      // 每 MP 含 2 个 L2 slice
    parameter int NUM_L2_SLICES         = NUM_MP * L2_SLICES_PER_MP; // 12（全片 L2 slice 总数）
    parameter int WARP_SIZE             = 32;     // warp 固定 32 线程（01 §2）
    parameter int MAX_WARPS_PER_NSM     = 64;     // 每 NSM 最多驻留 64 warp（= 2048 线程）
    parameter int MAX_CTAS_PER_NSM     = 16;      // 每 NSM 最多 16 个 CTA 槽位
    parameter int MAX_THREADS_PER_CTA   = 1024;   // 单 CTA 线程数上限

    // -------------------------------------------------------------------------
    // 寄存器堆物理参数（依据 00 册 §3；03 册 §7 寄存器堆物理组织）
    // -------------------------------------------------------------------------
    // 全 NSM 共 256 KiB = 65536 个 32 位物理寄存器。RF 划分为 4 个单端口逻辑 bank，
    // bank 映射采用 swizzled 方案 bank(w, regnum) = (regnum + warp_slot) mod 4，
    // 目的是让连续 warp 的同号寄存器自然分散到不同 bank，减少 bank 冲突概率。
    // 每线程架构寄存器 R0..R254（255 个）+ RZ（R255，常量零寄存器，写忽略）。
    parameter int RF_PHYSICAL_ENTRIES    = 65536;  // 256 KiB 总容量
    parameter int RF_BANKS               = 4;      // 4 个单端口逻辑 bank
    parameter int NUM_REGS              = 256;     // R0..R255
    parameter int RZ                    = 255;     // 零寄存器编号（02 册 §1）
    parameter int NUM_PREDICATES        = 7;       // 谓词 P0..P6（02 册 §1）
    parameter int PT                    = 7;       // PT 编码值（恒真谓词，02 §2.1）
    parameter int REG_NUM_BITS          = 8;       // 寄存器号位宽

    // -------------------------------------------------------------------------
    // 统一 SRAM 容量（依据 00 册 §3；04 册 §1）
    // -------------------------------------------------------------------------
    // 每 NSM 内 64 KiB SRAM 在 shared memory 与 L1D 之间静态切分：
    //   - 默认 48 KiB shared + 16 KiB L1D；
    //   - 二者总和必须严格等于 64 KiB（05 册 §4 合法性断言）。
    // shared 为 32 bank × 32 bit 宽，bank 号 = (字节地址 / 4) mod 32（04 册 §2.1）。
    parameter int UNIFIED_SRAM_SIZE      = 65536;   // 64 KiB
    parameter int SMEM_SIZE              = 48*1024; // 默认 shared 容量
    parameter int L1D_SIZE               = 16*1024; // 默认 L1D 容量
    parameter int SMEM_BANKS             = 32;      // shared bank 数（04 §2.1）
    parameter int SMEM_BANK_WIDTH_BYTES  = 4;       // 每 bank 4 字节宽（04 §2.1）

    // -------------------------------------------------------------------------
    // L1D 数据 cache 参数（依据 04 册 §4.1）
    // -------------------------------------------------------------------------
    // 16 KiB，4 路组相联，128 B/line，每 line 切为 4×32B sector（sector 有效位
    // 独立），共 32 组，LRU 替换。虚拟索引虚拟标识（VI/VT），用虚拟 sector
    // 地址直接查 tag，miss 出站时才经 MMU 翻译（基线恒等页映射，无 TLB 延迟）。
    // L1D_SETS = L1D_SIZE / (L1D_LINE_BYTES * L1D_ASSOC) = 16384 / (128 * 4) = 32 组
    // L1D_SECTORS_PER_LINE = L1D_LINE_BYTES / L1D_SECTOR_BYTES = 128 / 32 = 4
    parameter int L1D_LINE_BYTES        = 128;     // cache line 128 B
    parameter int L1D_SECTOR_BYTES      = 32;      // sector 32 B（突发原子）
    parameter int L1D_ASSOC             = 4;       // 4 路组相联
    parameter int L1D_SETS              = L1D_SIZE / (L1D_LINE_BYTES * L1D_ASSOC); // 32
    parameter int L1D_SECTORS_PER_LINE  = L1D_LINE_BYTES / L1D_SECTOR_BYTES; // 4
    parameter int L1D_MSHR_ENTRIES      = 32;      // PRT（Pending Request Table）32 项
    parameter int L1D_WB_ENTRIES         = 16;      // WDB（Write Data Buffer）16 项
    parameter int LSU_REPLAY_ENTRIES    = 16;      // LSU replay 队列 16 项（03 §9）

    // -------------------------------------------------------------------------
    // I-Cache 参数（依据 04 册 §5）
    // -------------------------------------------------------------------------
    // 8 KiB，4 路，128 B line，LRU。物理上独立阵列，不与 L1D 争端口。
    // 8 个 MSHR；缺失经 NoC 到 L2（指令在 global 代码段，L2 正常缓存）。
    parameter int ICACHE_SIZE           = 8192;
    parameter int ICACHE_ASSOC          = 4;
    parameter int ICACHE_LINE_BYTES     = 128;
    parameter int ICACHE_MSHRS          = 8;

    // 常量 cache 参数（依据 04 册 §6）：8 KiB，只读，warp 同址广播。
    parameter int CONST_CACHE_SIZE      = 8192;

    // CTA barrier 槽位（依据 03 册 §10）：每 NSM 16 个，按 (CTA槽, bar_id) 索引。
    parameter int BARRIER_SLOTS         = 16;

    // -------------------------------------------------------------------------
    // 前端参数（依据 00 册 §5 前端参数表；03 册 §3、§6、§8）
    // -------------------------------------------------------------------------
    // 每 warp I-Buffer 2 个槽（非阻塞滑动窗口，最多预取 2 条指令）。
    // Coon 式记分牌每 warp 4 项（追踪未完成目的寄存器，03 §6）。
    // Operand collector 8 个 unit × 4 操作数槽（03 §8.1）。
    parameter int IBUF_ENTRIES_PER_WARP = 2;
    parameter int SCOREBOARD_ENTRIES_PER_WARP = 4;
    parameter int NUM_COLLECTORS        = 8;
    parameter int OPERAND_SLOTS_PER_COLLECTOR = 4;

    // -------------------------------------------------------------------------
    // 名义延迟表（依据 00 册 §5；GPU 周期 @1 GHz，1ns ≈ 1 cycle）
    // -------------------------------------------------------------------------
    // 延迟 = 从发射（issue）到结果可写回寄存器堆的周期数，不含排队。
    // BRU 1 拍；SP INT32/FP32 4 拍；SFU/DP 8 拍；shared 命中 2 拍；
    // L1D 命中 30 拍；L2 命中 180 拍；DRAM 行命中 400 拍；BAR.SYNC 唤醒 4 拍。
    parameter int LAT_BRU               = 1;
    parameter int LAT_SP               = 4;
    parameter int LAT_SFU              = 8;
    parameter int LAT_DP               = 8;
    parameter int LAT_SMEM_HIT          = 2;
    parameter int LAT_L1_HIT            = 30;
    parameter int LAT_L2_HIT            = 180;
    parameter int LAT_DRAM_ROW_HIT      = 400;
    parameter int LAT_BARRIER_RESUME    = 4;

    // -------------------------------------------------------------------------
    // L2 cache 参数（依据 04 册 §9）
    // -------------------------------------------------------------------------
    // 每 slice 128 KiB，8 路，128B line，4×32B sector，128 组，LRU，64 MSHR/slice。
    // 全片 L2 = 128 KiB × 2 slice × 6 MP = 1536 KiB = 1.5 MiB。
    // ROP 原子 cache 每 MP 2 KiB（= 16 行 128B）。
    parameter int L2_SLICE_SIZE         = 131072;  // 128 KiB
    parameter int L2_ASSOC              = 8;
    parameter int L2_LINE_BYTES         = 128;
    parameter int L2_SECTOR_BYTES      = 32;
    parameter int L2_SETS              = L2_SLICE_SIZE / (L2_LINE_BYTES * L2_ASSOC); // 128
    parameter int L2_SECTORS_PER_LINE  = L2_LINE_BYTES / L2_SECTOR_BYTES; // 4
    parameter int L2_MSHR_PER_SLICE     = 64;
    parameter int L2_ROP_ATOM_CACHE    = 2048;

    // -------------------------------------------------------------------------
    // NoC 片上互连参数（依据 04 册 §8）
    // -------------------------------------------------------------------------
    // 全交叉开关：16 个 NSM 端口 + 6 个 MP 端口 = 22 端口。
    // flit 8 字节负载；2 条虚通道 REQ/RSP，每输入口各 4 flit 缓冲；
    // 路由+仲裁 1 周期（router_pipe），链路 1 周期/flit；地址交织粒度 256 B。
    parameter int NOC_NSM_PORTS        = NUM_NSM;  // 16
    parameter int NOC_MP_PORTS          = NUM_MP;   // 6
    parameter int NOC_PORTS             = NOC_NSM_PORTS + NOC_MP_PORTS; // 22
    parameter int NOC_FLIT_BYTES       = 8;
    parameter int NOC_VC_ENTRIES        = 4;
    parameter int NOC_ROUTER_PIPE      = 1;
    parameter int NOC_LINK_CYCLES      = 1;
    parameter int NOC_INTERLEAVE_BYTES  = 256;

    // -------------------------------------------------------------------------
    // DRAM GDDR5 通道时序参数（依据 04 册 §12；ns 数值，按 1ns ≈ 1 周期换算）
    // -------------------------------------------------------------------------
    // 6 个 GDDR5 通道，每通道 8 bank、32-bit 宽、1 GiB 容量，聚合 6 GiB。
    // 突发原子 32 B（=4 次传输）。tCK=0.4ns 取整为 1 周期；tBURST=1.6ns≈2 周期。
    // 时序符号含义：
    //   tCAS (CL)  = 列地址到数据输出的列延迟（12 周期）
    //   tRCD       = ACTIVATE 到 READ/WRITE 的行激活延迟（12 周期）
    //   tRP        = PRECHARGE 充电延迟（12 周期）
    //   tRAS       = ACTIVE 后最短维持时间才能 PRECHARGE（28 周期）
    //   tRC        = 同 bank 两次 ACTIVATE 之间最短间隔（40 周期）
    //   tWR         = WRITE 数据写入后到 PRECHARGE 的写恢复延迟（12 周期）
    // 行命中：仅付 tCAS + tBURST；行未命中：付 tRP + tRCD + tCAS。
    parameter int DRAM_CHANNELS         = 6;
    parameter int DRAM_BANKS_PER_CHAN   = 8;
    parameter int DRAM_CHANNEL_WIDTH_BITS = 32;
    parameter int DRAM_BURST_ATOM_BYTES = 32;
    parameter int DRAM_TCK              = 1;   // 0.4ns 取整为 1 周期
    parameter int DRAM_TBURST           = 2;   // 1.6ns ≈ 2 周期
    parameter int DRAM_TRCD             = 12;  // ACTIVE→READ/WRITE
    parameter int DRAM_TCAS             = 12;  // 列→数据
    parameter int DRAM_TRP              = 12;  // PRECHARGE
    parameter int DRAM_TRAS             = 28;  // ACTIVE 维持下限
    parameter int DRAM_TRC              = 40;  // ACTIVE→ACTIVE 同 bank
    parameter int DRAM_TWR              = 12;  // 写恢复
    parameter int DRAM_WRITE_DRAIN_THR  = 32;  // 写排空阈值：连续 32 读后强制排空写队列

    // -------------------------------------------------------------------------
    // SIMT 栈参数（依据 03 册 §4.1）
    // -------------------------------------------------------------------------
    // 每 warp 栈深最多 32 项；栈底项固定为 EXIT 哨兵 RPC=0xFFFF_FFFF_FFFF_FFF8。
    parameter int SIMT_STACK_DEPTH     = 32;
    parameter logic [63:0] EXIT_SENTINEL = 64'hFFFF_FFFF_FFFF_FFF8;

    // =========================================================================
    // 枚举类型定义
    // =========================================================================

    // NTAS1 操作码枚举（依据 02 册 §3 操作码表，OP 字段 7 位）
    // 编码值是 ABI 二进制兼容的一部分，不得更改。
    typedef enum logic [6:0] {
        // 数据传送 / warp 内通信（0x00–0x0F）
        OP_NOP    = 7'h00, OP_MOV    = 7'h01, OP_MOV32I = 7'h02, OP_S2R    = 7'h03,
        OP_LDC    = 7'h04, OP_CVTA   = 7'h05, OP_SHFL   = 7'h06, OP_VOTE   = 7'h07,
        OP_PRMT   = 7'h08, OP_SELP   = 7'h09,
        // 整数运算（0x10–0x2F）
        OP_IADD   = 7'h10, OP_IADD3  = 7'h11, OP_IMAD  = 7'h12, OP_IMUL   = 7'h13,
        OP_ISUB   = 7'h14, OP_IMNMX  = 7'h15, OP_LOP   = 7'h16, OP_SHF    = 7'h17,
        OP_SHL    = 7'h18, OP_SHR    = 7'h19, OP_SAR   = 7'h1A, OP_BFE    = 7'h1B,
        OP_BFI    = 7'h1C, OP_POPC   = 7'h1D, OP_BREV  = 7'h1E, OP_IABS    = 7'h1F,
        OP_INEG   = 7'h20, OP_IDIV   = 7'h21, OP_IREM  = 7'h22, OP_LOP3   = 7'h23,
        OP_BMSK   = 7'h24, OP_FIND   = 7'h25,
        // 浮点运算（0x30–0x3F）
        OP_FADD   = 7'h30, OP_FSUB   = 7'h31, OP_FMUL  = 7'h32, OP_FFMA   = 7'h33,
        OP_FMNMX  = 7'h34, OP_FSET   = 7'h35, OP_F2F   = 7'h36, OP_XCVT   = 7'h37,
        OP_RCP    = 7'h38, OP_RSQ    = 7'h39, OP_MUFU  = 7'h3A, OP_FRND   = 7'h3B,
        OP_FABS   = 7'h3C, OP_FNEG   = 7'h3D, OP_FCMP  = 7'h3E, OP_DADD   = 7'h3F,
        // 比较与谓词（0x40–0x4F）
        OP_SETP   = 7'h40, OP_SETPI  = 7'h41, OP_PLOP  = 7'h42, OP_PSET2  = 7'h43,
        // 控制流 / 同步（0x50–0x5F）
        OP_BRA    = 7'h50, OP_BRX    = 7'h51, OP_CALL  = 7'h52, OP_RET    = 7'h53,
        OP_SSY    = 7'h54, OP_BAR    = 7'h55, OP_MEMBAR= 7'h56, OP_EXIT   = 7'h57,
        OP_YIELD  = 7'h58, OP_TRAP   = 7'h59, OP_BRKPT = 7'h5A,
        // 访存与原子（0x60–0x6F）
        OP_LD     = 7'h60, OP_ST     = 7'h61, OP_LDU   = 7'h62, OP_ATOM   = 7'h63,
        OP_RED    = 7'h64, OP_PREFETCH=7'h65, OP_LD128 = 7'h66, OP_ST128  = 7'h67
    } op_e;

    // 指令格式枚举（依据 02 册 §2.2）：
    //   FMT_R  - 寄存器-寄存器（d/a/c/X/Y）
    //   FMT_I  - 立即数（d/imm32/X）
    //   FMT_B  - 跳转/SSY/CALL（imm26/X）
    //   FMT_M  - 访存寄存器偏移（X[7]=1）
    //   FMT_MI - 访存立即数偏移（X[7]=0）
    typedef enum logic [2:0] {
        FMT_R  = 3'd0,
        FMT_I  = 3'd1,
        FMT_B  = 3'd2,
        FMT_M  = 3'd3,
        FMT_MI = 3'd4
    } fmt_e;

    // 存储空间枚举（依据 02 册 §2.2 X[2:0]；00 册 §6 存储空间一览）
    typedef enum logic [2:0] {
        SP_GLOBAL = 3'd0,  // 全局设备 DRAM，经 L2；L1D 默认只缓存只读/local
        SP_SHARED = 3'd1,  // CTA 内 scratchpad，32 bank，不经 NoC/L2
        SP_LOCAL  = 3'd2,  // 单线程私有，DRAM 经 L1D（write-back, write-allocate）
        SP_CONST  = 3'd3,  // 只读常量；warp 同址广播
        SP_FLAT   = 3'd4   // flat 64 位地址，按窗口译码空间
    } space_e;

    // 数据宽度枚举（依据 02 册 §2.2 X[6:3]）
    typedef enum logic [3:0] {
        W_U8   = 4'd0,
        W_S8   = 4'd1,
        W_U16  = 4'd2,
        W_S16  = 4'd3,
        W_U32  = 4'd4,
        W_S32  = 4'd5,
        W_U64  = 4'd6,
        W_F32  = 4'd7,
        W_F64  = 4'd8,
        W_F16  = 4'd9,
        W_U128 = 4'd10
    } width_e;

    // Cache hint 枚举（依据 02 册 §2.2 Y[2:0]）
    typedef enum logic [2:0] {
        HINT_CA = 3'd0,  // 逐级缓存
        HINT_CG = 3'd1,  // 仅 L2（绕过 L1 数据数组）
        HINT_CS = 3'd2,  // 流式绕过
        HINT_CV = 3'd3,  // 易失
        HINT_LU = 3'd4   // 只读入 L1
    } hint_e;

    // 执行流水线枚举（依据 02 册 §7 执行部件表）
    //   SP  - 32 lane INT32/FP32 主流水线，1 warp/周期
    //   SFU - 16 lane 特殊函数，2 拍/warp
    //   DP  - 16 lane FP64，2 拍/warp
    //   LSU - Load/Store 单元，1 接收/周期
    //   BRU - 分支/控制/VOTE/SHFL，1 拍
    typedef enum logic [2:0] {
        PIPE_SP  = 3'd0,
        PIPE_SFU = 3'd1,
        PIPE_DP  = 3'd2,
        PIPE_LSU = 3'd3,
        PIPE_BRU = 3'd4
    } pipe_e;

    // 寄存器类别（用于记分牌区分 R 与 P）：
    //   CLASS_R - 通用寄存器 R0..R254
    //   CLASS_P - 谓词 P0..P6
    typedef enum logic {
        CLASS_R = 1'b0,
        CLASS_P = 1'b1
    } regclass_e;

    // 错误码枚举（依据 05 册 §6 错误模型）：
    // 错误码是 ABI 的一部分；S2/S3 在同一触发条件下给出同一码。
    typedef enum logic [3:0] {
        ERR_OK               = 4'd0,
        ERR_BAD_OPCODE       = 4'd1,   // 保留 OP
        ERR_BAD_ENCODING     = 4'd2,   // 非法修饰/保留特殊寄存器
        ERR_UNALIGNED_REG    = 4'd3,   // 64/128 位寄存器号奇对齐
        ERR_MISALIGNED_ADDR  = 4'd4,   // 访存自然对齐违例
        ERR_MISALIGNED_PC    = 4'd5,   // 跳转目标非 8B 对齐
        ERR_BAR_ID           = 4'd6,   // id ≥ 16
        ERR_BAR_PARTIAL      = 4'd7,   // 部分线程屏障
        ERR_NO_IPDOM         = 4'd8,   // 分歧 BRA 时栈顶 rpc 为哨兵（漏 SSY）
        ERR_STACK_OVERFLOW   = 4'd9,   // BRX 使栈深 > 32
        ERR_ADDR_OUT_OF_RANGE= 4'd10, // 地址越界
        ERR_LAUNCH           = 4'd11,  // 启动参数违例
        ERR_DEADLOCK         = 4'd12   // 无可推进 warp 且存在未完成 CTA
    } error_e;

    // 特殊寄存器编号（依据 02 册 §4.1 S2R 的 X[5:0]）
    // 3–7、23、25–31、其余编号保留；解码命中保留编号 → ERR_BAD_ENCODING。
    typedef enum logic [5:0] {
        SR_TID_X      = 6'd0,   // CTA 内线程 x 坐标（lane 粒度）
        SR_TID_Y      = 6'd1,
        SR_TID_Z      = 6'd2,
        SR_CTAID_X    = 6'd8,   // CTA 坐标（warp 内统一）
        SR_CTAID_Y    = 6'd9,
        SR_CTAID_Z    = 6'd10,
        SR_NTID_X     = 6'd11,  // blockDim（warp 内统一）
        SR_NTID_Y     = 6'd12,
        SR_NTID_Z     = 6'd13,
        SR_NCTAID_X   = 6'd14,  // gridDim
        SR_SMID       = 6'd16,  // 物理 NSM 号 0..15
        SR_CLOCKLO    = 6'd17,  // 自由运行计数器低 32 位（硬件返回周期数）
        SR_CLOCKHI    = 6'd18,
        SR_NCTAID_Y   = 6'd19,
        SR_NCTAID_Z   = 6'd20,
        SR_LANEID     = 6'd21,  // lane 号 0..31（lane 粒度）
        SR_WARPID     = 6'd22,  // CTA 内 warp 号 0..31
        SR_LR         = 6'd24,  // CALL 链接寄存器（每 lane）
        SR_PARAM_BASE = 6'd32,  // 参数区 u64（两寄存器）
        SR_SMEM_BASE  = 6'd33,  // 本 CTA shared flat 基址 u64
        SR_LMEM_BASE  = 6'd34,  // 本 lane local flat 基址 u64
        SR_WARPID_IN_GRID = 6'd40,
        SR_GRIDID     = 6'd41   // 多 kernel 扩展，基线恒 0
    } sreg_e;

    // =========================================================================
    // 结构体定义
    // =========================================================================

    // 解码后的指令（依据 05 册 §2.1 Inst 结构；02 册 §2 字段定义）
    // packed struct 用于寄存器存储，方便整体赋值。
    typedef struct packed {
        logic [6:0]  op;          // 操作码（bit 63:57）
        logic        guard_en;    // 谓词守卫使能（bit 56，G）
        logic        guard_neg;   // 守卫取反（bit 55，GN）
        logic [2:0]  guard_pred;  // 守卫谓词号（bit 54:52，GP；7=PT）
        logic [3:0]  m;            // 类别修饰（bit 51:48，M）
        fmt_e        fmt;          // 指令格式 R/I/B/M/MI
        logic [7:0]  d;            // 目的寄存器（bit 47:40）
        logic [7:0]  a;            // 第一源（bit 39:32）
        logic [7:0]  c;            // 第二源（bit 31:24）
        logic [7:0]  x;            // 修饰 1（bit 23:16）
        logic [7:0]  y;            // 修饰 2（bit 15:8）
        logic [31:0] imm32;       // I 型 32 位立即数（bit 39:8）
        logic [25:0] imm26;       // B 型 26 位立即数（bit 47:22，有符号）
        logic [15:0] simm16;      // MI 型 16 位偏移（bit 15:0，有符号）
        logic [63:0] raw_pc;      // 该指令在指令存储中的 PC（字节地址）
        logic [63:0] raw;          // 原始 64 位编码（用于反汇编/调试）
    } inst_t;

    // SIMT 栈项（依据 03 册 §4.1 StackEntry 结构）
    //   rpc     - 再收敛 PC（由 SSY 写入；栈底项 = EXIT 哨兵）
    //   nextpc  - 本子路径下一条指令 PC
    //   mask    - 本子路径 active lane 掩码（32 位）
    //   dflag   - 该项是否曾作为"分歧父项"（其 nextpc 被改写过）
    typedef struct packed {
        logic [63:0] rpc;
        logic [63:0] nextpc;
        logic [31:0] mask;
        logic        dflag;
    } stack_entry_t;

    // 记分牌项（依据 03 册 §6 Coon 式顺序记分牌）
    // 每 warp 4 项，每项追踪一个未完成的目的寄存器（或谓词）。
    typedef struct packed {
        logic        valid;       // 该项是否占用
        logic        is_pred;     // 0=R 通用寄存器，1=P 谓词
        logic [7:0]  reg_num;     // 寄存器号 0..255
    } sb_entry_t;

    // 操作数描述符（collector 内部用）
    //   valid   - 该操作数槽是否有效
    //   is_pred - 是谓词还是通用寄存器
    //   regnum  - 寄存器号
    //   ready   - 操作数是否已就绪（RF 读到或写回捕获）
    //   data    - 操作数数据（32 位/ lane，此处存 lane 0 或代表值）
    typedef struct packed {
        logic        valid;
        logic        is_pred;
        logic [7:0]  regnum;
        logic        ready;
        logic [31:0] data;
    } operand_t;

    // Collector unit 项（依据 03 册 §8.1，非 packed，因含 operand 数组）
    //   valid       - 该 collector 是否被占用
    //   warp        - 所属 warp 槽号
    //   pc          - 指令 PC
    //   exec_mask   - 执行掩码（active mask & guard）
    //   op/pipe     - 操作码与目标流水线
    //   dst/dst_is_pred/has_dst - 目的寄存器信息
    //   ops[NUM_OPERAND_SLOTS] - 操作数槽数组
    typedef struct {
        logic        valid;
        logic [$clog2(MAX_WARPS_PER_NSM)-1:0] warp;
        logic [63:0] pc;
        logic [31:0] exec_mask;
        logic [6:0]  op;
        logic [2:0]  pipe;
        logic [7:0]  dst;
        logic        dst_is_pred;
        logic        has_dst;
        operand_t    ops [OPERAND_SLOTS_PER_COLLECTOR];
    } collector_entry_t;

    // 访存请求包（LSU ↔ L1D ↔ NoC 之间传递）
    //   sectors - 128B line 内 4 个 sector 的有效位向量
    //   be      - 每 4B word 的字节使能（合并写时用）
    //   subid   - 出站请求在本 NSM 内的序号，随包携带，用于 fill 路由回 NSM
    typedef struct packed {
        logic        valid;
        logic        is_write;
        logic [63:0] addr;
        logic [3:0]  sectors;
        logic [31:0] wdata;
        logic [3:0]  be;
        logic [7:0]  space;
        logic [7:0]  width;
        logic [7:0]  hint;
        logic [15:0] subid;
        logic [$clog2(MAX_WARPS_PER_NSM)-1:0] warp;
        logic [7:0]  dst_reg;
        logic [63:0] pc;
    } mem_req_t;

    // NoC flit（依据 04 册 §8 头 flit 字段）
    //   is_head/is_tail - 标识包头/包尾（多 flit 包）
    //   src/dst         - 源/目的端口号
    //   mp/slice        - 目标 MP 与 slice（由地址散列得出）
    //   cmd             - 0=read, 1=write
    //   n_flit          - 本包总 flit 数
    typedef struct packed {
        logic        valid;
        logic        is_head;
        logic        is_tail;
        logic [7:0]  src;
        logic [7:0]  dst;
        logic [63:0] addr;
        logic [7:0]  mp;
        logic        slice;
        logic [15:0] subid;
        logic [3:0]  sectors;
        logic        cmd;
        logic [5:0]  n_flit;
        logic [63:0] payload;
    } noc_flit_t;

    // =========================================================================
    // 解码函数（依据 02 册 §2；05 册 §2.1 decode 签名）
    // =========================================================================
    // 将 64 位原始指令字解码为 inst_t 结构。失败时上层抛 NtasError(code,pc)。
    // 字段切片严格按 02 册 §2.1 公共头 + §2.2 格式体定义。
    function automatic inst_t decode_inst(input logic [63:0] raw);
        inst_t inst;
        logic [6:0] op;
        logic [7:0] x;
        op = raw[63:57];       // OP 字段（bit 63:57）
        x  = raw[23:16];       // X 字段（bit 23:16），访存格式判定要用 X[7]
        // 公共头字段（bit 56:48）
        inst.op         = op;
        inst.guard_en   = raw[56];       // G：谓词守卫使能
        inst.guard_neg  = raw[55];       // GN：守卫取反
        inst.guard_pred = raw[54:52];   // GP：守卫谓词号
        inst.m          = raw[51:48];    // M：类别修饰
        // 寄存器与立即数字段（多种格式共用切片，最终由 fmt 决定语义）
        inst.d     = raw[47:40];         // 目的寄存器 d
        inst.a     = raw[39:32];         // 第一源 a
        inst.c     = raw[31:24];         // 第二源 c
        inst.x     = x;                  // 修饰 1 X
        inst.y     = raw[15:8];          // 修饰 2 Y
        inst.imm32 = raw[39:8];          // I 型 32 位立即数
        inst.imm26 = raw[47:22];         // B 型 26 位立即数
        inst.simm16= raw[15:0];          // MI 型 16 位偏移
        inst.raw   = raw;
        inst.raw_pc= 64'h0;
        // 格式判定（02 册 §2.2）
        case (op)
            OP_MOV32I, OP_SETPI, OP_TRAP:
                inst.fmt = FMT_I;        // I 型：d + imm32 + X
            OP_BRA, OP_CALL, OP_SSY:
                inst.fmt = FMT_B;        // B 型：imm26 + X
            OP_LD, OP_ST, OP_LDU, OP_LD128, OP_ST128, OP_PREFETCH: begin
                if (x[7]) inst.fmt = FMT_M;  // X[7]=1 → M 型（带 c 偏移寄存器）
                else      inst.fmt = FMT_MI; // X[7]=0 → MI 型（simm16 偏移）
            end
            OP_LDC:
                inst.fmt = FMT_MI;       // LDC 固定 MI 型
            default:
                inst.fmt = FMT_R;        // 默认 R 型
        endcase
        return inst;
    endfunction

    // =========================================================================
    // 辅助函数集合
    // =========================================================================

    // 26 位有符号立即数符号扩展到 64 位（B 型跳转用）
    function automatic logic [63:0] sext26_64(input logic [25:0] imm);
        return {{38{imm[25]}}, imm};
    endfunction

    // 16 位有符号立即数符号扩展到 64 位（MI 型偏移用）
    function automatic logic [63:0] sext16_64(input logic [15:0] imm);
        return {{48{imm[15]}}, imm};
    endfunction

    // 16 位有符号立即数符号扩展到 32 位
    function automatic logic [31:0] sext16(input logic [15:0] imm);
        return {{16{imm[15]}}, imm};
    endfunction

    // 分支目标地址计算（依据 02 册 §2.2 B 格式）
    //   absolute=0（默认）：相对跳转 target = PC + 8*sext(imm26)
    //   absolute=1：绝对跳转 target = 8*sext(imm26)
    // 强制 8 字节对齐（清除低 3 位）
    function automatic logic [63:0] branch_target(input logic [63:0] pc,
                                                   input logic [25:0] imm26,
                                                   input logic        absolute);
        logic [63:0] target;
        if (absolute)
            target = {sext26_64(imm26) >> 3, 3'b0} << 3; // 8*sext, aligned
        else
            target = pc + (sext26_64(imm26) << 3); // PC + 8*sext(imm26)
        target[2:0] = 3'b0; // 确保 8 字节对齐
        return target;
    endfunction

    // 寄存器堆 bank 映射（依据 03 册 §7 swizzled 布局）
    // bank(w, regnum) = (regnum + warp_slot) mod 4
    // 目的：让连续 warp 的同号寄存器自然落在不同 bank，减少 bank 冲突。
    // 教学注记：这种 swizzled 映射是 GPU 寄存器堆设计的关键技巧——
    //   虽然物理上只有 4 个单端口 bank，但通过 warp 槽号偏移使不同 warp 的
    //   同号寄存器读访问天然分散，配合 operand collector 的多 bank 仲裁，
    //   能让 4 单端口 bank 在 8 个 collector 面前表现出接近多端口的带宽。
    function automatic logic [$clog2(RF_BANKS)-1:0] rf_bank(
        input logic [$clog2(MAX_WARPS_PER_NSM)-1:0] warp_slot,
        input logic [7:0] regnum);
        return (regnum + warp_slot) % RF_BANKS;
    endfunction

    // L1D 组号散列（依据 04 册 §7.1 XOR 散列）
    // l1set = ((A>>7) ^ (A>>13) ^ (A>>19)) & 0x1F
    // 目的：削弱 warp 内连续 sector 访问导致的组冲突（ aliasing）。
    function automatic logic [$clog2(L1D_SETS)-1:0] l1d_set_hash(input logic [63:0] addr);
        logic [$clog2(L1D_SETS)-1:0] s;
        s = (addr[11:7] ^ addr[17:13] ^ addr[24:19]) & (L1D_SETS-1);
        return s;
    endfunction

    // L2 组号散列（依据 04 册 §7.1）
    // l2set = ((A>>7) ^ (A>>14) ^ (A>>21) ^ (mp<<2)) & 0x7F
    // 注入 mp 低位是为了让同一地址在不同 MP 的 slice 上散列到不同组，
    // 避免跨 MP 的 aliasing。
    function automatic logic [$clog2(L2_SETS)-1:0] l2_set_hash(input logic [63:0] addr,
                                                                input logic [2:0] mp);
        logic [$clog2(L2_SETS)-1:0] s;
        s = (addr[13:7] ^ addr[20:14] ^ addr[27:21] ^ {mp, 2'b00}) & (L2_SETS-1);
        return s;
    endfunction

    // 物理地址 → MP（memory partition）映射（依据 04 册 §7.1）
    // b256 = A[31:8]（256B 交织块号）
    // fold = b256[5:0] ^ b256[11:6] ^ b256[17:12] ^ b256[23:18]  （XOR 折叠到 6 位）
    // mp   = fold % 6   （0..5：memory partition，= GDDR5 通道号）
    // 教学注记：用 XOR 折叠而非简单取模，能让连续 256B 块均匀分布到 6 个 MP，
    //   避免热 点；256B 粒度与 cache line（128B）的 2 倍关系保证跨 line 访问
    //   也能均匀分担到多通道带宽上。
    function automatic logic [2:0] addr_to_mp(input logic [63:0] addr);
        logic [23:0] b256;
        logic [5:0]  fold;
        b256 = addr[31:8];
        fold = b256[5:0] ^ b256[11:6] ^ b256[17:12] ^ b256[23:18];
        return fold % NUM_MP;
    endfunction

    // MP 内 slice 选择（依据 04 册 §7.1）
    // slice = (A[7] ^ A[14] ^ b256[3]) & 1
    // 每 MP 含 2 个 L2 slice，用地址位异或做 1 位散列。
    function automatic logic addr_to_slice(input logic [63:0] addr);
        logic [23:0] b256;
        b256 = addr[31:8];
        return (addr[7] ^ addr[14] ^ b256[3]) & 1'b1;
    endfunction

    // DRAM bank 映射（依据 04 册 §7.2）
    // bank = (A[14:12] ^ A[18:16] ^ mp) & 0b111
    // 注入 mp 是为了避免不同 MP 的同 bank 同时被命中（减少 bank conflict）。
    function automatic logic [2:0] addr_to_dram_bank(input logic [63:0] addr,
                                                      input logic [2:0] mp);
        return (addr[14:12] ^ addr[18:16] ^ mp) & 3'b111;
    endfunction

    // DRAM 列地址（依据 04 册 §7.2）
    // col = A[11:5]（每次突发 32B atom，列号 7 位）
    function automatic logic [6:0] addr_to_dram_col(input logic [63:0] addr);
        return addr[11:5];
    endfunction

    // DRAM 行地址（依据 04 册 §7.2）
    // row = A[35:15]（去掉 bank/col 位后紧缩，21 位行号）
    function automatic logic [20:0] addr_to_dram_row(input logic [63:0] addr);
        return addr[35:15];
    endfunction

    // 判断指令是否写目的寄存器（用于记分牌分配与写回控制）
    // NOP/BRA/BRX/SSY/BAR/MEMBAR/EXIT/YIELD/BRKPT/TRAP/ST/RED/PREFETCH/ST128
    // 这些指令不写架构寄存器。
    function automatic logic has_destination(input logic [6:0] op);
        case (op)
            OP_NOP, OP_BRA, OP_BRX, OP_SSY, OP_BAR, OP_MEMBAR, OP_EXIT,
            OP_YIELD, OP_BRKPT, OP_TRAP, OP_ST, OP_RED, OP_PREFETCH, OP_ST128:
                return 1'b0;
            default:
                return 1'b1;
        endcase
    endfunction

    // 指令 → 执行流水线派发（依据 02 册 §7 部件/延迟表）
    // BRU：控制流 + S2R + NOP + YIELD + BRKPT + SHFL + VOTE
    // SFU：RCP/RSQ/MUFU
    // DP ：DADD（FP64 算术槽）
    // LSU：LD/ST/LDU/ATOM/RED/PREFETCH/LD128/ST128/LDC/CVTA
    // SP ：其余全部（INT32/FP32/SETP 等）
    function automatic logic [2:0] get_pipeline(input logic [6:0] op);
        case (op)
            OP_NOP, OP_S2R, OP_SHFL, OP_VOTE,
            OP_BRA, OP_BRX, OP_CALL, OP_RET, OP_SSY, OP_BAR, OP_MEMBAR,
            OP_EXIT, OP_YIELD, OP_BRKPT:
                return PIPE_BRU;
            OP_RCP, OP_RSQ, OP_MUFU:
                return PIPE_SFU;
            OP_DADD:
                return PIPE_DP;
            OP_LD, OP_ST, OP_LDU, OP_ATOM, OP_RED, OP_PREFETCH,
            OP_LD128, OP_ST128, OP_LDC, OP_CVTA:
                return PIPE_LSU;
            default:
                return PIPE_SP;
        endcase
    endfunction

    // 取指令源操作数寄存器号（最多 4 个源，用于记分牌依赖检查）
    // 依据 03 册 §6.1：取指令所有源与目的操作数的 (类,号) 集合。
    //   R 型：源 a 与 c；BFI/BFE 还用 d_old；U64 还用 a+1/c+1 寄存器对
    //   I 型：MOV32I/SETPI 无寄存器源；LOP3 用 a 与 c
    //   B 型：无寄存器源（纯立即数跳转）
    //   M/MI 型：基址 a；M 型还有偏移寄存器 c；ST/ST128 用 d 作数据源；
    //           ATOM 用 c 作数据源
    function automatic void get_sources(input inst_t inst,
        output logic [7:0] s0, output logic s0_valid,
        output logic [7:0] s1, output logic s1_valid,
        output logic [7:0] s2, output logic s2_valid,
        output logic [7:0] s3, output logic s3_valid);
        s0 = 8'h0; s1 = 8'h0; s2 = 8'h0; s3 = 8'h0;
        s0_valid = 1'b0; s1_valid = 1'b0;
        s2_valid = 1'b0; s3_valid = 1'b0;
        case (inst.fmt)
            FMT_R: begin
                // R 型：源 a 与 c
                s0 = inst.a; s0_valid = (inst.a != RZ);
                s1 = inst.c; s1_valid = (inst.c != RZ);
                // BFI/BFE 用 d_old 作为源
                if (inst.op == OP_BFI || inst.op == OP_BFE) begin
                    s2 = inst.d; s2_valid = (inst.d != RZ);
                end
                // U64 操作使用寄存器对（a+1, c+1 也作源）
                if (inst.op == OP_IADD && inst.x[2:0] == 3'd6) begin // U64
                    s2 = inst.a + 8'd1; s2_valid = 1'b1;
                    s3 = inst.c + 8'd1; s3_valid = 1'b1;
                end
            end
            FMT_I: begin
                // I 型：MOV32I/SETPI 无寄存器源；LOP3 用 a 与 c
                if (inst.op == OP_LOP3) begin
                    s0 = inst.a; s0_valid = (inst.a != RZ);
                    s1 = inst.c; s1_valid = (inst.c != RZ);
                end
            end
            FMT_B: begin
                // B 型：无寄存器源（纯立即数跳转）
            end
            FMT_M, FMT_MI: begin
                // M/MI 型：基址 a；M 型还有偏移寄存器 c
                s0 = inst.a; s0_valid = (inst.a != RZ);
                if (inst.fmt == FMT_M) begin
                    s1 = inst.c; s1_valid = (inst.c != RZ);
                end
                // ST/ST128/RED 用 d 作数据源
                if (inst.op == OP_ST || inst.op == OP_ST128 || inst.op == OP_RED) begin
                    s2 = inst.d; s2_valid = (inst.d != RZ);
                end
                // ATOM 用 c 作数据源
                if (inst.op == OP_ATOM) begin
                    s2 = inst.c; s2_valid = (inst.c != RZ);
                end
            end
            default: ;
        endcase
    endfunction

    // 判断是否控制流指令（用于 SIMT 栈操作）
    function automatic logic is_ctrl_flow(input logic [6:0] op);
        case (op)
            OP_BRA, OP_BRX, OP_CALL, OP_RET, OP_SSY,
            OP_BAR, OP_MEMBAR, OP_EXIT, OP_YIELD:
                return 1'b1;
            default:
                return 1'b0;
        endcase
    endfunction

    // 判断是否访存指令（用于 LSU 派发）
    function automatic logic is_mem_op(input logic [6:0] op);
        case (op)
            OP_LD, OP_ST, OP_LDU, OP_ATOM, OP_RED,
            OP_PREFETCH, OP_LD128, OP_ST128, OP_LDC:
                return 1'b1;
            default:
                return 1'b0;
        endcase
    endfunction

endpackage : NutShellGPU_pkg
