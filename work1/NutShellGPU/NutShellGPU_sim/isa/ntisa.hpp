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
// ntisa.hpp - NTAS1 指令集共享库: 解码器 + 指令语义辅助函数
// =============================================================================
// 本头文件被功能模拟器 (func_sim) 和周期级模拟器 (cycle_sim) 共同使用,
// 提供 NTAS1 指令集的解码、编码、语义计算等基础设施。
//
// 参考: NutShellGPU_spec 02_NTAS1指令集手册.md (指令编码/语义定义)
//       NutShellGPU_spec 05_模拟器实现契约.md §6 (错误码定义)
//
// NTAS1 指令集特点:
//   - 64 位定长编码, 所有指令占 8 字节
//   - 5 种指令格式: R(寄存器) / I(立即数) / B(分支) / M(内存-寄存器偏移) / MI(内存-立即数偏移)
//   - 谓词保护 (guard predicate): 每条指令可带 @Pn 或 @!Pn 守卫
//   - SIMT 执行模型: 32 线程组成一个 warp, 共用 PC, 用 active mask 控制每条 lane
//
// 命名空间 ntisa 包含所有 ISA 相关定义, 避免全局污染
// =============================================================================
// 教材引用: 第 2 章 §2.2 GPU Instruction Set Architectures (p.14) — 本文件定义 NTAS1 (SASS 教学简化版) 的 ISA, 对应教材对 NVIDIA SASS ISA 的总览介绍
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — 64 位定长编码与 R/I/B/M/MI 五种格式对应教材描述的 SASS 编码方案
// 教材引用: 第 2 章 §2.1.2 Threading Model (p.12) — SIMT 模型与 32 线程 warp 对应教材的线程模型描述
// 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) — 谓词守卫 @Pn/@!Pn 与 active mask 对应教材的 SIMT 掩码机制
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <utility>
#if defined(_MSC_VER)
#include <intrin.h>  // MSVC 内建函数: __popcnt, _BitScanReverse 等
#endif

namespace ntisa {

// =============================================================================
// 错误码 (05 册 §6)
// -----------------------------------------------------------------------------
// 模拟器在解码/执行/访存/调度各环节可能检测到错误, 用统一的错误码标识
// 约定: 0 = 成功 (ERR_OK), 非 0 = 各类错误
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) & 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 错误码覆盖保留 OP / 编码非法 / 寄存器对齐 / 屏障 ID 等校验, 对应教材对 SASS 编码合法性与字段约束的描述
enum ErrCode : int {
    ERR_OK              = 0,   // 成功, 无错误
    ERR_BAD_OPCODE      = 1,   // 非法操作码 (使用了保留的 OP 编号)
    ERR_BAD_ENCODING    = 2,   // 编码错误 (字段组合非法, 如 S2R 的 SR 编号无效)
    ERR_UNALIGNED_REG   = 3,   // 寄存器未对齐 (64/128 位操作要求偶数寄存器号)
    ERR_MISALIGNED_ADDR = 4,   // 内存地址未对齐 (访问地址非数据宽度的倍数)
    ERR_MISALIGNED_PC   = 5,   // PC 未对齐 (分支目标地址非 8 字节倍数)
    ERR_BAR_ID          = 6,   // 屏障 ID 非法 (BAR 指令的 id >= 16)
    ERR_BAR_PARTIAL     = 7,   // 屏障部分到达 (不应在功能模拟中出现)
    ERR_NO_IPDOM        = 8,   // 无重汇聚点 (BRA 发散但栈顶 rpc=EXIT_SENTINEL)
    ERR_STACK_OVERFLOW  = 9,   // SIMT 栈溢出 (超过 SIMT_STACK_MAX=32)
    ERR_ADDR_OUT_OF_RANGE = 10, // 地址越界 (访存超出共享内存/常量内存范围)
    ERR_LAUNCH          = 11,  // 启动失败 (CTA 资源不足: warp/寄存器/共享内存)
    ERR_DEADLOCK        = 12,  // 死锁 (所有 warp 都在等待屏障, 无可推进的 warp)
};

// 错误码 → 可读字符串 (用于日志/测试输出)
inline const char* err_name(int c) {
    switch(c) {
    case ERR_OK: return "OK";
    case ERR_BAD_OPCODE: return "ERR_BAD_OPCODE";
    case ERR_BAD_ENCODING: return "ERR_BAD_ENCODING";
    case ERR_UNALIGNED_REG: return "ERR_UNALIGNED_REG";
    case ERR_MISALIGNED_ADDR: return "ERR_MISALIGNED_ADDR";
    case ERR_MISALIGNED_PC: return "ERR_MISALIGNED_PC";
    case ERR_BAR_ID: return "ERR_BAR_ID";
    case ERR_BAR_PARTIAL: return "ERR_BAR_PARTIAL";
    case ERR_NO_IPDOM: return "ERR_NO_IPDOM";
    case ERR_STACK_OVERFLOW: return "ERR_STACK_OVERFLOW";
    case ERR_ADDR_OUT_OF_RANGE: return "ERR_ADDR_OUT_OF_RANGE";
    case ERR_LAUNCH: return "ERR_LAUNCH";
    case ERR_DEADLOCK: return "ERR_DEADLOCK";
    default: return "ERR_UNKNOWN";
    }
}

// =============================================================================
// 操作码定义 (02 册 §3)
// -----------------------------------------------------------------------------
// NTAS1 操作码占 7 位 (bits 63:57), 范围 0x00-0x7F
// 按 功能类别 分段:
//   0x00-0x09: 数据搬运与 warp 原语 (NOP/MOV/MOV32I/S2R/LDC/CVTA/SHFL/VOTE/PRMT/SELP)
//   0x0A-0x0F: 保留
//   0x10-0x25: 整数 ALU (IADD/IMAD/IMUL/ISUB/LOP/SHF/SHL/SHR/...)
//   0x26-0x2F: 保留
//   0x30-0x3F: 浮点 ALU (FADD/FMUL/FFMA/RCP/RSQ/MUFU/F2F/...)
//   0x40-0x43: 谓词操作 (SETP/SETPI/PLOP/PSET2)
//   0x50-0x5A: 控制流 (BRA/BRX/CALL/RET/SSY/BAR/MEMBAR/EXIT/YIELD/TRAP/BRKPT)
//   0x60-0x67: 内存 (LD/ST/LDU/ATOM/RED/PREFETCH/LD128/ST128)
//   0x68-0x7F: 保留
// =============================================================================
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 7 位操作码按功能类别分段, 对应教材对 SASS 操作码分类的描述
enum OP : uint8_t {
    // ---- 数据搬运与 warp 原语 (0x00-0x09) ----
    OP_NOP   = 0x00,  // 空操作
    OP_MOV   = 0x01,  // 寄存器搬运: Rd ← Ra
    OP_MOV32I= 0x02,  // 32位立即数搬运: Rd ← imm32 (X[0]=1 时扩展为 64 位对)
    // 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — S2R 读取特殊寄存器 (SR_TID/CTAID/...), 对应教材描述的特殊寄存器机制
    OP_S2R   = 0x03,  // 特殊寄存器到通用寄存器: Rd ← SR[x]
    // 教材引用: 第 4 章 §4.1.1 Constant Memory (p.68) — LDC 从常量内存 (只读, 64KB, 广播访问) 加载, 对应教材对常量内存的描述
    OP_LDC   = 0x04,  // 常量加载: Rd ← const_mem[bank][offset]
    OP_CVTA  = 0x05,  // 地址空间转换: 在 flat 与具体空间间转换地址
    OP_SHFL  = 0x06,  // warp 内交换: 跨 lane 读取寄存器 (IDX/UP/DOWN/BFLY)
    OP_VOTE  = 0x07,  // warp 投票: ALL/ANY/EQ/BALLOT 跨 lane 聚合
    OP_PRMT  = 0x08,  // 字节重排: 按 c 的索引从 a/c 中选取字节
    OP_SELP  = 0x09,  // 谓词选择: Rd ← P[y] ? Ra : Rc
    // ---- 整数 ALU (0x10-0x25) ----
    OP_IADD  = 0x10,  // 整数加: Rd ← Ra + Rc (支持 U32/U64)
    OP_IADD3 = 0x11,  // 三操作数加: Rd ← Ra + Rc + imm8
    OP_IMAD  = 0x12,  // 整数乘加: Rd ← Ra * Rc + Rd_old (LO/HI/WIDE 模式)
    OP_IMUL  = 0x13,  // 整数乘: Rd ← Ra * Rc (LO/HI/WIDE)
    OP_ISUB  = 0x14,  // 整数减: Rd ← Ra - Rc
    // 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — IMNMX (整数最小/最大) 对应教材提到的 FMNMX 整数/浮点 min/max 指令族
    OP_IMNMX = 0x15,  // 整数最小/最大: Rd ← min/max(Ra, Rc)
    OP_LOP   = 0x16,  // 逻辑运算: AND/OR/XOR/ANDN
    OP_SHF   = 0x17,  // 漏斗移位: 64位 (c:a) 移位取 32 位
    OP_SHL   = 0x18,  // 逻辑左移: Rd ← Ra << (Rc & 31)
    OP_SHR   = 0x19,  // 逻辑右移: Rd ← Ra >> (Rc & 31) (无符号)
    OP_SAR   = 0x1A,  // 算术右移: Rd ← (int)Ra >> (Rc & 31) (有符号)
    OP_BFE   = 0x1B,  // 位域提取: 从 Ra 中提取 [pos, pos+len) 位
    OP_BFI   = 0x1C,  // 位域插入: 将 Ra 的低位插入 Rd_old 的 [pos, pos+len)
    OP_POPC  = 0x1D,  // 位计数: Rd ← popcount(Ra)
    OP_BREV  = 0x1E,  // 位反转: Rd ← bitrev32(Ra)
    OP_IABS  = 0x1F,  // 整数绝对值: Rd ← |(int)Ra|
    OP_INEG  = 0x20,  // 整数取负: Rd ← -(int)Ra
    OP_IDIV  = 0x21,  // 整数除法: Rd ← (int)Ra / (int)Rc (微码, 高延迟)
    OP_IREM  = 0x22,  // 整数取余: Rd ← (int)Ra % (int)Rc (微码, 高延迟)
    OP_LOP3  = 0x23,  // 三输入查找表: 8位LUT(Ra, Rc, imm8)
    OP_BMSK  = 0x24,  // 位掩码生成: Rd ← (1 << (Ra & 31)) - 1
    OP_FIND  = 0x25,  // 位查找: MSB/CLZ/CTZ
    // ---- 浮点 ALU (0x30-0x3F) ----
    OP_FADD  = 0x30,  // 浮点加: Rd ← Ra + Rc (IEEE 754, 带舍入模式)
    OP_FSUB  = 0x31,  // 浮点减: Rd ← Ra - Rc
    OP_FMUL  = 0x32,  // 浮点乘: Rd ← Ra * Rc
    // 教材引用: 第 2 章 §2.2.3 Fused Multiply-Add (p.20) — FFMA 单次舍入的融合乘加, 第三源通过 Rd 字段传入 (Rd ← Ra * Rc + Rd_old), 对应教材对 FFMA 单次舍入语义的描述
    OP_FFMA  = 0x33,  // 浮点乘加: Rd ← Ra * Rc + Rd_old (单次舍入)
    // 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — FMNMX (浮点最小/最大) 对应教材明确列出的 SASS 特殊浮点指令
    OP_FMNMX = 0x34,  // 浮点最小/最大: Rd ← min/max(Ra, Rc)
    OP_FSET  = 0x35,  // 浮点比较: Rd ← (Ra cmp Rc) ? 0xFFFFFFFF : 0
    OP_F2F   = 0x36,  // 浮点格式转换: F16↔F32, F32↔F64
    OP_XCVT  = 0x37,  // 定点↔浮点转换: S32↔F32 等
    OP_RCP   = 0x38,  // 浮点倒数: Rd ← 1.0 / Ra (SFU, 迭代近似)
    OP_RSQ   = 0x39,  // 浮点平方根倒数: Rd ← 1.0 / sqrt(Ra)
    OP_MUFU  = 0x3A,  // 数学函数: sin/cos/exp2/log2 (SFU)
    OP_FRND  = 0x3B,  // 浮点舍入: floor/ceil/trunc/nearbyint
    OP_FABS  = 0x3C,  // 浮点绝对值: 清除符号位
    OP_FNEG  = 0x3D,  // 浮点取负: 翻转符号位
    OP_FCMP  = 0x3E,  // 浮点比较 (结果写入谓词)
    OP_DADD  = 0x3F,  // 双精度浮点加 (DP 单元, 寄存器对)
    // ---- 谓词操作 (0x40-0x43) ----
    // 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) & 第 3 章 §3.1.1 SIMT Execution Masking (p.23) — SETP 设置谓词寄存器, 用于后续指令的 @Pn/@!Pn 守卫
    OP_SETP  = 0x40,  // 谓词设置: Pd ← (Ra cmp Rc) op Ps
    OP_SETPI = 0x41,  // 谓词设置(立即数): Pd ← (Ra cmp imm32)
    OP_PLOP  = 0x42,  // 谓词逻辑: Pd ← Pa op Pc
    OP_PSET2 = 0x43,  // 双谓词设置
    // ---- 控制流 (0x50-0x5A) ----
    // SIMT 栈机制: SSY 压栈设重汇聚点, BRA 按谓词发散, 到达 rpc 时重汇聚
    // 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — BRA/SSY 实现 SIMT 栈式发散与重汇聚, 对应教材对 warp 分支发散栈机制的描述
    OP_BRA   = 0x50,  // 分支: @Pn BRA target (按谓词掩码可能发散)
    OP_BRX   = 0x51,  // 间接分支: 每条 lane 的目标来自 Ra (多路发散)
    OP_CALL  = 0x52,  // 函数调用: LR ← PC+8, 跳转 target
    OP_RET   = 0x53,  // 函数返回: 跳转到 LR
    // 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — SSY 压栈设置重汇聚点 rpc, 对应教材的 SIMT 重汇聚点 (reconvergence PC) 机制
    OP_SSY   = 0x54,  // 设置重汇聚点: 压入 {rpc=target, nextpc=PC+8, mask=TOS.mask}
    OP_BAR   = 0x55,  // 屏障: SYNC/ARRIVE/WAIT (CTA 内 warp 同步)
    OP_MEMBAR= 0x56,  // 内存屏障: CTA/GL/SYS 级可见性
    OP_EXIT  = 0x57,  // 退出: 清除已退出 lane, warp 全部退出则标记完成
    OP_YIELD = 0x58,  // 让步: 提示调度器可切换 warp (功能模拟中无操作)
    OP_TRAP  = 0x59,  // 陷阱: 触发异常 (功能模拟中当作 NOP)
    OP_BRKPT = 0x5A,  // 断点: 调试用 (功能模拟中当作 NOP)
    // ---- 内存 (0x60-0x67) ----
    // M 格式: [Ra + Rc] (寄存器偏移), MI 格式: [Ra + simm16] (立即数偏移)
    // X[2:0]=地址空间, X[6:3]=数据宽度, Y[2:0]=缓存提示
    OP_LD    = 0x60,  // 加载: Rd ← mem[addr]
    OP_ST    = 0x61,  // 存储: mem[addr] ← Rd
    OP_LDU   = 0x62,  // 不可缓存加载 (绕过 L1)
    OP_ATOM  = 0x63,  // 原子操作: old ← mem[addr]; mem[addr] ← f(old, Rc); Rd ← old
    OP_RED   = 0x64,  // 原子归约: mem[addr] ← f(mem[addr], Rc) (无返回值)
    OP_PREFETCH=0x65, // 预取: 将数据拉入缓存 (功能模拟中无操作)
    OP_LD128 = 0x66,  // 128位加载: Rd~Rd+3 ← mem[addr] (4 个寄存器)
    OP_ST128 = 0x67,  // 128位存储: mem[addr] ← Rd~Rd+3
};

// =============================================================================
// 指令格式种类 (02 册 §2.1)
// -----------------------------------------------------------------------------
// NTAS1 有 5 种指令格式, 由操作码隐式决定:
//   FK_R  : 寄存器格式 - d, a, c, x, y, b 共 6 个 8 位字段
//           布局: [OP|g|gn|gp|m=16b] [d=8b] [a=8b] [c=8b] [x=8b] [y=8b] [b=8b]
//   FK_I  : 立即数格式 - d, imm32, x (用于 MOV32I, LOP3 等)
//           布局: [OP|g|gn|gp|m=16b] [d=8b] [imm32=32b] [x=8b]
//   FK_B  : 分支格式 - imm26 (有符号偏移, 单位为 8 字节), x
//           布局: [OP|g|gn|gp|m=16b] [imm26=26b] [x=6b]
//   FK_M  : 内存格式(寄存器偏移) - d, a, c, x, y, b; 地址 = Ra + Rc
//           X[7]=1 区分 M 与 MI; X[6:3]=宽度, X[2:0]=地址空间
//   FK_MI : 内存格式(立即数偏移) - d, a, x, y, simm16; 地址 = Ra + simm16
//           X[7]=0; X[6:3]=宽度, X[2:0]=地址空间
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — 5 种指令格式由操作码隐式决定, 对应教材描述的 SASS 格式分类 (R/I/B/M/MI)
enum FmtKind : uint8_t { FK_R=0, FK_I=1, FK_B=2, FK_M=3, FK_MI=4 };

// =============================================================================
// 地址空间 (02 册 §2.2, X[2:0])
// -----------------------------------------------------------------------------
// NutShellGPU 的内存模型分为 4 个独立地址空间 + 1 个 flat 虚拟空间:
//   AS_GLOBAL: 全局内存 (DRAM), 所有 CTA 可见, 经 L1/L2 缓存
//   AS_SHARED: 共享内存 (片上 SRAM), CTA 内可见, 低延迟, 32 bank
//   AS_LOCAL : 本地内存 (实际位于 DRAM, 每线程私有), 经 L1 缓存
//   AS_CONST : 常量内存 (片上, 64KB, 只读), 广播式访问, 高效
//   AS_FLAT  : 虚拟统一地址空间, 高位编码空间类型 (04 册 §2)
// =============================================================================
// 教材引用: 第 4 章 §4.1.1 Constant Memory (p.68) & 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 4 个独立地址空间 (Global/Shared/Local/Const) 对应教材对 GPU 内存层次的描述
enum ASpace : uint8_t {
    AS_GLOBAL=0,  // 全局内存: 经 L1→L2→DRAM, 大容量高延迟
    AS_SHARED=1,  // 共享内存: 片上 SRAM, CTA 内共享, 32 bank, ~20 周期
    AS_LOCAL=2,   // 本地内存: 每线程私有, 实际在 DRAM, 经 L1 缓存
    AS_CONST=3,   // 常量内存: 只读, 64KB, 广播访问 (8 bank × 8KB)
    AS_FLAT=4     // 统一地址: 高位区分空间, 见 decode_flat()
};

// =============================================================================
// 数据宽度 (02 册 §2.2, X[6:3])
// -----------------------------------------------------------------------------
// 定义访存指令的数据大小, 决定读写的字节数和寄存器对齐要求
// 64/128 位操作要求寄存器对齐 (偶数寄存器号)
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — 数据宽度由 X[6:3] 字段编码, 对应教材描述的指令字段提取与宽度修饰符
enum Width : uint8_t {
    W_U8=0,   // 无符号 8 位 (1 字节)
    W_S8=1,   // 有符号 8 位 (加载时符号扩展)
    W_U16=2,  // 无符号 16 位 (2 字节)
    W_S16=3,  // 有符号 16 位
    W_U32=4,  // 无符号 32 位 (4 字节, 最常用)
    W_S32=5,  // 有符号 32 位
    W_U64=6,  // 无符号 64 位 (8 字节, 寄存器对 Rd:Rd+1)
    W_F32=7,  // 32 位浮点 (IEEE 754 single)
    W_F64=8,  // 64 位浮点 (IEEE 754 double, 寄存器对)
    W_F16=9,  // 16 位浮点 (IEEE 754 half)
    W_U128=10 // 128 位 (16 字节, 四寄存器 Rd~Rd+3)
};

// 宽度枚举 → 字节数 (用于计算访存大小和寄存器对齐)
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — Width 字段到字节数的映射, 决定访存大小与寄存器对齐要求
inline unsigned width_bytes(uint8_t w) {
    switch(w) {
    case W_U8: case W_S8: return 1;
    case W_U16: case W_S16: case W_F16: return 2;
    case W_U32: case W_S32: case W_F32: return 4;
    case W_U64: case W_F64: return 8;
    case W_U128: return 16;
    default: return 4;  // 默认 4 字节
    }
}

// =============================================================================
// 缓存提示 (02 册 §2.2, Y[2:0])
// -----------------------------------------------------------------------------
// LD/ST 指令的 Y[2:0] 字段提示缓存策略:
//   CA: Cache All - 正常缓存, 读时分配 L1 cache line (默认)
//   CG: Cache Global - 只缓存到 L2 (L1 绕过), 适合流式数据
//   CS: Cache Streaming - 流式访问, 不污染 L1 (evict-first)
//   CV: Cache Volatile - 不缓存, 每次直接访问 L2/DRAM
//   LU: Last Use - 最后一次使用, 读后从 L1 淘汰
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — 缓存提示由 Y[2:0] 字段编码, 对应教材描述的指令字段提取中的缓存修饰符
enum CacheHint : uint8_t { CH_CA=0, CH_CG=1, CH_CS=2, CH_CV=3, CH_LU=4 };

// =============================================================================
// 比较码 (02 册 §5.2)
// -----------------------------------------------------------------------------
// SETP/SETPI/FSET 指令使用 4 位比较码, 支持有序/无序(NaN处理)两种语义:
//   有序 (ordered): 遇到 NaN 返回 false
//   无序 (unordered, 后缀 U): 遇到 NaN 返回 true
// =============================================================================
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 4 位比较码支持有序/无序 (NaN 处理) 语义, 用于 SETP/SETPI/FSET 指令
enum CmpCode : uint8_t {
    CC_EQ=0,   // 等于 (有序)
    CC_NE=1,   // 不等于 (无序, NaN → true)
    CC_LT=2,   // 小于 (有序)
    CC_LE=3,   // 小于等于 (有序)
    CC_GT=4,   // 大于 (有序)
    CC_GE=5,   // 大于等于 (有序)
    CC_HI=6,   // 高于 (无符号, 高于)
    CC_HS=7,   // 高于等于 (无符号, 不低于)
    CC_LO=8,   // 低于 (无符号)
    CC_LS=9,   // 低于等于 (无符号)
    CC_FE=10,  // 浮点等于 (有序, 同 EQ)
    CC_GTU=11, // 大于 (无序, NaN → true)
    CC_LTU=12, // 小于 (无序)
    CC_LEU=13, // 小于等于 (无序)
    CC_NUM=14, // 非 NaN (有序: 操作数均为数字)
    CC_NAN=15  // 是 NaN (无序: 存在 NaN 操作数)
};

// =============================================================================
// 原子操作码 (02 册 §5.3, Y[3:0])
// -----------------------------------------------------------------------------
// ATOM/RED 指令的 Y[3:0] 字段指定原子操作类型:
// 读-改-写序列: old ← mem[addr]; new ← f(old, arg); mem[addr] ← new; (ATOM: Rd ← old)
// =============================================================================
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 原子操作码由 Y[3:0] 字段指定, 对应教材 SASS 操作码分类中的原子操作族
enum AtomOp : uint8_t {
    AOP_EXCH=0,  // 交换: mem ← arg, 返回 old
    AOP_ADD=1,   // 加法: mem ← old + arg
    AOP_MIN=2,   // 有符号最小: mem ← min(old, arg)
    AOP_MAX=3,   // 有符号最大: mem ← max(old, arg)
    AOP_UMIN=4,  // 无符号最小
    AOP_UMAX=5,  // 无符号最大
    AOP_INC=6,   // 递增: mem ← (old >= arg) ? 0 : old + 1
    AOP_DEC=7,   // 递减: mem ← (old == 0 || old > arg) ? arg : old - 1
    AOP_CAS=8,   // 比较交换: mem ← (old == cmp) ? arg : old (cmp 在 b 字段)
    AOP_AND=9,   // 按位与
    AOP_OR=10,   // 按位或
    AOP_XOR=11   // 按位异或
};

// =============================================================================
// 特殊寄存器编号 (02 册 §4.1)
// -----------------------------------------------------------------------------
// S2R 指令通过 X[5:0] 字段指定要读取的特殊寄存器
// 特殊寄存器提供线程/CTA/grid 的坐标信息和硬件状态:
//   TID: Thread ID - 线程在 CTA 内的三维坐标
//   CTAID: CTA ID - CTA 在 grid 内的三维坐标
//   NTID: CTA 维度 - 每个维度上的线程数
//   NCTAID: Grid 维度 - 每个维度上的 CTA 数
//   SMID: NSM 编号, 用于区分执行核心
//   CLOCKLO/HI: 64 位时钟计数器 (低/高 32 位)
//   LANEID: lane 在 warp 内的编号 (0-31)
//   WARPID: warp 在 CTA 内的编号
//   LR: Link Register (CALL/RET 用)
//   PARAM_BASE/SMEM_BASE/LMEM_BASE: 各地址空间的 flat 基地址 (64 位)
// =============================================================================
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 特殊寄存器 (SR_TID/CTAID/NTID/NCTAID/LANEID/...) 由 S2R 读取, 对应教材描述的 SASS 特殊寄存器机制
enum SR : uint8_t {
    // 线程坐标 (CTA 内, 3D)
    SR_TID_X=0,  SR_TID_Y=1,  SR_TID_Z=2,    // threadIdx.{x,y,z}
    // CTA 坐标 (grid 内, 3D)
    SR_CTAID_X=8,SR_CTAID_Y=9,SR_CTAID_Z=10, // blockIdx.{x,y,z}
    // CTA 维度 (每维线程数)
    SR_NTID_X=11,SR_NTID_Y=12,SR_NTID_Z=13,  // blockDim.{x,y,z}
    // Grid 维度 (每维 CTA 数)
    SR_NCTAID_X=14,SR_NCTAID_Y=19,SR_NCTAID_Z=20, // gridDim.{x,y,z}
    // 硬件状态
    SR_SMID=16,      // NSM 编号 (0~15)
    SR_CLOCKLO=17,   // 64 位时钟低 32 位
    SR_CLOCKHI=18,   // 64 位时钟高 32 位
    SR_LANEID=21,    // lane 编号 (0-31)
    SR_WARPID=22,    // warp 编号 (CTA 内)
    SR_LR=24,        // 链接寄存器 (CALL 保存返回地址)
    // 地址空间基址 (64 位, flat 地址转换用)
    SR_PARAM_BASE=32, // 参数区基址 (const_mem bank 0)
    SR_SMEM_BASE=33,  // 共享内存 flat 基址
    SR_LMEM_BASE=34,  // 本地内存 flat 基址 (每线程不同)
    // 全局标识
    SR_WARPID_IN_GRID=40, // warp 在整个 grid 中的编号
    SR_GRIDID=41          // grid 编号
};

inline bool sr_valid(uint8_t s) {
    switch(s) {
    case SR_TID_X: case SR_TID_Y: case SR_TID_Z:
    case SR_CTAID_X: case SR_CTAID_Y: case SR_CTAID_Z:
    case SR_NTID_X: case SR_NTID_Y: case SR_NTID_Z:
    case SR_NCTAID_X: case SR_NCTAID_Y: case SR_NCTAID_Z:
    case SR_SMID: case SR_CLOCKLO: case SR_CLOCKHI:
    case SR_LANEID: case SR_WARPID: case SR_LR:
    case SR_PARAM_BASE: case SR_SMEM_BASE: case SR_LMEM_BASE:
    case SR_WARPID_IN_GRID: case SR_GRIDID:
        return true;
    default: return false;
    }
}

inline bool sr_is_u64(uint8_t s) {
    return s==SR_PARAM_BASE || s==SR_SMEM_BASE || s==SR_LMEM_BASE;
}

// =============================================================================
// 执行单元类型 (03 册 §6.1)
// -----------------------------------------------------------------------------
// NutShellGPU NSM (SIMT Core) 内有 5 种功能单元, 不同指令路由到不同单元:
//   EU_SP  : Single Precision - 32 路 SP ALU, 处理整数和单精度浮点 (主算力)
//   EU_SFU : Special Function Unit - 特殊函数单元, 处理 RCP/RSQ/MUFU (迭代)
//   EU_DP  : Double Precision - 双精度浮点单元 (吞吐量低于 SP)
//   EU_BRU : Branch/Control Unit - 分支与控制流单元, 处理 SIMT 栈操作
//   EU_LSU : Load/Store Unit - 访存单元, 处理 LD/ST/ATOM 等
// 调度器根据指令的目标单元和单元空闲状态决定发射 (03 册 §6.3 轮转调度)
// =============================================================================
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 5 种执行单元 (SP/SFU/DP/BRU/LSU) 对应教材对 SASS 流水线功能单元划分
enum ExecUnit : uint8_t { EU_SP=0, EU_SFU=1, EU_DP=2, EU_BRU=3, EU_LSU=4 };

// 操作码 → 执行单元映射 (决定指令路由到哪个流水线)
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 操作码到执行单元的路由, 对应教材对 SASS 指令到流水线单元映射的描述
inline ExecUnit op_unit(uint8_t op) {
    switch(op) {
    // 控制流与 warp 原语 → BRU (1 周期, 立即处理)
    case OP_NOP: case OP_S2R: case OP_SHFL: case OP_VOTE:
    case OP_BRA: case OP_BRX: case OP_CALL: case OP_RET:
    case OP_SSY: case OP_BAR: case OP_MEMBAR: case OP_EXIT:
    case OP_YIELD: case OP_BRKPT:
        return EU_BRU;
    // 特殊函数 → SFU (8 周期, 2 周期间隔)
    case OP_RCP: case OP_RSQ: case OP_MUFU:
        return EU_SFU;
    // 双精度 → DP (8 周期, 2 周期间隔)
    case OP_DADD:
        return EU_DP;
    // 访存 → LSU (标称 30 周期, 实际由缓存/DRAM 决定)
    case OP_LDC: case OP_CVTA: case OP_LD: case OP_ST: case OP_LDU:
    case OP_ATOM: case OP_RED: case OP_PREFETCH: case OP_LD128: case OP_ST128:
        return EU_LSU;
    // 除法/取余 → SP (微码实现, 24 周期)
    case OP_IDIV: case OP_IREM:
        return EU_SP; // microcode
    // 其他整数/浮点 ALU → SP (4 周期, 1 周期间隔)
    default:
        return EU_SP;
    }
}

// =============================================================================
// 指令时序参数 (00 册 §5, 02 册 §7)
// -----------------------------------------------------------------------------
// latency: 从发射到结果可用的周期数 (流水线深度)
// interval: 连续发射同单元指令的最小周期间隔 (吞吐量倒数)
// 周期级模拟器使用这些参数建模流水线时序 (03 册 §6)
// =============================================================================
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 指令时序参数 (latency/interval) 对应教材对不同功能单元流水线深度的描述
struct OpTiming { int latency; int interval; ExecUnit unit; };

// 操作码 → 时序参数
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 操作码到时序参数的映射, 对应教材对 SASS 不同指令族延迟/吞吐差异的描述
// 1.1: 此函数保留原 ISA 教学默认表; CycleSim 使用 timing_model.hpp 的
// execution_latency 和 Config 启动间隔, 不调用本表决定当前执行时间。
inline OpTiming op_timing(uint8_t op) {
    switch(op) {
    // 整数除法: 微码实现, 高延迟低吞吐 (24 周期, 20 间隔)
    case OP_IDIV: case OP_IREM: return {24, 20, EU_SP};
    // 特殊函数: 迭代算法 (8 周期, 2 间隔)
    case OP_RCP: case OP_RSQ: case OP_MUFU: return {8, 2, EU_SFU};
    // 双精度: (8 周期, 2 间隔)
    case OP_DADD: return {8, 2, EU_DP};
    // 控制流: 1 周期完成 (分支解析/SIMT栈操作)
    case OP_S2R: case OP_NOP: case OP_YIELD: case OP_BRKPT:
    case OP_SHFL: case OP_VOTE: case OP_BRA: case OP_BRX:
    case OP_CALL: case OP_RET: case OP_SSY: case OP_BAR:
    case OP_MEMBAR: case OP_EXIT:
        return {1, 1, EU_BRU};
    // 常量加载: 30 周期 (从 const cache 读取)
    case OP_LDC: return {30, 1, EU_LSU};
    // 访存: 标称 30 周期, 实际由缓存命中/缺失动态决定
    case OP_LD: case OP_ST: case OP_LDU: case OP_LD128: case OP_ST128:
    case OP_ATOM: case OP_RED: case OP_PREFETCH: case OP_CVTA:
        return {30, 1, EU_LSU}; // nominal; mem path overrides
    // 默认: SP ALU 整数/浮点 (4 周期, 1 间隔)
    default:
        return {4, 1, EU_SP};
    }
}

// =============================================================================
// 全局常量 (00 册 §4, 01 册 §3)
// -----------------------------------------------------------------------------
// 这些常量定义了 NutShellGPU 的硬件配置参数, 在编译期固定
// =============================================================================
// 教材引用: 第 2 章 §2.1.2 Threading Model (p.12) — WARP_SZ=32 对应教材对 SIMT 32 线程 warp 的基本定义
// 教材引用: 第 2 章 §2.1 Programming Model (p.10) — MAX_CTAS_PER_SM/NSM_COUNT/MP_COUNT 对应教材对 Grid/CTA/SM 划分的描述
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — RZ=255 零寄存器 (读为 0, 写丢弃), 对应教材明确提到的 RZ 特殊寄存器
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — SIMT_STACK_MAX=32 限定嵌套发散栈深度, 对应教材对 SIMT 栈式重汇聚机制的描述
static constexpr int WARP_SZ = 32;           // warp 大小: 32 线程 (SIMT 基本调度单元)
static constexpr int MAX_WARPS_PER_SM = 64;  // 每 NSM 最多 64 个并发 warp (占用率上限)
static constexpr int MAX_CTAS_PER_SM = 16;   // 每 NSM 最多 16 个并发 CTA
static constexpr int NSM_COUNT = 16;          // 全芯片 16 个 NSM (SIMT Core)
static constexpr int MP_COUNT = 6;            // 6 个内存分区 (Memory Partition)
static constexpr int RF_REGS = 256;           // 每线程 256 个寄存器 (R0-R254 + RZ)
static constexpr int RZ = 255;                // RZ: 零寄存器, 读始终为 0, 写丢弃
static constexpr int SIMT_STACK_MAX = 32;     // SIMT 栈最大深度 32 (嵌套发散上限)
// EXIT_SENTINEL: 特殊 rpc 值, 表示栈底 (无重汇聚点)
// 值 0xFFF...F8 是 8 字节对齐的"全 1"地址, 不会与真实 PC 冲突
static constexpr uint64_t EXIT_SENTINEL = 0xFFFFFFFFFFFFFFF8ULL;
// -----------------------------------------------------------------------------
// Flat 地址空间布局 (04 册 §2)
// 64 位 flat 地址空间按高位划分为 4 个 1 TiB 窗口:
//   [0x0000_0000_0000_0000, 0x0000_FFFF_FFFF_FFFF) : Global (1 TiB)
//   [0x0001_0000_0000_0000, 0x0001_FFFF_FFFF_FFFF) : Shared (1 TiB)
//   [0x0002_0000_0000_0000, 0x0002_FFFF_FFFF_FFFF) : Local (1 TiB)
//   [0x0003_0000_0000_0000, 0x0003_FFFF_FFFF_FFFF) : Const (1 TiB)
// decode_flat() 通过高位判断地址空间
// -----------------------------------------------------------------------------
// 教材引用: 第 4 章 §4.1.1 Constant Memory (p.68) — flat 地址空间按高位划分 4 个 1 TiB 窗口, 对应教材对 GPU 多地址空间统一编址的描述
static constexpr uint64_t FLAT_GLOBAL_BASE = 0x0000000000000000ULL; // Global 窗口基址
static constexpr uint64_t FLAT_SHARED_BASE = 0x0000010000000000ULL; // Shared 窗口基址
static constexpr uint64_t FLAT_LOCAL_BASE  = 0x0000020000000000ULL; // Local 窗口基址
static constexpr uint64_t FLAT_CONST_BASE  = 0x0000030000000000ULL; // Const 窗口基址
static constexpr uint64_t FLAT_GLOBAL_MASK = 0x000000FFFFFFFFFFULL; // Global 窗口掩码 (1 TiB)
static constexpr uint64_t FLAT_WINDOW_SZ   = 0x10000000000ULL;       // 每个窗口 1 TiB

// =============================================================================
// 解码后的指令结构 (05 册 §2.1)
// -----------------------------------------------------------------------------
// decode() 函数将 64 位原始指令字解码为此结构
// 字段含义取决于指令格式 (kind):
//   R 格式: d=目的, a=源1, c=源2, x/y/b=修饰符
//   I 格式: d=目的, imm32=立即数, x=修饰符
//   B 格式: imm26=分支偏移(×8), x=修饰符
//   M 格式: d=目的, a=基址, c=偏移寄存器, x=空间+宽度, y=提示
//   MI格式: d=目的, a=基址, simm16=偏移, x=空间+宽度, y=提示
//
// 谓词保护 (guard): g=使能, gn=取反, gp=谓词号(0-6=P0-P6, 7=PT恒真)
// 每条指令执行前, 根据谓词计算 active mask (03 册 §4 SIMT 执行)
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — 解码后的 Inst 结构组织各字段, 对应教材描述的 64 位编码字段提取 (公共头 + 格式特定字段)
// 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) — g/gn/gp 三字段编码谓词守卫 @Pn/@!Pn, 对应教材对 active mask 与谓词保护机制的描述
struct Inst {
    uint8_t  op;       // 操作码 (7 位, bits 63:57)
    uint8_t  g;        // 谓词保护使能 (bit 56): 1=使用谓词守卫
    uint8_t  gn;       // 谓词取反 (bit 55): 1=@!Pn (否定)
    uint8_t  gp;       // 谓词编号 (bits 54:52): 0-6=P0-P6, 7=PT(恒真)
    uint8_t  m;        // 修饰符 (bits 51:48): 舍入模式等
    uint8_t  kind;     // 指令格式 (FmtKind)
    uint8_t  d, a, c;  // 寄存器号: d=目的, a=源1, c=源2 (RZ=255 表示零寄存器)
    uint8_t  x, y;     // 修饰字段: 宽度/空间/比较码/模式等
    uint8_t  b;        // R 格式 bits 7:0: CAS 比较寄存器或保留
    int32_t  imm32;    // I 格式: 32 位立即数
    int32_t  imm26;    // B 格式: 26 位有符号分支偏移 (单位 8 字节)
    int16_t  simm16;   // MI 格式: 16 位有符号内存偏移
    uint64_t raw;      // 原始 64 位指令字 (用于往返校验)
    uint64_t raw_pc;   // 该指令的 PC 地址 (用于分支目标计算)

    // ---- 内存格式辅助函数 ----
    // X[2:0]: 地址空间 (AS_GLOBAL/AS_SHARED/AS_LOCAL/AS_CONST/AS_FLAT)
    uint8_t mem_space() const { return x & 0x7; }
    // X[6:3]: 数据宽度 (W_U8..W_U128)
    uint8_t mem_width() const { return (x >> 3) & 0xF; }
    // X[7]: M 类型标志 (1=M格式寄存器偏移, 0=MI格式立即数偏移)
    bool    mem_is_mtype() const { return (x >> 7) & 1; }
    // Y[2:0]: 缓存提示 (CH_CA/CH_CG/CH_CS/CH_CV/CH_LU)
    uint8_t mem_hint() const { return y & 0x7; }
    // LDC 专用: X[6:5] 宽度 (0=U32, 1=U64, 2=F32)
    uint8_t ldc_width() const { return (x >> 5) & 0x3; }
    // LDC 专用: X[3:0] 常量 bank 编号 (0=参数区, 1-7=常量区)
    uint8_t ldc_bank() const { return x & 0xF; }

    // ---- SETP/FSET 比较辅助函数 ----
    // X[3:0]: 比较码 (CC_EQ..CC_NAN)
    uint8_t cmp_code() const { return x & 0xF; }
    // X[6:4]: 比较类型 (0=S32, 1=U32, 2=F32, ...)
    uint8_t cmp_type() const { return (x >> 4) & 0x7; }
    // Y[1:0]: 谓词组合方式 (0=AND, 1=OR, 2=XOR, 3=SET直接赋值)
    uint8_t pred_combine() const { return y & 0x3; }
    // Y[7]: 是否将结果写入谓词目的 (SETP 标志)
    bool    setp_is_pred_dst() const { return (y >> 7) & 1; }
    // C[2:0]: 源谓词编号 (用于谓词组合)
    uint8_t setp_src_pred() const { return c & 0x7; }

    // ---- 谓词守卫计算 ----
    // 给定 lane 的谓词值, 返回该 lane 是否被本指令使能
    // pred_val: 该 lane 的谓词值 (0 或 1)
    // 返回: true=该 lane 执行本指令, false=该 lane 被屏蔽
    // 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) — guard_lane 按 @Pn/@!Pn/PT 计算 lane 是否被使能, 是 active mask 生成的核心
    bool guard_lane(bool pred_val) const {
        if (!g) return true;           // 无守卫: 所有 lane 使能
        bool pv = (gp == 7) ? true : pred_val;  // PT(7)=恒真
        return gn ? !pv : pv;          // gn=1 取反
    }
};

// =============================================================================
// 指令解码器 (05 册 §2.2)
// -----------------------------------------------------------------------------
// 将 64 位原始指令字 (word) 解码为 Inst 结构
// pc: 该指令的程序计数器值 (用于分支目标计算和错误报告)
// out: 输出解码后的指令
// 返回: 错误码 (ERR_OK=成功, 其他=解码错误)
//
// 解码步骤:
//   1. 提取公共头部字段 (OP, guard, modifier) - bits 63:48
//   2. 根据操作码确定指令格式 (R/I/B/M/MI)
//   3. 按格式提取特定字段 (d/a/c/x/y/b/imm32/imm26/simm16)
//   4. 合法性检查 (保留OP, SR有效性, 寄存器对齐, BAR ID)
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — decode 从 64 位原始字提取公共头/格式特定字段, 对应教材描述的字段提取与往返一致性 (encode→decode→encode) 要求
inline int decode(uint64_t word, uint64_t pc, Inst& out) {
    out.raw = word;
    out.raw_pc = pc;
    // ---- 第 1 步: 提取 16 位公共头部 (bits 63:48) ----
    uint16_t hdr = (uint16_t)(word >> 48);
    out.op = (hdr >> 9) & 0x7F;      // bits 63:57: 操作码 (7 位)
    out.g  = (hdr >> 8) & 1;        // bit 56: 谓词守卫使能
    out.gn = (hdr >> 7) & 1;        // bit 55: 谓词取反
    out.gp = (hdr >> 4) & 0x7;      // bits 54:52: 谓词编号
    out.m  = hdr & 0xF;             // bits 51:48: 修饰符 (舍入模式等)

    // 清零所有格式特定字段 (后面按格式填充)
    out.d = out.a = out.c = out.x = out.y = out.b = 0;
    out.imm32 = 0; out.imm26 = 0; out.simm16 = 0;

    // ---- 第 2 步: 根据操作码确定指令格式 ----
    auto determine_kind = [](uint8_t op) -> std::pair<FmtKind,bool> {
        // 返回 (格式, 是否内存格式需进一步区分 M/MI)
        switch(op) {
        case OP_MOV32I: case OP_SETPI: case OP_TRAP: case OP_LOP3:
            return {FK_I, false};   // 立即数格式
        case OP_BRA: case OP_CALL: case OP_SSY:
            return {FK_B, false};   // 分支格式
        case OP_LDC:
            return {FK_MI, false};  // LDC 固定使用 MI 格式
        case OP_LD: case OP_ST: case OP_LDU:
        case OP_PREFETCH: case OP_LD128: case OP_ST128:
            return {FK_M, true};    // 内存格式: 需用 X[7] 区分 M 与 MI
        default:
            return {FK_R, false};   // 默认寄存器格式
        }
    };

    auto [k, is_mem] = determine_kind(out.op);
    if (is_mem) {
        // ---- M 与 MI 格式的区分 (02 册 §2.1) ----
        // M 格式布局:  d[47:40] a[39:32] c[31:24] X[23:16] Y[15:8] 0x00[7:0]
        // MI 格式布局: d[47:40] a[39:32] X[31:24] Y[23:16] simm16[15:0]
        //
        // 判别位: bit 23
        //   - M 格式中 bit 23 = X[7] (M 格式的 X 在 bits 23:16)
        //   - MI 格式中 bit 23 = Y[7] (MI 格式的 Y 在 bits 23:16, Y[7:4] 保留为 0)
        // 因此: bit 23 = 1 → M 格式 (寄存器偏移), bit 23 = 0 → MI 格式 (立即数偏移)
        bool is_m = (word >> 23) & 1;
        k = is_m ? FK_M : FK_MI;
    }
    out.kind = k;

    // ---- 第 3 步: 按格式提取特定字段 ----
    switch(k) {
    case FK_R: {
        // R 格式: d[47:40] a[39:32] c[31:24] x[23:16] y[15:8] b[7:0]
        out.d = (word >> 40) & 0xFF;
        out.a = (word >> 32) & 0xFF;
        out.c = (word >> 24) & 0xFF;
        out.x = (word >> 16) & 0xFF;
        out.y = (word >> 8)  & 0xFF;
        out.b = word & 0xFF;
        break;
    }
    case FK_I: {
        // I 格式: d[47:40] imm32[39:8] x[7:0]
        out.d = (word >> 40) & 0xFF;
        out.imm32 = (int32_t)((word >> 8) & 0xFFFFFFFF);
        out.x = word & 0xFF;
        break;
    }
    case FK_B: {
        // B 格式: imm26[47:22] x[21:16]
        uint32_t u26 = (uint32_t)((word >> 22) & 0x3FFFFFF);
        // 符号扩展 26 位 → 32 位有符号数
        out.imm26 = (int32_t)(u26 << 6) >> 6;
        out.x = (word >> 16) & 0x3F;
        break;
    }
    case FK_M: {
        // M 格式: d[47:40] a[39:32] c[31:24] x[23:16] y[15:8] b[7:0]
        out.d = (word >> 40) & 0xFF;
        out.a = (word >> 32) & 0xFF;
        out.c = (word >> 24) & 0xFF;
        out.x = (word >> 16) & 0xFF;
        out.y = (word >> 8) & 0xFF;
        out.b = word & 0xFF;
        break;
    }
    case FK_MI: {
        // MI 格式: d[47:40] a[39:32] x[31:24] y[23:16] simm16[15:0]
        out.d = (word >> 40) & 0xFF;
        out.a = (word >> 32) & 0xFF;
        out.x = (word >> 24) & 0xFF;
        out.y = (word >> 16) & 0xFF;
        out.simm16 = (int16_t)(word & 0xFFFF);
        break;
    }
    }

    // ---- 第 4 步: 合法性检查 ----

    // 检查操作码是否在保留区间 (保留区间不允许使用)
    // 保留: 0x0A-0x0F, 0x26-0x2F, 0x5B-0x5F, 0x68-0x6F, 0x70-0x7F
    auto is_reserved_op = [](uint8_t op) -> bool {
        if (op >= 0x0A && op <= 0x0F) return true;
        if (op >= 0x26 && op <= 0x2F) return true;
        if (op >= 0x5B && op <= 0x5F) return true;
        if (op >= 0x68 && op <= 0x6F) return true;
        if (op >= 0x70) return true;
        return false;
    };
    if (is_reserved_op(out.op)) return ERR_BAD_OPCODE;

    // S2R 指令: 检查特殊寄存器编号是否有效
    if (out.op == OP_S2R) {
        uint8_t sr = out.x & 0x3F;
        if (!sr_valid(sr)) return ERR_BAD_ENCODING;
    }

    // 64/128 位操作的寄存器对齐检查 (要求偶数寄存器号, 因使用 Rd:Rd+1 对)
    if (out.kind == FK_R || out.kind == FK_M) {
        auto check_pair = [&](uint8_t reg) -> int {
            if (reg != RZ && (reg & 1)) return ERR_UNALIGNED_REG; // 非偶数
            return 0;
        };
        uint8_t w = out.mem_width();
        bool needs_pair = (w == W_U64 || w == W_F64);    // 64 位需寄存器对
        bool needs_quad = (w == W_U128);                   // 128 位需四寄存器
        if (out.kind == FK_M) {
            if (needs_pair || needs_quad) {
                int e;
                if ((e = check_pair(out.d))) return e;
                if ((e = check_pair(out.a))) return e;
                if (out.c != RZ && (e = check_pair(out.c))) return e;
            }
        }
        // DADD (双精度加): 使用寄存器对
        if (out.op == OP_DADD) {
            int e;
            if ((e = check_pair(out.d))) return e;
            if ((e = check_pair(out.a))) return e;
            if ((e = check_pair(out.c))) return e;
        }
        // CVTA (地址转换): 64 位结果使用寄存器对
        if (out.op == OP_CVTA) {
            int e;
            if ((e = check_pair(out.d))) return e;
            if ((e = check_pair(out.a))) return e;
        }
    }
    if (out.kind == FK_MI) {
        // 1.1 修复 F11: LDC 的宽度编码在 x[6:5] (ldc_width), 与 LD/ST 的 x[6:3] (mem_width) 不同
        // 因此 LDC 不使用 mem_width() 做对齐检查, 而由下方 ldc_width() 专门处理
        uint8_t w = out.mem_width();
        if (out.op != OP_LDC && (w == W_U64 || w == W_F64 || w == W_U128)) {
            auto check_pair = [&](uint8_t reg) -> int {
                if (reg != RZ && (reg & 1)) return ERR_UNALIGNED_REG;
                return 0;
            };
            int e;
            if ((e = check_pair(out.d))) return e;
            if ((e = check_pair(out.a))) return e;
        }
        // LDC U64 宽度: 目的寄存器需对齐
        if (out.op == OP_LDC && out.ldc_width() == 1) {
            if (out.d != RZ && (out.d & 1)) return ERR_UNALIGNED_REG;
        }
    }
    // MOV32I: X[0]=1 表示 64 位扩展, d 必须偶数
    if (out.op == OP_MOV32I && (out.x & 1)) {
        if (out.d != RZ && (out.d & 1)) return ERR_UNALIGNED_REG;
    }
    // BAR 指令 (02 册 §3.5): X[5:4]=sync 模式 (0 SYNC / 1 ARRIVE / 2 WAIT),
    // X[3:0]=屏障 ID (0-15), X[7:6] 保留须为 0
    if (out.op == OP_BAR) {
        if (out.x & 0xC0) return ERR_BAR_ID;                // X[7:6] 保留位非 0
        if ((out.x & 0xF0) > 0x20) return ERR_BAD_ENCODING; // sync 模式 > WAIT 非法
    }

    return ERR_OK;
}

// =============================================================================
// 分支目标地址计算 (02 册 §5.4)
// -----------------------------------------------------------------------------
// B 格式指令 (BRA/CALL/SSY) 的 imm26 字段编码分支偏移量:
//   imm26 为有符号 26 位整数, 单位为 8 字节 (因为所有指令定长 8 字节)
//   target = PC + imm26 * 8        (相对分支, 最常用)
//   target = imm26 * 8             (绝对分支, X[0]=1 时)
//
// 例: SSY Ldone=0xB0, 当前 PC=0x50
//   imm26 = (0xB0 - 0x50) / 8 = 12, target = 0x50 + 12*8 = 0xB0 ✓
// =============================================================================
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — 分支目标计算 (target = PC + imm26*8), 用于 BRA 按谓词发散与 SSY 设置重汇聚点
inline uint64_t branch_target(const Inst& ins, uint64_t pc) {
    bool absolute = ins.x & 1;            // X[0]: 1=绝对地址, 0=相对地址
    if (absolute) {
        return (uint64_t)(ins.imm26) * 8; // 绝对: 目标 = imm26 * 8
    }
    return pc + (int64_t)ins.imm26 * 8;   // 相对: 目标 = PC + imm26 * 8
}

// =============================================================================
// 内存有效地址计算 (02 册 §5.5, 04 册 §3)
// -----------------------------------------------------------------------------
// LD/ST 指令的访存地址 = 基址寄存器值 + 偏移
//   M  格式: 地址 = Ra + Rc      (寄存器偏移, 两寄存器相加)
//   MI 格式: 地址 = Ra + simm16  (立即数偏移, 16 位有符号)
// 注意: 返回的是空间内偏移, 具体物理地址需由上层根据地址空间映射
//   - Global/Local: 经 flat 转换后访问 DRAM
//   - Shared: 直接作为共享内存 bank 内偏移
//   - Const: 作为常量缓存 bank 内偏移
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) & 第 4 章 §4.1.1 Constant Memory (p.68) — M/MI 格式地址 = Ra+Rc 或 Ra+simm16, 用于 LD/ST/LDC 访问各地址空间
inline uint64_t mem_addr(const Inst& ins, uint64_t ra_val, uint64_t rc_val) {
    uint64_t addr = ra_val;                // 基址 = Ra 寄存器值
    if (ins.kind == FK_M) {
        addr += rc_val;                     // M 格式: + Rc 寄存器偏移
    } else if (ins.kind == FK_MI) {
        addr += (int64_t)ins.simm16;        // MI 格式: + simm16 立即数偏移
    }
    return addr;
}

// =============================================================================
// 加载后的符号/零扩展 (02 册 §5.6)
// -----------------------------------------------------------------------------
// LD 指令加载数据后, 需将窄数据扩展到 32 位寄存器宽度
//   无符号 (U8/U16): 高位补零 (零扩展)
//   有符号 (S8/S16): 高位补符号位 (符号扩展)
//   F16: 保持原样 (浮点格式转换由 F2F 指令处理)
// 参数 val: 从内存读入的原始值 (可能含高位垃圾)
// 参数 width: 数据宽度枚举 (W_U8/W_S8/W_U16/W_S16/W_F16)
// 返回: 扩展后的 32 位值
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — 窄加载的符号/零扩展由 Width 字段决定, 对应教材对加载指令宽度修饰符语义的描述
inline uint32_t extend_load(uint32_t val, uint8_t width) {
    switch(width) {
    case W_U8:  return val & 0xFF;                              // 零扩展 8→32
    case W_S8:  return (uint32_t)(int32_t)(int8_t)(val & 0xFF); // 符号扩展 8→32
    case W_U16: return val & 0xFFFF;                            // 零扩展 16→32
    case W_S16: return (uint32_t)(int32_t)(int16_t)(val & 0xFFFF); // 符号扩展 16→32
    case W_F16: return val & 0xFFFF;                            // F16 原样保留
    default: return val;                                        // 32/64 位无需扩展
    }
}

// =============================================================================
// 整数比较 (02 册 §5.2)
// -----------------------------------------------------------------------------
// SETP/SETPI/FSET 指令使用此函数执行整数比较, 结果为布尔值
// 参数 a, c: 两个 32 位无符号操作数 (比较时按需解释为有符号)
// 参数 code: 比较码 (CC_EQ/CC_NE/CC_LT/CC_LE/CC_GT/CC_GE/CC_HI/CC_HS/CC_LO/CC_LS)
// 参数 is_unsigned: true=无符号比较, false=有符号比较
// 返回: 比较结果 (true/false)
//
// 有序比较码 (EQ/NE/LT/LE/GT/GE) 根据 is_unsigned 切换有/无符号语义
// 无符号比较码 (HI/HS/LO/LS) 始终用无符号比较 (忽略 is_unsigned)
// =============================================================================
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 整数比较 SETP/SETPI 按 CmpCode 执行, 对应教材对 SETP 谓词设置指令语义的描述
inline bool int_cmp(uint32_t a, uint32_t c, uint8_t code, bool is_unsigned) {
    switch(code) {
    case CC_EQ: return a == c;   // 等于 (有序和无序相同)
    case CC_NE: return a != c;   // 不等于
    case CC_LT: return is_unsigned ? (a < c) : (int32_t)a < (int32_t)c;   // 小于
    case CC_LE: return is_unsigned ? (a <= c) : (int32_t)a <= (int32_t)c; // 小于等于
    case CC_GT: return is_unsigned ? (a > c) : (int32_t)a > (int32_t)c;   // 大于
    case CC_GE: return is_unsigned ? (a >= c) : (int32_t)a >= (int32_t)c; // 大于等于
    case CC_HI: return a > c;    // 无符号高于 (Higher)
    case CC_HS: return a >= c;   // 无符号高于等于 (Higher or Same)
    case CC_LO: return a < c;    // 无符号低于 (Lower)
    case CC_LS: return a <= c;   // 无符号低于等于 (Lower or Same)
    default: return false;       // 未知比较码, 返回 false
    }
}

// =============================================================================
// 浮点比较 (02 册 §5.2, IEEE 754 语义)
// -----------------------------------------------------------------------------
// FCMP/SETP.F32 指令使用此函数执行浮点比较
// 参数 a_bits, c_bits: 操作数的 32 位位模式 (reinterpret 为 float)
// 参数 code: 比较码, 有序 (CC_EQ/CC_LT/...) 和无序 (CC_GTU/CC_LTU/...) 两类
//
// IEEE 754 NaN 处理规则:
//   有序比较 (ordered): 任一操作数为 NaN → 结果为 false
//   无序比较 (unordered, 后缀 U): 任一操作数为 NaN → 结果为 true
//   CC_NUM: 两操作数都不是 NaN 时为 true
//   CC_NAN: 存在 NaN 操作数时为 true
// =============================================================================
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 浮点比较 FCMP/SETP.F32 按 IEEE 754 有序/无序语义执行, 对应教材对 SETP 浮点变体的描述
inline bool fp_cmp(uint32_t a_bits, uint32_t c_bits, uint8_t code) {
    float fa, fc;
    std::memcpy(&fa, &a_bits, 4);  // 位模式重解释为 float (避免类型双关 UB)
    std::memcpy(&fc, &c_bits, 4);
    bool a_nan = std::isnan(fa);   // 检测 NaN 操作数
    bool c_nan = std::isnan(fc);
    bool any_nan = a_nan || c_nan; // 任一为 NaN
    switch(code) {
    case CC_EQ: return !any_nan && fa == fc;   // 有序等于 (NaN→false)
    case CC_NE: return any_nan || fa != fc;    // 无序不等于 (NaN→true)
    case CC_LT: return !any_nan && fa < fc;    // 有序小于
    case CC_LE: return !any_nan && fa <= fc;   // 有序小于等于
    case CC_GT: return !any_nan && fa > fc;    // 有序大于
    case CC_GE: return !any_nan && fa >= fc;   // 有序大于等于
    case CC_FE: return !any_nan && fa == fc;   // 浮点等于 (同 CC_EQ)
    case CC_GTU: return any_nan || fa > fc;    // 无序大于 (NaN→true)
    case CC_LTU: return any_nan || fa < fc;    // 无序小于
    case CC_LEU: return any_nan || fa <= fc;   // 无序小于等于
    case CC_NUM: return !any_nan;              // 是数字 (非 NaN)
    case CC_NAN: return any_nan;               // 含 NaN
    default: return false;
    }
}

// =============================================================================
// 浮点舍入模式 (02 册 §5.1)
// -----------------------------------------------------------------------------
// IEEE 754 定义 4 种舍入模式, 由指令的 m 字段 (bits 51:48) 低 2 位指定:
//   RM_RN: Round to Nearest, ties to Even - 就近舍入, 0.5 向偶数舍入 (默认)
//   RM_RZ: Round toward Zero - 向零舍入 (截断)
//   RM_RM: Round toward Minus infinity - 向负无穷舍入 (向下取整)
//   RM_RP: Round toward Plus infinity - 向正无穷舍入 (向上取整)
// =============================================================================
// 教材引用: 第 2 章 §2.2.3 Fused Multiply-Add (p.20) — IEEE 754 4 种舍入模式由 m 字段低 2 位指定, FFMA 单次舍入依赖此模式
enum RoundMode : uint8_t { RM_RN=0, RM_RZ=1, RM_RM=2, RM_RP=3 };

// 浮点舍入函数: 按指定模式对 float 值舍入
// 参数 v: 待舍入的浮点值
// 参数 m: 修饰符字节 (取低 2 位作为舍入模式)
// 教材引用: 第 2 章 §2.2.3 Fused Multiply-Add (p.20) — round_fp32 实现 FFMA/FADD/FMUL 等浮点指令的单次舍入, 对应教材对融合乘加单次舍入语义的描述
// Input is already FP32; arithmetic rounds during the operation, never to integer.
// =============================================================================
// 1.1 修复说明 F07 FP32 舍入辅助函数
// -----------------------------------------------------------------------------
// FP32 输入已经具有浮点精度, 不能再次按整数 floor/ceil/trunc 处理。
// 显式 FRND 与算术指令的浮点舍入是不同语义。
// =============================================================================
inline float round_fp32(float v, uint8_t) { return v; }

// =============================================================================
// 位计数 popcount (02 册 §5.7, POPC 指令)
// -----------------------------------------------------------------------------
// 计算 32 位整数中置位 (1) 的个数
// 实现: 优先使用编译器内建函数 (1 周期指令), 否则用 SWAR 算法回退
//   GCC/Clang: __builtin_popcount
//   MSVC: __popcnt (需 /arch:SSE4.2 或更高, 否则软件实现)
//   回退: SWAR (SIMD Within A Register) 位并行算法
//
// SWAR 算法原理 (5 步):
//   1. 每 2 位一组, 统计组内 1 的个数: v = v - ((v>>1) & 0x55555555)
//   2. 每 4 位一组累加: v = (v & 0x33333333) + ((v>>2) & 0x33333333)
//   3. 每 8 位一组累加: v = (v + (v>>4)) & 0x0F0F0F0F
//   4. 乘以 0x01010101 将 4 个字节并行累加到高字节
//   5. 右移 24 位取出总数
// =============================================================================
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — popcount32 实现 POPC 指令的位计数语义, 对应教材 SASS 操作码分类中的位操作族
inline uint32_t popcount32(uint32_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_popcount(v);   // GCC/Clang 内建: 单指令
#elif defined(_MSC_VER)
    return (uint32_t)__popcnt(v);             // MSVC 内建
#else
    // SWAR 回退算法 (无硬件 popcount 时使用)
    v = v - ((v >> 1) & 0x55555555u);          // 2 位组内计数
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u); // 4 位组累加
    v = (v + (v >> 4)) & 0x0F0F0F0Fu;         // 8 位组累加
    return (v * 0x01010101u) >> 24;           // 字节并行求和
#endif
}

// =============================================================================
// 位反转 bitrev32 (02 册 §5.7, BREV 指令)
// -----------------------------------------------------------------------------
// 将 32 位整数的位序反转: bit[0]↔bit[31], bit[1]↔bit[30], ...
// 用途: FFT 等算法中的位反转寻址 (bit-reversal addressing)
//
// 实现: 分治法, 5 步交换:
//   1. 奇偶位交换:   abcd... → badc...
//   2. 2 位组交换:   bbad... → dcba...
//   3. 4 位组交换
//   4. 8 位组交换
//   5. 16 位组交换 (高低半字互换)
// =============================================================================
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — bitrev32 实现 BREV 指令的位反转语义, 对应教材 SASS 操作码分类中的位操作族
inline uint32_t bitrev32(uint32_t x) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1); // 奇偶位交换
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2); // 2位组交换
    x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4); // 4位组交换
    x = ((x >> 8) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8); // 8位组交换
    return (x >> 16) | (x << 16);                            // 16位组交换
}

// =============================================================================
// 位查找函数 (02 册 §5.7, FIND 指令)
// -----------------------------------------------------------------------------
// 三个相关函数, 用于位扫描:
//   find_msb(x): 返回最高置位的位置 (0-31), x=0 时返回 32
//                 例: find_msb(0b1000) = 3, find_msb(0b10000000) = 7
//   clz32(x):    返回前导零个数 (Count Leading Zeros), x=0 时返回 32
//                 例: clz32(0b1) = 31, clz32(0x80000000) = 0
//   ctz32(x):    返回末尾零个数 (Count Trailing Zeros), x=0 时返回 32
//                 例: ctz32(0b1000) = 3, ctz32(0b10100) = 2
//
// 关系: find_msb(x) = 31 - clz32(x)
// 实现: 优先用编译器内建 (BSR/BSF 指令), 否则软件循环回退
// =============================================================================
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — find_msb/clz32/ctz32 实现 FIND 指令的位扫描语义, 对应教材 SASS 操作码分类中的位查找族
inline uint32_t find_msb(uint32_t x) { // 最高置位位置; 无则返回 32
    if (x == 0) return 32;
#if defined(__GNUC__) || defined(__clang__)
    return 31 - (uint32_t)__builtin_clz(x);   // GCC/Clang 内建
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse(&idx, x);                 // MSVC: BitScanReverse (BSR 指令)
    return 31 - (uint32_t)idx;
#else
    uint32_t r = 0;
    while (x >>= 1) r++;                      // 软件回退: 逐位移位计数
    return r;
#endif
}
inline uint32_t clz32(uint32_t x) { // 前导零个数
    if (x == 0) return 32;
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_clz(x);        // GCC/Clang 内建
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse(&idx, x);                 // MSVC: 31 - BSR 结果
    return 31 - (uint32_t)idx;
#else
    uint32_t n = 0;
    while (!(x & 0x80000000u)) { x <<= 1; n++; } // 软件回退: 左移找首1
    return n;
#endif
}
inline uint32_t ctz32(uint32_t x) { // 末尾零个数
    if (x == 0) return 32;
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_ctz(x);        // GCC/Clang 内建
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward(&idx, x);                 // MSVC: BitScanForward (BSF 指令)
    return (uint32_t)idx;
#else
    uint32_t n = 0;
    while (!(x & 1u)) { x >>= 1; n++; }       // 软件回退: 右移找末1
    return n;
#endif
}

// =============================================================================
// 操作码 → 助记符字符串 (02 册 §3)
// -----------------------------------------------------------------------------
// 将 7 位操作码映射为人类可读的助记符, 用于反汇编和调试输出
// 返回: 指向字符串字面量的 const char* (静态存储, 无需释放)
// 未知操作码返回 "UNK"
// =============================================================================
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 操作码到助记符的映射, 用于反汇编 SASS 风格的指令文本
inline const char* op_mnemonic(uint8_t op) {
    switch(op) {
    // ---- 数据搬运与 warp 原语 (0x00-0x09) ----
    case OP_NOP: return "NOP";     case OP_MOV: return "MOV";     case OP_MOV32I: return "MOV32I";
    case OP_S2R: return "S2R";    case OP_LDC: return "LDC";     case OP_CVTA: return "CVTA";
    case OP_SHFL: return "SHFL";   case OP_VOTE: return "VOTE";   case OP_PRMT: return "PRMT";
    case OP_SELP: return "SELP";
    // ---- 整数 ALU (0x10-0x25) ----
    case OP_IADD: return "IADD";   case OP_IADD3: return "IADD3"; case OP_IMAD: return "IMAD";
    case OP_IMUL: return "IMUL";   case OP_ISUB: return "ISUB";   case OP_IMNMX: return "IMNMX";
    case OP_LOP: return "LOP";     case OP_SHF: return "SHF";     case OP_SHL: return "SHL";
    case OP_SHR: return "SHR";     case OP_SAR: return "SAR";      case OP_BFE: return "BFE";
    case OP_BFI: return "BFI";     case OP_POPC: return "POPC";   case OP_BREV: return "BREV";
    case OP_IABS: return "IABS";   case OP_INEG: return "INEG";   case OP_IDIV: return "IDIV";
    case OP_IREM: return "IREM";   case OP_LOP3: return "LOP3";   case OP_BMSK: return "BMSK";
    case OP_FIND: return "FIND";
    // ---- 浮点 ALU (0x30-0x3F) ----
    case OP_FADD: return "FADD";   case OP_FSUB: return "FSUB";   case OP_FMUL: return "FMUL";
    case OP_FFMA: return "FFMA";   case OP_FMNMX: return "FMNMX"; case OP_FSET: return "FSET";
    case OP_F2F: return "F2F";     case OP_XCVT: return "XCVT";   case OP_RCP: return "RCP";
    case OP_RSQ: return "RSQ";     case OP_MUFU: return "MUFU";   case OP_FRND: return "FRND";
    case OP_FABS: return "FABS";   case OP_FNEG: return "FNEG";   case OP_FCMP: return "FCMP";
    case OP_DADD: return "DADD";
    // ---- 谓词操作 (0x40-0x43) ----
    case OP_SETP: return "SETP";   case OP_SETPI: return "SETPI"; case OP_PLOP: return "PLOP";
    case OP_PSET2: return "PSET2";
    // ---- 控制流 (0x50-0x5A) ----
    case OP_BRA: return "BRA";     case OP_BRX: return "BRX";     case OP_CALL: return "CALL";
    case OP_RET: return "RET";     case OP_SSY: return "SSY";     case OP_BAR: return "BAR";
    case OP_MEMBAR: return "MEMBAR"; case OP_EXIT: return "EXIT"; case OP_YIELD: return "YIELD";
    case OP_TRAP: return "TRAP";   case OP_BRKPT: return "BRKPT";
    // ---- 内存 (0x60-0x67) ----
    case OP_LD: return "LD";       case OP_ST: return "ST";       case OP_LDU: return "LDU";
    case OP_ATOM: return "ATOM";   case OP_RED: return "RED";     case OP_PREFETCH: return "PREFETCH";
    case OP_LD128: return "LD128";  case OP_ST128: return "ST128";
    default: return "UNK";   // 未知/保留操作码
    }
}

// =============================================================================
// 反汇编器 (简化版, 02 册 §8)
// -----------------------------------------------------------------------------
// 将解码后的 Inst 结构格式化为汇编文本, 用于调试输出和测试验证
// 输出格式: [@Pn] MNEMONIC[.SPACE[.HINT]] operands
//
// 例:
//   @P0 BRA 0x10            → 谓词守卫分支
//   LDG.CA.F32 R0, [R8+0]   → 全局内存加载, CA 缓存, F32 宽度
//   FFMA.RN.F32 R0, R0, R1, R2 → 浮点乘加
//
// 简化: 不输出修饰符 m (舍入模式等), 仅输出关键信息
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — 反汇编器将解码字段格式化为 [@Pn] MNEMONIC[.SPACE[.HINT]] operands, 对应教材对 SASS 指令文本表示的描述
inline std::string disassemble(const Inst& ins) {
    std::ostringstream os;
    // ---- 谓词守卫前缀 (@Pn 或 @!Pn) ----
    if (ins.g) {
        os << "@";
        if (ins.gn) os << "!";             // 取反前缀
        if (ins.gp == 7) os << "PT";       // PT = 恒真谓词
        else os << "P" << (int)ins.gp;    // P0-P6
        os << " ";
    }
    os << op_mnemonic(ins.op);              // 助记符 (如 LD, FFMA, BRA)

    // ---- 内存指令的空间和缓存提示后缀 ----
    if (ins.kind == FK_M || ins.kind == FK_MI) {
        const char* sp = "G";               // 默认全局
        switch(ins.mem_space()) {
        case AS_GLOBAL: sp="G"; break; case AS_SHARED: sp="S"; break;
        case AS_LOCAL: sp="L"; break; case AS_CONST: sp="C"; break;
        case AS_FLAT: sp="F"; break;
        }
        os << "." << sp;                   // 如 LDG, LDS, LDC
        const char* hints[] = {"CA","CG","CS","CV","LU"};
        uint8_t h = ins.mem_hint();
        if (h < 5) os << "." << hints[h];  // 如 LDG.CA
    }
    // ---- 按格式输出操作数 ----
    switch(ins.kind) {
    case FK_R:
        os << " R" << (int)ins.d << ", R" << (int)ins.a << ", R" << (int)ins.c;
        break;
    case FK_I:
        os << " R" << (int)ins.d << ", R" << (int)ins.a << ", 0x" << std::hex << (uint32_t)ins.imm32 << std::dec;
        break;
    case FK_B:
        os << " 0x" << std::hex << (uint64_t)ins.imm26 << std::dec;
        break;
    case FK_M:
        os << " R" << (int)ins.d << ", [R" << (int)ins.a << "+R" << (int)ins.c << "]";
        break;
    case FK_MI:
        os << " R" << (int)ins.d << ", [R" << (int)ins.a << "+" << (int)ins.simm16 << "]";
        break;
    }
    return os.str();
}

// =============================================================================
// ╔══════════════════════════════════════════════════════════════════════════╗
// ║           NTAS1 指令编码格式 ASCII 位图 (教学版)                         ║
// ╠══════════════════════════════════════════════════════════════════════════╣
// ║ 所有指令 64 位定长 [63:0], 高 16 位公共头 + 低 48 位按格式区分            ║
// ╚══════════════════════════════════════════════════════════════════════════╝
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — ASCII 位图直观展示 64 位定长编码与公共头/格式特定字段布局, 对应教材对 SASS 编码字段位置图的描述
//
// ┌───────────────── 公共头部 bits[63:48] (所有格式共有) ─────────────────┐
// │ 63    57 56  55 52  51    48                                        │
// │┌─────────┬──┬──┬──────┬────────┐                                     │
// ││  OP[6:0] │g │gn│ gp[2:0]│  m[3:0] │  ← 16 bits                     │
// │└─────────┴──┴──┴──────┴────────┘                                     │
// │  操作码     谓 取 谓     修饰码                                        │
// │  7 bits    词 反 词     4 bits                                        │
// │             使  护 号                                                 │
// │             能  卫 码                                                 │
// └───────────────────────────────────────────────────────────────────────┘
//
// ════════════════════════════════════════════════════════════════════════
// 格式 1: FK_R (寄存器-寄存器) — 算术/逻辑指令 (FFMA/IADD/LOP/...)
// ════════════════════════════════════════════════════════════════════════
//  63          48 47    40 39    32 31    24 23    16 15     8 7     0
//  ┌──────────────┬────────┬────────┬────────┬────────┬─────────┬───────┐
//  │   公共头部    │d[7:0]  │a[7:0]  │c[7:0]  │x[7:0]  │ y[7:0]  │b[7:0] │
//  └──────────────┴────────┴────────┴────────┴────────┴─────────┴───────┘
//    16 bits        8         8        8        8        8        8
//  例: FFMA R2, R0, R1, R2 (d=2 是第三源! func_sim 用 fma(a,c,d))
//
// ════════════════════════════════════════════════════════════════════════
// 格式 2: FK_I (立即数) — MOV32I/SETP 等
// ════════════════════════════════════════════════════════════════════════
//  63          48 47    40 39                  8 7     0
//  ┌──────────────┬────────┬────────────────────────┬───────┐
//  │   公共头部    │d[7:0]  │    imm32[31:0]         │x[7:0] │
//  └──────────────┴────────┴────────────────────────┴───────┘
//    16 bits        8              32 bits              8
//  例: MOV32I R5, 0x40400000 (d=5, imm32=1.0f 的 IEEE754)
//
// ════════════════════════════════════════════════════════════════════════
// 格式 3: FK_B (分支) — BRA/SSY/BAR/EXIT 等
// ════════════════════════════════════════════════════════════════════════
//  63          48 47    32 31              6 5     0
//  ┌──────────────┬────────────────┬────────────────────┬───────┐
//  │   公共头部    │   保留 0x0000  │ imm25[24:0] (符号) │ op_ext│
//  └──────────────┴────────────────┴────────────────────┴───────┘
//    16 bits           16 bits             25 bits            8
//  例: @P0 BRA PC+0x38  编码 imm25=(target-PC)/8, 有符号
//
// ════════════════════════════════════════════════════════════════════════
// 格式 4: FK_M (内存-寄存器偏移) — LD/ST, 偏移在寄存器 c
// ════════════════════════════════════════════════════════════════════════
//  63          48 47    40 39    32 31    24 23    16 15     8 7     0
//  ┌──────────────┬────────┬────────┬────────┬────────┬─────────┬───────┐
//  │   公共头部    │d[7:0]  │a[7:0]  │c[7:0]  │x[7:0]  │ y[7:0]  │  保留  │
//  └──────────────┴────────┴────────┴────────┴────────┴─────────┴───────┘
//    16 bits        8         8        8        8        8        8
//  语义: addr = R[a] + R[c]   例: LDG.CA.F32 R2, [R8+R4]
//
// ════════════════════════════════════════════════════════════════════════
// 格式 5: FK_MI (内存-立即数偏移) — LD/ST 立即偏移
// ════════════════════════════════════════════════════════════════════════
//  63          48 47    40 39    32 31    24 23    16 15         0
//  ┌──────────────┬────────┬────────┬────────┬────────┬─────────────────┐
//  │   公共头部    │d[7:0]  │a[7:0]  │c[7:0]  │x[7:0]  │ simm15[14:0]   │
//  └──────────────┴────────┴────────┴────────┴────────┴─────────────────┘
//    16 bits        8         8        8        8        15 bits (有符号)
//  语义: addr = R[a] + simm15   例: LDC.C0 U64 R12, 0x0 (R12=xp 指针)
//
// =========================================================================
// ║ 注意事项                                                              ║
// ║   • 所有指令必须 8 字节对齐 (PC[2:0]=0)                                ║
// ║   • 64 位寄存器对编号必须是偶数 (R8:R9, R12:R13...)                    ║
// ║   • R255 是 RZ (硬连线 0), 写 RZ 无效                                  ║
// ║   • 谓词守卫 @Pn: g=1, gn=0, gp=n;  @!Pn: g=1, gn=1, gp=n             ║
// ║   • RZ 做源寄存器时值恒为 0, 做目的寄存器时写入被忽略                    ║
// =========================================================================

// =============================================================================
// 指令编码器 (用于测试和内核构建, 05 册 §2.3)
// -----------------------------------------------------------------------------
// 将各字段组装为 64 位指令字, 是 decode() 的逆操作
// 主要用途:
//   1. 测试中构造指令验证 decode() 往返一致性
//   2. 内核构建器 (build_saxpy_kernel 等) 生成机器码
//
// 参数说明:
//   op:  操作码 (7 位, 0-127)
//   g:   谓词守卫使能 (0/1)
//   gn:  谓词取反 (0/1)
//   gp:  谓词编号 (0-7, 7=PT)
//   m:   修饰符 (4 位, 舍入模式等)
//   kind: 指令格式 (FK_R/FK_I/FK_B/FK_M/FK_MI)
//   d, a, c: 寄存器号 (8 位, 0-255, 255=RZ)
//   x, y, b: 修饰字段 (8 位)
//   imm32:  I 格式立即数 (32 位)
//   imm26:  B 格式分支偏移 (26 位有符号)
//   simm16: MI 格式内存偏移 (16 位有符号)
// 返回: 组装后的 64 位指令字
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — encode 是 decode 的逆操作, 用于往返一致性校验 (encode→decode→encode) 与内核构建
inline uint64_t encode(uint8_t op, uint8_t g, uint8_t gn, uint8_t gp, uint8_t m,
                       FmtKind kind, uint8_t d, uint8_t a, uint8_t c,
                       uint8_t x, uint8_t y, uint8_t b,
                       int32_t imm32, int32_t imm26, int16_t simm16) {
    // ---- 组装 16 位公共头部 (bits 63:48) ----
    uint64_t hdr = ((uint64_t)op & 0x7F) << 9;   // bits 63:57: 操作码
    hdr |= ((uint64_t)g & 1) << 8;               // bit 56: 谓词使能
    hdr |= ((uint64_t)gn & 1) << 7;              // bit 55: 谓词取反
    hdr |= ((uint64_t)gp & 0x7) << 4;            // bits 54:52: 谓词号
    hdr |= (uint64_t)m & 0xF;                    // bits 51:48: 修饰符
    uint64_t word = hdr << 48;                    // 头部左移到高位

    // ---- 按格式填充剩余字段 (bits 47:0) ----
    switch(kind) {
    case FK_R:   // R 格式: d[47:40] a[39:32] c[31:24] x[23:16] y[15:8] b[7:0]
        word |= ((uint64_t)d) << 40;
        word |= ((uint64_t)a) << 32;
        word |= ((uint64_t)c) << 24;
        word |= ((uint64_t)x) << 16;
        word |= ((uint64_t)y) << 8;
        word |= (uint64_t)b;
        break;
    case FK_I:   // I 格式: d[47:40] imm32[39:8] x[7:0]
        word |= ((uint64_t)d) << 40;
        word |= ((uint64_t)(uint32_t)imm32) << 8;
        word |= (uint64_t)x;
        break;
    case FK_B: {  // B 格式: imm26[47:22] x[21:16]
        uint32_t u26 = (uint32_t)imm26 & 0x3FFFFFF;  // 取低 26 位
        word |= ((uint64_t)u26) << 22;
        word |= ((uint64_t)(x & 0x3F)) << 16;          // x 取低 6 位
        break;
    }
    case FK_M:   // M 格式: d[47:40] a[39:32] c[31:24] x[23:16] y[15:8] b[7:0]
        word |= ((uint64_t)d) << 40;
        word |= ((uint64_t)a) << 32;
        word |= ((uint64_t)c) << 24;
        word |= ((uint64_t)x) << 16;
        word |= ((uint64_t)y) << 8;
        word |= (uint64_t)b;
        break;
    case FK_MI:  // MI 格式: d[47:40] a[39:32] x[31:24] y[23:16] simm16[15:0]
        word |= ((uint64_t)d) << 40;
        word |= ((uint64_t)a) << 32;
        word |= ((uint64_t)x) << 24;
        word |= ((uint64_t)y) << 16;
        word |= (uint64_t)(uint16_t)simm16;
        break;
    }
    return word;
}

// =============================================================================
// 便捷重载: 从 Inst 结构编码为 64 位指令字
// -----------------------------------------------------------------------------
// 用途: 解码 → 修改 → 重新编码, 或测试中往返校验
// 例: decode(word) → inst; encode_inst(inst) == word (校验解码正确性)
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — encode_inst 从 Inst 结构重新组装 64 位字, 用于 decode/encode 往返一致性校验
inline uint64_t encode_inst(const Inst& ins) {
    return encode(ins.op, ins.g, ins.gn, ins.gp, ins.m,
                  (FmtKind)ins.kind, ins.d, ins.a, ins.c,
                  ins.x, ins.y, ins.b, ins.imm32, ins.imm26, ins.simm16);
}

} // namespace ntisa
