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
// ntas_enc.hpp - NTAS1 编码辅助函数 + 测试向量 + 简易汇编器
// =============================================================================
// 本头文件提供 NTAS1 指令集的"构建端"工具:
//   1. mk_R/mk_I/mk_B/mk_M/mk_MI: 按格式构造 Inst 的便捷函数
//   2. test_vectors(): 解码器单元测试用的标准测试向量 (02 册 §9)
//   3. asm_line(): 简易文本汇编器, 将汇编行解析为 Inst (测试用)
//
// 参考: NutShellGPU_spec 02_NTAS1指令集手册.md §9 (测试向量)
//       NutShellGPU_spec 05_模拟器实现契约.md §2.3 (编码器)
//
// 与 ntisa.hpp 的关系:
//   ntisa.hpp 提供 encode()/decode() (字段级编解码)
//   本文件提供 mk_*() (语义级构建, 自动组装 X/Y 字段) 和测试基础设施
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — 本文件整体实现 NTAS1 指令编码辅助函数, 对应教材 R/I/B/M/MI 多种编码格式与字段提取; 测试向量服务于 §5.3 (p.129) 模拟器验证
// =============================================================================
#pragma once
#include "ntisa.hpp"
#include <vector>
#include <tuple>

namespace ntisa {

// =============================================================================
// 编码辅助函数 (Builder Functions, 05 册 §2.3)
// -----------------------------------------------------------------------------
// 这组函数封装了各指令格式的字段组装逻辑, 让测试和内核构建器
// 不必手动拼装 X/Y 修饰字段, 而是用语义化参数 (space/width/hint) 构造指令
//
// 所有 mk_* 函数的共同步骤:
//   1. 填充 Inst 结构的各字段
//   2. 调用 encode_inst() 生成 raw 64 位指令字
//   3. 设置 raw_pc = 0 (调用者可后续覆盖)
// =============================================================================

// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — R 格式构建器, 三源寄存器型指令编码 (ALU/控制流), 通过 encode_inst() 组装 64 位机器码
// ---- R 格式构建器: 寄存器型指令 (ALU/控制流) ----
// 参数: op=操作码, d/a/c=寄存器号, x/y/b=修饰字段, g/gn/gp=谓词, m=修饰符
inline Inst mk_R(uint8_t op, uint8_t d, uint8_t a, uint8_t c,
                  uint8_t x=0, uint8_t y=0, uint8_t b=0,
                  uint8_t g=0, uint8_t gn=0, uint8_t gp=0, uint8_t m=0) {
    Inst ins{};
    ins.op=op; ins.g=g; ins.gn=gn; ins.gp=gp; ins.m=m;
    ins.kind=FK_R; ins.d=d; ins.a=a; ins.c=c; ins.x=x; ins.y=y; ins.b=b;
    ins.imm32=0; ins.imm26=0; ins.simm16=0;
    ins.raw = encode_inst(ins);   // 生成 64 位机器码
    ins.raw_pc = 0;
    return ins;
}

// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — I 格式构建器, 含 32 位立即数的指令编码, 对应教材 64 位 I 格式字段 (MOV32I/LOP3)
// ---- I 格式构建器: 立即数型指令 (MOV32I/LOP3 等) ----
// 参数: op=操作码, d=目的寄存器, a=源寄存器, imm32=32位立即数, x=修饰
inline Inst mk_I(uint8_t op, uint8_t d, uint8_t a, int32_t imm32,
                  uint8_t x=0, uint8_t g=0, uint8_t gn=0, uint8_t gp=0, uint8_t m=0) {
    Inst ins{};
    ins.op=op; ins.g=g; ins.gn=gn; ins.gp=gp; ins.m=m;
    ins.kind=FK_I; ins.d=d; ins.a=a; ins.imm32=imm32; ins.x=x;
    ins.raw = encode_inst(ins);
    ins.raw_pc = 0;
    return ins;
}

// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) & 第 3 章 §3.1.4 Divergence (p.32) — B 格式构建器, 分支型指令 (BRA/CALL/SSY) 编码; SSY/BRA 用于 SIMT 分支发散与重汇聚栈管理
// ---- B 格式构建器: 分支型指令 (BRA/CALL/SSY) ----
// 参数: op=操作码, imm26=26位分支偏移(单位8字节), x=修饰
inline Inst mk_B(uint8_t op, int32_t imm26, uint8_t x=0,
                  uint8_t g=0, uint8_t gn=0, uint8_t gp=0, uint8_t m=0) {
    Inst ins{};
    ins.op=op; ins.g=g; ins.gn=gn; ins.gp=gp; ins.m=m;
    ins.kind=FK_B; ins.imm26=imm26; ins.x=x;
    ins.raw = encode_inst(ins);
    ins.raw_pc = 0;
    return ins;
}

// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) & 第 4 章 §4.1 First-Level Memory Structures (p.67) — M 格式构建器, 寄存器偏移访存指令编码, X 字段组装 space/width 对应教材 global/shared/local 地址空间
// ---- M 格式构建器: 内存型-寄存器偏移 (LD/ST [Ra+Rc]) ----
// 自动组装 X 字段: X[7]=1(M格式标志), X[6:3]=width, X[2:0]=space
// 自动组装 Y 字段: Y[3:0]=hint(缓存提示)
// 参数: space=地址空间(AS_GLOBAL等), width=数据宽度(W_U32等), hint=缓存提示(CH_CA等)
inline Inst mk_M(uint8_t op, uint8_t d, uint8_t a, uint8_t c,
                  uint8_t space, uint8_t width, uint8_t hint=0,
                  uint8_t y2=0, uint8_t g=0, uint8_t gn=0, uint8_t gp=0, uint8_t m=0) {
    Inst ins{};
    ins.op=op; ins.g=g; ins.gn=gn; ins.gp=gp; ins.m=m;
    ins.kind=FK_M; ins.d=d; ins.a=a; ins.c=c;
    // X 字段组装: bit7=1(M格式), bits[6:3]=width, bits[2:0]=space
    ins.x = 0x80 | ((width & 0xF) << 3) | (space & 0x7);
    ins.y = (y2 << 4) | (hint & 0x7);   // Y 字段: 高4位保留, 低3位=hint
    ins.raw = encode_inst(ins);
    ins.raw_pc = 0;
    return ins;
}

// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) & 第 4 章 §4.1.1 Constant Memory (p.68) — MI 格式构建器, 立即数偏移访存指令编码; LDC 也用此格式访问常量内存 (c[0][off])
// ---- MI 格式构建器: 内存型-立即数偏移 (LD/ST [Ra+simm16]) ----
// X 字段: bit7=0(MI格式), bits[6:3]=width, bits[2:0]=space
// 注意: 与 mk_M 的唯一区别是 bit7=0 且偏移用 simm16 而非 Rc 寄存器
inline Inst mk_MI(uint8_t op, uint8_t d, uint8_t a, int16_t simm16,
                   uint8_t space, uint8_t width, uint8_t hint=0,
                   uint8_t y2=0, uint8_t g=0, uint8_t gn=0, uint8_t gp=0, uint8_t m=0) {
    Inst ins{};
    ins.op=op; ins.g=g; ins.gn=gn; ins.gp=gp; ins.m=m;
    ins.kind=FK_MI; ins.d=d; ins.a=a; ins.simm16=simm16;
    // X 字段: bit7=0(MI格式, 无M标志), bits[6:3]=width, bits[2:0]=space
    ins.x = ((width & 0xF) << 3) | (space & 0x7);
    ins.y = (y2 << 4) | (hint & 0x7);
    ins.raw = encode_inst(ins);
    ins.raw_pc = 0;
    return ins;
}

// =============================================================================
// 测试向量 (02 册 §9)
// -----------------------------------------------------------------------------
// 为解码器单元测试提供的标准测试向量集合
// 每个向量包含: 64位指令字 + PC值 + 描述 + 期望错误码
// t_decode.cpp 用这些向量验证 decode() 的正确性 (编码→解码往返一致性)
// 教材引用: 第 5 章 §5.3 Validation (p.129) — TestVec 是模拟器验证基础设施, 用标准指令字验证解码器与编码器的往返一致性 (round-trip)
// =============================================================================
struct TestVec {
    uint64_t word;       // 64 位原始指令字
    uint64_t pc;         // 该指令的 PC (用于分支目标计算)
    const char* desc;    // 人类可读描述
    int expect_err;      // 期望错误码 (0 = OK, 非0 = 期望解码失败)
};

// 教材引用: 第 5 章 §5.3 Validation (p.129) & 第 2 章 §2.2.2 Instruction Encoding (p.18) — 标准测试向量集合覆盖各格式典型指令, 用于验证 decode() 与 encode_inst() 的往返一致性
inline std::vector<TestVec> test_vectors() {
    return {
        // ---- 分支指令 (B 格式) ----
        // 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — @P0 BRA 是谓词保护分支, 用于 SIMT 栈管理发散分支的 taken/fall-through 路径
        // @P0 BRA Ldone, PC+0x38, imm26=7, 相对分支
        // 头部 0xA100 (OP=0x50, g=1, gp=0), 完整 0xA100000001C00000
        {0xA100000001C00000ULL, 0x00, "@P0 BRA PC+0x38", 0},
        // ---- 整数乘加 (R 格式) ----
        // IMAD R3, R6, R8 (LO+U32): d=3,a=6,c=8; OP=0x12<<9=0x2400
        {0x2400030608000000ULL, 0x00, "IMAD R3,R6,R8", 0},
        // ---- 全局内存加载 (MI 格式) ----
        // LDG.CA.U32 R2, [R4] (MI, simm=0; X=0x20: space=0,width=4); OP=0x60<<9=0xC000
        {0xC000020420000000ULL, 0x00, "LDG.CA.U32 R2,[R4]", 0},
        // ---- 谓词比较设置 (R 格式, 特殊Y字段) ----
        // 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) & 第 2 章 §2.2.1 NVIDIA GPU ISA (p.14) — SETP 设置谓词寄存器, 用于 SIMT 谓词保护 (@P0/@!P3) 控制分支执行
        // SETP.GE.S32 P0, R3, R2 (GE=5, S32=0, Y=0x83: SET+pred dst flag)
        // SETP.GE.S32 P0,R3,R2: OP=0x40 首字节=0x80, d=0,a=3,c=2,X=0x05,Y=0x83
        {0x8000030302058300ULL, 0x00, "SETP.GE.S32 P0,R3,R2", 0},
        // ---- 空操作 (R 格式, 全零) ----
        {0x0000000000000000ULL, 0x00, "NOP", 0},
        // ---- 退出指令 (R 格式) ----
        {encode(OP_EXIT,0,0,0,0,FK_R,0,0,0,0,0,0,0,0,0), 0x00, "EXIT", 0},
        // ---- 设置重汇聚点 (B 格式) ----
        // 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — SSY 设置重汇聚点压入 SIMT 栈, 是分支前必须指令 (教材图 3.3 重汇聚栈结构)
        // SSY PC+0x40, imm26=8
        {encode(OP_SSY,0,0,0,0,FK_B,0,0,0,0,0,0,0,8,0), 0x00, "SSY PC+0x40", 0},
        // ---- 立即数搬运 (I 格式) ----
        // MOV32I R5, 0x3F800000 (IEEE 754 单精度 1.0f)
        {encode(OP_MOV32I,0,0,0,0,FK_I,5,0,0,0,0,0,(int32_t)0x3F800000,0,0), 0x00, "MOV32I R5,1.0f", 0},
        // ---- 全局内存存储 (MI 格式) ----
        // STG.CS.F32 [R4], R0 (space=0, width=7 F32, hint=2 CS)
        // X = (7<<3)|0 = 0x38, Y = (0<<4)|2 = 0x02, simm16=0
        {encode(OP_ST,0,0,0,0,FK_MI,0,4,0,0x38,0x02,0,0,0,0), 0x00, "STG.CS.F32 [R4],R0", 0},
        // ---- 共享内存加载 (M 格式, 寄存器偏移) ----
        // LDS.U32 R5, [R6+R7] (space=1 SHARED, width=4 U32, hint=0 CA)
        // X = 0x80 | (4<<3) | 1 = 0xA1 (bit7=1 表示 M 格式)
        {encode(OP_LD,0,0,0,0,FK_M,5,6,7,0xA1,0x00,0,0,0,0), 0x00, "LDS.U32 R5,[R6+R7]", 0},
        // ---- 屏障同步 (R 格式) ----
        // BAR.SYNC 0 (X=0x00: SYNC模式+id0, a=RZ=255)
        {encode(OP_BAR,0,0,0,0,FK_R,0,RZ,0,0x00,0,0,0,0,0), 0x00, "BAR.SYNC 0", 0},
        // ---- 原子操作 (R 格式) ----
        // 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — ATOM 原子操作在 L2/DRAM 分区单元执行读-改-写, 用于并发线程同步 (L2 ROP 原子缓存)
        // ATOM.GLOBAL.ADD.U32 R2, [R4], R6 (X: space=0,width=4; Y: op=1 ADD)
        // X = (4<<3)|0 = 0x20, Y = (0<<4)|1 = 0x01
        {encode(OP_ATOM,0,0,0,0,FK_R,2,4,6,0x20,0x01,0,0,0,0), 0x00, "ATOM.GLOBAL.ADD R2,[R4],R6", 0},
        // ---- 内存屏障 (R 格式) ----
        // MEMBAR.GL (X[1:0]=1 表示 GL 级别)
        {encode(OP_MEMBAR,0,0,0,0,FK_R,0,0,0,0x01,0,0,0,0,0), 0x00, "MEMBAR.GL", 0},
        // ---- 特殊寄存器读取 (R 格式) ----
        // 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — S2R 读取特殊寄存器 (SR_TID/SR_CTAID 等), 对应教材 SASS 中的特殊寄存器访问机制
        // S2R R6, SR_CTAID.X (X=8: CTAID.X 的 SR 编号)
        {encode(OP_S2R,0,0,0,0,FK_R,6,0,0,SR_CTAID_X,0,0,0,0,0), 0x00, "S2R R6,SR_CTAID.X", 0},
        // ---- 浮点乘加 (R 格式) ----
        // 教材引用: 第 2 章 §2.2.3 Fused Multiply-Add (p.20) — FFMA 单次舍入融合乘加, 第三源通过 Rd 字段隐式传递 (教材 §2.2.3 FFMA 编码说明)
        // FFMA.RN.F32 R0, R0, R1, R2 (M[1:0]=0: 就近舍入 RN)
        {encode(OP_FFMA,0,0,0,0,FK_R,0,0,1,0,0,0,0,0,0), 0x00, "FFMA R0,R0,R1,R2", 0},
        // ---- 64位整数加 (R 格式) ----
        // IADD.U64 R8, R8, R10 (X[2:0]=6 U64, 需寄存器对)
        {encode(OP_IADD,0,0,0,0,FK_R,8,8,10,6,0,0,0,0,0), 0x00, "IADD.U64 R8,R8,R10", 0},
        // ---- 三输入查找表 (I 格式) ----
        // LOP3 R0, R1, R2, imm8=0xF8 (imm32=0xF8: LUT 真值表, 0xF8=AND)
        {encode(OP_LOP3,0,0,0,0,FK_I,0,1,0,0,0,0,(int32_t)0xF8,0,0), 0x00, "LOP3 R0,R1,R2,0xF8", 0},
    };
}

// =============================================================================
// 简易文本汇编器 (测试辅助工具)
// -----------------------------------------------------------------------------
// 将一行汇编文本解析为 Inst 结构, 供测试使用
// 支持的语法子集:
//   [@Pn[!]] MNEM[.SUFFIX...] operands
//   例: @P0 BRA 0xB0
//       LDG.CA.U32 R2, [R4+0]
//       SETP.GE.S32 P0, R3, R2
//
// 支持的后缀: .U32/.S32/.U64/.F32 (宽度), .CA/.CG/.CS/.CV (缓存提示),
//             .LO/.HI/.WIDE (乘法模式), .GE/.LT/.EQ (比较码) 等
//
// 返回 AsmResult: {inst, err}; err 为空字符串表示成功
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — AsmResult 是简易汇编器的返回值, 解析一行文本得到 Inst 结构 (字段级编码), 配合测试向量使用
struct AsmResult {
    Inst inst;          // 解析结果
    std::string err;    // 错误信息 (空=成功)
};

// =============================================================================
// asm_line - 简易汇编器主函数
// -----------------------------------------------------------------------------
// 解析流程:
//   1. 预处理: 去除注释 (//...), 去除首尾空白
//   2. 解析谓词守卫: @Pn / @!Pn / @PT
//   3. 提取助记符并按 '.' 分割为基础助记符 + 后缀列表
//   4. 根据基础助记符分派到对应的解析逻辑
//   5. 后缀用于确定宽度/空间/缓存提示/比较码等修饰字段
//
// 这是一个教学用最小汇编器, 仅支持测试中使用的指令子集
// 生产级汇编器需要更完整的语法分析和符号表管理
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) & 第 2 章 §2.2.1 NVIDIA GPU ISA (p.14) — asm_line() 简易文本汇编器, 将汇编行解析为 Inst 字段编码, 涵盖 RZ/PT 特殊寄存器、谓词守卫、各种修饰后缀解析
// =============================================================================
inline AsmResult asm_line(const std::string& line, uint64_t pc) {
    AsmResult r{};

    // ---- 步骤 1: 预处理 (去注释, 去空白) ----
    std::string s = line;
    auto cpos = s.find("//");                    // 查找注释起点
    if (cpos != std::string::npos) s = s.substr(0, cpos); // 截断注释
    size_t b = s.find_first_not_of(" \t");        // 跳过前导空白
    if (b == std::string::npos) { r.err = "empty"; return r; }
    s = s.substr(b);
    size_t e = s.find_last_not_of(" \t;");        // 去尾部空白和分号
    if (e == std::string::npos) { r.err = "empty"; return r; }
    s = s.substr(0, e+1);
    if (s.empty()) { r.err = "empty"; return r; }

    // 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) — @Pn / @!Pn 谓词守卫解析, 对应 SIMT 谓词保护执行掩码机制 (@P0/@!P3 教材示例)
    // ---- 步骤 2: 解析谓词守卫 (@Pn / @!Pn / @PT) ----
    uint8_t g=0, gn=0, gp=0;
    if (s[0] == '@') {
        g = 1;                                    // 启用谓词守卫
        size_t sp = s.find(' ');
        if (sp == std::string::npos) { r.err = "bad guard"; return r; }
        std::string gs = s.substr(1, sp-1);       // 提取谓词名
        s = s.substr(sp+1);                       // 剩余部分
        if (!gs.empty() && gs[0]=='!') { gn=1; gs=gs.substr(1); } // 取反标志
        if (gs == "PT") gp=7;                     // PT = 恒真谓词
        else if (gs.size()>=2 && gs[0]=='P') gp = (uint8_t)std::stoul(gs.substr(1)); // P0-P6
        else { r.err = "bad guard pred"; return r; }
    }

    // ---- 步骤 3: 提取助记符和操作数部分 ----
    size_t ms = s.find(' ');
    std::string mnem = (ms==std::string::npos) ? s : s.substr(0, ms);
    std::string rest = (ms==std::string::npos) ? "" : s.substr(ms+1);
    for (auto& ch : mnem) if (ch>='a'&&ch<='z') ch -= 32; // 转大写

    // ---- 步骤 3b: 按 '.' 分割助记符为 [base, suffix1, suffix2, ...] ----
    // 例: "LDG.CA.U32" → ["LDG", "CA", "U32"]
    std::vector<std::string> parts;
    {
        std::string cur;
        for (char ch : mnem) {
            if (ch == '.') { parts.push_back(cur); cur.clear(); }
            else cur += ch;
        }
        parts.push_back(cur);
    }
    std::string base = parts[0];                 // 基础助记符 (如 LDG)

    // ---- 辅助函数: 将寄存器/谓词名解析为编号 ----
    // RZ → 255 (零寄存器), PT → 7 (恒真谓词)
    // R0-R254 → 0-254, P0-P6 → 0-6
    auto reg = [](const std::string& tok) -> int {
        if (tok == "RZ") return RZ;
        if (tok == "PT") return 7;
        if (tok.size()>=2 && tok[0]=='R') return std::stoul(tok.substr(1));
        if (tok.size()>=2 && tok[0]=='P') return std::stoul(tok.substr(1));
        return -1;
    };

    // =============================================================================
    // 步骤 4: 按基础助记符分派解析逻辑
    // -----------------------------------------------------------------------------
    // 每个分支负责解析一种或一组指令, 提取操作数和修饰后缀,
    // 然后调用 mk_R/mk_I/mk_B/mk_M/mk_MI 构造 Inst
    // =============================================================================

    // ---- NOP: 空操作, 无操作数 ----
    if (base == "NOP") {
        r.inst = mk_R(OP_NOP, 0, 0, 0, 0, 0, 0, g, gn, gp);
    } else if (base == "MOV") {
        // MOV.U32 Rd, Ra: 寄存器搬运, 2 个操作数
        std::vector<std::string> ops;
        std::string cur;
        for (char ch : rest) { if (ch==',') { ops.push_back(cur); cur.clear(); } else if (ch!=' ') cur+=ch; }
        if (!cur.empty()) ops.push_back(cur);
        if (ops.size()<2) { r.err="MOV needs 2 ops"; return r; }
        r.inst = mk_R(OP_MOV, (uint8_t)reg(ops[0]), (uint8_t)reg(ops[1]), 0, 0, 0, 0, g, gn, gp);
    } else if (base == "MOV32I") {
        // MOV32I Rd, imm: 立即数搬运, I 格式
        std::vector<std::string> ops;
        std::string cur;
        for (char ch : rest) { if (ch==',') { ops.push_back(cur); cur.clear(); } else if (ch!=' ') cur+=ch; }
        if (!cur.empty()) ops.push_back(cur);
        if (ops.size()<2) { r.err="MOV32I needs 2 ops"; return r; }
        int32_t imm = (int32_t)std::stoul(ops[1], nullptr, 0);
        r.inst = mk_I(OP_MOV32I, (uint8_t)reg(ops[0]), 0, imm, 0, g, gn, gp);
    } else if (base == "S2R") {
        // 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — S2R 读取特殊寄存器 (SR_TID/SR_CTAID 等), 对应教材 SASS 中的特殊寄存器访问机制
        // S2R Rd, SR_XXX: 特殊寄存器读取
        // 第二操作数为特殊寄存器名, 需查表映射到 SR 编号
        std::vector<std::string> ops;
        std::string cur;
        for (char ch : rest) { if (ch==',') { ops.push_back(cur); cur.clear(); } else if (ch!=' ') cur+=ch; }
        if (!cur.empty()) ops.push_back(cur);
        if (ops.size()<2) { r.err="S2R needs 2 ops"; return r; }
        uint8_t sr = 0;
        // 特殊寄存器名 → 编号 映射表
        if (ops[1]=="SR_TID.X") sr=SR_TID_X; else if (ops[1]=="SR_TID.Y") sr=SR_TID_Y;
        else if (ops[1]=="SR_TID.Z") sr=SR_TID_Z; else if (ops[1]=="SR_CTAID.X") sr=SR_CTAID_X;
        else if (ops[1]=="SR_CTAID.Y") sr=SR_CTAID_Y; else if (ops[1]=="SR_CTAID.Z") sr=SR_CTAID_Z;
        else if (ops[1]=="SR_NTID.X") sr=SR_NTID_X; else if (ops[1]=="SR_NTID.Y") sr=SR_NTID_Y;
        else if (ops[1]=="SR_NTID.Z") sr=SR_NTID_Z; else if (ops[1]=="SR_NCTAID.X") sr=SR_NCTAID_X;
        else if (ops[1]=="SR_NCTAID.Y") sr=SR_NCTAID_Y; else if (ops[1]=="SR_NCTAID.Z") sr=SR_NCTAID_Z;
        else if (ops[1]=="SR_LANEID") sr=SR_LANEID; else if (ops[1]=="SR_WARPID") sr=SR_WARPID;
        else if (ops[1]=="SR_SMID") sr=SR_SMID; else if (ops[1]=="SR_CLOCKLO") sr=SR_CLOCKLO;
        else if (ops[1]=="SR_PARAM_BASE") sr=SR_PARAM_BASE; else if (ops[1]=="SR_SMEM_BASE") sr=SR_SMEM_BASE;
        else if (ops[1]=="SR_LMEM_BASE") sr=SR_LMEM_BASE;
        else { r.err="unknown SR"; return r; }
        r.inst = mk_R(OP_S2R, (uint8_t)reg(ops[0]), 0, 0, sr, 0, 0, g, gn, gp);
    } else if (base == "IADD" || base=="ISUB" || base=="IMAD" || base=="IMUL" ||
               base=="IABS" || base=="INEG" || base=="POPC" || base=="BREV" ||
               base=="SHL" || base=="SHR" || base=="SAR" || base=="LOP" ||
               base=="BMSK" || base=="FADD" || base=="FSUB" || base=="FMUL" ||
               base=="FFMA" || base=="FABS" || base=="FNEG" || base=="FMNMX" ||
               base=="FSET" || base=="FRND" || base=="SETP" || base=="SELP" ||
               base=="IMNMX") {
        // 教材引用: 第 2 章 §2.2.1 NVIDIA GPU ISA (p.14) & 第 2 章 §2.2.3 Fused Multiply-Add (p.20) — R 格式 3 操作数 ALU/浮点指令 (含 FFMA 单次舍入融合乘加、SETP 谓词设置、FMNMX 极值选择)
        // ---- R 格式 3 操作数指令 (整数/浮点 ALU + 谓词) ----
        std::vector<std::string> ops;
        std::string cur;
        for (char ch : rest) { if (ch==',') { ops.push_back(cur); cur.clear(); } else if (ch!=' ') cur+=ch; }
        if (!cur.empty()) ops.push_back(cur);
        // 助记符 → 操作码 映射
        uint8_t op=0;
        if (base=="IADD") op=OP_IADD; else if (base=="ISUB") op=OP_ISUB;
        else if (base=="IMAD") op=OP_IMAD; else if (base=="IMUL") op=OP_IMUL;
        else if (base=="IABS") op=OP_IABS; else if (base=="INEG") op=OP_INEG;
        else if (base=="POPC") op=OP_POPC; else if (base=="BREV") op=OP_BREV;
        else if (base=="SHL") op=OP_SHL; else if (base=="SHR") op=OP_SHR;
        else if (base=="SAR") op=OP_SAR; else if (base=="LOP") op=OP_LOP;
        else if (base=="BMSK") op=OP_BMSK; else if (base=="FADD") op=OP_FADD;
        else if (base=="FSUB") op=OP_FSUB; else if (base=="FMUL") op=OP_FMUL;
        else if (base=="FFMA") op=OP_FFMA; else if (base=="FABS") op=OP_FABS;
        else if (base=="FNEG") op=OP_FNEG; else if (base=="FMNMX") op=OP_FMNMX;
        else if (base=="FSET") op=OP_FSET; else if (base=="FRND") op=OP_FRND;
        else if (base=="SETP") op=OP_SETP; else if (base=="SELP") op=OP_SELP;
        else if (base=="IMNMX") op=OP_IMNMX;
        // 从后缀解析 X/Y 修饰字段
        uint8_t xval=0, yval=0;
        for (size_t i=1;i<parts.size();i++) {
            const std::string& sfx = parts[i];
            // 宽度后缀
            if (sfx=="U32" || sfx=="S32") xval = (sfx[0]=='U') ? 0 : 1; // 简化: U32=0, S32=1
            else if (sfx=="U64") xval = 6;
            // 乘法模式后缀 (IMAD/IMUL)
            else if (sfx=="LO") xval |= 0;                    // LO: 取低 32 位
            else if (sfx=="HI") xval |= (1<<3);               // HI: 取高 32 位
            else if (sfx=="WIDE") xval |= (2<<3);              // WIDE: 64 位结果
            // 比较码后缀 (SETP/FSET)
            else if (sfx=="GE") xval = 5;                      // 大于等于
            else if (sfx=="LT") xval = 2;                      // 小于
            else if (sfx=="EQ") xval = 0;                      // 等于
            else if (sfx=="NE") xval = 1;                      // 不等于
            else if (sfx=="GT") xval = 4;                      // 大于
            else if (sfx=="LE") xval = 3;                      // 小于等于
            // 谓词组合方式 (PLOP)
            else if (sfx=="AND") yval = 0;
            else if (sfx=="OR") yval = 1;
            else if (sfx=="XOR") yval = 2;
            else if (sfx=="SET") yval = 3;                    // 直接赋值
            // 最小/最大选择 (IMNMX/FMNMX)
            else if (sfx=="MAX") xval |= 1;
            else if (sfx=="MIN") xval &= ~1;
        }
        if (ops.size()<2) { r.err="needs operands"; return r; }
        uint8_t dd = (uint8_t)reg(ops[0]);
        uint8_t aa = ops.size()>1 ? (uint8_t)reg(ops[1]) : (uint8_t)0;
        uint8_t cc = ops.size()>2 ? (uint8_t)reg(ops[2]) : (uint8_t)0;
        // SETP 特殊: Y=0x83 表示 SET 模式 + 谓词目的标志
        if (base=="SETP") yval = 0x83;
        r.inst = mk_R(op, dd, aa, cc, xval, yval, 0, g, gn, gp);
    } else if (base == "BRA" || base == "SSY" || base == "CALL") {
        // 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — B 格式分支指令, SSY 压栈设置重汇聚点, BRA 触发散路压入 taken/fall-through 子栈帧
        // ---- B 格式分支指令: 操作数为目标地址 (立即数) ----
        std::string tok;
        for (char ch : rest) if (ch!=' ' && ch!=';') tok += ch;
        int32_t imm = (int32_t)std::stoul(tok, nullptr, 0);
        uint8_t op = (base=="BRA") ? OP_BRA : (base=="SSY") ? OP_SSY : OP_CALL;
        r.inst = mk_B(op, imm, 0, g, gn, gp);
    } else if (base == "EXIT") {
        // EXIT: 退出指令, 无操作数
        r.inst = mk_R(OP_EXIT, 0, 0, 0, 0, 0, 0, g, gn, gp);
    } else if (base == "RET") {
        // RET: 函数返回, 无操作数
        r.inst = mk_R(OP_RET, 0, 0, 0, 0, 0, 0, g, gn, gp);
    } else if (base == "MEMBAR") {
        // MEMBAR.CTA/GL/SYS: 内存屏障, 后缀决定级别
        uint8_t xv=0;
        for (size_t i=1;i<parts.size();i++) {
            if (parts[i]=="CTA") xv=0; else if (parts[i]=="GL") xv=1; else if (parts[i]=="SYS") xv=2;
        }
        r.inst = mk_R(OP_MEMBAR, 0, 0, 0, xv, 0, 0, g, gn, gp);
    } else if (base == "BAR") {
        // 教材引用: 第 3 章 §3.1 One-Loop Approximation (p.22) — BAR.SYNC 实现 CTA 内 warp 间屏障同步, 功能模拟器单循环调度模型必须建模屏障等待
        // BAR.SYNC/ARRIVE/WAIT id: 屏障同步
        uint8_t xv=0; // SYNC=0
        for (size_t i=1;i<parts.size();i++) {
            if (parts[i]=="SYNC") xv = 0x00;        // 全体到达并等待
            else if (parts[i]=="ARRIVE") xv = 0x10; // 仅到达 (不等待)
            else if (parts[i]=="WAIT") xv = 0x20;  // 仅等待 (不到达)
        }
        std::string tok;
        for (char ch : rest) if (ch!=' '&&ch!=';') tok+=ch;
        uint8_t bid = tok.empty() ? 0 : (uint8_t)std::stoul(tok);
        xv |= (bid & 0xF);                          // 屏障 ID (0-15)
        r.inst = mk_R(OP_BAR, 0, RZ, 0, xv, 0, 0, g, gn, gp);
    } else if (base == "LDG" || base == "STG" || base == "LDL" || base == "STL" ||
               base == "LDS" || base == "STS" || base == "LDC") {
        // 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) & 第 4 章 §4.1.1 Constant Memory (p.68) — 内存指令按地址空间分派 (G/L/S/C), LDC 通过常量缓存读取 c[0][off]; M/MI 格式按偏移类型自动选择
        // ---- 内存指令: 自动选择 M 或 MI 格式 ----
        // 助记符前缀决定地址空间: G=Global, L=Local, S=Shared, C=Const
        uint8_t space=0, width=4, hint=0;
        if (base=="LDG"||base=="STG") space=AS_GLOBAL;
        else if (base=="LDL"||base=="STL") space=AS_LOCAL;
        else if (base=="LDS"||base=="STS") space=AS_SHARED;
        else if (base=="LDC") space=AS_CONST;
        // 后缀解析: 宽度 + 缓存提示
        for (size_t i=1;i<parts.size();i++) {
            const std::string& sfx = parts[i];
            if (sfx=="U8") width=W_U8; else if (sfx=="S8") width=W_S8;
            else if (sfx=="U16") width=W_U16; else if (sfx=="S16") width=W_S16;
            else if (sfx=="U32") width=W_U32; else if (sfx=="S32") width=W_S32;
            else if (sfx=="U64") width=W_U64; else if (sfx=="F32") width=W_F32;
            else if (sfx=="F64") width=W_F64; else if (sfx=="F16") width=W_F16;
            else if (sfx=="U128") width=W_U128;
            else if (sfx=="CA") hint=CH_CA; else if (sfx=="CG") hint=CH_CG;
            else if (sfx=="CS") hint=CH_CS; else if (sfx=="CV") hint=CH_CV;
            else if (sfx=="LU") hint=CH_LU;
        }
        uint8_t op = (base[0]=='L') ? OP_LD : OP_ST;
        if (base=="LDC") op = OP_LDC;
        // 解析操作数: Rd, [Ra+imm] 或 Rd, [Ra+Rc] 或 Rd, [Ra]
        std::vector<std::string> ops;
        std::string cur;
        for (char ch : rest) {
            if (ch==',') { ops.push_back(cur); cur.clear(); }
            else if (ch!=' ') cur+=ch;
        }
        if (!cur.empty()) ops.push_back(cur);
        if (ops.size()<2) { r.err="needs 2 ops"; return r; }
        uint8_t dd = (uint8_t)reg(ops[0]);
        // 解析地址表达式 [Ra+imm] 或 [Ra+Rc]
        std::string addr = ops[1];
        if (addr[0]=='[') addr = addr.substr(1);
        if (!addr.empty() && addr.back()==']') addr.pop_back();
        // 按 '+' 分割地址表达式
        std::vector<std::string> aparts;
        std::string ac;
        for (char ch : addr) {
            if (ch=='+') { aparts.push_back(ac); ac.clear(); }
            else ac+=ch;
        }
        if (!ac.empty()) aparts.push_back(ac);
        uint8_t aa=0, cc=RZ;
        int16_t simm=0;
        bool is_m=false;   // 是否为 M 格式 (有寄存器偏移)
        if (!aparts.empty()) aa = (uint8_t)reg(aparts[0]);  // 基址寄存器
        for (size_t i=1;i<aparts.size();i++) {
            if (aparts[i][0]=='R' || aparts[i]=="RZ") {
                cc = (uint8_t)reg(aparts[i]);               // 偏移寄存器 → M 格式
                is_m = true;
            } else {
                simm = (int16_t)std::stol(aparts[i], nullptr, 0); // 立即数偏移 → MI 格式
            }
        }
        if (base=="LDC") {
            // LDC.Param.U32 Rd, c[0][off]: 常量加载, MI 格式
            r.inst = mk_MI(OP_LDC, dd, 0, simm, AS_CONST, width, hint);
            // LDC 特殊 X 字段: X[7:5]=宽度编码 (0=U32, 1=U64, 2=F32), X[3:0]=bank
            r.inst.x = ((width==W_U64?1:(width==W_F32?2:0))<<5) | 0; // bank 0 = param
        } else if (is_m) {
            // M 格式: [Ra+Rc] 寄存器偏移
            r.inst = mk_M(op, dd, aa, cc, space, width, hint);
        } else {
            // MI 格式: [Ra+simm16] 立即数偏移
            r.inst = mk_MI(op, dd, aa, simm, space, width, hint);
        }
    } else if (base == "ATOM" || base == "RED") {
        // 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — ATOM/RED 原子操作在内存分区单元执行, RED 不返回值; 支持 ADD/MIN/MAX/CAS/EXCH 等原子算子
        // ---- 原子操作: ATOM.GLOBAL.ADD.U32 Rd, [Ra], Rc ----
        uint8_t space=0, width=4, aop=1;
        for (size_t i=1;i<parts.size();i++) {
            const std::string& sfx = parts[i];
            if (sfx=="GLOBAL") space=AS_GLOBAL; else if (sfx=="SHARED") space=AS_SHARED;
            else if (sfx=="U32") width=W_U32; else if (sfx=="U64") width=W_U64;
            // 原子操作类型 (Y[3:0])
            else if (sfx=="ADD") aop=AOP_ADD; else if (sfx=="EXCH") aop=AOP_EXCH;
            else if (sfx=="MIN") aop=AOP_MIN; else if (sfx=="MAX") aop=AOP_MAX;
            else if (sfx=="CAS") aop=AOP_CAS; else if (sfx=="AND") aop=AOP_AND;
            else if (sfx=="OR") aop=AOP_OR; else if (sfx=="XOR") aop=AOP_XOR;
            else if (sfx=="INC") aop=AOP_INC; else if (sfx=="DEC") aop=AOP_DEC;
        }
        uint8_t op = (base=="ATOM") ? OP_ATOM : OP_RED;
        std::vector<std::string> ops;
        std::string cur;
        for (char ch : rest) { if (ch==',') { ops.push_back(cur); cur.clear(); } else if (ch!=' ') cur+=ch; }
        if (!cur.empty()) ops.push_back(cur);
        if (ops.size()<3) { r.err="needs 3 ops"; return r; }
        uint8_t dd = (uint8_t)reg(ops[0]);
        std::string addr = ops[1];
        if (addr[0]=='[') addr = addr.substr(1);
        if (!addr.empty() && addr.back()==']') addr.pop_back();
        uint8_t aa = (uint8_t)reg(addr);
        uint8_t cc = (uint8_t)reg(ops[2]);
        uint8_t xv = ((width&0xF)<<3) | (space&0x7);
        uint8_t yv = aop & 0xF;
        r.inst = mk_R(op, dd, aa, cc, xv, yv, 0, g, gn, gp);
    } else if (base == "YIELD") {
        // YIELD: 让步指令, 无操作数
        r.inst = mk_R(OP_YIELD, 0, 0, 0, 0, 0, 0, g, gn, gp);
    } else if (base == "SHFL" || base == "VOTE" || base == "BRX") {
        // 教材引用: 第 2 章 §2.1.2 Threading Model (p.12) — SHFL/VOTE/BRX 是 warp 级原语, 在 32 线程 SIMT 调度单元内进行跨 lane 通信; BRX 间接分支可触发多路发散
        // SHFL/VOTE/BRX: warp 级原语, 简化为 R 格式
        std::vector<std::string> ops;
        std::string cur;
        for (char ch : rest) { if (ch==',') { ops.push_back(cur); cur.clear(); } else if (ch!=' ') cur+=ch; }
        if (!cur.empty()) ops.push_back(cur);
        uint8_t op = (base=="SHFL")?OP_SHFL : (base=="VOTE")?OP_VOTE : OP_BRX;
        uint8_t dd = ops.size()>0 ? (uint8_t)reg(ops[0]) : 0;
        uint8_t aa = ops.size()>1 ? (uint8_t)reg(ops[1]) : 0;
        uint8_t cc = ops.size()>2 ? (uint8_t)reg(ops[2]) : 0;
        r.inst = mk_R(op, dd, aa, cc, 0, 0, 0, g, gn, gp);
    } else {
        // 未知助记符
        r.err = "unknown mnemonic: " + base;
        return r;
    }
    r.inst.raw_pc = pc;   // 设置指令的 PC (用于分支目标计算)
    return r;
}

} // namespace ntisa
