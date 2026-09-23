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
// t_decode.cpp - 指令译码器单元测试 (02 册 §9) ★★ ISA 验证
// -----------------------------------------------------------------------------
// 本文件测试 NTAS1 指令集的编码与译码正确性:
//   1. test_basic_vectors: 基础测试向量 (来自 ntas_enc.hpp 的 test_vectors)
//   2. test_specific_vectors: 特定指令的精确编码验证 (BRA/IMAD/LD/SETP 等)
//   3. test_guard_decoding: 保护谓词 (guard predicate) 译码
//   4. test_error_codes: 错误码检测 (保留操作码/BAR ID 越界)
//   5. test_encoding_helpers: 编码辅助函数 (mk_R/mk_I/mk_B/mk_MI/mk_M)
//
// 测试方法:
//   - 编码 → 译码 → 检查字段 → 重新编码 → 检查往返一致性 (round-trip)
//   - CHECK: 布尔断言
//   - CHECK_EQ: 相等断言 (输出实际值与期望值)
//
// 教学要点:
//   - 指令编码是 ISA 的核心: 二进制格式 → 字段提取
//   - 往返测试 (round-trip): encode(decode(x)) == x, 验证一致性
//   - 保护谓词 (guard): GPU 特有, 支持谓词化执行 (@P0/@!P3)
// =============================================================================
#include "../isa/ntisa.hpp"      // ISA 定义 (操作码/格式/Inst 结构)
#include "../isa/ntas_enc.hpp"   // 编码辅助与测试向量
#include <iostream>
#include <cstring>

using namespace ntisa;

// 全局测试计数器
static int pass_count = 0, fail_count = 0;

// CHECK: 布尔断言宏 (成功 pass++, 失败 fail++ 并输出错误)
#define CHECK(cond, msg) do { \
    if (cond) { pass_count++; } \
    else { fail_count++; std::cerr << "FAIL: " << msg << " at " << __LINE__ << std::endl; } \
} while(0)

// CHECK_EQ: 相等断言宏 (失败时输出实际值与期望值, 16 进制)
#define CHECK_EQ(a, b, msg) do { \
    auto _a = (a); auto _b = (b); \
    if (_a == _b) { pass_count++; } \
    else { fail_count++; std::cerr << "FAIL: " << msg << " got=" << std::hex << (uint64_t)_a << " exp=" << (uint64_t)_b << std::dec << " at " << __LINE__ << std::endl; } \
} while(0)

// =============================================================================
// test_basic_vectors: 基础测试向量 (02 册 §9) ★ 批量验证
// -----------------------------------------------------------------------------
// 从 ntas_enc.hpp 的 test_vectors() 获取所有测试向量, 验证:
//   1. 译码错误码与预期一致 (expect_err)
//   2. 往返一致性: 重新编码后与原字一致 (round-trip)
//
// 测试向量覆盖:
//   - 各格式 (R/I/B/M/MI)
//   - 各指令组 (ALU/内存/控制流/谓词)
//   - 边界情况 (保留操作码/非法字段)
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — test_basic_vectors 验证往返一致性 round-trip: encode(decode(x))==x; §5.4 Methodology (p.131) 断言驱动 CHECK_EQ
void test_basic_vectors() {
    std::cout << "--- test_basic_vectors ---" << std::endl;
    auto vecs = test_vectors();
    CHECK_EQ(vecs.size() >= 16, true, "at least 16 test vectors");  // 至少 16 个测试向量

    for (const auto& v : vecs) {
        Inst ins;
        int err = decode(v.word, v.pc, ins);
        CHECK_EQ(err, v.expect_err, std::string("decode: ") + v.desc);
        if (err == ERR_OK) {
            // 往返测试: 重新编码后应与原字一致
            uint64_t re = encode_inst(ins);
            CHECK_EQ(re, v.word, std::string("round-trip: ") + v.desc);
        }
    }
    std::cout << "  " << vecs.size() << " vectors tested" << std::endl;
}

// =============================================================================
// test_specific_vectors: 特定指令精确验证 (02 册 §9) ★ 逐字段检查
// -----------------------------------------------------------------------------
// 对每条指令进行精确的字段级验证, 确保译码正确:
//   - BRA: 保护谓词/格式/imm26/分支目标
//   - IMAD: 操作码/寄存器/格式
//   - LDG: 操作码/寄存器/格式/内存空间/宽度/缓存提示
//   - SETP: 操作码/寄存器/比较码/类型/谓词目的
//   - NOP/EXIT/SSY/MOV32I/STG/LDS/BAR/ATOM/MEMBAR/S2R/IADD/LOP3
// =============================================================================
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU ISA (p.14) + §2.2.2 Instruction Encoding (p.18) — 逐字段验证 BRA/IMAD/LDG/SETP 等指令编码, RZ 寄存器 (R31 硬连线 0), 特殊寄存器 SR_CTAID/SR_TID
void test_specific_vectors() {
    std::cout << "--- test_specific_vectors ---" << std::endl;

    // ===== @P0 BRA Ldone, PC+0x38, imm26=7, 相对寻址 =====
    // 规范: 0xA100000001C00000
    {
        uint64_t word = 0xA100000001C00000ULL;
        Inst ins;
        int err = decode(word, 0x00, ins);
        CHECK_EQ(err, ERR_OK, "BRA decode");
        CHECK_EQ(ins.op, (uint8_t)OP_BRA, "BRA op");
        CHECK_EQ(ins.g, (uint8_t)1, "BRA guard enable");       // 有保护谓词
        CHECK_EQ(ins.gn, (uint8_t)0, "BRA guard negate");       // 不取反
        CHECK_EQ(ins.gp, (uint8_t)0, "BRA guard pred P0");      // 谓词 P0
        CHECK_EQ(ins.kind, (uint8_t)FK_B, "BRA format B");      // B 格式
        CHECK_EQ(ins.imm26, 7, "BRA imm26=7");                   // 偏移 7
        uint64_t target = branch_target(ins, 0x00);
        CHECK_EQ(target, (uint64_t)0x38, "BRA target PC+0x38");  // 目标 = 0x38
    }

    // ===== IMAD R3, R6, R8 (LO+U32) =====
    // 编码: OP=0x12 放在 bits 63:57 (左移 9 位), header=0x2400
    //       0x2400_0306_0800_0000
    {
        uint64_t word = 0x2400030608000000ULL;
        Inst ins;
        int err = decode(word, 0x00, ins);
        CHECK_EQ(err, ERR_OK, "IMAD decode");
        CHECK_EQ(ins.op, (uint8_t)OP_IMAD, "IMAD op");
        CHECK_EQ(ins.d, (uint8_t)3, "IMAD d=3");   // 目的寄存器 R3
        CHECK_EQ(ins.a, (uint8_t)6, "IMAD a=6");   // 源寄存器 R6
        CHECK_EQ(ins.c, (uint8_t)8, "IMAD c=8");   // 源寄存器 R8
        CHECK_EQ(ins.kind, (uint8_t)FK_R, "IMAD format R");
    }

    // ===== LDG.CA.U32 R2, [R4] (MI 格式, simm=0, X=0x20) =====
    // 编码: OP=0x60 放在 bits 63:57 (左移 9 位), header=0xC000
    //       0xC000_0204_2000_0000
    {
        uint64_t word = 0xC000020420000000ULL;
        Inst ins;
        int err = decode(word, 0x00, ins);
        CHECK_EQ(err, ERR_OK, "LDG decode");
        CHECK_EQ(ins.op, (uint8_t)OP_LD, "LD op");
        CHECK_EQ(ins.d, (uint8_t)2, "LD d=2");
        CHECK_EQ(ins.a, (uint8_t)4, "LD a=4");
        CHECK_EQ(ins.kind, (uint8_t)FK_MI, "LD format MI");
        CHECK_EQ(ins.mem_space(), (uint8_t)AS_GLOBAL, "LD space GLOBAL");   // 全局内存
        CHECK_EQ(ins.mem_width(), (uint8_t)W_U32, "LD width U32");          // 32 位无符号
        CHECK_EQ(ins.mem_hint(), (uint8_t)CH_CA, "LD hint CA");              // 缓存提示 CA
        CHECK_EQ(ins.simm16, (int16_t)0, "LD simm16=0");                    // 偏移 0
    }

    // ===== SETP.GE.S32 P0, R3, R2 =====
    // 规范: 0x4000_0003_0205_8300 (已修正)
    // 构建: OP=0x40 header=0x4000; R 格式: d=0, a=3, c=2, X=0x05, Y=0x83
    {
        uint64_t word = encode(OP_SETP, 0, 0,0,0, FK_R, 0, 3, 2, 0x05, 0x83, 0, 0, 0, 0);
        Inst ins;
        int err = decode(word, 0x00, ins);
        CHECK_EQ(err, ERR_OK, "SETP decode");
        CHECK_EQ(ins.op, (uint8_t)OP_SETP, "SETP op");
        CHECK_EQ(ins.d, (uint8_t)0, "SETP d=0 (pred P0)");   // 目的是谓词 P0
        CHECK_EQ(ins.a, (uint8_t)3, "SETP a=3");
        CHECK_EQ(ins.c, (uint8_t)2, "SETP c=2");
        CHECK_EQ(ins.cmp_code(), (uint8_t)CC_GE, "SETP cmp GE");              // 比较: 大于等于
        CHECK_EQ(ins.cmp_type(), (uint8_t)0, "SETP type S32");                // 类型: 有符号 32 位
        CHECK_EQ(ins.setp_is_pred_dst(), true, "SETP pred dst flag");         // 目的是谓词
        CHECK_EQ(ins.pred_combine(), (uint8_t)3, "SETP combine SET");        // 谓词组合: SET
    }

    // ===== NOP: 全零 =====
    {
        uint64_t word = 0;
        Inst ins;
        int err = decode(word, 0x00, ins);
        CHECK_EQ(err, ERR_OK, "NOP decode");
        CHECK_EQ(ins.op, (uint8_t)OP_NOP, "NOP op");
    }

    // ===== EXIT =====
    {
        uint64_t word = encode(OP_EXIT, 0,0,0,0, FK_R, 0,0,0, 0,0,0, 0,0,0);
        Inst ins;
        decode(word, 0x00, ins);
        CHECK_EQ(ins.op, (uint8_t)OP_EXIT, "EXIT op");
    }

    // ===== SSY 到 PC+0x40 (imm26=8) =====
    {
        uint64_t word = encode(OP_SSY, 0,0,0,0, FK_B, 0,0,0, 0,0,0, 0,8,0);
        Inst ins;
        decode(word, 0x00, ins);
        CHECK_EQ(ins.op, (uint8_t)OP_SSY, "SSY op");
        CHECK_EQ(ins.imm26, 8, "SSY imm26=8");
        uint64_t t = branch_target(ins, 0x00);
        CHECK_EQ(t, (uint64_t)0x40, "SSY target PC+0x40");
    }

    // ===== MOV32I R5, 0x3F800000 (1.0f) =====
    {
        uint64_t word = encode(OP_MOV32I, 0,0,0,0, FK_I, 5,0,0, 0,0,0, (int32_t)0x3F800000, 0, 0);
        Inst ins;
        decode(word, 0x00, ins);
        CHECK_EQ(ins.op, (uint8_t)OP_MOV32I, "MOV32I op");
        CHECK_EQ(ins.d, (uint8_t)5, "MOV32I d=5");
        CHECK_EQ(ins.imm32, (int32_t)0x3F800000, "MOV32I imm32=0x3F800000");
        float f; std::memcpy(&f, &ins.imm32, 4);
        CHECK_EQ(f, 1.0f, "MOV32I 1.0f value");   // 验证浮点值
    }

    // ===== STG.CS.F32 [R4], R0 =====
    {
        uint64_t word = encode(OP_ST, 0,0,0,0, FK_MI, 0, 4, 0, 0x38, 0x02, 0, 0, 0, 0);
        Inst ins;
        decode(word, 0x00, ins);
        CHECK_EQ(ins.op, (uint8_t)OP_ST, "STG op");
        CHECK_EQ(ins.d, (uint8_t)0, "STG d=0 (data)");      // 数据寄存器
        CHECK_EQ(ins.a, (uint8_t)4, "STG a=4 (base)");       // 基址寄存器
        CHECK_EQ(ins.mem_space(), (uint8_t)AS_GLOBAL, "STG space GLOBAL");
        CHECK_EQ(ins.mem_width(), (uint8_t)W_F32, "STG width F32");           // 32 位浮点
        CHECK_EQ(ins.mem_hint(), (uint8_t)CH_CS, "STG hint CS");             // 缓存提示 CS
    }

    // ===== LDS.U32 R5, [R6+R7] (M 格式) =====
    {
        uint64_t word = encode(OP_LD, 0,0,0,0, FK_M, 5, 6, 7, 0xA1, 0x00, 0, 0, 0, 0);
        Inst ins;
        decode(word, 0x00, ins);
        CHECK_EQ(ins.op, (uint8_t)OP_LD, "LDS op");
        CHECK_EQ(ins.kind, (uint8_t)FK_M, "LDS format M");   // M 格式 (双源地址)
        CHECK_EQ(ins.d, (uint8_t)5, "LDS d=5");
        CHECK_EQ(ins.a, (uint8_t)6, "LDS a=6");               // 基址
        CHECK_EQ(ins.c, (uint8_t)7, "LDS c=7");               // 索引
        CHECK_EQ(ins.mem_space(), (uint8_t)AS_SHARED, "LDS space SHARED");  // 共享内存
        CHECK_EQ(ins.mem_width(), (uint8_t)W_U32, "LDS width U32");
        CHECK_EQ(ins.mem_is_mtype(), true, "LDS is M-type");
    }

    // ===== BAR.SYNC 0 =====
    {
        uint64_t word = encode(OP_BAR, 0,0,0,0, FK_R, 0, RZ, 0, 0x00, 0, 0, 0, 0, 0);
        Inst ins;
        decode(word, 0x00, ins);
        CHECK_EQ(ins.op, (uint8_t)OP_BAR, "BAR op");
        CHECK_EQ(ins.x & 0xF, (uint8_t)0, "BAR id=0");          // 屏障 ID = 0
        CHECK_EQ((ins.x >> 4) & 0x3, (uint8_t)0, "BAR SYNC");   // 类型 = SYNC
    }

    // ===== ATOM.GLOBAL.ADD.U32 R2, [R4], R6 =====
    {
        uint64_t word = encode(OP_ATOM, 0,0,0,0, FK_R, 2, 4, 6, 0x20, 0x01, 0, 0, 0, 0);
        Inst ins;
        decode(word, 0x00, ins);
        CHECK_EQ(ins.op, (uint8_t)OP_ATOM, "ATOM op");
        CHECK_EQ(ins.d, (uint8_t)2, "ATOM d=2");
        CHECK_EQ(ins.a, (uint8_t)4, "ATOM a=4");
        CHECK_EQ(ins.c, (uint8_t)6, "ATOM c=6");
        CHECK_EQ(ins.mem_space(), (uint8_t)AS_GLOBAL, "ATOM space GLOBAL");
        CHECK_EQ(ins.mem_width(), (uint8_t)W_U32, "ATOM width U32");
        CHECK_EQ(ins.y & 0xF, (uint8_t)AOP_ADD, "ATOM op ADD");  // 原子操作 = ADD
    }

    // ===== MEMBAR.GL =====
    {
        uint64_t word = encode(OP_MEMBAR, 0,0,0,0, FK_R, 0,0,0, 0x01, 0,0, 0,0,0);
        Inst ins;
        decode(word, 0x00, ins);
        CHECK_EQ(ins.op, (uint8_t)OP_MEMBAR, "MEMBAR op");
        CHECK_EQ(ins.x & 0x3, (uint8_t)1, "MEMBAR GL");   // 范围 = GL (全局)
    }

    // ===== S2R R6, SR_CTAID.X =====
    {
        uint64_t word = encode(OP_S2R, 0,0,0,0, FK_R, 6,0,0, SR_CTAID_X, 0,0, 0,0,0);
        Inst ins;
        decode(word, 0x00, ins);
        CHECK_EQ(ins.op, (uint8_t)OP_S2R, "S2R op");
        CHECK_EQ(ins.d, (uint8_t)6, "S2R d=6");
        CHECK_EQ(ins.x & 0x3F, (uint8_t)SR_CTAID_X, "S2R SR_CTAID.X");  // 特殊寄存器
    }

    // ===== IADD.U64 R8, R8, R10 =====
    {
        uint64_t word = encode(OP_IADD, 0,0,0,0, FK_R, 8, 8, 10, 6, 0,0, 0,0,0);
        Inst ins;
        decode(word, 0x00, ins);
        CHECK_EQ(ins.op, (uint8_t)OP_IADD, "IADD op");
        CHECK_EQ(ins.x & 0x7, (uint8_t)6, "IADD U64");   // 宽度 = U64 (64 位)
    }

    // ===== LOP3 R0, R1, R2, imm8=0xF8 =====
    {
        uint64_t word = encode(OP_LOP3, 0,0,0,0, FK_I, 0, 1, 0, 0,0,0, (int32_t)0xF8, 0, 0);
        Inst ins;
        decode(word, 0x00, ins);
        CHECK_EQ(ins.op, (uint8_t)OP_LOP3, "LOP3 op");
        CHECK_EQ(ins.kind, (uint8_t)FK_I, "LOP3 format I");
        CHECK_EQ(ins.imm32 & 0xFF, 0xF8, "LOP3 imm8=0xF8");   // 查找表立即数
    }
}

// =============================================================================
// test_guard_decoding: 保护谓词译码 (02 册 §3) ★ GPU 特有
// -----------------------------------------------------------------------------
// GPU 指令支持保护谓词 (guard predicate), 实现谓词化执行:
//   - @P0: 当 P0 为真时执行
//   - @!P3: 当 P3 为假时执行 (取反)
//   - @PT: 总是为真 (gp=7, PT=真谓词)
//   - 无保护: 总是执行 (g=0)
//
// 字段说明:
//   - g: 保护使能 (1=有保护, 0=无)
//   - gn: 取反标志 (1=取反, 0=不取反)
//   - gp: 谓词编号 (0-6 普通谓词, 7=PT 真谓词)
// =============================================================================
// 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) — 保护谓词 (guard predicate) 实现谓词化执行 @P0/@!P3/@PT, 对应 exec_mask 三层与中的谓词层
void test_guard_decoding() {
    std::cout << "--- test_guard_decoding ---" << std::endl;

    // @P0 (保护 P0, 不取反)
    {
        uint64_t word = encode(OP_BRA, 1, 0, 0, 0, FK_B, 0,0,0, 0,0,0, 0, 5, 0);
        Inst ins;
        decode(word, 0, ins);
        CHECK_EQ(ins.g, (uint8_t)1, "guard enabled");
        CHECK_EQ(ins.gn, (uint8_t)0, "not negated");
        CHECK_EQ(ins.gp, (uint8_t)0, "P0");
    }

    // @!P3 (保护 !P3, 取反)
    {
        uint64_t word = encode(OP_BRA, 1, 1, 3, 0, FK_B, 0,0,0, 0,0,0, 0, 5, 0);
        Inst ins;
        decode(word, 0, ins);
        CHECK_EQ(ins.g, (uint8_t)1, "guard enabled");
        CHECK_EQ(ins.gn, (uint8_t)1, "negated");
        CHECK_EQ(ins.gp, (uint8_t)3, "P3");
    }

    // 无保护
    {
        uint64_t word = encode(OP_NOP, 0, 0, 0, 0, FK_R, 0,0,0, 0,0,0, 0,0,0);
        Inst ins;
        decode(word, 0, ins);
        CHECK_EQ(ins.g, (uint8_t)0, "no guard");
    }

    // @PT (保护 PT, gp=7)
    {
        uint64_t word = encode(OP_BRA, 1, 0, 7, 0, FK_B, 0,0,0, 0,0,0, 0, 5, 0);
        Inst ins;
        decode(word, 0, ins);
        CHECK_EQ(ins.g, (uint8_t)1, "guard enabled");
        CHECK_EQ(ins.gp, (uint8_t)7, "PT");
        CHECK_EQ(ins.guard_lane(true), true, "PT guard always true");   // PT 总是为真
    }
}

// =============================================================================
// test_error_codes: 错误码检测 (02 册 §9) ★ 异常处理
// -----------------------------------------------------------------------------
// 验证译码器能正确检测非法指令:
//   - ERR_BAD_OPCODE: 保留操作码 (0x0A, 0x70)
//   - ERR_BAR_ID: BAR 屏障 ID 越界 (>= 16)
//
// 教学要点:
//   - ISA 必须定义保留操作码的处理 (报错而非执行)
//   - 屏障 ID 范围检查防止越界访问
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — 保留操作码 (0x0A/0x70) 与 BAR 保留位违规检测; §5.4 Methodology (p.131) 异常路径覆盖测试
void test_error_codes() {
    std::cout << "--- test_error_codes ---" << std::endl;

    // 保留操作码 0x0A
    {
        uint64_t word = encode(0x0A, 0,0,0,0, FK_R, 0,0,0, 0,0,0, 0,0,0);
        Inst ins;
        int err = decode(word, 0, ins);
        CHECK_EQ(err, ERR_BAD_OPCODE, "reserved OP 0x0A");
    }

    // 保留操作码 0x70
    {
        uint64_t word = ((uint64_t)0x70 << 9) << 48;
        Inst ins;
        int err = decode(word, 0, ins);
        CHECK_EQ(err, ERR_BAD_OPCODE, "reserved OP 0x70");
    }

    // BAR 保留位 X[7:6] 非零 (id 字段仅 4 位, 无法表达 >=16, 改测保留位违规)
    {
        uint64_t word = encode(OP_BAR, 0,0,0,0, FK_R, 0,0,0, 0x40, 0,0, 0,0,0);
        Inst ins;
        int err = decode(word, 0, ins);
        CHECK_EQ(err, ERR_BAR_ID, "BAR reserved bits != 0");
    }
}

// =============================================================================
// test_encoding_helpers: 编码辅助函数 (02 册 §9) ★ 构造器验证
// -----------------------------------------------------------------------------
// 验证 ntas_enc.hpp 中的编码辅助函数 (mk_R/mk_I/mk_B/mk_MI/mk_M):
//   - mk_R: R 格式 (ALU 指令)
//   - mk_I: I 格式 (立即数指令)
//   - mk_B: B 格式 (分支指令)
//   - mk_MI: MI 格式 (内存指令, 立即数偏移)
//   - mk_M: M 格式 (内存指令, 寄存器偏移)
// =============================================================================
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — 编码辅助函数构造 R/I/B/M/MI 五种格式指令, 验证字段提取与编码一致性
void test_encoding_helpers() {
    std::cout << "--- test_encoding_helpers ---" << std::endl;

    // mk_R: 构造 R 格式指令
    {
        Inst ins = mk_R(OP_IADD, 1, 2, 3);
        CHECK_EQ(ins.op, (uint8_t)OP_IADD, "mk_R op");
        CHECK_EQ(ins.d, (uint8_t)1, "mk_R d");
        CHECK_EQ(ins.a, (uint8_t)2, "mk_R a");
        CHECK_EQ(ins.c, (uint8_t)3, "mk_R c");
    }

    // mk_I: 构造 I 格式指令 (立即数)
    {
        Inst ins = mk_I(OP_MOV32I, 5, 0, 42);
        CHECK_EQ(ins.op, (uint8_t)OP_MOV32I, "mk_I op");
        CHECK_EQ(ins.d, (uint8_t)5, "mk_I d");
        CHECK_EQ(ins.imm32, 42, "mk_I imm32");
    }

    // mk_B: 构造 B 格式指令 (分支)
    {
        Inst ins = mk_B(OP_BRA, 10);
        CHECK_EQ(ins.op, (uint8_t)OP_BRA, "mk_B op");
        CHECK_EQ(ins.imm26, 10, "mk_B imm26");
    }

    // mk_MI: 构造 MI 格式指令 (内存, 立即数偏移)
    {
        Inst ins = mk_MI(OP_LD, 2, 4, 0, AS_GLOBAL, W_U32, CH_CA);
        CHECK_EQ(ins.op, (uint8_t)OP_LD, "mk_MI op");
        CHECK_EQ(ins.d, (uint8_t)2, "mk_MI d");
        CHECK_EQ(ins.a, (uint8_t)4, "mk_MI a");
        CHECK_EQ(ins.mem_space(), (uint8_t)AS_GLOBAL, "mk_MI space");
        CHECK_EQ(ins.mem_width(), (uint8_t)W_U32, "mk_MI width");
    }

    // mk_M: 构造 M 格式指令 (内存, 寄存器偏移)
    {
        Inst ins = mk_M(OP_LD, 5, 6, 7, AS_SHARED, W_U32);
        CHECK_EQ(ins.op, (uint8_t)OP_LD, "mk_M op");
        CHECK_EQ(ins.kind, (uint8_t)FK_M, "mk_M format M");
        CHECK_EQ(ins.mem_is_mtype(), true, "mk_M is M-type");
    }
}

// =============================================================================
// main: 测试入口
// -----------------------------------------------------------------------------
// 运行所有测试, 输出通过/失败计数, 返回 0=全过 / 1=有失败
// =============================================================================
// 教材引用: 第 5 章 §5.4 Methodology (p.131) — 断言驱动测试方法学, 通过/失败计数; §5.3 Validation (p.129) ISA 译码验证
int main() {
    std::cout << "NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education\n"
              << "Version 1.10\n"
              << "Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN\n"
              << "Improved by: Tonghui Ming\n"
              << "References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018\n"
              << "September 2026\n";
    std::cout << "===== t_decode =====" << std::endl;
    test_basic_vectors();
    test_specific_vectors();
    test_guard_decoding();
    test_error_codes();
    test_encoding_helpers();

    std::cout << "\n===== Summary =====" << std::endl;
    std::cout << "Passed: " << pass_count << ", Failed: " << fail_count << std::endl;
    return fail_count > 0 ? 1 : 0;
}
