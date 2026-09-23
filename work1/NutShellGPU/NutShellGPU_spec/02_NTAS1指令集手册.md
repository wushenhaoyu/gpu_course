> **1.1 阅读说明：** 本册列出原架构参数与接口。周期模拟器的已实现行为、模拟延时和简化边界以 [06_1.1时序与修复说明](06_1.1时序与修复说明.md) 为准。

NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education  
Version 1.10  
Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN  
Improved by: Tonghui Ming  
References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018  
September 2026

# NTAS1 指令集架构手册（第 2 册）

> NTAS1 是 NutShellGPU 的**硬件级 ISA**（对标 NVIDIA SASS 的角色）：定长 64 位、RISC、load/store、谓词执行、寄存器号显式。本手册给出程序员可见状态、二进制编码、全部指令的汇编形式与逐 lane 语义、执行部件/延迟表、文本汇编语法。功能模拟器必须逐条实现本册全部语义。

---

## 1. 程序员可见状态

| 状态 | 数量/宽度 | 说明 |
|---|---|---|
| 通用寄存器 R | R0..R254（每线程 255 个 32 位）+ RZ（R255，恒 0，写忽略） | 64 位值用**偶奇寄存器对**：偶数号存低 32 位，紧接奇数号存高 32 位 |
| 谓词寄存器 P | P0..P6（每线程 7 个 1 位）+ PT（恒 1） | SETP 写，守卫读 |
| active mask | 32 位/warp | 当前执行 lane；SIMT 栈管理，软件不可直接写（VOTE/BALLOT 可读信息） |
| PC | 每 warp（逻辑上每 lane 一个 rPC，基线由栈派生） | 字节地址，永远 8 字节对齐 |
| SIMT 栈 | 每 warp 最多 32 项 | `{RPC, NextPC, mask}`，见 03 册 §4 |
| 特殊寄存器 | 见 01 册 §6 | 只读，S2R 访问 |
| 条件码 CC | 无 | 本 ISA **不设**条件码/标志位；64 位运算由寄存器对原生支持（不设 .CC/.X） |
| LR | SR_LR（CALL 链接寄存器，每 lane） | S2R 可读 |

每线程独立拥有一份 R/P；warp 指令对 32 个 lane 锁步执行同一条指令。R0 启动时为 0，可用作普通寄存器（无保留），仅 R254 按 ABI 作 SP（硬件不强制）。

---

## 2. 指令字与字段总览

每条指令 8 字节，小端。16 位公共头 + 48 位格式体。

### 2.1 公共头（bit 63..48）

```
 63       57 56  55 54    52 51    48
+----------+---+---+------+-------+
| OP[6:0]  | G | GN| GP[2:0]| M[3:0] |
+----------+---+---+------+-------+
```

- `OP`：操作码（§6 给号）。
- `G`：谓词守卫使能。G=0：无守卫。G=1：仅守卫为真的 lane 执行。
- `GN/GP`：守卫谓词号。0..6=P0..P6，7=PT。守卫条件 = G ? (GN ? ¬P[GP] : P[GP]) : 1。
- 每 lane 最终写使能 = `active_mask[lane] AND guard[lane]`。被掩掉的 lane：不产生任何存储写、不写寄存器、不参与原子的地址/数据，但**照常随流水线流动**（占发射/执行槽）。
- `M[3:0]`：类别相关修饰：
  - FP 类：M[1:0] 舍入（0=RN 就近偶数，1=RZ 向零，2=RM 向下，3=RP 向上）；M[2]=`.SAT`（饱和，限 F32→I 转换）；M[3] 保留。
  - 整数类：M[3:0] 一般为 0。

### 2.2 格式体（bit 47..0）

**R 型**（寄存器-寄存器；也用于 ATOM/BAR/S2R 等）

```
 47    40 39    32 31    24 23    16 15     8 7      0
+--------+--------+--------+--------+--------+--------+
| d[7:0] | a[7:0] | c[7:0] | X[7:0] | Y[7:0] | (注1) |
+--------+--------+--------+--------+--------+--------+
```
注 1：R 型实际只用 40 位（d/a/c/X/Y），bit 7:0 保留为 0。三寄存器操作数语义 `d = op(a, c)`；两操作数指令用 a、c，b 位置即字段 c。统一约定：**第一源 a = bit39:32，第二源 c = bit31:24，目的 d = bit47:40**。

**I 型**（立即数）

```
 47    40 39                         8 7      0
+--------+------------------------------+--------+
| d[7:0] | imm32[31:0]                  | X[7:0] |
+--------+------------------------------+--------+
```

**B 型**（跳转/SSY/CALL）

```
 47                     22 21    16 15                  0
+--------------------------+--------+---------------------+
| imm26[25:0]（有符号）    | X[5:0] | reserved=0          |
+--------------------------+--------+---------------------+
```
目标地址：X[0]=0（默认）相对跳转 `target = PC + 8*sext(imm26)`；X[0]=1 绝对跳转 `target = 8*sext(imm26)`。

访存指令的两种变体，高位均为 d 与基址 a，区别由 X[7] 标识；汇编器与解码器必须照此实现：

```
M  型（X[7]=1，寄存器偏移）：
 47:40 d(数据)  39:32 a(基址)  31:24 c(偏移寄存器)
 23:16 X(修饰1) 15:8  Y(修饰2) 7:0  0x00

MI 型（X[7]=0，立即数偏移）：
 47:40 d(数据)  39:32 a(基址)  31:24 X(修饰1)
 23:16 Y(修饰2) 15:0  simm16（有符号字节偏移，范围 −32768..32767）
```

访存修饰编码：

- X[2:0] = 地址空间：`0 GLOBAL / 1 SHARED / 2 LOCAL / 3 CONST / 4 FLAT`
- X[6:3] = 宽度：`0 U8 1 S8 2 U16 3 S16 4 U32 5 S32 6 U64 7 F32 8 F64 9 F16 10 U128`
- X[7] = 1 表示本指令为 M 型（带 c 偏移寄存器）；0 = MI 型
- Y[2:0] = cache hint：`0 CA(逐级缓存) 1 CG(仅 L2) 2 CS(流式绕过) 3 CV(易失) 4 LU(只读入 L1)`
- Y[7:4] 保留 0

有效地址：`addr = R[a] (+ R[c]) (+ simm16)`，始终为字节地址；U64/F64/U128 使用 d/a/c 起的寄存器对/四元组（d、a、c 必须偶数对齐，否则解码异常 `ERR_UNALIGNED_REG`）。

### 2.3 SETP 的字段约定（R 型）

- Y[7]=1 表示 d 字段是谓词号：目的谓词 = d[2:0]。
- c[2:0] = 布尔组合源谓词（7=PT/不用）；Y[1:0] = 组合方式 `0 AND 1 OR 2 XOR 3 SET(直接覆盖)`。
- X[3:0] = 比较码（见 §5.2）；X[6:4] = 操作数类型 `0 S32 1 U32 2 F32 3 S64 4 U64 5 F64`。

---

## 3. 操作码表（OP 字段数值是二进制兼容的一部分）

### 3.1 数据传送 / warp 内通信（0x00–0x0F）

| OP | 助记 | 格式 | 部件 | 语义（逐 lane，简写） |
|---|---|---|---|---|
| 0x00 | NOP | R | BRU | 无操作 |
| 0x01 | MOV | R | SP | d ← a |
| 0x02 | MOV32I | I | SP | d ← imm32（F32 立即数按位重解释；X[0]=1 时符号扩展到偶奇对高半部） |
| 0x03 | S2R | R | BRU | d ← SpecialReg[X[5:0]]（编号表 §4.3）；64 位特殊寄存器写 d/d+1 |
| 0x04 | LDC | MI | LSU/常量 | d ← const_bank[X[3:0]][simm16]；X[7:5] 宽度 `0 U32 1 U64 2 F32`；warp 同址广播 |
| 0x05 | CVTA | R | LSU/ALU | X[2:0] 给定源空间，d_pair ← to_flat(R[a_pair] + (c≠RZ ? R[c_pair] : 0))；反向 X[3]=1：flat→指定空间偏移 |
| 0x06 | SHFL | R | BRU+交叉 | d ← shuffle(X[1:0])：0 IDX 1 UP 2 DOWN 3 BFLY；源 a，索引起 c；越界时 X[2]=1 返回自身否则按 mod32 |
| 0x07 | VOTE | R | BRU | X[1:0]：0 ALL 1 ANY 2 EQ 3 BALLOT；输入谓词 Y[2:0]；ALL/ANY/EQ 写 d=0/1，BALLOT 写 d=active 中为真 lane 的 32 位掩码 |
| 0x08 | PRMT | R | SP | d 字节置换：对 c 的每个字节，低 4 位在 {a 四字节,b 四字节} 共 8 字节中选 1 字节，bit4 取符号扩展 |
| 0x09 | SELP | R | SP | d ← P[Y[2:0]] ? a : c（GN 语义由谓词号最高位表达：编码 15=取反，7=PT） |

### 3.2 整数（0x10–0x2F）

| OP | 助记 | 语义 | 备注 |
|---|---|---|---|
| 0x10 | IADD | d ← a + c | X[2:0] 类型 4=U32(默认) 6=U64（寄存器对） |
| 0x11 | IADD3 | d ← a + c + imm8(X 扩到 imm? ) | 基线限制：第三操作数必须是 X 中 8 位立即数（bit15:8），无寄存第三源 |
| 0x12 | IMAD | d ← a*c + d（旧值） | X[5:3]：0 LO（低32）1 HI（高32）2 WIDE（写偶奇对）；X[2:0] 符号/宽度：0 U32 1 S32 |
| 0x13 | IMUL | d ← a*c | X[5:3] LO/HI/WIDE；X[2:0] U/S |
| 0x14 | ISUB | d ← a − c | U64 同 IADD |
| 0x15 | IMNMX | d ← min/max | X[0]=1 取 max；X[1]=1 无符号 |
| 0x16 | LOP | d ← 位逻辑 | X[1:0]：0 AND 1 OR 2 XOR 3 ANDN(a&~c)；X[2]=1 时 c 字段解释为零扩展 imm8（位 23:16） |
| 0x17 | SHF | 漏斗移位 | X[0] 左/右；X[1] 无符号；64 位拼接 (c:a) 移 X[7:2]&63 取 32 位 |
| 0x18 | SHL | d ← a << (c&31) | U64 时 &63 |
| 0x19 | SHR | d ← a >>> (c&31) | 逻辑 |
| 0x1A | SAR | d ← a >> (c&31) | 算术 |
| 0x1B | BFE | d ← 位域提取 | a=源，c 低 5 位=pos，d 旧值低 5 位=len（无符号，符号由 X[0] 选） |
| 0x1C | BFI | d ← (d_old & ~mask(pos,len)) | 插入 a 的低位域；pos=c[4:0]，len=d_old[4:0]，目的/源同为 d |
| 0x1D | POPC | d ← popcount(a) | |
| 0x1E | BREV | d ← bit_reverse32(a) | |
| 0x1F | IABS | d ← abs_s32(a) | −2³¹ 回绕为 −2³¹ |
| 0x20 | INEG | d ← −a | |
| 0x21 | IDIV | d ← s32(a)/s32(c) | SP 微码，20 周期，c=0 返回 0 |
| 0x22 | IREM | d ← s32(a)%s32(c) | 同上，c=0 返回 a |
| 0x23 | LOP3 | d ← LUT8(a,c,imm8) | I 型：imm32 低 8 位为三输入真值表（第三输入恒 0/由 Y 选 d_old） |
| 0x24 | BMSK | d ← (1<<(a&31))-1 | 生成低位掩码；a&31=0 时 d=0 |
| 0x25 | FIND | d ← 位查找 | X[0]：0 最高位 1 的位置（无则 32），1 前导零计数，2 尾零 |

### 3.3 浮点（0x30–0x3F）

舍入由 M[1:0] 给；除 MUFU 外 IEEE 精确（0.5 ULP），subnormal 不刷新。

| OP | 助记 | 语义 |
|---|---|---|
| 0x30 | FADD | d ← a + c |
| 0x31 | FSUB | d ← a − c |
| 0x32 | FMUL | d ← a × c |
| 0x33 | FFMA | d ← a × c + d_old（单次舍入融合乘加） |
| 0x34 | FMNMX | X[0]=1 max；NaN 按 IEEE 传播；−0/+0 取非负 |
| 0x35 | FSET | d ← compare(a,c) 为真 ? 0xFFFFFFFF : 0（比较码同 SETP，Y[1:0] 与 d 旧值 AND/OR/XOR/SET） |
| 0x36 | F2F | d ← 浮点间转换；X[3:0]：0 F16→F32 1 F32→F16(RN) 2 F32→F64(对) 3 F64→F32 |
| 0x37 | XCVT | 整↔浮点转换；X[2:0] 源类型（S32/U32/S64/U64/F32/F64），X[6:4] 目标类型；M[2]=SAT 时饱和到目标整数范围 |
| 0x38 | RCP | SFU：d ← 1/a（近似，相对误差 ≤ 2⁻¹²） |
| 0x39 | RSQ | SFU：d ← 1/√a（近似，同上） |
| 0x3A | MUFU | SFU：X[2:0] 选 0 SIN 1 COS 2 EX2(2^x) 3 LG2(log2x)；输入规约由软件完成；误差 ≤ 2 ULP |
| 0x3B | FRND | X[1:0]：0 floor 1 ceil 2 trunc 3 nearbyint |
| 0x3C | FABS | d ← |a| |
| 0x3D | FNEG | d ← −a（仅翻符号位） |
| 0x3E | FCMP | d ← compare(a,c) ? d_old : a（用于无谓词选择，X=比较码；可由 SELP 替代，保留助记） |
| 0x3F | DADD | F64 加（寄存器对，走 DP 管线下同样编码空间）；助记法：0x3F 为 FP64 算术槽，X[1:0] 0 ADD 1 SUB 2 MUL 3 FMA |

### 3.4 比较与谓词（0x40–0x4F）

| OP | 助记 | 语义 |
|---|---|---|
| 0x40 | SETP | R 型，Y[7]=1：Pd ← (a cmp c) {AND/OR/XOR/SET} Pc；比较码/类型在 X（§5.2） |
| 0x41 | SETPI | I 型，d 字段为谓词号：Pd ← (a cmp sext(imm32)) |
| 0x42 | PLOP | Pd（d[2:0]）← Pa(a[2:0]) op Pc(c[2:0])；X[1:0] AND/OR/XOR/SET/MOV_NOT |
| 0x43 | PSET2 | 谓词选择：Pd ← Ps ? Pa : Pc（Y[4:2]=选择谓词） |

### 3.5 控制流 / 同步（0x50–0x5F）

| OP | 助记 | 格式 | 语义（概要，完整机器规则在 03 册 §4） |
|---|---|---|---|
| 0x50 | BRA | B | 守卫在时按**每 lane 谓词**求值；全一致直接跳/不跳；分歧时操作 SIMT 栈（03 册 §4.3）。无守卫=无条件跳转 |
| 0x51 | BRX | R | 间接跳转 target ← R[a]（8B 对齐，否则 `ERR_MISALIGNED_PC`）；分歧语义同 BRA（lane 各自 rPC） |
| 0x52 | CALL | B | SR_LR ← PC+8；PC ← target；调用不切栈帧（软件在 local 自建） |
| 0x53 | RET | R | PC ← SR_LR |
| 0x54 | SSY | B | 在 SIMT 栈压入再收敛标记 RPC=target（不改变 PC） |
| 0x55 | BAR | R | X[5:4]：0 SYNC 1 ARRIVE 2 WAIT；X[3:0]=id；a=计数寄存器（RZ=本 CTA 全部有效 lane） |
| 0x56 | MEMBAR | R | X[1:0]：0 CTA 1 GL 2 SYS（01 册 §7.2） |
| 0x57 | EXIT | R | 当前 lane 线程结束；warp 全部 lane EXIT 后 warp 消亡 |
| 0x58 | YIELD | R | 调度提示（自旋退让）；功能=NOP；时序模型降低该 warp 一拍优先级（见 03 册 §5.4） |
| 0x59 | TRAP | I | 以 imm32 低 16 位为错误码陷入，grid 失败（05 册 §8） |
| 0x5A | BRKPT | R | 调试断点；仿真器断点回调，发布硬件为空操作 |
| 0x5B–0x5F | — | | 保留，解码命中 → `ERR_BAD_OPCODE` |

### 3.6 访存与原子（0x60–0x6F）

| OP | 助记 | 格式 | 语义 |
|---|---|---|---|
| 0x60 | LD | M/MI | R[d] ← mem<space>[addr]（宽度/符号/hint 见 X/Y） |
| 0x61 | ST | M/MI | mem<space>[addr] ← R[d]（d 为数据，a 为基址） |
| 0x62 | LDU | M/MI | uniform 只读 LD.GLOBAL.LU：warp 内地址须一致（不一致时按各自地址取，性能模型记非一致惩罚），强制经 L1 只读路径 |
| 0x63 | ATOM | R | old←mem[addr]；mem[addr]←f(old, R[c])；d←old；CAS 时 R[c]=新值、b=比较值；Y[3:0]=op（§5.3），空间/宽度在 X |
| 0x64 | RED | R | 同 ATOM 但不返回 old（d 字段必须为 RZ） |
| 0x65 | PREFETCH | M/MI | 提示预取到 Y 指定层级；无副作用、不 faults（越界静默丢弃） |
| 0x66 | LD.128 | M/MI | 宽加载：d..d+3 ← 16 字节（仅 GLOBAL，hint CS/CG），地址须 16B 对齐 |
| 0x67 | ST.128 | M/MI | d..d+3 → 16 字节，同上对齐；全掩或部分写按字节使能不支持（要求 16B 全 lane 合并，硬件按 warp 内连续段拆分，见 04 册） |
| 0x68–0x6F | — | | 保留 |

助记书写约定（汇编/反汇编统一）：空间写在助记后：`LDG`=LD.GLOBAL、`LDS`=LD.SHARED、`LDL`=LD.LOCAL、`STG`/`STS`/`STL`；扩展修饰点写：`LDG.E.U32`（E=evict-first 流式 CS 的别名）、`LDG.CA.F32`、`STS.U32`。

---

## 4. 详细字段编号

### 4.1 特殊寄存器编号（S2R 的 X[5:0]）

| 编号 | 名称 | 编号 | 名称 |
|---|---|---|---|
| 0 | SR_TID.X | 16 | SR_SMID |
| 1 | SR_TID.Y | 17 | SR_CLOCKLO |
| 2 | SR_TID.Z | 18 | SR_CLOCKHI |
| 8 | SR_CTAID.X | 19 | SR_NCTAID.Y |
| 9 | SR_CTAID.Y | 20 | SR_NCTAID.Z |
| 10 | SR_CTAID.Z | 21 | SR_LANEID |
| 11 | SR_NTID.X | 22 | SR_WARPID（CTA 内） |
| 12 | SR_NTID.Y | 24 | SR_LR |
| 13 | SR_NTID.Z | 32 | SR_PARAM_BASE（u64） |
| 14 | SR_NCTAID.X | 33 | SR_SMEM_BASE（u64） |
| — | — | 34 | SR_LMEM_BASE（u64） |
| — | — | 40 | SR_WARPID_IN_GRID |
| — | — | 41 | SR_GRIDID（多 kernel 扩展，基线恒 0） |

3–7、23、25–31、其余编号保留；解码命中保留编号 → `ERR_BAD_ENCODING`。汇编器始终以名称为准。

### 4.2 LDC 常量 bank

- bank 0：kernel 参数（param），偏移即 ABI 参数偏移。
- bank 1：用户 `.const` 常量表（≤ 64 KiB）。
- 其余保留。

### 4.3 谓词守卫编码

GP=0..6 → P0..P6；GP=7 → PT。GN=1 取反。汇编形式 `@P3` / `@!P3` / 无守卫。

---

## 5. 运算修饰与比较码

### 5.1 类型后缀

整数：`.S8 .U8 .S16 .U16 .S32 .U32 .S64 .U64`（默认 U32/S32 按助记语境）。
浮点：`.F16 .F32 .F64`（默认 F32）。
浮点舍入：`.RN .RZ .RM .RP`（默认 RN），仅 FADD/FSUB/FMUL/FFMA/F2F/XCVT 合法。
饱和：`.SAT`（仅 XCVT 整型目标）。
宽度不符的寄存器对/四元组未对齐 → 解码异常。

### 5.2 SETP/FSET 比较码（X[3:0]）

| 码 | 整数助记 | 浮点助记 | 为真条件 |
|---|---|---|---|
| 0 | EQ | EQ | a == c |
| 1 | NE | NE | a != c |
| 2 | LT | LT | a < c |
| 3 | LE | LE | a <= c |
| 4 | GT | GT | a > c |
| 5 | GE | GE | a >= c |
| 6 | HI | — | unsigned a > c |
| 7 | HS | — | unsigned a >= c |
| 8 | LO | — | unsigned a < c |
| 9 | LS | — | unsigned a <= c |
| 10 | — | FE | 两者都有序且 a==c（false if NaN） |
| 11 | — | GTU | a>c 或任一 NaN（unordered） |
| 12 | — | LTU/GEU 族 | unordered 变体（12=LTU 13=LEU 14=NUM(都非NaN) 15=NAN(任一NaN)） |

无符号比较走 HI/HS/LO/LS；有符号走 LT/LE/GT/GE。浮点 NaN：除带 U 后缀外，含 NaN 比较一律为假。

### 5.3 原子操作码（ATOM/RED 的 Y[3:0]）

| 码 | op | 32/64 语义 |
|---|---|---|
| 0 | EXCH | mem←val，old 返回 |
| 1 | ADD | mem←old+val（整数；F32 仅 GLOBAL 可选，基线不支持浮点原子，命中报 `ERR_BAD_ENCODING`） |
| 2 | MIN | 有符号 min |
| 3 | MAX | 有符号 max |
| 4 | UMIN | 无符号 min |
| 5 | UMAX | 无符号 max |
| 6 | INC | (old>=val)?0:old+1 |
| 7 | DEC | (old==0||old>val)?val:old−1 |
| 8 | CAS | old==cmp?mem←val; 返回 old（b=cmp，c=val） |
| 9 | AND | old&val |
| 10 | OR | old|val |
| 11 | XOR | old^val |

CAS 不支持 F32/F64；F16 不支持原子。

---

## 6. 未定义/非法情形与异常优先级

解码阶段（按序）检查，命中即取对应错误码，不产生架构状态修改：

1. 保留 OP / 非法修饰组合 → `ERR_BAD_OPCODE` / `ERR_BAD_ENCODING`
2. 64/128 位寄存器号奇偶未对齐 → `ERR_UNALIGNED_REG`
3. 访存 simm 偏移使宽度跨 16B 边界之外的对齐违例（U64 须 8B 对齐、U128 须 16B 对齐、F64 8B） → `ERR_MISALIGNED_ADDR`
4. BRA/BRX/SSY 目标非 8 字节对齐 → `ERR_MISALIGNED_PC`
5. BAR id ≥ 16 → `ERR_BAR_ID`

执行阶段：越界访问（超出分配/窗口）→ `ERR_ADDR_OUT_OF_RANGE`，整 grid 终止并记录首条故障指令的 PC、lane、地址。除 IDIV/IREM（除零返回定义值）外不产生算术异常。

---

## 7. 执行部件、延迟与吞吐（供硬件模拟器译码派发）

| 指令类 | 部件 | 发射间隔（周期/warp） | 结果延迟（周期） |
|---|---|---|---|
| 整数 ALU（0x10–0x20, 0x23–0x25, 0x01,0x08,0x09） | SP（32 lanes） | 1 | 4 |
| IDIV/IREM | SP 微码序列 | 20 | 24 |
| 浮点 FADD/FSUB/FMUL/FFMA/FMNMX/FSET/FRND/FABS/FNEG/XCVT/F2F(F32) | SP | 1 | 4 |
| F64 DADD 槽（0x3F）、F2F 含 F64 | DP（16 lanes，跨 2 拍） | 2 | 8 |
| RCP/RSQ/MUFU | SFU（16 lanes，跨 2 拍） | 2 | 8 |
| S2R/NOP/YIELD/BRKPT/SHF/VOTE/BRA 解析 | BRU | 1 | 1（分支）/ 随 ALU |
| LDC | LSU→常量 cache | 1 | 命中 30（广播） |
| LD/ST/LDU/ATOM(SHARED)/RED(SHARED) | LSU→shared/L1 | 1 接收 | smem 2；L1 30；见 04 册 |
| ATOM/RED.GLOBAL、PREFETCH | LSU→NoC→L2/ROP | 1 接收 | L2/DRAM 延迟 |
| BAR/MEMBAR/EXIT/SSY | BRU + 同步逻辑 | 1 | 见 03 册 |
| CALL/RET/BRX | BRU | 1 | 取指重定向 2 周期气泡 |

一条指令只派发到一个部件；硬件每周期总发射 ≤ 1 条 warp 指令（无双发射）。SFU/DP 指令占用对应部件 2 个发射周期（第一拍低 16 lane，第二拍高 16 lane），期间该部件忙。

---

## 8. 文本汇编语法（`ntas-as` 输入）

### 8.1 词法

- 指令每行一条；`//` 行注释；`/* */` 块注释。
- 数字：十进制或 `0x` 十六进制；浮点立即数在 MOV32I 中写作 `0f32:2.0`（汇编器转 IEEE 位型）。
- 标签：`name:`，代表其指令的字节地址；`@` 前缀引用仅用于指令修饰（不作为标签语法）。
- 寄存器：`R0`..`R255`、`P0`..`P6`、`RZ`、`PT`；常量 `c[bank][offset]`；特殊寄存器用助记名。
- 助记大小写不敏感；后缀大小写敏感（F16 等）。

### 8.2 指令通用形式

```
[守卫]  MNEMONIC[.后缀...]  Rd, Ra, Rc [; ]
[守卫]  MNEMONIC  Rd, Ra, [Rc + imm]      // 访存
[守卫]  BRA   label
SSY label
BAR.SYNC  R0          // RZ 表示全部 CTA lane
```

### 8.3 汇编指示（directives）

```
.target ntas1
.entry saxpy
.regs 12                 // 每线程 32 位寄存器需求（ABI）
.regs64 2                // 64 位值个数
.smem 0                  // 静态 shared
.param 20                // 参数大小
.bar 1                   // barrier 槽
.flags flat
    ... 指令 ...
.endentry
```

段：`.text` 代码；`.smem align=16` 后接 `label: .zero 256` 静态 shared 布局；`.const` 常量数据。多个 `.entry` 可同文件，名称对应容器 n_kernels。

### 8.4 反汇编输出规范

`ntas-dis` 每行格式固定（供差分测试）：

```
/*0000*/ @!P0 LDG.CA.U32 R2, [R4] ; /* 0x........ */
```

地址 4 位十六进制，冒号前无空格；编码 16 位十六进制；未知字段必须原样报告而非省略。

---

## 9. 完整编码示例（解码器测试向量）

`@P0 BRA Ldone`，假设 Ldone 在 PC+0x38（即 imm26 = 7，相对）：

| 字段 | 值 |
|---|---|
| OP=0x50（bit63:57） | 1010000 |
| G=1 GN=0 GP=0（bit56:52） | 1 0 000 |
| M=0（bit51:48） | 0000 |
| imm26=7（bit47:22） | 0x0000007，左移 22 位 |
| X[0]=0 相对，其余 0（bit21:16） | 0x00 |

16 位头 = `1010 0001 0000 0000` = `0xA100`。

拼出 64 位：`0xA100000001C00000`（文件内小端字节序 `00 00 C0 01 00 00 00 A1`）。

`IMAD R3, R6, R8`（d=3,a=6,c=8, LO+U32：X=0,Y=0，无守卫）：

- 头：OP=0x12 → `0010010`，G/N/GP=00000，M=0000 → 16 位头 = `0x1200`
- d=0x03 a=0x06 c=0x08 X=0x00 Y=0x00
- 64 位：`0x1200_0306_0800_0000`

`LDG.CA.U32 R2, [R4]`（MI，simm=0；X：space=0,width=4(U32)<<3=0x20，X7=0 → X=0x20；Y=0）：

- OP=0x60 头 `0x6000`；d=2,a=4,X=0x20,Y=0,simm16=0 → `0x6000_0204_2000_0000`

`SETP.GE.S32 P0, R3, R2`（GE=5，S32=0；Y: SET=3, Y7=1 → Y=0x83；d=0,a=3,c=2；X=0x05）：

- OP=0x40 头 `0x4000`；R 型 Y 在 bit15:8 → `0x4000_0003_0205_8300`

这些向量必须包含在模拟器自测中（05 册 §8 的 `t_decode`）。

---

## 10. 指令序列的功能性伪代码（功能模拟器主循环）

```text
对每个被调度执行一拍的 warp：
  inst = mem_i[PC_of_TOS]              // 取 TOS.NextPC 处指令（I-cache 功能模型恒命中）
  解码 inst；非法 → 设备错误
  guard_mask = inst.G ? (inst.GN ? ~P[GP] : P[GP]) : 0xFFFFFFFF
  exec_mask  = TOS.mask & guard_mask
  按 OP 对 32 lanes 中 exec_mask 置位者逐条 lane 计算：
      读 R[a],R[c]（注意寄存器对、RZ 读 0、写忽略）
      执行 §3 语义；LD/ST/ATOM 生成带 {lane,addr,size,data} 的访问集合
  控制类（BRA/SSY/BAR/EXIT/...）按 03 册规则更新 SIMT 栈/屏障/PC
  非控制类：TOS.NextPC += 8
  （功能模型没有"周期"概念：一次 step 推进一个 warp 一拍；选择哪个 warp 由 §11 给的功能调度序）
```

### 10.1 功能模拟器的 warp 服务顺序（与时序无关的确定性）

功能模拟器采用**固定 round-robin**：每"大轮"按 NSM 号升序、NSM 内 warp 槽号升序，让每个非阻塞 warp 推进一拍；屏障等待、空 I-Buffer 在功能模型中立即视为可越过（不计数）；访存立即完成。分歧路径的推进严格按 SIMT 栈 TOS 展开（栈的处理天然确定）。该顺序只影响调试轨迹，不影响最终结果。

---

## 11. NPTX1 → NTAS1 主要映射

| NPTX1 | NTAS1 |
|---|---|
| 无限 `%r/%f/%p` 虚拟寄存器 | ntas-ptxas 寄存器分配到 R0..R254 / P0..P6 |
| `ld.param` | LDC.Param c[0x0][off] |
| `mov %tid` 等 | S2R |
| `mad.lo.s32` | IMAD.LO.S32 |
| `setp.ge.s32` + `@p bra` | ISETP(=SETP.GE.S32) + 守卫 BRA（汇编器在分支前插 SSY IPDOM） |
| `mul.wide.s32` | IMAD.WIDE.S32（c=RZ, d 旧值=0 先 MOV RZ） |
| `cvta.to.global` | CVTA |
| `ld.global / st.global` | LDG/STG（汇编器按局部性选 .CA/.CG/.CS hint） |
| `atom.global.cas` | ATOM.GLOBAL.CAS |
| `bar.sync 0` | BAR.SYNC |
| `membar.gl` | MEMBAR.GL |
| `ret/exit` | RET / EXIT |
| `vote.all/ballot/shfl` | VOTE/SHFL |

NPTX 不暴露 SIMT 栈；`.syncwarp`、再收敛点由编译器插入 SSY。本包不包含该编译器；所有测试直接使用 NTAS1 汇编。

---

## 12. 编码与执行接口

- NTAS1 指令定长 64 位，包含寄存器字段与谓词守卫。
- 编码不含显式 stall count 和读写等待屏障字段；依赖状态由周期模拟器维护。
- 寄存器对表示 64 位数值，不包含条件码寄存器。
