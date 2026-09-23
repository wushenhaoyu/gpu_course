> **1.1 阅读说明：** 本册列出原架构参数与接口。周期模拟器的已实现行为、模拟延时和简化边界以 [06_1.1时序与修复说明](06_1.1时序与修复说明.md) 为准。

NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education  
Version 1.10  
Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN  
Improved by: Tonghui Ming  
References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018  
September 2026

# 编程模型与 ABI（第 1 册）

> 适用：功能模拟器与硬件模拟器的共同程序视图。硬件必须为本册描述的全部语义提供实现；模拟器作者只需读本册 + 02 册即可写出正确的功能模拟器。

---

## 1. 执行模型总览

应用在主机 CPU 上启动，典型流程：

1. host 分配设备内存并初始化数据；
2. host 经驱动把输入数据拷入设备内存（离散显卡，独立地址空间）；
3. host 启动一个 **kernel**：一份在 GPU 上执行的函数，由一个 **grid** 的海量标量线程执行；
4. 每个线程执行**同一份程序**，但可用自己的线程号选择不同数据、走不同控制流；
5. grid 全部线程结束后，控制权返回 host；host 拷回结果。

编程模型对程序员呈现 MIMD 外观（每个标量线程仿佛独立执行），硬件以 SIMT 方式把 32 个线程组成 warp 锁步执行。这种"独立外观"由编译器 + SIMT 栈 + 谓词共同保证（见 03 册 §4）。

---

## 2. 线程层次

```
Grid
 ├── CTA(0,0,0)            cooperative thread array = thread block
 │    ├── warp 0 = thread (0..31)
 │    ├── warp 1 = thread (32..63)
 │    └── ...
 ├── CTA(1,0,0)
 └── ...
```

- 每个线程持有固定、唯一的坐标三元组：grid 维 `gridDim=(Gx,Gy,Gz)` 中 CTA 坐标 `ctaID=(bx,by,bz)`；CTA 维 `blockDim=(Dx,Dy,Dz)` 中线程坐标 `tid=(tx,ty,tz)`。
- 所有维度为非负整数，x 维变化最快。

**线性化（两种模拟器必须使用同一公式）：**

```
linear_tid   = tx + Dx*(ty + Dy*tz)          // CTA 内线程号，范围 0..B-1（B = DxDyDz）
linear_cta   = bx + Gx*(by + Gy*bz)          // grid 内 CTA 号
global_tid   = linear_cta*B + linear_tid     // grid 内全局线程号
```

- **warp 的形成是静态的**：`warp_id_in_cta = linear_tid / 32`，lane = `linear_tid % 32`。一个 CTA 的 warp 数 `W = ceil(B/32)`；最后一个 warp 不满 32 人时，不存在的 lane 的 active mask 位**永久为 0**（资源以整 warp 认购，但掩码保护）。
- warp 一经形成，其成员在整个生命周期不变（基线无 dynamic warp formation）。

### 2.1 限制

| 限制项 | 值 |
|---|---|
| 每 CTA 线程数 B | ≤ **1024** |
| blockDim 各维 | Dx ≤ 1024，Dy ≤ 1024，Dz ≤ 64，且乘积 ≤ 1024 |
| gridDim 各维 | ≤ 2³¹−1 |
| 每 NSM 驻留 warp / CTA / 线程 | 64 / 16 / 2048 |
| kernel 参数区 | 4096 字节 |
| shared/CTA（默认配置） | 48 KiB |
| barrier 槽位/CTA | 16（id 0..15） |
| 每线程寄存器数 R | ≤ 255（64 位值占 2 个连续编号） |

### 2.2 通信与同步

- **同 CTA 内线程**：通过 shared memory 通信，用硬件 barrier（`BAR.SYNC`）同步；延迟低。
- **跨 CTA 线程**：只通过 global 地址空间通信，代价高；同步只能依赖原子操作（`ATOM`）与内存栅栏（`MEMBAR`）。**基线不提供 grid 级 barrier**（跨 CTA 无内建屏障；错误地假设不同 CTA 间锁步是未定义行为）。
- 全局内存对所有线程可见，但 L1D **不保证跨 NSM 一致**：一个 NSM 上缓存的旧值可能被另一个 NSM 的写绕过。需要共享的数据必须走原子、`MEMBAR.GL`，或只读（基线 L1D 只放入可证明只读的 global 数据与 local 数据，从硬件策略上规避此问题）。

---

## 3. 存储（地址空间）模型

### 3.1 Global

- host 用 `ntMalloc` 分配的 64 位设备地址，对全 grid 所有线程与所有 kernel 持久（直到 host 释放）。
- 支持 8/16/32/64 位加载存储与原子。
- 经 L2；L1D 默认策略：global 写 write-through + no-write-allocate 并**绕过 L1D 数据数组**（经 L2），global 只读数据（编译器标记 `.const`/`LDU` 或只读推断）可进 L1D。
- 多个线程/CTA 对同一地址的访问顺序未定义，除非以原子指令或栅栏建立顺序。

### 3.2 Local

- 每线程私有的地址空间，用于寄存器溢出与函数栈；其他线程不可见。
- 物理上在 DRAM，经 L1D（write-back, write-allocate）与 L2 缓存。
- 对线程呈现 512 KiB 窗口；flat 地址由 local 窗口基址 + `(warp_slot*32+lane)*512KiB + offset` 重定位（功能模拟器按此实现隔离）。

### 3.3 Shared（scratchpad）

- 每 CTA 私有，CTA 结束即失效；同一 NSM 上多个驻留 CTA 瓜分 48 KiB（默认），由硬件按 CTA 槽重定位基址。
- 软件可控 SRAM，32 个 bank，每 bank 宽 32 bit、单读单写端口；一次 warp 访问的 bank 冲突规则与 replay 详见 04 册 §1。
- 支持 8/16/32/64 位访问与原子；64 位访问跨相邻两 bank。
- 生命周期内不清零：启动时内容**未定义**，编译器生成的 `SMEM_BAR`/显式初始化由程序负责。

### 3.4 Constant / Param

- 只读；`LD.CONST` 走常量 cache。一个 warp 中所有 lane 访问同一地址时只访问一次并广播；访问不同地址时退化为逐拍串行（功能模拟器按 lane 独立返回即可；性能模型记录 replay/多拍）。
- **Param 是 constant 的 bank 0**：kernel 参数由启动命令拷贝到设备内存参数区，硬件在 CTA 启动时把其映射到常量 bank `c[0x0]`，SASS 用 `LDC.Param R, c[0x0][off]` 读取（非访存指令也可带 `c[]` 操作数，由汇编器改写为 LDC 序列）。

### 3.5 Flat 与 CVTA

LD/ST 默认带显式空间后缀（`.GLOBAL/.SHARED/.LOCAL/.CONST`）。提供 `CVTA` 指令把带空间地址转换成 00 册 §6 的 flat 64 位地址，也支持 `LD/ST.FLAT` 由硬件按窗口判定空间（功能模拟器按窗口范围 if-else 路由；硬件模拟器在 LSU 做高位译码）。

---

## 4. Host 运行时语义（驱动接口）

功能模拟器必须提供下列 API（语言无关，语义等价即可）；硬件模拟器在仿真宿主侧用同样的调用序列驱动。

| 调用 | 语义 |
|---|---|
| `ntMalloc(size) -> dev_addr` | 在设备 global 内存分配，64 位对齐，返回设备虚拟地址；失败返回 0 |
| `ntFree(dev_addr)` | 释放 |
| `ntMemcpy(dst, src, n, H2D/D2H/D2D)` | 显式拷贝；语义在 launch 队列中**顺序提交**；H2D/D2H 经 PCIe（功能模型立即完成，时序模型按配置计延迟） |
| `ntLoadModule("a.ntas") -> mod` | 装载 NTAS1 目标（代码段 + 元数据） |
| `ntLaunch(mod, "kernel", gridDim, blockDim, smem_dyn, args...)` | 入命令队列：异步 |
| `ntSync()` | 等待队列中所有命令完成 |
| `ntMemset(addr, val, n)` | 设备侧按字节填充 |
| `ntErrorStr()` | 取最近错误（见 05 册 §8） |

**内存序**：同一队列里 launch 之前的所有 H2D 拷贝对该 kernel 可见；kernel 完成（ntSync 返回）后其全部写对之后的 D2H 拷贝可见。队列内命令之间不重排。

---

## 5. Kernel 启动 ABI

### 5.1 目标文件 `.ntas`（容器格式）

```
魔数 16B : "NTAS1\0\0\0\0\0\0\0\0\0\0\0"
version  : u16 = 1
n_kernels: u16
对每个 kernel：
  name_len   u16 ; name[name_len]（不含\0）
  code_off   u64 ; 相对文件头
  code_size  u64 ; 字节，8 字节指令，首指令位于偏移0
  regs_u32   u16 ; 每线程 32 位寄存器槽需求 R（ABI 声明）
  regs_u64   u16 ; 64 位值计数（占双槽）
  static_smem u32; 静态 shared 字节
  param_size u16; 参数字节（≤ 4096）
  bar_slots  u8 ; 需要的 barrier id 个数（≤16）
  flags      u32 ; bit0: 使用 flat；bit1: 使用 local；其余保留 0
```

代码段即 NTAS1 机器码数组（每条 8 字节，小端）。另支持文本汇编（见 02 册 §12）由汇编器 `ntas-as` 生成该容器。

### 5.2 Launch 描述符（驱动写入设备内存，64 字节，小端）

| 偏移 | 类型 | 字段 |
|---|---|---|
| 0 | u64 | code_entry（设备虚拟地址） |
| 8 | u64 | param_base（参数区设备地址） |
| 16 | u32 | Gx |
| 20 | u32 | Gy |
| 24 | u32 | Gz |
| 28 | u32 | Dx |
| 32 | u32 | Dy |
| 36 | u32 | Dz |
| 40 | u32 | dyn_smem_bytes |
| 44 | u32 | regs_per_thread（来自目标文件） |
| 48 | u16 | bar_slots |
| 50 | u16 | flags |
| 52 | u64 | completion_addr（完成后写 1 的设备地址；0=不写） |
| 60 | u32 | reserved=0 |

命令队列项 = `{u32 type=1(launch), u32 desc_hi? , u64 desc_addr}`（共 16 字节），队列在设备内存中是环形（容量 256 项），doorbell 为一个主机可写的 MMIO 寄存器；功能模拟器可用直接函数调用替代。

### 5.3 调用约定（NTAS1 ABI v1）

1. 参数由 host 拷入 param 区；函数入口前几条指令用 `LDC.Param` 取到 R 寄存器。参数按 8 字节对齐排列：≤8 字节的标量按原值；指针为 u64；结构体按 8 字节对齐平铺。
2. R0..R254 全部为调用者保存（kernel 内无跨调用 ABI；函数内调用由汇编器自行安排，基线不做硬件 call/ret 强制约定——`CALL/RET` 见 02 册，软件约定 R0 传返回值）。
3. 栈：需要栈的函数把 `SP`（汇编器虚拟，实体用某专用寄存器 R254 承载，记为 `R_SP`）指向 local 窗口；栈帧 16 字节对齐，向下生长。
4. 硬件在每个 warp 启动时初始化的内容：
   - 所有 R 寄存器、谓词 P0..P6 为 0；PT 恒为 1；
   - active mask = 本 warp 有效 lane（尾包 warp 高位为 0）；
   - `SIMT 栈`：单条 `{RPC=−, NextPC=code_entry, mask=active}`；
   - 特殊寄存器（见 §6）按坐标填入。
5. 线程退出：执行 `EXIT`。一个 warp 所有未屏蔽 lane 都 EXIT 后 warp 消亡；CTA 的全部 warp 消亡即 CTA 完成。

---

## 6. 特殊寄存器（`S2R` 读取）

| 助记 | 含义 | 粒度 |
|---|---|---|
| `SR_TID.X/Y/Z` | CTA 内线程坐标 tx,ty,tz | lane |
| `SR_CTAID.X/Y/Z` | CTA 坐标 bx,by,bz | warp 内统一 |
| `SR_NTID.X/Y/Z` | blockDim | warp 内统一 |
| `SR_NCTAID.X/Y/Z` | gridDim | warp 内统一 |
| `SR_LANEID` | lane 号 0..31 | lane |
| `SR_WARPID` | CTA 内 warp 号 0..31 | warp 内统一 |
| `SR_SMID` | 物理 NSM 号 0..15 | warp 内统一 |
| `SR_CLOCKLO` | 自由运行计数器低 32 位（功能模拟器返回已执行指令数；硬件模拟器返回周期数） | lane |
| `SR_PARAM_BASE` | 参数区 u64（两寄存器） | warp 内统一 |
| `SR_SMEM_BASE` | 本 CTA 的 shared flat 基址 u64 | warp 内统一 |
| `SR_LMEM_BASE` | 本 lane 的 local flat 基址 u64 | lane |

---

## 7. 同步、顺序与原子

### 7.1 CTA 屏障 `BAR.SYNC`

- 编码携带 barrier id（0..15）与参与线程数（=B，汇编器填 `SR_NTID` 乘积，也可硬件从 CTA 描述符取，编码中允许填 0 表示"本 CTA 全部有效 lane"）。
- 语义：到达的 warp/lane 在屏障上等待；当属于该 CTA 的全部有效 lane 都到达后，所有等待者同时被释放。硬件为每个 id 维护到达计数与等待 warp 集合（见 03 册 §10）。
- 规则：barrier 必须被 CTA 中所有线程**无条件到达**；把它放进有分歧的分支（某些 lane 不到达）是未定义行为（硬件死锁或返回错误码 `ERR_BAR_PARTIAL`，由配置选择；默认死锁，以暴露程序错误）。

### 7.2 内存栅栏

| 指令 | 顺序约束 |
|---|---|
| `MEMBAR.CTA` | 本线程之前的 shared/global 访存对本 CTA 其他线程先于其后的访存可见 |
| `MEMBAR.GL` | 本线程之前的 global 访存对全设备其他线程先于其后的访存可见 |
| `MEMBAR.SYS` | GL 之外再加 host 可见序（功能模拟器等同 GL） |

栅栏只约束**顺序**，不等待别人；无获取端配合时不构成同步。获取端用原子或 BAR.SYNC。

### 7.3 原子操作

`ATOM.{EXCH,ADD,MIN,MAX,INC,DEC,CAS,AND,OR,XOR}` 可作用于 `.GLOBAL`（在 ROP/L2 串行化）与 `.SHARED`（在 NSM 内串行化）。32/64 位。同一地址的多个原子（含同 warp 不同 lane）**逐个串行**，顺序未定义但每个原子整体不可分割；`CAS` 返回旧值。`RED.` 变体不返回旧值（只做归约，省一次写回）。

### 7.4 warp 级原语（无屏障，立即执行）

- `VOTE.ALL/ANY/EQ p`：对谓词在 32 lane 上投票，结果写入每 lane 的目的寄存器（0/1）。
- `BALLOT p -> R`：32 位掩码写入每 lane 的 R。
- `SHFL.IDX Rd, Rs, Ridx`：按每 lane 的索引 lane 号取数（越界回绕或返回自身，由后缀定，默认回绕）。
- 这些原语读取的是**当前 active mask 内**的 lane；被屏蔽 lane 不参与投票/不提供源。
- 注意：基线**不保证** warp 内分歧路径的隐式锁步会合（教材 3.1.1）。需要会合时由编译器在 IPDOM 放置再收敛点；程序不应依赖 lane 间在某条语句上的隐式同步（基线不提供 `__syncwarp` 等价物；BAR.SYNC 是 CTA 级屏障）。

---

## 8. NPTX1：虚拟指令集（驱动层）简述

类比 PTX：NPTX1 是面向驱动的稳定虚拟 ISA（无限虚拟寄存器 `%r/%f/%p`，RISC 风格 load/store），驱动内的 `ntas-ptxas` 在装载时把它编译成硬件 NTAS1（有限寄存器、显式寄存器号、控制指令）。本系列模拟器**以 NTAS1 为唯一强制输入**；NPTX1 仅作为前端表示，其指令语义是 NTAS1 的超集且一一对应（映射表在 02 册 §13 给出）。本包测试使用 NTAS1 文本汇编；本包不包含 NPTX 编译器。

---

## 9. 编程示例：SAXPY（端到端）

主机侧伪代码：

```c
ntInit();
devx = ntMalloc(n*4); devy = ntMalloc(n*4);
ntMemcpy(devx, hx, n*4, H2D);
ntMemcpy(devy, hy, n*4, H2D);
mod = ntLoadModule("saxpy.ntas");
int blocks = (n+255)/256;
ntLaunch(mod, "saxpy", {blocks,1,1}, {256,1,1}, 0, /*args*/ n, 2.0f, devx, devy);
ntSync();
ntMemcpy(hy, devy, n*4, D2H);
```

NPTX1（虚拟层示意）：

```
.entry saxpy(.param.u32 n, .param.f32 a, .param.u64 xp, .param.u64 yp) {
  %r<8> %rd<4> %f<4> %p<2>
  ld.param.u32  %r2, n;
  ld.param.f32  %f1, a;
  ld.param.u64  %rd8, xp;
  ld.param.u64  %rd4, yp;
  mov.u32 %r6, %ctaid.x;
  mov.u32 %r7, %ntid.x;
  mov.u32 %r9, %tid.x;
  mad.lo.s32 %r3, %r6, %r7, %r9;   // i = bx*Dx + tx
  setp.ge.u32 %p1, %r3, %r2;
  @%p1 bra Ldone;
  mul.wide.s32 %rd10, %r3, 4;      // i*4，零扩展为 64 位
  add.u64 %rd8, %rd8, %rd10;
  add.u64 %rd4, %rd4, %rd10;
  ld.global.f32 %f0, [%rd8];       // x[i]
  ld.global.f32 %f2, [%rd4];       // y[i]
  fma.rn.f32 %f0, %f0, %f1, %f2;   // a*x[i]+y[i]
  st.global.f32 [%rd4], %f0;
Ldone:
  exit;
}
```

编译后的 NTAS1 文本（02 册给出完整编码规则；地址以字节计、8 字节递增）：

```
// params: c[0x0][0x00]=n(u32)  [0x04]=a(f32)  [0x08]=xp(u64)  [0x10]=yp(u64)
00: LDC.Param.U32  R2,  c[0x0][0x00]
08: LDC.Param.F32  R1,  c[0x0][0x04]
10: LDC.Param.U64  R8,  c[0x0][0x08]   // 写寄存器对 R8/R9
18: LDC.Param.U64  R4,  c[0x0][0x10]   // 写寄存器对 R4/R5
20: S2R  R6, SR_CTAID.X
28: S2R  R7, SR_NTID.X
30: S2R  R9, SR_TID.X
38: MOV.U32 R3, R9                    // IMAD 累加项必须先放入目的寄存器
40: IMAD.LO.U32 R3, R6, R7            // R3 = R6*R7 + R3（旧值）= bx*Dx+tx
48: SETP.GE.S32 P0, R3, R2            // 默认组合方式 SET
50: SSY  Ldone
58: @P0  BRA Ldone
60: MOV32I R12, 4                     // 常数 4（移位量/比例因子必须在寄存器中）
68: MOV32I R10, 0
70: MOV32I R11, 0                     // 先清偶奇对（IMAD 含 d 旧值累加）
78: IMAD.WIDE.U32 R10, R3, R12        // R10:R11 = (u64)R3*4 + 0
80: IADD.U64 R8, R8, R10              // &x[i]（偶奇对相加）
88: IADD.U64 R4, R4, R10              // &y[i]
90: LDG.CA.F32 R0, [R8]
98: LDG.CA.F32 R2, [R4]
A0: FFMA.RN.F32 R0, R0, R1, R2
A8: STG.CG.F32 [R4], R0               // 写穿透，经 L2
B0: Ldone: EXIT
```

注意：

1. `SSY Ldone` 由编译器在守卫分支前显式发射（03 册 §4.2）；
2. IMAD 语义是 `d ← a*c + d（旧值）`（02 册 §4），所以 0x38 先把累加项 R9 放进 R3，再发射 0x40；
3. 分支偏移：`SSY Ldone`（pc=0x50,target=0xB0）imm26=12；`BRA Ldone`（pc=0x58,target=0xB0）imm26=11。

该程序是功能/硬件两类模拟器的**强制基准测试之一**（完整测试清单见 05 册 §8）。
