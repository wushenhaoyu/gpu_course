> **1.1 阅读说明：** 本册列出原架构参数与接口。周期模拟器的已实现行为、模拟延时和简化边界以 [06_1.1时序与修复说明](06_1.1时序与修复说明.md) 为准。

NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education  
Version 1.10  
Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN  
Improved by: Tonghui Ming  
References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018  
September 2026

# NSM SIMT 核心微结构规格（第 3 册）

> 本册定义每个 NSM 内部、周期级硬件模拟器必须实现的全部结构与算法。教材第 3 章的三个"调度循环"（取指循环 / 发射循环 / 寄存器访问循环）是本册骨架。功能模拟器只需实现其中带【F】标记的语义。

---

## 1. NSM 内部框图与流水线

```
                     SIMT 前端（取指 + 发射循环）
 +---------+   +---------+   +--------+   +----------+   +-------+
 | Fetch 1 |-->| I-Cache |-->|Decode 2|-->| I-Buffer |-->|Issue 3|
 |  (PC)   |   |  8 KiB  |   |        |   | 2/warp   |   | (RR)  |
 +---------+   +---------+   +--------+   +----------+   +-------+
     ^                                         |
     |                              +----------+----------+
     |                              v                     v
 +---+-----+                  +------------+       +-------------+
 |SIMT 栈  |<-----------------| Scoreboard |       | BRU(分支/   |
 |32级/warp|  mask/目标        | 4 项/warp  |       | BAR/VOTE)   |
 +---------+                  +------------+       +-------------+
                                   |
                            SIMD 后端（寄存器访问循环）
                                   v
                          +------------------+
                          | Operand Collector|   8 unit × 4 操作数槽
                          +--------+---------+
                                   | 每周期最多 4 bank 访问 + 1 写回
                                   v
                  +------------------------------------+
                  |   寄存器堆 RF  65536×32b，4 bank    |
                  +--------+---------------------------+
                                   |
        +--------------------------+--------------------------+
        v                          v                          v
 +-------------+            +---------------+          +--------------+
 | SP 管 32 宽 |            | SFU/DP 16 宽  |          | LSU -> shared |
 | INT32/FP32  |            | (各2拍/warp)  |          | /L1（第4册）  |
 +------+------+            +-------+-------+          +-------+------+
        |                           |                          |
        +---------------------------+--------------------------+
                                    v
                              WB 写回（清记分牌）
```

五级逻辑流水：`F（取指/查 I-Cache）→ D（译码入 I-Buffer）→ I（记分牌就绪后发射）→ R（collector 读操作数）→ E（执行）→ W（写回）`。访存指令的 E 在 LSU 内继续，结果在数十至数百周期后才 W。

每周期各阶段独立推进，用"边沿更新"（本周期产生的信号下周期生效），禁止组合穿透。

---

## 2. 【F】warp 与 lane 的硬件状态

每个 NSM 驻留最多 64 个 warp 槽。每槽硬件状态：

| 状态 | 说明 |
|---|---|
| `valid` | 该槽是否被某 CTA 的某 warp 占用 |
| `cta_id_lo` | 所属 CTA（本 NSM 局部 CTA 槽号 0..15） |
| `pc` 派生 | 永远等于 `stack[tos].nextpc`（不另设寄存器） |
| `stack[0..31]` | SIMT 栈，见 §4 |
| `ibuf[0..1]` | 两条已取指指令槽（含 depvec、PC、valid、done 位） |
| `sb[0..3]` | 记分牌未完成目的寄存器项 |
| `fetch_wait_msrh` | 正在等待的 I-cache MSHR 号（无=−1） |
| `bar_wait` | −1 自由；否则在某 barrier 上等待 |
| `lane_valid[32]` | 尾包 warp 永久无效 lane=0；EXIT 后清 0 |
| `exit_done` | warp 全部 lane 已 EXIT |

每 lane 私有：R0..R254、P0..P6、SR_LR、`sp`（ABI 用 R254 承载）。RF 物理实现见 §7。

---

## 3. 取指循环（第一调度循环）

### 3.1 每周期算法

```
若本周期 I-cache 有 fill 返回：
    写入对应 fill 缓冲，下周期重查 tag（同 §9 replay 规则的 icache 版）
按 RR 指针在 64 个 warp 槽中找第一个满足全部条件的 warp w：
    a) valid 且 !exit_done
    b) ibuf 存在空槽
    c) fetch_wait_msrh == -1
    d) TOS.nextpc 不在 ibuf 已有项的 PC 中（不重复取）
发起 I-cache 读(w, TOS.nextpc)：
    - 命中（功能模型恒命中）：下周期 D 阶段译码，写入 ibuf
    - 缺失：分配 1 个 I-MSHR（共 8 个），向 L2 发请求；置 fetch_wait_msrh；
      fill 回来后清除，该 warp 下一轮再取；其他 warp 完全不受影响
RR 指针 = (w+1) mod 64
```

- I-cache：8 KiB、4 路、128 B line、LRU；**一次 I-cache 访问返回 128 B line，ibuf 每次只译码写入 1 条**（PC 对应指令）；line 在 cache 内，后续顺序取指 1 周期命中。取指宽度=1 条指令/周期。
- 分支重定向：BRA/BRX/RET/CALL 改变 TOS.nextpc 后，清空该 warp ibuf 中旧 PC 路径的有效项，取指从新 PC 重新开始；产生 **2 周期取指气泡**（重定向当拍 + 下一拍无该 warp 指令可发）。
- ibuf 是**非阻塞滑动窗口**：warp 最多预取 2 条；遇到自己的屏障等待时停止取指。

### 3.2 译码

译码在 D 拍完成：解公共头/格式体（02 册 §2），非法编码立即设备错误（不进 ibuf）。同时进行**记分牌预查询**（§6）：把 depvec 与指令一起存入 ibuf。

---

## 4. 【F】SIMT 栈与分歧控制流（核心语义）

本节是功能正确性的关键，两类模拟器必须逐字实现同一状态机。

### 4.1 栈项结构

```
struct StackEntry {
   u64   rpc;     // 再收敛 PC（由 SSY 写入）；栈底项 = EXIT 哨兵 0xFFFF_FFFF_FFFF_FFF8
   u64   nextpc;  // 本子路径下一条指令
   u32   mask;    // 本子路径 active lane
   bool  dflag;   // 该项是否曾作为"分歧父项"（其 nextpc 被改写过）
}
```

warp 启动时：`stack[0] = {EXIT哨兵, entry, 有效lane掩码, false}`，tos=0，深度 1。

### 4.2 SSY（编译器在分歧区前发射）

```
push { rpc = target, nextpc = PC+8, mask = stack[tos].mask, dflag=false }
nextpc(PC) 推进：新 TOS 的 nextpc 即 PC+8
```

### 4.3 守卫 BRA / BRX 的分歧判定

令 `m = stack[tos].mask`，`g[32]` 为该指令守卫谓词的逐 lane 值（无守卫则 g=全 1，仅在 m 内有效）。

- `taken = m & g`，`fall  = m & ~g`（BRX：按每 lane 目标地址分组，taken_i 各自目标不同）
- **情形 1：taken == m**（全部活跃 lane 同去目标）：`tos.nextpc = target`，栈不变，记一次统一跳转。
- **情形 2：taken == 0**：`tos.nextpc = PC+8`，栈不变。
- **情形 3：分歧**（BRA 恰 2 路；BRX 可 k 路，k≤32，每路一个表项）：
  1. 若 `tos.rpc == EXIT 哨兵`：说明编译器漏发 SSY → `ERR_NO_IPDOM`。
  2. `join = tos.rpc`；
  3. `tos.nextpc = join`；`tos.dflag = true`（父项掩码保持 m 不动）；
  4. 依次压入各子路径项 `{join, 子路径PC, 子路径mask, false}`：
     - BRA：先压 fall 路径（PC+8），再压 taken 路径（target）；
     - BRX：按目标地址数值升序压入。
  5. **路径顺序裁定（可复现性）**：压入完成后，若栈顶项的 mask popcount 小于其下一项的 mask popcount，则交换这两个新压入项（lane 多的路径先执行）；相等时保持 taken/小地址路径在栈顶。该规则使最坏栈深度为 ⌈log₂32⌉ 量级，且与调度顺序无关。
  6. 下条执行指令 = 新 TOS.nextpc，active mask = 新 TOS.mask。

### 4.4 再收敛（出栈规则）

每条指令在 TOS 上执行完并算出 `newpc`（顺序推进 PC+8、统一跳转目标、CALL 目标等）后，**按顺序应用以下规则，每拍最多连续弹出至规则不再满足**：

- **R1（子路径走到 join）**：若 `newpc == tos.rpc`，弹出 TOS（本子路径不再执行 join 处指令，join 交还给暴露出来的父项）。
- **R2（父项执行完 join 点指令）**：若未弹出且 `tos.dflag == true` 且**刚执行指令的 PC == tos.rpc**，则该指令执行完毕（其 newpc 已离开 join）后弹出 TOS——dflag 父项在 join 点只代表一次会合，执行完 join 点那一条指令即归还下层。
- R1/R2 弹出后露出的新 TOS 若其 `nextpc` 恰好等于它自己的 rpc（即它也是被会合到该点的项），**本拍不做任何弹出**：它尚未在 join 点执行指令；等它执行后由 R2 处理。

栈底哨兵项永不弹出。

### 4.5 EXIT 与 lane 回收

- EXIT 对 exec_mask 中每个 lane：在**从 TOS 到栈底的所有栈项 mask 中清该位**（退出的 lane 不得在任何再收敛点复活），并清 `lane_valid`。
- 然后：若 `tos.mask == 0`，弹出该项；重复直到 tos.mask≠0 或栈空。
- 栈空（或所有 lane_valid=0）：warp `exit_done=1`。
- warp 所属 CTA 的全部 warp exit_done → CTA 结束（§11）。

### 4.6 谓词与栈的关系

谓词守卫只影响"本指令哪些 lane 写使能"，**不改栈**；只有控制类指令（BRA/BRX/SSY/CALL/RET）改栈。普通被守卫不执行的 lane 随流不取数不存数（LD/ST/ATOM 的地址集合只含 exec lane）。

### 4.7 教材例程的期望轨迹

教材图 3.2/3.3/3.4 的 4 lane 例子（A/1111→分支 B/1110 与 F/0001→内层 C/1000、D/0110→E 会合→G 全会合）必须作为强制测试 `t_simt_stack`：模拟器输出每次分支后的完整栈快照序列与本册规则逐行一致。

---

## 5. 【F】warp 发射调度（第二调度循环的选择策略）

### 5.1 合格条件

warp w 在某周期"可发射"当且仅当：

1. `valid && !exit_done && bar_wait == -1`；
2. ibuf 头部槽 valid（下一条指令已译码）；
3. 该 ibuf 项 `depvec == 0`（§6）；
4. 指令不是需要等待 replay 而被 LSU 扣留的状态（LSU 对尚未完成的 LD/ST 在 ibuf 项上置 `mem_pending`，见 04 册）；
5. 目标执行部件当前有空：
   - SP：每周期 1 条；
   - SFU/DP：部件忙（2 拍/warp）则不可发射；
   - LSU：接收队列未满（16 项 replay/接收容量）；
   - BRU：每周期 1 条。

### 5.2 基线策略：round-robin

- 每 NSM 一个 issue RR 游标。每周期从游标起按 warp 槽号升序扫描，发射第一个合格 warp 的 1 条指令，游标移到 `(该warp+1) mod 64`。
- 全部不合格：该周期不发射，统计 `stall_cycle`。
- 不做 GTO/ICOUNT 等策略（可选扩展 EXT-SCHED，05 册附录 E；切换策略不得改变功能结果）。
- YIELD 提示：执行到 YIELD 的 warp，其扫描优先级在接下来 4 周期内排到合格队列末尾（仅时序模型；功能模型忽略）。

### 5.3 发射时动作

- ALU/访存：分配 collector unit（无空 collector 不算"发射"，跳过该 warp，不消耗周期外事件，记 `stall_no_collector`）；把指令、warp 号、PC、exec_mask、操作数编号交给 collector。
- BRU 类：直接在 BRU 执行（S2R、SETP 走 SP 的除外——SETP 走 SP 写谓词；VOTE/SHFL/BAR/BRA/SSY/EXIT/MEMBAR 走 BRU）。
- 在 sb 中为目的寄存器分配一项（含谓词目的标志位）；LD/ST 的目的直到写回才释放。

---

## 6. 【F】Coon 式顺序记分牌

每 warp 4 项 `sb[j] = {reg(9b: 1bit P/R + 8bit 号), valid}`。

### 6.1 依赖位向量生成（D 拍，指令入 ibuf 时）

- 取指令**所有源操作数与目的操作数**的（类,号）集合（含谓词守卫号、SETP 组合谓词；RZ/PT 不入集合；64 位寄存器对按两个号都入集合）。
- 与该 warp 的 4 个 sb 项逐一比较：`depvec[j]=1`  iff  sb[j].valid 且其（类,号）∈ 操作数集合。
- depvec（4 bit）随 ibuf 项保存。指令合格 ⇔ depvec==0。

### 6.2 分配（I 拍发射时）

- 目的（类,号）写入某个 invalid 的 sb 项并置 valid。
- 4 项全 valid：**该指令不发射也不重新取指**（指令已在 ibuf），等任一项释放；取指循环可继续为其他 warp 服务，但本 warp 因 ibuf 将满最终自然停取（教材："fetch stalls for that warp / 丢弃重取"，基线取"等待"方案）。

### 6.3 释放（W 拍写回时）

- 找到 sb 中（类,号）等于写回目的的项，清 valid；
- 扫描该 warp 的全部 ibuf 项，把对应 j 的 depvec 位清 0；
- 顺序保证：同一 warp 指令按程序序经过 I→R→E→W（§8.2 强制同 warp 按序离开 collector），因此该简单结构同时防 RAW/WAW/WAR。

---

## 7. 寄存器堆物理组织

- 容量 65 536 个 32 位物理寄存器（256 KiB），按 CTA 分派时按 `R·B` 连续预留（00 册 §8.1）。
- 4 个**单端口逻辑 bank**，每 bank 每周期接受 1 次访问（读或写）；一次访问是"一个 warp 的一个操作数"（32 lanes 宽）。
- bank 映射（**swizzled**，教材图 3.16）：

```
bank(w, regnum) = (regnum + warp_slot) mod 4
```

  64 位操作数（偶奇对）占相邻两个 bank。谓词不进 RF 阵列（用独立的小谓词堆，每 lane 7 bit×64 warp，单周期读改写）。

- 逻辑行：`(w, regnum)` 在 bank 内的行号 = 该 CTA 寄存器基址 + regnum（同 warp 同号寄存器在 bank 内连续存放，宽端口一次铺开 32 lanes——仿真器只需维护 bank 占用，不必建模行内 lane 排布）。

---

## 8. 第三调度循环：Operand Collector

### 8.1 collector unit

共 8 个。每个保存：`{valid, warp, PC, exec_mask, opcode, pipe, dst, op[4]}`，其中 `op[k]={regnum, class, ready}`。I 拍分配，W 拍（或 LSU 接收后）释放。

### 8.2 每周期仲裁算法

```
输入：所有 collector 中未就绪的 op、本周期到达的写回请求
1) 先排写回：若有写回，占用其 bank(w,dst)，优先级最高（写回不可推迟）
2) 余下最多 3 个 bank 访问名额，在待读 op 中按
   (collector 年龄最老, warp 槽号小, 操作数序号小) 排序，
   依次选取与已选 bank 不冲突的 op 读 RF，置 ready
3) 一个 collector 的 4 个 op 全 ready：
   - 检查执行部件空闲（SP 空 / SFU 空 / LSU 可接收）
   - 同一 warp 的多个 collector 必须按其在 ibuf 中的程序序先后送往 E（WAR 防护，教材 3.3.1）
   - 满足则送入 E 阶段
```

- 单条指令两个源落在同一 bank（例：bank 数为 4 时 r5 与 r1 同 bank）→ 分两个周期读完，collector 用缓存吸收，**不产生 replay**（这是 bank 冲突在寄存器侧的处理：仅延迟）。
- 写回与读冲突：写回优先，被压的读顺延（教材图 3.14 的现象由本算法自然复现）。
- 该结构使 4 个单端口 bank 在多 collector 面前表现出接近多端口的带宽。

### 8.3 执行（E）

- SP：32 lanes 全宽，1 周期发起、流水线 4 拍后整 warp 结果就绪。
- SFU/DP：仅 16 lanes，第 1 拍执行 lane0..15、第 2 拍 lane16..31；两拍内部件 busy；结果 8 拍后可用。
- 浮点/整数语义按 02 册；只对 exec_mask 内 lane 计算，其余 lane 输出保持。

### 8.4 写回（W）

- 结果经 collector 登记的 dst 在 §8.2 仲裁中写 RF；仅 exec_mask lane 写。
- 谓词结果写谓词堆；同时按 §6.3 清记分牌。
- LD 类的 W 发生在 fill/replay 命中之后（04 册），不走 SP 延迟。

---

## 9. 指令 replay（结构性/访存冒险处理）

**资源重试：不向流水线深处传播全局停顿信号；资源不满足的指令就地重试。**

| 触发点 | 处理 |
|---|---|
| I 拍无空闲 collector | 不发射，下周期重新参与调度（指令留 ibuf，不重新译码） |
| LSU shared bank 冲突 | LSU 把 warp 访问切成"接受子集 + replay 子集"，replay 子集进 LSU replay 队列（16 项），下周期重新仲裁；队列满则留在 ibuf 重走 collector 之外的 LSU 接收口（记 `replay_from_ibuf`） |
| L1 tag 发现 miss | 在 PRT 登记，指令标 `mem_pending` 留 ibuf；fill 回来后由 fill unit 触发 replay，replay 时保证命中（line 锁定） |
| PRT 满 / 目标 set 所有 way 被未完成 miss 占用（associativity stall） | 当次拒绝 → replay |
| WDB 满（写缓冲） | ST replay |
| I-cache miss | §3.1 的 MSHR 等待（语义同上，结构独立） |
| 屏障未满足 | warp parked（不是 replay，是合格性排除） |

硬件模拟器统计每种 replay 次数；功能模拟器全部退化为"立即成功一次完成"。

---

## 10. 【F】CTA barrier 硬件

每 NSM 16 个 barrier 槽，按 `(CTA槽, bar_id)` 索引：

```
BarrierState { u32 expected; u32 arrived; list<(warp,mask)> waiters; bool active; }
```

- `BAR.SYNC id, count`（count=RZ ⇒ expected=本 CTA 有效 lane 总数 B）：等价 ARRIVE 后立即 WAIT。
- ARRIVE：`arrived |= exec_mask`（按 lane 计点）。
- WAIT：
  - `arrived == expected`：越过屏障并执行释放：清 active/arrived，把 waiters 中全部 warp 的 `bar_wait=-1`（同一周期末批量唤醒）；
  - 否则：本 warp 记 `bar_wait=id` 并把 `(warp, exec_mask)` 入 waiters。
- 唤醒的 warp 从 PC+8 继续（BAR 指令在 ARRIVE 语义已完成时视为完成）。
- 到达计数只增到 expected；若 `arrived` 中出现 expected 之外的 lane（掩码错误）⇒ `ERR_BAR_PARTIAL`。
- 部分 CTA 到达（分支内 BAR 使部分 lane 永不到达）：默认**死锁**（仿真器检测到 grid 无任何可推进 warp 时报告死锁 PC 集合）；配置 `bar_partial=error` 时直接 `ERR_BAR_PARTIAL`。
- CTA 结束时释放其全部 barrier 槽。

---

## 11. CTA 装载、结束与资源回收

- 全局分发引擎把 CTA 分派给 NSM 时（00 册 §4.1），NSM 原子预留 warp 槽（连续 W 槽，warp 号=linear_tid/32）、RF 区间、shared 基址、barrier 命名空间；初始化每 warp 栈与特殊寄存器；CTA 状态置 running。
- 每周期收集本 NSM 的 CTA 完成事件：某 CTA 的 W 个 warp 全部 exit_done →
  1. 写回脏数据的责任在存储侧（WB 策略，04 册），CTA 结束前本 CTA 的 ST 必须已离开 WDB（保证 local/writeback 数据已进 L2）；
  2. 释放 warp 槽、collector 残留、sb、ibuf、shared 分区、barrier、L1 中该 CTA 的 local 数据无效化；
  3. 上报分发引擎"CTA(linear_cta) 完成于本 NSM"，可接收新 CTA。
- kernel 全部 CTA 完成后，命令处理器写 completion token。

---

## 12. 周期级参考主循环（硬件模拟器直接照写）

```
每个上升沿：
  // ---- 存储侧先行（可能产生 fill/解锁事件，第 4 册）----
  l1_tick(); noc_tick(); l2_tick(); dram_tick();
  处理 fill 返回：清 PRT、置对应 ibuf 项 mem_pending=0、line 锁定

  // ---- W：写回（结果就绪且拿到 bank 仲裁）----
  wb_tick();        // §8.2/§8.4，清 sb/depvec

  // ---- E：流水线推进 ----
  sp_advance(); sfu_dp_advance();
  bru_execute();   // 改 SIMT 栈、bar_wait、唤醒（§4、§10）

  // ---- R：collector 仲裁 ----
  collector_tick(); // §8.2，可能送新指令入 E

  // ---- I：发射（每周期最多 1 条 warp 指令）----
  issue_tick();     // §5，分配 collector/sb/LSU 接收

  // ---- D：译码 ----
  decode_tick();    // §3.2，做 sb 预查生成 depvec

  // ---- F：取指 ----
  fetch_tick();     // §3.1

  // ---- 顶层 ----
  cta_retire_tick();// §11
  work_distributor_tick(); // 全片 1 个
  周期计数 +1；采集统计
```

事件时序约定（必须统一，否则两个硬件模拟器结果不一致）：

1. fill 与 WB 在"本周期前半"可见，使被唤醒指令最早**下周期**发射；
2. 同周期内 collector 刚读齐操作数的指令，最早**本周期末**进 E（E 结果按延迟表之后周期产生）；
3. 屏障批量唤醒在周期末生效，最早下周期可发射；
4. 取指命中：F 拍发起，下周期 D，再下周期才可 I（顺序 PC 下 ibuf 流水线允许每周期 1 条持续供给）。

---

## 13. 必须输出的微结构统计（每 NSM / 每 kernel）

`total_cycles`、`issue_cycles`、`stall_cycles` 及细分（no_collector / sb_wait / barrier_wait / lsu_full / icache_miss_wait / no_eligible_warp）、`inst_issued{按OP}`、`simt_divergent_branch`、`simt_stack_maxdepth`、`replays{smem_conflict, l1_miss, prt_full, assoc_stall, wdb_full}`、`rf_bank_conflict_cycles`、`sp/sfu/dp_busy_cycles`、`active_lane_sum / (32*inst_issued)`（SIMD 效率）、驻留 warp 数时间平均（占用率）、每拍合格 warp 数直方图。

---

## 14. 与功能模拟器的对应（【F】语义清单）

功能模拟器必须实现：栈状态机（§4 全部）、lane 掩码与写使能、谓词、warp 静态组成、RR 服务序（02 册 §10.1）、barrier 计数语义、EXIT/CTA 结束、特殊寄存器。功能模拟器**不实现**：I-cache、ibuf、记分牌、collector、bank、replay、部件忙、周期统计——所有依赖立即满足、所有访存立即完成。
