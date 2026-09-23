# NutShellGPU 260921 相对 260920 代码变更记录

NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education  
Version 1.10  
Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN  
Improved by: Tonghui Ming  
References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018  
September 2026


> 本文由两个版本目录的严格逐文件 diff（git diff --no-index，忽略 CR/LF 行尾差异）整理，
> 仅记录实际代码改动，不引用规格文档描述。
> 对比基线：`260920/` → `260921/`（VERSION: 1.1.0）。

## 0. 变更范围总览

| 范围 | 文件 | 结论 |
|---|---|---|
| RTL 硬件 | rtl/ 下 11 个 .sv/.svh（约 4600 行） | **零实质改动**，仅 LF→CRLF 行尾差异 |
| ISA | isa/ntisa.hpp | 11 增 10 删：F07 浮点舍入修复 + 注释 |
| 功能模拟器 | func/func_sim.cpp、func_sim.hpp | +313 / −488 行，多项正确性修复 |
| 时序模拟器 | timing/cycle_sim.cpp/.hpp | 引擎重写，净减约 587 行 |
| 时序模型 | timing/timing_model.hpp | **新增** 98 行 |
| 配置 | configs/timing_v11*.ini | **新增** 2 份 |
| 测试 | tests/ | 新增 4 个 C++ 测试 + runner + 3 个 Python 脚本 + 2 套 fixtures |
| 构建 | CMakeLists.txt | +44 行，新增 5 个测试目标 |

---

## 1. ISA 层（isa/ntisa.hpp）

- **F07 浮点舍入修复**
  - 旧 `round_fp32(v, m)` 对 FP32 结果调用 `nearbyint/trunc/floor/ceil`，会把小数结果错误舍入为整数。
  - 新版 `inline float round_fp32(float v, uint8_t) { return v; }`，舍入交由运算时的宿主浮点环境（`fesetround`）控制。
- `op_timing()` 注释标明：CycleSim 不再使用该默认表，改由 timing_model.hpp 的 `execution_latency` 决定延时。
- F11（LDC 宽度按 ldc_width 解码）逻辑在 260920 已存在，260921 仅补充修复标记注释。

## 2. 功能模拟器（func/）

### 2.1 Config（func_sim.hpp）

- 新增 `[timing_v11]` 配置字段：`core_clock_ghz`、move/integer/imul/idiv/shuffle latency、
  `rf_read_ports_per_bank`、`writeback_ports`、`execution_slots`、`lsu_slots`、`atomic_latency`。
- `load_ini_string` 新增 `timing_v11` 段解析。

### 2.2 PageMem 稀疏内存（F09）

- 页表键类型 `uint32_t → uint64_t`，消除 64 位高地址截断别名。
- 新增 recent page 单级缓存（`recent_key/recent_page`）。
- r16/r32/r64、w16/w32/w64 增加同页快速路径与正确的跨页边界处理。
- `read_bytes/write_bytes` 改为按页 `memcpy`。
- 自定义拷贝构造/赋值/`clear()`，复制与清空后使内部页指针失效。

### 2.3 内核启动与 CTA 派发（F01/F02/F03）

- `launch()` 新增 `module_idx` 参数，按模块身份取 code；旧版写死 `modules[0]`，多模块同名入口会混用。
- 启动时统一校验：grid/block 非零与上限、寄存器需求（`regs_u32 + 2*regs_u64`）、
  静态+动态共享内存、barrier 槽、参数大小、`code_offset%8` 及范围；失败返回 `ERR_LAUNCH`。
- 支持非零 `code_offset`，warp 入口 PC 由 0 改为 `ki.code_offset`。
- 启动时预解码全部指令至 `decoded_code/decoded_errors`，`fetch_inst` 走缓存。
- 新增 `dispatch_pending()` 与 `next_cta`：CTA 增量派发，NSM 满载后在每轮执行后补发；
  旧版 launch 时一次性派发、派不下即静默丢失，大 grid 被截断。
- `sync()` 收紧：无进展且 CTA 未全部完成即报 `ERR_DEADLOCK`；删除无效的死代码分支。
- `step_one_round()` 对 `cur_ctas==0` 的 NSM 跳过。

### 2.4 资源记账（F04）

- dispatch 时寄存器需求计入 u64 对、smem 需求计入动态部分；新增 `cur_regs` 占用检查。
- CTA 记录 `allocated_smem`；回收按实际分配量扣减 smem 与 regs（旧版回收写死 modules[0] 的 static_smem）。
- 共享内存按 `smem_need` 实际大小分配，不再固定分配整块 smem_size。
- local 基址改为 `(nsm_id * MAX_WARPS_PER_SM + ws) * 32 + lane`，消除跨 NSM local 地址别名。

### 2.5 取指（F05）

- `fetch_inst` 增加 PC 8 字节对齐与代码段越界检查，分别返回
  `ERR_MISALIGNED_PC` / `ERR_ADDR_OUT_OF_RANGE`，不再静默返回。

### 2.6 屏障（F06）

- `BarrierState` 新增 `generation` 与每 warp `arrival_phase[64]`。
- `bar_sync` 重写为代次模型：同一代每 warp 只登记一次，到齐后唤醒并进入下一代。
- 显式拒绝发散屏障（`mask != lane_valid` → `ERR_BAR_PARTIAL`），sub>2 拒绝。
- execute_inst 中 OP_BAR 的 79 行重复内联实现删除，统一调用 `bar_sync`。

### 2.7 浮点执行（F07）

- FADD/FSUB/FMUL/FFMA 执行前按指令 m 字段 `fesetround`（RN/RZ/RM/RP），执行后恢复。
- 文件头新增 `#include <cfenv>` 与 `#pragma STDC FENV_ACCESS ON`。
- 构建配套：MSVC 加 `/fp:strict`，GCC/Clang 加 `-frounding-math -ffp-contract=off`。

### 2.8 RZ 与 SHFL（F08）

- `WarpState::reg()` 判定由 `r==RZ` 收紧为 `r>=RZ || r<0`，宽指令 RZ 高字不越界 RF。
- RF 索引布局由 `rf[lane*256+r]` 转置为 `rf[r*32+lane]`。
- SHFL 执行前快照全部 lane 的源值与索引，再逐条提交；修复源/目寄存器重叠时读到已污染值。
- execute_inst 的读寄存器 lambda 改为走 const 版 `reg()`。

### 2.9 访存（F13）

- `decode_flat`：local offset 改为 `flat - w.lmem_base[lane]`，与 CVTA 互逆（旧版减全局基址导致重复叠加）。
- `load_mem/store_mem` 四个地址空间的重复 switch 合并为统一字节访问路径：
  - 修复 flat 解码出的 offset 被原始 flat 地址覆盖；
  - shared/const 统一边界检查（旧 const/local 宽访存高字丢失）；
  - 全部改字节读写，消除宿主未对齐解引用 UB；
  - store 高字源为 RZ 时写 0。
- LD/LDU/ST/LD128/ST128 合并为单一实现：写目的前快照地址；
  目的寄存器对越过 RZ 返回 `ERR_UNALIGNED_REG`；宽加载按 32 位逐字读取再回写。

### 2.10 原子（F14）

- `atom_op` 只接受 32 位宽度（其他宽度旧版静默接受），返回 `ERR_BAD_ENCODING`。
- 读失败立即返回；写失败错误向调用方传播（旧版吞掉）。
- CVTA 只做地址转换，不产生访存请求（由时序层统计侧保证）。

## 3. 时序模拟器（timing/，引擎重写）

### 3.1 提交模型翻转（T01）

- 旧：collector 阶段同步调用 `func.execute_inst()` 修改架构状态，writeback 仅清 SB。
- 新：issue/collector 不修改任何架构状态；ExecPipe::Slot 新增 `ready_cycle`，
  只有到期且抢到写回端口时才在 writeback 执行功能语义并释放互锁。

### 3.2 每 warp 顺序互锁（T03）

- `CycWarpState` 新增 `pending_commit`：每 warp 最多一条未提交指令，
  fetch/issue 遇之停顿。天然覆盖 RAW/WAW/WAR、谓词、宽寄存器对、FFMA 累加器 d 源依赖。
- 旧版仅 4 项 SB + depvec 只查 a/c，BRU 在 issue 期立即执行，依赖覆盖不全。

### 3.3 指令差异化延时（T02）

- 新增 timing_model.hpp `execution_latency()`：move/整数/乘法/除法/SP 浮点/SFU/DP/
  shuffle/BRU/LDC/BAR 分别映射可配置延时；ns→周期经 `ns_cycles()` 按 core_clock_ghz 换算。

### 3.4 寄存器组与 collector（T04）

- `CollectorUnit::ops` 容量 4→12，并保存完整 `Inst`。
- 新增 `timing_sources()` 精确列出功能读集合（含 FFMA/IMAD 的 d 源、ST 多字源、
  原子 CAS 的 b 源、宽访存高字源等），set 去重。
- collector 真正建模 GPR bank：`reg % rf_logical_banks`、每 bank 读端口限额、
  相同源同 collector 内只广播读一次、`collector_cursor` 跨单元轮询仲裁，
  冲突计入 `rf_conflict_stalls/rf_bank_conflict_cycles`。

### 3.5 执行资源（T05）

- 延时与启动间隔分离；interval 按 lane 数（sp/sfu/dp_lanes）折算吞吐下限。
- 槽位数由 `execution_slots/lsu_slots` 配置（旧版 SP 槽=lanes/32、LSU 写死 latency=30 槽=1）。
- writeback 按最早 ready_cycle 选择提交，受 `writeback_ports` 限制，竞争计入 `writeback_stalls`；
- 执行槽满计入 `pipeline_full_stalls`。

### 3.6 内存时序（T06，全新预约模型）

- 旧 `l1d_tick/noc_tick/l2_tick/dram_tick` 均为空壳/未连线占位（含不产生效果的 FR-FCFS 状态机），
  新版保留为空函数，逻辑全部移入 `memory_ready()`：
  - 新增 `TimingCache`：组相联 + 扇区 + LRU + 在途填充时间戳；未完成填充可合并，
    替换在途行使请求 start 顺延；只存 tag/时间，数据唯一保存在 FuncSim。
  - 按 lane 收集扇区集合实现 global 合并；stride 4 → 4 扇区，stride 128 → 32 扇区。
  - shared 按 bank 宽度统计冲突轮数（同字读广播）。
  - store/atom/LDU/CG/CV 绕过 L1，CV 同时绕 L2；const 按字串行计数。
  - 有限 MSHR（L1/L2）耗尽产生背压（prt_full）。
  - NoC 按内存分区预约 flit 往返链路时间（router_pipe + link_cycles）。
  - DRAM 按 channel/bank/open row/数据总线预约：行命中与行缺失（tRP+tRCD）分别计时。
  - 原子额外按活跃 lane 数加 atomic_latency；local tag 含 lane 基址避免跨 lane 合并。

### 3.7 取指/发射/回收（F10）

- fetch 每 warp 只保留单个 ibuf 表项语义（`ibuf_empty()`），`pending_commit` 时停取。
- issue 不再做 SB depvec/BRU 立即执行，只建 collector、算三层谓词 mask、置互锁。
- `cta_retire_tick`：所有 warp EXIT 且 collector 与全部流水槽排空后才回收；
  回收清空周期状态并按实扣 smem/regs，杜绝在途指令槽被复用。

### 3.8 预算、统计与配置校验（T07/F12）

- `run(max_cycles)` 改为本次调用的增量预算，0=不限；超限返回 ERR_DEADLOCK 且可续跑。
  旧版固定 100 万周期上限并输出调试打印。
- tick 顺序：writeback→execute→collector→fetch→decode→issue→retire + dispatch_pending。
- 占用率由"末尾瞬时 int 快照"改为按周期时间积分的 double（occupancy_sum 累计）。
- 新增 `validate_timing_config()`：拓扑、数组容量、延时参数全面启动校验，非法抛异常。
- stats JSON schema 升级为 `NutShellGPU-stats-1.1`：
  - 新增 `simulated_seconds = cycles / (core_clock_ghz × 1e9)`；
  - 新增 rf/writeback/pipeline stall、l1_hits/l1_pending_merges/memory_sectors、
    serial_dependency_stalls、unsupported_timing_ops、core_clock_ghz 等字段；
  - 修复 DRAM 有效带宽换算（旧 `/1e9` → `× core_clock_ghz`）。

## 4. 构建与测试

### 4.1 CMakeLists.txt

- project 声明版本 1.1.0。
- 新增目标：`t_runtime_v2`、`t_timing_v11`、`t_repairs`、`t_stats_time`、`model_runner`，
  及 `cycle_sim --test` CTest 项。
- numpy 存在时注册 tiny_lenet / residual_small / depthwise_medium 的模拟与比对 CTest。

### 4.2 新增测试

| 文件 | 覆盖点（断言级） |
|---|---|
| t_timing_v11.cpp | 到期前结果不可见、RAW/WAW/WAR 保序、指令差异化延时、RF bank 冲突与广播、FFMA 读旧累加器、store 完成前不可见/写后读顺序、L1 命中、扇区合并、shared bank 冲突、跨 warp 屏障保序、谓词关闭零访存、增量预算超时与续跑、宽寄存高字依赖、flat shared/local、跨 NSM local 隔离、宽 RZ 写不越界、128 位 local、shared 越界拒绝、INI 与 ns 换算、执行槽跨 warp 重叠、写回端口竞争、MSHR 背压、DRAM 行命中/缺失、发散重汇聚、CVTA 零访存、原子串行与错误传播、非法配置拒绝 |
| t_repairs.cpp | 跨页/非对齐 u64、跨页批量往返、惰性零页、超驻留 CTA 补发、多模块资源元数据、SHFL BFLY 源目别名 |
| t_runtime_v2.cpp | 64 位页号不别名、PageMem 深拷贝与 clear 失效、FADD/FMUL 小数保留、RZ 丢弃、四种 FP 舍入模式、非零入口、动态 smem 与回收、超额/零网格拒绝、寄存器驻留限制、PC 对齐错误、周期级 CTA 补发/重发与资源归零、时间平均占用率、屏障两代协同 |
| t_stats_time.cpp | 0.5/1/2 GHz × 多组周期的 simulated_seconds 换算精度 |

### 4.3 测试与工具链其他改动

- t_diverge.cpp：修正用例自身错误编码（原指令误覆盖 lane ID 的 R0），
  补 `static_smem=128` 与 32 lane shared 写后读断言。
- 新增 model_runner.cpp：夹具驱动逐层（conv/pool/fc/residual/argmax）周期模拟，
  导出每层 cycles/instructions CSV 与逐层张量、stats.json。
- 新增 benchmark_process.py：进程外 start-to-exit 墙钟测量、重复次数、退出码、
  模拟周期/simulated_seconds、平台与文件 SHA256 存证。
- 新增 check_model.py、generate_fixtures.py；configs 两份 ini；
  tests/fixtures（residual_small、depthwise_medium）与 examples/tiny_lenet（含 CUDA 对照）。

## 5. 未改动项（明确排除）

- 全部 RTL（nsm/scoreboard/operand_collector/simt_stack/l1d/l2/noc/dram_ctrl/tb 等）
  代码逻辑无任何变化。
- func/main.cpp、timing/main.cpp、isa/ntas_enc.hpp、tests/t_decode.cpp、tests/t_saxpy.cpp
  仅行尾差异，无实质改动。
