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
// cycle_sim.hpp - 周期级模拟器头文件 (S3 阶段) ★★★ 性能建模核心
// -----------------------------------------------------------------------------
// 本文件定义了 NutShellGPU 周期级模拟器的所有数据结构和接口。
// 与功能模拟器 (func_sim) 不同, 周期级模拟器关注 "时间" 维度:
//   - 每条指令占用多少时钟周期?
//   - 流水线各阶段的延迟与吞吐?
//   - 缓存命中率对性能的影响?
//   - 内存队列拥塞导致的停顿?
//   - bank 冲突引发的 replays?
//
// 架构对应关系 (参考 NutShellGPU_spec):
//   03 册 §6: NSM 流水线 (F-D-I-R-E-W 六级)
//   04 册 §3-§5: 内存层次 (L1D/L2/DRAM)
//   04 册 §7: 地址映射与 NoC 路由
//   05 册 §3: 模拟器实现契约 (S3 周期级)
//
// 设计哲学:
//   1. 复用功能模拟器 (FuncSim) 维护架构状态 (寄存器/内存/PC)
//   2. 在其上叠加 "时序状态" (流水线槽位/记分牌/缓存/MSHR)
//   3. 每个时钟周期 (tick) 推进所有组件一步
//   4. 统计停顿原因, 输出性能指标 (IPC/带宽/命中率)
//
// 教学要点:
//   - 周期级 = 功能正确 + 时序建模
//   - 流水线并行: 多条指令同时处于不同阶段
//   - 记分牌 (scoreboard): 解决 RAW/WAR/WAW 数据依赖
//   - MSHR (Miss Status Holding Register): 缓存未命中合并
//   - DRAM: 按预约顺序模拟行命中/未命中与通道竞争
// =============================================================================
#pragma once
#include "../func/func_sim.hpp"   // 复用功能模拟器的所有架构状态
#include "timing_model.hpp"
#include <queue>                   // std::queue 用于 NoC 虚通道
#include <deque>                   // std::deque 用于 DRAM 请求队列
#include <list>                    // std::list 用于 LRU 替换
#include <unordered_set>           // 用于跟踪活跃 warp 集合

namespace ntisa {

// =============================================================================
// Stage: 流水线阶段枚举 (03 册 §6.2) ★ 基础概念
// -----------------------------------------------------------------------------
// NutShellGPU NSM 采用经典六段流水线 (类似 Fermi/Kepler):
//   F  (Fetch)      - 取指: 从 I-Cache 读取指令
//   D  (Decode)     - 译码: 解析指令字段, 生成控制信号
//   I  (Issue)      - 发射: 记分牌检查, 选定执行单元, 进入操作数收集
//   R  (Read)       - 读寄存器: 操作数收集器读源操作数
//   E  (Execute)    - 执行: ALU/FPU/MEM/SFU 运算
//   W  (Writeback)  - 写回: 结果写回寄存器堆
//   DONE            - 完成 (虚状态, 用于状态机终止判断)
//
// 流水线时序示例 (单发射, 理想情况):
//   cycle:  1  2  3  4  5  6  7  8
//   I1:     F  D  I  R  E  W
//   I2:        F  D  I  R  E  W
//   I3:           F  D  I  R  E  W
//   理想 IPC = 1 (每周期发射一条指令)
//
// 教学要点:
//   - 流水线 ≠ 乱序执行, NutShellGPU 是顺序流水线 (按 warp 顺序发射)
//   - 停顿 (stall) 会在某段产生气泡 (bubble), 降低 IPC
//   - 实际 IPC < 1 的原因: 数据依赖/缓存未命中/bank 冲突/资源争用
// =============================================================================

// =============================================================================
// 1.1 NSM 流水线示意 ★ 延时为可配置的模拟值
// -----------------------------------------------------------------------------
// I-Cache(命中) -> I-Buffer -> 译码/发射 -> Collector/RF bank 仲裁
//                                         |
//                +----------+-------------+-------------+----------+
//                v          v             v             v          v
//               SP         SFU           DP            BRU        LSU
//                |          |             |             |          |
//                +----------+-------------+-------------+----------+
//                                         v
//                      ready_cycle 到期 + 提交端口仲裁
//                                         v
//                       功能语义写回 -> 释放 warp 互锁
//
// LSU: shared bank / L1 -> NoC 预约 -> L2 -> DRAM bank/总线预约。
// 每 warp 保持提交顺序; 不同 warp 可在多段流水线内同时运行。
// 原有 SB/PRT/WDB/Flit/DRAM 状态结构保留作教学接口, 实际启用范围
// 见 06_1.1时序与修复说明, 不代表每个接口结构均驱动当前时序。
// =============================================================================

// =============================================================================
// 教材引用: 第 3 章 §3.3.2 Pipeline (p.41) — Stage 枚举定义 F-D-I-R-E-W 六级流水线阶段, 对应教材图 3.11 流水线推进顺序
enum class Stage : uint8_t { F, D, I, R, E, W, DONE };

// =============================================================================
// IBufEntry: 取指缓冲表项 (03 册 §6.3) ★ I-Buffer
// -----------------------------------------------------------------------------
// I-Buffer 是 Fetch 与 Decode 之间的双拍 FIFO, 容量 2 条指令:
//   - Fetch 段每个周期填入一条指令 (若 I-Cache 命中)
//   - Decode 段每个周期取出一条指令进行译码
//   - 容量 2 允许 1 拍 Fetch 延迟而不阻塞流水线
//
// 字段说明:
//   valid       - 该表项是否有效 (有指令)
//   pc          - 指令地址 (用于分支预测/调试)
//   inst       - 译码后的 Inst 结构 (来自 ntisa::decode)
//   depvec      - 依赖向量 (4 位), 记录与未完成指令的依赖关系
//                 bit0-3: 对应记分牌 4 个槽位, 若置 1 表示本指令等待该槽
//   mem_pending - 是否有内存操作数未就绪 (LD/ST 等待地址翻译或填充)
//
// 教学要点:
//   - depvec 实现 "简化版记分牌": 4 个槽位跟踪 4 个未完成目的寄存器
//   - depvec = 0 表示指令所有源操作数已就绪, 可进入 Issue 段
// =============================================================================
// 教材引用: 第 3 章 §3.3.2 Pipeline (p.41) — IBufEntry 实现 I-Buffer 解耦 Fetch 与 Decode, 容量 2 允许 1 拍取指延迟不阻塞流水线
struct IBufEntry {
    bool valid = false;
    uint64_t pc = 0;
    Inst inst;
    uint32_t depvec = 0;    // 记分牌依赖向量 (4 位, 对应 4 个 SBEntry)
    bool mem_pending = false;  // 内存操作数未就绪标记
};

// =============================================================================
// SBEntry: 记分牌表项 (03 册 §6.4) ★ Scoreboard
// -----------------------------------------------------------------------------
// 记分牌用于检测数据依赖 (RAW/WAR/WAW), 共 4 个表项:
//   - 每个表项记录一个 "未完成的目的寄存器"
//   - Issue 段在发射前检查源寄存器是否命中记分牌
//   - Writeback 段在写回时清除对应表项, 并唤醒等待的指令
//
// 字段说明:
//   is_pred - true 表示这是谓词寄存器, false 表示普通寄存器
//   reg     - 寄存器编号 (0-255) 或谓词编号 (0-7)
//   valid   - 该表项是否在使用中
//
// 工作流程:
//   1. Issue: 指令 I 的 dst_reg 进入记分牌 (占用一个空表项)
//   2. 后续指令 J 若 src_reg == I.dst_reg, J.depvec 对应位置 1
//   3. Writeback: I 完成, 清除记分牌表项, J.depvec 对应位清 0
//   4. 当 J.depvec == 0 时, J 可进入 Issue
//
// 教学要点:
//   - NutShellGPU 用 4 槽简化记分牌 (真实 GPU 有几十槽)
//   - 这是一种 "顺序流水线 + 记分牌" 的简化乱序机制
//   - 与完整 Tomasulo 算法的区别: 无重排缓冲, 无寄存器重命名
// =============================================================================
// 教材引用: 第 3 章 §3.3 Three-Loop Approximation (p.35) — SBEntry 实现 Coon 式记分牌, 4 槽跟踪未完成目的寄存器以检测 RAW/WAR/WAW 依赖 (教材图 3.14)
struct SBEntry {
    bool is_pred = false;
    uint8_t reg = 0;
    bool valid = false;
};

// =============================================================================
// CollectorUnit: 操作数收集单元 (03 册 §6.5) ★ Operand Collector
// -----------------------------------------------------------------------------
// 操作数收集器是 GPU 的关键设计: 多个 warp 共享少量读端口以节省面积。
//   - NutShellGPU 每个 NSM 有 8 个 CollectorUnit
//   - 每个 CollectorUnit 一次最多收集 4 个操作数
//   - 通过轮询机制共享 4 个读端口 (避免每个 warp 独占端口)
//
// 工作流程:
//   1. Issue 段选定一条指令, 分配到一个空闲 CollectorUnit
//   2. CollectorUnit 记录指令的所有源操作数 (最多 4 个)
//   3. 在 Read 段, 每周期尝试读 1 个未就绪的操作数
//   4. 所有操作数就绪后, 指令进入对应执行单元 (SP/SFU/DP/LSU/BRU)
//
// 字段说明:
//   valid      - 该 Collector 是否在使用
//   warp       - 所属 warp 索引
//   pc         - 指令地址
//   exec_mask  - SIMT 活跃掩码 (哪些 lane 参与执行)
//   op         - 操作码 (用于路由到执行单元)
//   unit       - 执行单元类型 (SP/SFU/DP/LSU/BRU)
//   ops[]      - 4 个源操作数表项
//     regnum   - 寄存器编号
//     is_pred  - 是否谓词寄存器
//     ready    - 操作数是否已就绪
//   n_ops      - 实际操作数个数
//   dst_reg    - 目的寄存器编号 (用于写回)
//   dst_is_pred - 目的是否谓词
//   warp_slot - 在 NSM 内的 warp 槽位编号 (写回时定位寄存器堆)
// =============================================================================
// 教材引用: 第 3 章 §3.3.1 Operand Collector (p.38) — CollectorUnit 多 warp 共享少量读端口节省面积, 收集 4 个源操作数后送执行单元
struct CollectorUnit {
    bool valid = false;
    int warp = -1;
    uint64_t pc = 0;
    uint32_t exec_mask = 0;
    uint8_t op = 0;
    ExecUnit unit = EU_SP;
    struct Op {
        uint8_t regnum = 0;
        bool is_pred = false;
        bool ready = false;
    };
    // 1.1: 存储/宽指令可能超过四个源, 不能截断依赖和 bank 访问。
    std::array<Op, 12> ops;
    Inst inst{};
    int n_ops = 0;
    uint8_t dst_reg = 0;
    bool dst_is_pred = false;
    int warp_slot = -1;
};

// =============================================================================
// ExecPipe: 执行单元流水线 (03 册 §6.6) ★ 流水化执行
// -----------------------------------------------------------------------------
// NutShellGPU NSM 内有 5 类执行单元, 每类有自己的流水线:
//   SP  (Single Precision) - 单精度浮点/整数 ALU, latency=4, interval=1
//   SFU (Special Function Unit) - 特殊函数 (sin/cos/rsq/rcp), latency=4, interval=4
//   DP  (Double Precision) - 双精度浮点, latency=8, interval=2
//   BRU (BRanch Unit)       - 控制流, latency=1, interval=1
//   LSU (Load Store Unit)   - 内存访问, latency=4+ (依赖缓存命中)
//
// 字段说明:
//   slots[]      - 流水线槽位, 每个槽位容纳一条在途指令
//     busy       - 槽位是否被占用
//     warp       - 所属 warp
//     pc         - 指令地址 (用于结果路由)
//     remaining  - 距离结果就绪剩余周期数 (倒计时到 0 时完成)
//     exec_mask  - SIMT 掩码 (写回时知道哪些 lane 有效)
//     dst_reg    - 目的寄存器编号
//     dst_is_pred - 目的是否谓词
//     warp_slot  - warp 槽位 (写回定位寄存器堆)
//     result_lane[32]  - 每 lane 的结果值 (SIMT 32 lane 并行)
//     result_pred[32]  - 每 lane 的谓词结果
//     is_pred    - 结果是谓词还是普通寄存器
//   latency      - 该单元的执行延迟 (周期数)
//   interval     - 发射间隔 (流水化程度), 1=全流水, latency=非流水
//   busy_until   - 下次可发射周期 (用于 interval 控制)
//
// 教学要点:
//   - latency vs interval: latency 是单条指令用时, interval 是连续发射间隔
//   - interval=1 表示全流水化 (每周期可发新指令), 高吞吐
//   - interval=latency 表示非流水化 (一条完成才能发下一条), 低吞吐
//   - SFU 通常非流水 (interval=latency=4), 因为查找表读取慢
// =============================================================================
// 教材引用: 第 3 章 §3.3 Three-Loop Approximation (p.35) — ExecPipe 流水化执行单元, 区分 latency(单条指令用时)/interval(连续发射间隔), 5 类 EU 对应教材表 3.1
struct ExecPipe {
    struct Slot {
        bool busy = false;
        int warp = -1;
        uint64_t pc = 0;
        Inst inst{};
        uint64_t ready_cycle = 0; // 1.1: 到期后才允许提交功能语义
        int remaining = 0;    // 距离结果就绪剩余周期
        uint32_t exec_mask = 0;
        uint8_t dst_reg = 0;
        bool dst_is_pred = false;
        int warp_slot = -1;
        uint32_t result_lane[32] = {};  // 每 lane 的寄存器结果
        bool result_pred[32] = {};       // 每 lane 的谓词结果
        bool is_pred = false;
    };
    std::vector<Slot> slots;
    int latency;
    int interval;
    uint64_t busy_until = 0;   // 下次可发射周期 (interval 节流)

    void init(int lat, int iv, int n_slots) {
        latency = lat; interval = iv; busy_until = 0;
        slots.assign(n_slots, Slot{});
    }
};

// =============================================================================
// L1DCache: L1 数据缓存 (04 册 §3) ★ 一级缓存
// -----------------------------------------------------------------------------
// 每个 NSM 内置一个 L1D Cache, 用于缓存全局/局部/共享内存数据:
//   - 默认 16KB, 4 路, 128B 行, 32B 扇区 (可配置)
//   - 一个 cache line = 4 个 sector (每 sector 32B)
//   - 扇区化设计: 部分有效, 节省带宽
//
// 字段说明:
//   Line          - cache 行
//     valid       - 行是否有效
//     dirty       - 是否被写过 (写回策略)
//     tag         - 行的标签 (地址高位)
//     sector_valid - 4 位扇区有效掩码 (bit i = sector i 是否已填充)
//     lru         - LRU 计数器 (同 set 内最近最少使用)
//   size          - 总容量 (字节)
//   assoc         - 相联度 (路数)
//   line_bytes    - 每行字节数
//   sector_bytes - 每扇区字节数
//   n_sets        - 组数 = size / (assoc × line_bytes)
//   sets          - 缓存数据: sets[set_index][way_index]
//   mshr_entries  - MSHR (Miss Status Holding Register) 数量
//   wdb_entries   - 写数据缓冲 (Write Data Buffer) 数量
//
// 关键方法:
//   l1_set(addr) - 哈希映射地址到 set 索引 (XOR 索引减少冲突)
//   tag(addr)    - 提取地址的标签 (高 7 位以上, 因为 line=128B=2^7)
//
// 教学要点:
//   - 扇区化 (sector): 一个 line 可部分有效, 未命中只需填充缺失扇区
//   - MSHR: 多个未命中合并到同一行, 避免重复 DRAM 请求
//   - XOR 索引: (addr>>7) ^ (addr>>13) ^ (addr>>19), 减少连续地址冲突
// =============================================================================
// 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — L1DCache 一级数据缓存, 扇区化设计部分有效节省带宽, MSHR 合并未命中避免重复 DRAM 请求
struct L1DCache {
    struct Line {
        bool valid = false;
        bool dirty = false;
        uint64_t tag = 0;
        uint8_t sector_valid = 0;   // 4 位扇区有效掩码
        uint8_t lru = 0;
    };
    int size = 16384;       // 16KB
    int assoc = 4;          // 4 路组相联
    int line_bytes = 128;   // 128B/行
    int sector_bytes = 32;  // 32B/扇区
    int n_sets;
    std::vector<std::vector<Line>> sets;
    int mshr_entries = 32;  // MSHR 数量 (未命中合并)
    int wdb_entries = 16;   // 写数据缓冲数量

    void init(int sz, int as, int lb, int sb) {
        size = sz; assoc = as; line_bytes = lb; sector_bytes = sb;
        n_sets = size / (assoc * line_bytes);
        sets.resize(n_sets, std::vector<Line>(assoc));
    }

    // XOR 索引: 减少连续地址冲突 (相比直接取模)
    uint32_t l1_set(uint64_t addr) const {
        return (uint32_t)(((addr >> 7) ^ (addr >> 13) ^ (addr >> 19)) & (n_sets - 1));
    }
    // 行标签: 地址右移 7 位 (因为 line=128B=2^7)
    uint64_t tag(uint64_t addr) const {
        return addr >> 7;
    }
};

// =============================================================================
// L2Slice: L2 缓存切片 (04 册 §4) ★ 二级缓存
// -----------------------------------------------------------------------------
// L2 缓存被划分为多个切片 (slice), 分布在内存分区 (MP) 中:
//   - NutShellGPU 有 6 个 MP, 每个 MP 含 2 个 L2 切片, 共 12 切片
//   - 每切片默认 128KB, 8 路, 128B 行
//   - 切片间地址交错, 实现并行访问与负载均衡
//
// 字段说明:
//   Line          - L2 行 (比 L1 简单, 无 LRU 字段, 用 RRIP 替换)
//     valid, dirty, tag, sector_valid - 同 L1
//   size, assoc, line_bytes, n_sets, sets - 同 L1
//   mshr          - MSHR 数量 (L2 通常更多, 因为并发未命中多)
//
// 教学要点:
//   - L2 是共享缓存 (所有 NSM 共享), 通过 NoC 访问
//   - 切片化 (banked): 多个切片并行服务不同地址, 提高带宽
//   - 地址到切片的映射: addr_to_l2slice() 用哈希函数分散
// =============================================================================
// 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — L2Slice 切片分布于内存分区, 地址交错实现并行访问与负载均衡, 通过 NoC 共享给所有 NSM
struct L2Slice {
    struct Line {
        bool valid = false;
        bool dirty = false;
        uint64_t tag = 0;
        uint8_t sector_valid = 0;
    };
    int size = 131072;      // 128KB/切片
    int assoc = 8;          // 8 路组相联
    int line_bytes = 128;
    int n_sets;
    std::vector<std::vector<Line>> sets;
    int mshr = 64;          // 64 个 MSHR (L2 并发未命中多)

    void init(int sz, int as, int lb) {
        size = sz; assoc = as; line_bytes = lb;
        n_sets = size / (assoc * line_bytes);
        sets.resize(n_sets, std::vector<Line>(assoc));
    }
};

// =============================================================================
// DRAMBank: DRAM 储存块状态 (04 册 §5) ★ DRAM 行缓冲
// -----------------------------------------------------------------------------
// DRAM 被划分为多个 bank, 每个 bank 内部有行缓冲 (row buffer):
//   - 命中行缓冲: 仅需列访问 (CL), 延迟低 (~10ns)
//   - 未命中行缓冲: 需激活+列访问 (ACT+CL), 延迟高 (~50ns)
//   - 关闭行缓冲: 需预充电+激活+列访问 (PRE+ACT+CL), 延迟最高
//
// 状态机 (FR-FCFS 调度基础):
//   IDLE     - 空闲, 行缓冲关闭, 可接受新请求
//   ACTIVING - 正在激活行 (等待 tRCD), 即将进入 ACTIVE
//   ACTIVE   - 行已打开, 可读/写
//   READING  - 正在读 (等待 CL+tBD)
//   WRITING  - 正在写 (等待 CL+WR+tBD)
//   PRECHG   - 正在预充电 (等待 tRP), 即将回到 IDLE
//
// 字段说明:
//   state  - 当前状态
//   row    - 行缓冲中打开的行号 (-1 表示无行打开)
//   timer  - 距离状态转换剩余周期数 (倒计时)
//
// 教学要点:
//   - 行缓冲命中 (row hit): 已打开的行再次被访问, 延迟最低
//   - 行缓冲冲突 (row conflict): 需关闭旧行打开新行, 延迟最高
//   - FR-FCFS 调度: 优先调度行缓冲命中的请求 (即使非先到)
// =============================================================================
// 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — DRAMBank 行缓冲状态机, 行命中低延迟/行冲突高延迟, 是 FR-FCFS 调度策略的基础
struct DRAMBank {
    enum State { IDLE, ACTIVING, ACTIVE, READING, WRITING, PRECHG };
    State state = IDLE;
    int row = -1;       // 当前打开的行号
    int timer = 0;      // 状态转换倒计时
};

// =============================================================================
// DRAMChannel: DRAM 通道 (04 册 §5) ★ DRAM 控制器
// -----------------------------------------------------------------------------
// 一个 DRAM 通道包含多个 bank, 共享地址/数据总线:
//   - NutShellGPU 有 6 个 DRAM 通道 (可配置)
//   - 每通道默认 8 个 bank
//   - 通道内 bank 间可并行 (行缓冲独立), 但总线串行
//
// 字段说明:
//   banks             - 该通道下所有 bank 状态
//   Req               - 待处理请求
//     addr, is_write, bank, row, col, slice, subid, nsm, is_read - 请求元数据
//   read_queue        - 读请求队列 (FR-FCFS 调度)
//   write_queue       - 写请求队列 (写回策略)
//   reads_since_drain - 距上次写排空的读数 (写饥饿防止)
//   data_bus_busy     - 数据总线是否占用
//   data_bus_timer    - 总线占用剩余周期
//
// 调度策略 (FR-FCFS, First-Ready First-Come-First-Served):
//   1. 优先调度行缓冲命中的请求 (READY)
//   2. 同等优先级按先到先服务 (FCFS)
//   3. 定期排空写队列防止写饥饿
//
// 教学要点:
//   - 通道 vs bank: 通道是物理边界, bank 是逻辑分组
//   - 行缓冲命中率是 DRAM 性能的关键指标
//   - FR-FCFS 是 GPU/CPU 通用 DRAM 调度策略
// =============================================================================
// 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — DRAMChannel 通道含多 bank 共享总线, FR-FCFS 优先调度行缓冲命中请求, 定期排空写队列防饥饿
struct DRAMChannel {
    std::vector<DRAMBank> banks;
    struct Req {
        uint64_t addr;
        bool is_write;
        int bank;
        int row;
        int col;
        int slice;
        int subid;
        int nsm;
        bool is_read;
    };
    std::deque<Req> read_queue;     // 读请求 (优先调度行命中)
    std::deque<Req> write_queue;    // 写请求 (批量回写)
    int reads_since_drain = 0;      // 防止写饥饿的计数器
    bool data_bus_busy = 0;          // 数据总线占用标记
    int data_bus_timer = 0;         // 总线占用剩余周期

    void init(int n_banks) {
        banks.resize(n_banks);
    }
};

// =============================================================================
// Flit: NoC 虚单元 (04 册 §7) ★ 片上网络包
// -----------------------------------------------------------------------------
// NoC (Network on Chip) 用 flit (flow control unit) 传输消息:
//   - NutShellGPU 用虫孔路由 (wormhole): 大包切分为多个 flit 顺序传输
//   - 每 flit 32B (一个 sector 大小)
//   - 虚通道 (VC) 避免死锁: 请求/响应/原子/失效分别走不同 VC
//
// 类型:
//   REQ  - 请求 (NSM → L2/DRAM): 读/写/原子
//   RSP  - 响应 (L2/DRAM → NSM): 数据填充
//   ATOM - 原子操作 (带操作码的读改写)
//   INV  - 失效 (用于缓存一致性, NutShellGPU 简化为写回无效)
//
// 字段说明:
//   type     - 包类型 (决定虚通道)
//   src, dst - 源/目的节点编号
//   addr     - 内存地址
//   mp, slice, subid - 内存分区/切片/子 ID (路由与响应匹配)
//   sectors  - 涉及的扇区数 (1-4)
//   cmd      - 命令码 (读/写/原子等)
//   n_flit   - 本包占用的 flit 数 (1-5, 取决于数据大小)
//   gen      - 生成周期 (用于超时统计)
//   data[32] - 32B 数据负载 (一个 sector)
//   is_write - 读/写标记
//   nsm      - 源 NSM 编号 (响应路由)
//
// 教学要点:
//   - NoC 是 GPU 内存带宽的关键: 6 NSM × 多个 MP 切片
//   - 虚通道避免协议死锁: 请求/响应分离
//   - 虫孔路由: 头 flit 建路, 体 flit 跟随, 尾 flit 释放
// =============================================================================
// 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — Flit 是 NoC 虫孔路由基本单元, 虚通道 (REQ/RSP/ATOM/INV) 分离避免协议死锁
struct Flit {
    enum Type { REQ, RSP, ATOM, INV };
    Type type = REQ;
    int src = 0, dst = 0;
    uint64_t addr = 0;
    int mp = 0, slice = 0, subid = 0;
    uint8_t sectors = 0;
    uint8_t cmd = 0;
    int n_flit = 1;
    int gen = 0;
    uint8_t data[32] = {};
    bool is_write = false;
    int nsm = 0;
};

// =============================================================================
// MemReq: 待处理内存请求 (04 册 §6) ★ 内存合并请求
// -----------------------------------------------------------------------------
// 一条 warp 指令的内存访问经过合并 (coalescing) 后, 生成一个或多个 MemReq:
//   - 合并: warp 内 32 lane 的地址, 落在同一 sector 的合并为一个请求
//   - 路由: 全局内存走 NoC→L2→DRAM, 共享内存走 bank 直接访问
//
// 字段说明:
//   nsm, warp, lane        - 请求来源 (用于响应路由)
//   addr                   - 物理地址
//   space                  - 内存空间 (GLOBAL/LOCAL/SHARED/CONST)
//   width                  - 访问宽度 (4B/8B/16B)
//   is_store               - 读/写
//   is_atom, atom_op       - 原子操作标记与类型
//   data, cmp              - 写入数据, CAS 比较值
//   subid                  - 子请求 ID (一条指令可生成多个 MemReq)
//   prt_entry              - 在 PRT 中的索引 (未命中合并)
//   vline                  - 虚拟 cache 行 (合并键)
//   need_sectors           - 需要填充的扇区掩码 (4 位)
//   dst_reg, dst_is_pred   - 完成后写回的目的寄存器
//   pc, exec_mask          - 指令上下文
//   lanes                  - 该请求合并了哪些 lane (用于写回结果分发)
//
// 教学要点:
//   - 内存合并 (coalescing) 是 GPU 内存优化的核心
//   - 32 lane 访问若连续, 合并为 1-4 个请求; 若随机, 最多 32 个请求
//   - 合并好 → 带宽高; 合并差 → 带宽浪费, 性能下降
// =============================================================================
// 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — MemReq 是 warp 内存访问经合并 (coalescing) 后的请求, 32 lane 同 sector 合并以提升带宽利用率
struct MemReq {
    int nsm;
    int warp;
    int lane;
    uint64_t addr;
    uint8_t space;
    uint8_t width;
    bool is_store;
    bool is_atom;
    uint8_t atom_op;
    uint32_t data;
    uint32_t cmp;
    int subid;
    int prt_entry;
    uint64_t vline;        // 虚拟行 (合并键)
    uint8_t need_sectors;  // 待填充扇区掩码
    int dst_reg;
    bool dst_is_pred;
    uint64_t pc;
    uint32_t exec_mask;
    std::vector<int> lanes;  // 合并的 lane 列表 (用于写回)
};

// =============================================================================
// PRTEntry: 待处理请求表项 (04 册 §3.3) ★ PRT
// -----------------------------------------------------------------------------
// PRT (Pending Request Table) 用于 L1 未命中合并:
//   - 当多个 warp 请求同一 cache 行, 只发送一次到 L2
//   - 后续请求 "挂" 在 PRT 表项的 waiters 列表, 等待同一填充
//   - 填充到达后, 广播给所有 waiters
//
// 字段说明:
//   valid         - 该表项是否在使用
//   vline         - 虚拟行 (地址归一化后的键, 用于匹配新请求)
//   need_sectors  - 仍需填充的扇区掩码 (可能多次部分填充)
//   busy_way      - 该行将被填入的 cache 路 (防止替换自身)
//   waiters       - 等待该行的所有 MemReq (挂起队列)
//   subid         - 子请求 ID (对应 L2 请求)
//   state         - 0=SENT (已发往 L2), 1=FILLING (正在被填充)
//
// 教学要点:
//   - PRT 类似 MSHR, 用于未命中合并
//   - 合并减少 L2/DRAM 流量, 提高带宽利用率
//   - 当 PRT 满 (32 项全占用), 新请求需 replay (重新发射)
// =============================================================================
// 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — PRTEntry 类似 MSHR, L1 未命中合并到同一行只发一次 L2 请求, 填充到达后广播给所有 waiters
struct PRTEntry {
    bool valid = false;
    uint64_t vline = 0;
    uint8_t need_sectors = 0;
    int busy_way = -1;
    std::vector<MemReq> waiters;  // 等待该行填充的请求列表
    int subid = 0;
    int state = 0;  // 0=SENT (已发往 L2), 1=FILLING (正在被填充)
};

// =============================================================================
// WDBEntry: 写数据缓冲表项 (04 册 §3.4) ★ WDB
// -----------------------------------------------------------------------------
// WDB (Write Data Buffer) 缓存写操作, 支持合并写:
//   - 多次写同一行的不同 sector 可在 WDB 中合并
//   - 当 WDB 满, 必须驱逐 (evict) 一个表项到 L2
//   - 驱逐策略: LRU
//
// 字段说明:
//   valid       - 该表项是否在使用
//   addr        - 行对齐地址
//   data[128]   - 128B 行数据 (4 个 sector)
//   byte_en[16] - 字节使能 (16 个 8B 字段, 标记哪些字节有效)
//   warp        - 来源 warp (调试用)
//   space       - 内存空间 (GLOBAL/LOCAL)
//
// 教学要点:
//   - WDB 减少 L2 写流量: 多次写合并为一次
//   - 字节使能支持部分写 (sub-word write)
//   - 当 WDB 满, 新写必须等待驱逐, 引起 stall
// =============================================================================
// 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — WDBEntry 写数据缓冲, 多次写同一行不同 sector 在 WDB 合并, 减少向 L2 的写流量
struct WDBEntry {
    bool valid = false;
    uint64_t addr;
    uint8_t data[128];        // 128B 行数据 (4 sector)
    uint8_t byte_en[16];      // 16 个 8B 字段使能
    int warp;
    uint8_t space;
};

// =============================================================================
// CycStats: 周期级性能统计 (05 册 §4) ★★ 性能指标
// -----------------------------------------------------------------------------
// 模拟器运行过程中累计所有性能指标, 用于分析瓶颈:
//
//   1. 整体指标:
//      - total_cycles: 总运行周期数
//      - instructions: 执行的指令数 (1 条 warp 指令 = 1)
//      - ipc: 指令/周期 (越大越好, 理想=1)
//      - simd_efficiency: 活跃 lane / 32 (反映发散损失)
//      - occupancy_avg_warps: 平均活跃 warp 数
//
//   2. 停顿原因分解 (stall_breakdown):
//      - no_collector: 无空闲操作数收集器
//      - sb_wait: 记分牌等待 (数据依赖)
//      - barrier_wait: 屏障等待
//      - lsu_full: LSU 队列满
//      - icache_miss: I-Cache 未命中
//      - no_eligible: 无可发射 warp (全部在等待)
//
//   3. 指令分布:
//      - inst_issued: 每种操作码的发射次数 (映射)
//
//   4. SIMT 栈:
//      - simt_divergent_branch: 发散分支数
//      - simt_stack_maxdepth: 栈最大深度
//
//   5. 重演 (replays):
//      - smem_conflict: 共享内存 bank 冲突
//      - l1_miss: L1 未命中 (需 replay)
//      - prt_full: PRT 满引起 replay
//      - assoc_stall: 相联度 stall (set 满无法插入)
//      - wdb_full: WDB 满
//
//   6. 执行单元占用:
//      - rf_bank_conflict_cycles: 寄存器堆 bank 冲突
//      - sp_busy_cycles, sfu_busy_cycles, dp_busy_cycles: 各单元忙周期
//
//   7. 缓存与内存:
//      - smem_accesses, smem_conflict_replays: 共享内存统计
//      - l1_access_lines, l1_miss_sectors, l1_assoc_stall: L1 统计
//      - prt_full, wdb_full: 资源占用
//      - l2_accesses, l2_hits, l2_misses: L2 统计
//      - dram_reads, dram_writes, dram_row_hits, dram_row_misses: DRAM 统计
//      - bytes_to_dram, bytes_from_dram: DRAM 带宽
//      - noc_flits_sent, noc_stall_cycles: NoC 统计
//
// 计算辅助:
//   ipc():                        - instructions / total_cycles
//   simd_efficiency():            - active_lane_sum / (32 × instructions)
//   dram_effective_bw_gbps():     - (bytes_to_dram + bytes_from_dram) / cycles
//   row_buffer_hit_rate():        - dram_row_hits / (dram_row_hits + dram_row_misses)
//   to_json():                    - 序列化为 JSON 字符串
// =============================================================================
// 教材引用: 第 5 章 §5.1 Analyzing a GPU Microarchitecture (p.119) — CycStats 累计性能指标: IPC/SIMD 效率/停顿分解/DRAM 命中, 用于分析 GPU 微架构瓶颈
struct CycStats {
    uint64_t total_cycles = 0;       // 总周期数
    uint64_t issue_cycles = 0;       // 成功发射周期
    uint64_t stall_cycles = 0;       // 停顿周期
    uint64_t stall_no_collector = 0;
    uint64_t stall_sb_wait = 0;       // 记分牌等待
    uint64_t stall_barrier_wait = 0;
    uint64_t stall_lsu_full = 0;
    uint64_t stall_icache_miss = 0;
    uint64_t stall_no_eligible = 0;
    std::unordered_map<int, uint64_t> inst_issued;   // 操作码→次数
    uint64_t simt_divergent_branch = 0;
    int simt_stack_maxdepth = 0;
    uint64_t replays_smem_conflict = 0;
    uint64_t replays_l1_miss = 0;
    uint64_t replays_prt_full = 0;
    uint64_t replays_assoc_stall = 0;
    uint64_t replays_wdb_full = 0;
    uint64_t rf_bank_conflict_cycles = 0;
    uint64_t rf_conflict_stalls = 0, writeback_stalls = 0, pipeline_full_stalls = 0;
    uint64_t l1_hits = 0, l1_pending_merges = 0, memory_sectors = 0;
    uint64_t serial_dependency_stalls = 0, unsupported_timing_ops = 0;
    double core_clock_ghz = 1.0;
    uint64_t sp_busy_cycles = 0;
    uint64_t sfu_busy_cycles = 0;
    uint64_t dp_busy_cycles = 0;
    uint64_t active_lane_sum = 0;     // 活跃 lane 总数 (用于 SIMD 效率)
    uint64_t smem_accesses = 0;
    uint64_t smem_conflict_replays = 0;
    uint64_t l1_access_lines = 0;
    uint64_t l1_miss_sectors = 0;
    uint64_t l1_assoc_stall = 0;
    uint64_t prt_full = 0;
    uint64_t wdb_full = 0;
    uint64_t l2_accesses = 0;
    uint64_t l2_hits = 0;
    uint64_t l2_misses = 0;
    uint64_t dram_reads = 0;
    uint64_t dram_writes = 0;
    uint64_t dram_row_hits = 0;
    uint64_t dram_row_misses = 0;
    uint64_t dram_bank_busy_cycles = 0;
    uint64_t dram_bus_busy_cycles = 0;
    uint64_t bytes_to_dram = 0;
    uint64_t bytes_from_dram = 0;
    uint64_t noc_flits_sent = 0;
    uint64_t noc_stall_cycles = 0;
    uint64_t instructions = 0;        // 总指令数
    double occupancy_avg_warps = 0;      // 平均活跃 warp

    // IPC: 每周期指令数 (越大越好, 理想=1)
    double ipc() const {
        return total_cycles > 0 ? (double)instructions / total_cycles : 0;
    }
    // SIMD 效率: 活跃 lane 占比 (反映发散损失, 越接近 1 越好)
    double simd_efficiency() const {
        return instructions > 0 ? (double)active_lane_sum / (32.0 * instructions) : 0;
    }
    // DRAM 有效带宽 (字节/秒)
    double dram_effective_bw_gbps() const {
        if (total_cycles == 0) return 0;
        return (double)(bytes_to_dram + bytes_from_dram) / total_cycles;
    }
    // 行缓冲命中率 (反映 DRAM 访问模式优劣)
    double row_buffer_hit_rate() const {
        auto total = dram_row_hits + dram_row_misses;
        return total > 0 ? (double)dram_row_hits / total : 0;
    }

    std::string to_json() const;   // 序列化为 JSON (实现在 .cpp)
};

// =============================================================================
// CycWarpState: 周期级 warp 状态 (03 册 §6.7) ★ 每 warp 时序状态
// -----------------------------------------------------------------------------
// 在功能模拟器 WarpState 基础上, 周期级模拟器额外维护每个 warp 的:
//   - I-Buffer 内容 (2 项)
//   - 记分牌表项 (4 项)
//   - 取指状态 (是否在取指, 取指 PC, MSHR 等待)
//   - 内存请求在途计数 (vmem/smem)
//
// 字段说明:
//   func_warp          - 指向功能模拟器中的 WarpState (复用架构状态)
//   ibuf[2]            - 2 项 I-Buffer
//   sb[4]              - 4 项记分牌
//   fetch_wait_mshr    - 等待的 MSHR ID (-1 表示未等待)
//   is_fetching        - 是否正在取指
//   fetch_pc           - 正在取的 PC
//   fetch_timer        - 取指剩余周期 (I-Cache 命中延迟)
//   issue_priority_boost - YIELD 指令的优先级提升计数
//   vmem_inflight      - 全局/局部内存在途请求数
//   smem_inflight      - 共享内存在途请求数
//
// 方法:
//   ibuf_full()  - I-Buffer 是否已满 (2 项都有效)
//   ibuf_empty() - I-Buffer 是否为空
//   ibuf_head()  - 取最早的 I-Buffer 表项
// =============================================================================
// 教材引用: 第 3 章 §3.3.2 Pipeline (p.41) — CycWarpState 每 warp 时序状态, 在功能 WarpState 上叠加 I-Buffer/记分牌/取指状态, 驱动流水线推进
struct CycWarpState {
    WarpState* func_warp = nullptr;    // 指向功能模拟器的 WarpState
    std::array<IBufEntry, 2> ibuf;     // 2 项 I-Buffer
    std::array<SBEntry, 4> sb;          // 4 项记分牌
    int fetch_wait_mshr = -1;           // 等待的 MSHR (-1=无)
    // 1.1: 顺序互锁持有到写回, 覆盖 RAW/WAW/WAR、谓词和宽寄存器依赖。
    // 当前不支持同 warp 指令级并行; 跨 warp 流水重叠保持有效。
    bool pending_commit = false;
    bool is_fetching = false;
    uint64_t fetch_pc = 0;
    int fetch_timer = 0;
    int issue_priority_boost = 0;      // YIELD 指令的优先级

    int vmem_inflight = 0;             // 全局内存在途请求
    int smem_inflight = 0;             // 共享内存在途请求

    bool ibuf_full() const { return ibuf[0].valid && ibuf[1].valid; }
    bool ibuf_empty() const { return !ibuf[0].valid && !ibuf[1].valid; }
    IBufEntry& ibuf_head() { return ibuf[0].valid ? ibuf[0] : ibuf[1]; }
};

// =============================================================================
// CycleSim: 周期级模拟器主类 (05 册 §3) ★★★ 模拟器入口
// -----------------------------------------------------------------------------
// CycleSim 组合了所有时序组件, 提供 init/run/tick 三个核心方法:
//   - init(): 配置所有组件 (缓存/DRAM/流水线/NoC)
//   - run(): 主循环, 反复调用 tick() 直到所有内核完成
//   - tick(): 推进一个时钟周期, 处理所有组件
//
// 内部结构:
//   cfg          - 全局配置 (来自 Config 结构)
//   func         - 功能模拟器实例 (复用其架构状态)
//   nsm_cyc[]    - 每个 NSM 的周期级状态 (流水线/缓存/PRT/WDB)
//   l2_slices[]  - 12 个 L2 切片
//   dram_channels[] - 6 个 DRAM 通道
//   noc_req_vc[] - NoC 请求虚通道 (每个输入端口一个队列)
//   noc_rsp_vc[] - NoC 响应虚通道
//   pending_l2_resps - 待路由回 NSM 的 L2 响应
//   stats       - 性能统计
//
// 内部结构 NSMCycState (每 NSM):
//   cyc_warps[MAX_WARPS_PER_SM] - 每 warp 周期状态
//   collectors[8]                - 8 个操作数收集器
//   sp_pipe, sfu_pipe, dp_pipe, bru_pipe, lsu_pipe - 5 类执行流水线
//   l1d                          - L1 数据缓存
//   prt[32]                      - 32 项 PRT
//   wdb[16]                      - 16 项 WDB
//   issue_cursor                 - 发射轮询游标 (round-robin)
//   fetch_cursor                 - 取指轮询游标
//   n_active_warps               - 当前活跃 warp 数 (占用率)
//
// 阶段处理器 (每个 tick 调用一次):
//   fetch_tick     - 取指阶段
//   decode_tick    - 译码阶段
//   issue_tick     - 发射阶段 (含记分牌检查)
//   collector_tick - 操作数收集阶段
//   execute_tick   - 执行阶段 (推进流水线)
//   writeback_tick - 写回阶段 (释放记分牌)
//
// 内存系统 (每个 tick 调用一次):
//   l1d_tick       - L1 缓存处理
//   noc_tick       - NoC 路由
//   l2_tick        - L2 缓存处理
//   dram_tick      - DRAM 调度
//   cta_retire_tick - CTA 完成回收
//
// 辅助方法:
//   addr_to_mp(addr)           - 地址到内存分区映射
//   addr_to_l2slice(addr)      - 地址到 L2 切片映射
//   dram_addr_decode(addr, ...) - DRAM 地址解码 (channel/bank/row/col)
//
// 教学要点:
//   - 周期级模拟器是 "时间驱动" 的: 每周期推进所有组件
//   - 与事件驱动模拟器不同, 实现简单但效率较低
//   - 适合教学: 直观反映流水线/缓存/DRAM 时序
// =============================================================================
// 教材引用: 第 3 章 §3.3 Three-Loop Approximation (p.35) — CycleSim 完整周期模型三层循环: 外层 kernel/中层 cycle/内层流水线阶段, 组合流水线+记分牌+操作数采集+执行单元
class CycleSim {
public:
    Config cfg;
    FuncSim func;   // 复用功能模拟器 (架构状态)

    // 每 NSM 的周期级状态 (流水线/缓存/资源表)
// 教材引用: 第 3 章 §3.3 Three-Loop Approximation (p.35) — NSMCycState 每 NSM 时序状态, 5 类执行流水线+L1D+PRT+WDB, 对应教材完整周期模型的所有时序组件
    struct NSMCycState {
        std::array<CycWarpState, MAX_WARPS_PER_SM> cyc_warps;   // 每 warp 周期状态
        std::array<CollectorUnit, 8> collectors;                 // 8 个操作数收集器
        ExecPipe sp_pipe;       // 单精度 ALU 流水线
        ExecPipe sfu_pipe;      // 特殊函数单元流水线
        ExecPipe dp_pipe;       // 双精度流水线
        ExecPipe bru_pipe;      // 分支单元流水线
        ExecPipe lsu_pipe;      // 访存单元流水线
        L1DCache l1d;           // L1 数据缓存
        std::array<PRTEntry, 32> prt;   // 32 项 PRT (未命中合并)
        std::array<WDBEntry, 16> wdb;   // 16 项 WDB (写合并)
        TimingCache timing_l1;
        std::vector<uint64_t> mshr_ready;
        int collector_cursor = 0;
        int issue_cursor = 0;           // 发射轮询游标 (round-robin)
        int fetch_cursor = 0;           // 取指轮询游标
        int n_active_warps = 0;         // 活跃 warp 数 (占用率)
    };
    std::vector<NSMCycState> nsm_cyc;   // 所有 NSM 的周期状态

    // L2 切片 (12 个: 6 MP × 2 切片)
    std::vector<L2Slice> l2_slices;

    // DRAM 通道 (6 个)
    std::vector<DRAMChannel> dram_channels;

    // NoC 虚通道 (每输入端口一个队列)
    std::vector<std::deque<Flit>> noc_req_vc;   // 请求虚通道
    std::vector<std::deque<Flit>> noc_rsp_vc;   // 响应虚通道

    // 待路由回 NSM 的 L2 响应 (中介队列, 避免环依赖)
    std::vector<Flit> pending_l2_resps;

    // 性能统计
    uint64_t occupancy_sum = 0;
    CycStats stats;
    // 1.1: 有限资源预约。NoC 为按分区排队的链路带宽模型, 非逐 flit 网络。
    std::vector<TimingCache> timing_l2;
    std::vector<std::vector<uint64_t>> l2_mshr_ready;
    std::vector<uint64_t> noc_ready, dram_bus_ready, dram_bank_ready, dram_open_row;
    uint64_t memory_ready(NSMState&, NSMCycState&, const Inst&, uint32_t, int);
    void validate_timing_config() const;

    // 教材引用: 第 3 章 §3.3 Three-Loop Approximation (p.35) — init/run/tick 三层循环驱动: 外层遍历 kernel, 中层每周期推进, 内层调用各阶段处理器
    // ==================== 核心方法 ====================

    void init();                          // 初始化所有组件
    int run(uint64_t max_cycles = 0);     // 主循环 (0=无限制)
    void tick();                          // 推进一个时钟周期

    // 教材引用: 第 3 章 §3.3.2 Pipeline (p.41) — 流水线阶段处理器, tick 顺序: collector→fetch→issue→writeback, 记分牌检查在取指前 (教材图 3.11)
    // ==================== 阶段处理器 ====================

    void fetch_tick(NSMState& sm, NSMCycState& cyc);      // 取指
    void decode_tick(NSMState& sm, NSMCycState& cyc);     // 译码
    void issue_tick(NSMState& sm, NSMCycState& cyc);      // 发射
    void collector_tick(NSMState& sm, NSMCycState& cyc);  // 操作数收集
    void execute_tick(NSMState& sm, NSMCycState& cyc);    // 执行
    void writeback_tick(NSMState& sm, NSMCycState& cyc); // 写回

    // 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — 内存系统层级: L1D→NoC→L2→DRAM, NoC 路由连接 NSM 与内存分区切片
    // ==================== 内存系统 ====================

    void l1d_tick(NSMState& sm, NSMCycState& cyc);   // L1 缓存处理
    void noc_tick();                                  // NoC 路由
    void l2_tick();                                   // L2 缓存处理
    void dram_tick();                                 // DRAM 调度

    // CTA 完成回收 (检查 warp 是否 EXIT, 释放资源)
    void cta_retire_tick();

    // 教材引用: 第 5 章 §5.1 Analyzing a GPU Microarchitecture (p.119) — 统计输出: collect_stats 累计指标, output_stats_json 序列化为 JSON 供性能分析
    // ==================== 统计输出 ====================

    void collect_stats();                           // 收集最终统计
    void output_stats_json(const std::string& path); // 输出 JSON 文件

    // 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — 地址映射: 地址→内存分区/L2 切片/DRAM channel-bank-row-col, 决定请求路由路径
    // ==================== 地址映射辅助 ====================

    int addr_to_mp(uint64_t addr) const;             // 地址→内存分区
    int addr_to_l2slice(uint64_t addr) const;        // 地址→L2 切片
    void dram_addr_decode(uint64_t addr, int& channel, int& bank, int& row, int& col) const;
};

} // namespace ntisa
