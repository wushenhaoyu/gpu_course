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
// func_sim.hpp - 功能模拟器头文件 (S2, 无时序模型)
// =============================================================================
// 功能模拟器是 NutShellGPU 模拟器的基础层, 实现指令的功能语义:
//   - 每条指令立即执行, 不建模流水线/缓存时序
//   - 访存操作立即完成 (0 周期)
//   - 重点关注: 正确性 (SIMT 栈/指令语义/内存系统/屏障)
//
// 周期级模拟器 (cycle_sim) 复用本文件的功能逻辑, 仅在外层加时序包装
//
// 参考: NutShellGPU_spec 全 6 册
//   01 册: 编程模型与 ABI (内核启动/坐标系统)
//   02 册: NTAS1 指令集 (指令语义)
//   03 册: NSM 微结构 (SIMT 栈/屏障/warp 调度)
//   04 册: 内存系统 (地址空间/coalescing/flat 地址)
//   05 册: 模拟器实现契约 (S2 功能模拟器接口)
// 教材引用: 第 3 章 §3.1 One-Loop Approximation (p.22) — 本文件定义功能模拟器的数据结构, 对应教材单循环调度模型; WarpState/NSMState 直接服务于 §3.1.1 SIMT 掩码、§3.1.2 寄存器堆、§3.1.4 重汇聚栈等概念
// =============================================================================
#pragma once
#include "../isa/ntisa.hpp"
#include <vector>
#include <string>
#include <array>
#include <unordered_map>
#include <cstdint>
#include <functional>

namespace ntisa {

// =============================================================================
// 全局配置参数 (00 册 §4-§8)
// -----------------------------------------------------------------------------
// 模拟器的所有可配置参数集中在此结构中
// 功能模拟器仅使用其中的一部分 (如 warp_size, max_warps_per_core)
// 周期级模拟器使用全部参数 (含流水线/缓存/NoC/DRAM 时序)
// 配置可通过 INI 文件加载 (load_ini)
// 教材引用: 第 1 章 §1.2 GPU Hardware Basics (p.2) — Config 集中描述芯片级 SM 集群拓扑 (cluster/NSM/内存分区), 对应教材 GPU 硬件基础组织
// =============================================================================
struct Config {
    // ---- [gpgpu] 芯片级配置 (00 册 §4) ----
    int n_clusters = 4;              // 芯片内 cluster 数量 (4)
    int n_cores_per_cluster = 4;     // 每 cluster 的 NSM 数 (4)
    int n_nsm() const { return n_clusters * n_cores_per_cluster; } // 总 NSM 数 (16)
    int n_mem = 6;                   // 内存分区数 (6)
    int warp_size = 32;              // warp 大小 (32 线程, SIMT 调度单元)
    int max_warps_per_core = 64;    // 每 NSM 最大并发 warp 数 (占用率上限)
    int max_ctas_per_core = 16;     // 每 NSM 最大并发 CTA 数
    int max_threads_per_cta = 1024; // 每 CTA 最大线程数

    // 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — NSM 一级内存子系统 (shared memory / L1 / constant cache / barrier slots), 对应教材 L1/shared/constant memory 层次
    // ---- [sm_mem] NSM 内存子系统 (00 册 §5, 04 册 §3) ----
    int smem_size = 49152;           // 共享内存大小 (48 KiB)
    int l1d_size = 16384;            // L1 数据缓存大小 (16 KiB)
    int smem_num_banks = 32;         // 共享内存 bank 数 (32 bank, 避免冲突)
    int smem_bank_width = 4;         // 每 bank 宽度 (4 字节)
    int l1d_line = 128;              // L1 缓存行大小 (128 字节)
    int l1d_sector = 32;             // L1 扇区大小 (32 字节)
    int l1d_assoc = 4;               // L1 组相联度 (4 路)
    int l1d_mshr = 32;               // L1 MSHR 数 (缺失状态保持寄存器)
    int l1d_wdb = 16;                // L1 写数据缓冲 (Write Data Buffer)
    int icache_size = 8192;          // 指令缓存大小 (8 KiB)
    int const_cache_size = 8192;     // 常量缓存大小 (8 KiB)
    int barrier_slots = 16;          // 屏障槽位数 (0-15)

    // 教材引用: 第 3 章 §3.1.2 Register File Organization (p.27) & 第 3 章 §3.1.3 Warp Scheduling (p.31) — NSM 流水线参数 (SP/SFU/DP lane 数与延迟、寄存器堆物理表项、操作数采集器、记分板), 对应教材 warp 调度与寄存器堆组织
    // ---- [sm_pipeline] NSM 流水线 (00 册 §5, 03 册 §6) ----
    int sp_lanes = 32, sp_latency = 4, sp_interval = 1;  // SP 单元: 32 路, 4 周期, 1 间隔
    int sfu_lanes = 16, sfu_latency = 8, sfu_interval = 2; // SFU: 16 路, 8 周期, 2 间隔
    int dp_lanes = 16, dp_latency = 8, dp_interval = 2;  // DP: 16 路, 8 周期, 2 间隔
    int branch_latency = 1;          // 分支单元延迟 (1 周期)
    int num_collector_units = 8;     // 操作数采集器数量 (8 个)
    int operand_slots_per_collector = 4; // 每采集器的操作数槽 (4 个)
    int rf_logical_banks = 4;        // 寄存器堆逻辑 bank 数
    int rf_physical_entries = 65536; // 寄存器堆物理表项数
    int ibuffer_entries_per_warp = 2; // 每 warp 的指令缓冲表项 (2)
    int scoreboard_entries_per_warp = 4; // 每 warp 记分板表项 (4)
    int icache_mshrs = 8;            // 指令缓存 MSHR 数

    // ---- [timing_v11] 1.1 执行时序 ★ 延时为可配置的模拟值 ----
    // 单位: latency/interval/ports 为周期或每周期端口数, clock 为 GHz。
    // 保留每 warp 顺序互锁, 不推测执行; 不同 warp 可同时占用流水线。
    double core_clock_ghz = 1.0;
    int move_latency = 1, integer_latency = 2, imul_latency = 4;
    int idiv_latency = 32, shuffle_latency = 4;
    int rf_read_ports_per_bank = 1, writeback_ports = 1;
    int execution_slots = 16, lsu_slots = 32;
    int atomic_latency = 16;

    // ---- [latency_ns] 各级延迟 (纳秒, 04 册 §5-§8) ----
    double lat_smem_hit = 2.0;      // 共享内存命中延迟 (2 ns)
    double lat_l1_hit = 30.0;       // L1 命中延迟 (30 ns)
    double lat_l2_hit = 180.0;      // L2 命中延迟 (180 ns)
    double lat_dram_row_hit = 400.0; // DRAM 行命中延迟 (400 ns)
    double lat_barrier_resume = 4.0; // 屏障恢复延迟 (4 ns)
    double lat_pcie_base = 10000.0; // PCIe 基础延迟 (10000 ns)
    double lat_pcie_per4k = 1.0;    // PCIe 每传输 4KB 的增量延迟

    // ---- [noc] 片上网络 (04 册 §6) ----
    int noc_flit_bytes = 8;          // NoC flit 大小 (8 字节)
    int noc_req_vc = 4;             // 请求虚通道数
    int noc_rsp_vc = 4;             // 响应虚通道数
    int noc_router_pipe = 1;        // 路由器流水线级数
    int noc_link_cycles = 1;        // 链路延迟周期数
    int noc_interleave = 256;       // 地址交织粒度

    // ---- [l2] L2 缓存 (04 册 §5) ----
    int l2_slice_size = 131072;     // L2 切片大小 (128 KiB)
    int l2_assoc = 8;               // L2 组相联度 (8 路)
    int l2_line = 128;              // L2 缓存行 (128 字节)
    int l2_sector = 32;             // L2 扇区 (32 字节)
    int l2_mshr = 64;               // L2 MSHR 数
    int l2_rop_atom_cache = 2048;   // L2 ROP 原子缓存

    // ---- [dram] DRAM 模型 (04 册 §8) ----
    int dram_channels = 6;          // DRAM 通道数 (6)
    int dram_banks_per_channel = 8;  // 每通道 bank 数 (8)
    int dram_channel_width = 32;     // 通道宽度 (32 字节)
    double dram_data_rate = 5.0;    // 数据率 (5 GT/s)
    int dram_burst_atom = 32;        // 突发原子大小 (32 字节)
    // DRAM 时序参数 (ns): tCK=周期, tCAS=列访问, tRCD=RAS到CAS, tRP=预充电
    double tCK = 0.4, tCAS = 12.0, tRCD = 12.0, tRP = 12.0;
    double tRAS = 28.0, tWR = 12.0, tRC = 40.0; // tRAS=行激活, tWR=写恢复, tRC=行周期
    int dram_write_drain = 32;       // 写排空阈值

    // 获取默认配置 (使用上述内置默认值)
    static Config default_config() { return Config{}; }
    // 从 INI 文件加载配置
    void load_ini(const std::string& path);
    // 从 INI 格式字符串加载配置
    void load_ini_string(const std::string& content);
};

// =============================================================================
// SIMT 栈条目 (03 册 §4)
// -----------------------------------------------------------------------------
// SIMT 栈用于管理分支发散后的线程重汇聚
// 每个 warp 有一个 SIMT 栈, 栈顶 (TOS) 描述当前执行的指令流
//
// 字段说明:
//   rpc:     重汇聚点 PC (Reconvergence PC) - 到此 PC 时所有发散线程重汇聚
//            值为 EXIT_SENTINEL 表示栈底 (无重汇聚点, 即 warp 入口的隐式汇聚)
//   nextpc:  下一条要执行的指令 PC (当前栈帧的 PC)
//   mask:    活动掩码 - 32 位, 每位对应一个 lane (1=活跃, 0=屏蔽)
//   dflag:   发散标志 - true 表示此栈帧是由分支发散产生的 (用于 BRA 逻辑)
//
// 栈操作:
//   SSY: 压入新栈帧, 设置重汇聚点
//   BRA: 可能压入两个子栈帧 (taken/fall-through), 或原地更新
//   POP: 到达 rpc 时弹出栈帧, 合并掩码
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — StackEntry 是 SIMT 重汇聚栈帧, 存储 rpc/nextpc/mask 三元组 (教材图 3.3), 用于分支发散后的线程重汇聚
// =============================================================================
struct StackEntry {
    uint64_t rpc;       // 重汇聚点 PC
    uint64_t nextpc;    // 下一条指令 PC
    uint32_t mask;      // 32 位活动掩码 (每位=1 lane)
    bool dflag;         // 发散标志 (由 BRA 发散产生)
};

// =============================================================================
// 分页稀疏内存 (Page-based Sparse Memory, 04 册 §3)
// -----------------------------------------------------------------------------
// 模拟全局/本地内存 (DRAM) 的存储后端
// 使用页式管理 (每页 4 KiB), 仅分配被访问的页, 节省内存
//
// 设计动机: GPU 全局内存地址空间可达 1 TiB, 实际只用少量页
//   - 顺序分配: malloc 从 alloc_ptr 线性分配
//   - 稀疏映射: 未访问的页不分配物理内存
//
// 提供按字节/半字/字/双字 的读写接口, 以及批量读写
// 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) & 第 4 章 §4.3 Memory Partition Unit (p.75) — PageMem 实现全局/本地稀疏内存后端, 模拟 L2/DRAM 分区存储 (按需分页)
// =============================================================================
// =============================================================================
// 1.1 修复说明 F09 稀疏内存地址与生命周期
// -----------------------------------------------------------------------------
// 页编号保留 64 位, 避免高地址截断后别名; 跨页读写按字节边界处理。
// 复制和 clear 后使内部页指针失效, 避免悬挂指针或复制对象共享错误页。
// =============================================================================
class PageMem {
    static constexpr uint32_t PAGE_BITS = 12;             // 每页 12 位 = 4 KiB
    static constexpr uint32_t PAGE_SIZE = 1 << PAGE_BITS;  // 页大小 4096 字节
    static constexpr uint32_t PAGE_MASK = PAGE_SIZE - 1;   // 页内偏移掩码
    // 页表: 页号 → 页数据 (按需分配)
    std::unordered_map<uint64_t, std::vector<uint8_t>> pages;
    uint64_t recent_key=~uint64_t(0);
    std::vector<uint8_t>* recent_page=nullptr;
    // 获取指定页 (不存在则分配)
    std::vector<uint8_t>& get_page(uint64_t pi);
public:
    PageMem() = default;
    PageMem(const PageMem& other) : pages(other.pages) {}
    PageMem& operator=(const PageMem& other) {
        if(this!=&other){pages=other.pages;recent_page=nullptr;recent_key=~uint64_t(0);}
        return *this;
    }
    void clear() { recent_page=nullptr; recent_key=~uint64_t(0); pages.clear(); }    // 清空所有页 (释放内存)
    // 按宽度读取
    uint8_t  r8 (uint64_t a);          // 读 1 字节
    uint16_t r16(uint64_t a);          // 读 2 字节
    uint32_t r32(uint64_t a);          // 读 4 字节
    uint64_t r64(uint64_t a);          // 读 8 字节
    // 按宽度写入
    void w8 (uint64_t a, uint8_t v);
    void w16(uint64_t a, uint16_t v);
    void w32(uint64_t a, uint32_t v);
    void w64(uint64_t a, uint64_t v);
    // 批量读写
    void read_bytes(uint64_t a, void* dst, size_t n);
    void write_bytes(uint64_t a, const void* src, size_t n);
    void memset(uint64_t a, uint8_t val, size_t n);  // 内存填充
};

// =============================================================================
// 内核信息 (01 册 §6)
// -----------------------------------------------------------------------------
// 描述一个 GPU 内核的资源需求和元数据
// 启动内核时, 模拟器根据这些信息分配资源 (寄存器/共享内存)
// 教材引用: 第 2 章 §2.1 Programming Model (p.10) — KernelInfo 描述 CUDA 内核的资源需求 (寄存器/共享内存/参数), 对应教材 grid/CTA 启动模型
// =============================================================================
struct KernelInfo {
    std::string name;                // 内核名 (用于 launch_by_name)
    uint64_t code_offset = 0;        // 代码段在模块中的偏移
    uint64_t code_size = 0;          // 代码段大小 (字节)
    uint16_t regs_u32 = 0;           // 每线程所需 32 位寄存器数
    uint16_t regs_u64 = 0;           // 每线程所需 64 位寄存器对数
    uint32_t static_smem = 0;        // 静态共享内存大小 (字节)
    uint16_t param_size = 0;         // 参数区大小 (字节, 通过 LDC 访问)
    uint8_t  bar_slots = 0;          // 所需屏障槽数
    uint32_t flags = 0;              // 内核标志位
};

// =============================================================================
// 模块 (01 册 §6)
// -----------------------------------------------------------------------------
// 一个 .ntas 二进制模块, 可包含多个内核
// 教材引用: 第 2 章 §2.1 Programming Model (p.10) — Module 表示一个二进制内核模块, 可包含多个内核 (对应 CUDA 模块概念)
// =============================================================================
struct Module {
    std::vector<KernelInfo> kernels; // 模块内的内核列表
    std::vector<uint8_t> code;       // 原始代码段 (二进制机器码)
};

// =============================================================================
// 屏障状态 (03 册 §10)
// -----------------------------------------------------------------------------
// 实现 CTA 内 warp 间的屏障同步 (BAR.SYNC/ARRIVE/WAIT)
// 每个 CTA 有 16 个屏障槽 (bar_slots=16)
//
// 屏障机制:
//   1. BAR.SYNC id: warp 到达屏障 id, 等待所有参与 warp 到达
//   2. 所有 warp 到达后 (arrived >= expected), 释放所有等待的 warp
//   3. 屏障复位, 可重复使用
// 教材引用: 第 3 章 §3.1 One-Loop Approximation (p.22) — BarrierState 实现 CTA 内 warp 间屏障同步 (BAR.SYNC/ARRIVE/WAIT), 单循环调度模型必须建模屏障等待与释放
// =============================================================================
struct BarrierState {
    uint64_t generation=0;
    std::array<uint64_t,64> arrival_phase{};
    uint32_t expected = 0;          // 期望到达的 warp 数 (参与线程总数)
    uint32_t arrived = 0;            // 已到达的 warp 数
    // 等待队列: (warp_slot, mask) - 记录等待的 warp 及其活动掩码
    std::vector<std::pair<int, uint32_t>> waiters;
    bool active = false;            // 屏障是否激活
};

// =============================================================================
// CTA 状态 (01 册 §5, 03 册 §3)
// -----------------------------------------------------------------------------
// CTA (Cooperative Thread Array) = 一个线程块
// 一个 CTA 包含多个 warp, 共享共享内存和屏障
//
// CTA 调度: launch 时, CTA 被分配到 NSM (基于资源可用性)
//   - 检查: warp 槽位 / 寄存器 / 共享内存 / 屏障槽
//   - 分配成功: 在 NSM 上创建 CTAState, 初始化各 warp
// 教材引用: 第 2 章 §2.1 Programming Model (p.10) — CTAState 描述一个线程块 (Cooperative Thread Array), 含 blockIdx 坐标、共享内存、屏障槽, 对应教材 CTA/block 概念
// =============================================================================
struct CTAState {
    int slot = -1;                   // CTA 在 NSM 中的槽位号
    int nsm_id = -1;                // 所在 NSM 编号
    // CTA 坐标 (grid 内, 来自 blockIdx)
    uint32_t ctaid_x = 0, ctaid_y = 0, ctaid_z = 0;
    // CTA 维度 (每维线程数, 来自 blockDim)
    uint32_t ntid_x = 0, ntid_y = 0, ntid_z = 0;
    // Grid 维度 (每维 CTA 数, 来自 gridDim)
    uint32_t nctaid_x = 0, nctaid_y = 0, nctaid_z = 0;
    int num_warps = 0;              // CTA 内 warp 数
    int kernel_idx = -1;           // 关联的内核索引
    int allocated_smem = 0;
    int regs_per_thread = 0;      // 每线程寄存器数
    bool running = false;         // 是否正在运行
    std::vector<uint8_t> shared_mem; // 共享内存 (CTA 内可见)
    std::array<BarrierState, 16> barriers; // 16 个屏障槽
    std::vector<int> warp_slots;  // CTA 内各 warp 在 NSM 中的槽位
    uint64_t smem_flat_base = 0;  // 共享内存的 flat 地址基址
    int bar_slots_needed = 0;     // 所需屏障槽数
    int linear_cta = 0;           // CTA 在 grid 中的线性编号
};

// =============================================================================
// Warp 状态 (03 册 §3-§4)
// -----------------------------------------------------------------------------
// Warp 是 SIMT 执行的基本单元, 包含 32 个 lane (线程)
// 每 warp 有独立的: SIMT 栈 / 寄存器堆 / 谓词 / 本地内存基址
//
// 寄存器堆组织:
//   rf[lane * 256 + reg]: 32 lane × 256 寄存器 = 8192 个 32 位寄存器
//   64 位操作使用寄存器对 (Rd:Rd+1), 要求偶数寄存器号
//   RZ (255) 是零寄存器: 读=0, 写=丢弃
//
// 谓词堆:
//   pf[lane * 7 + p]: 32 lane × 7 谓词 (P0-P6), PT(7) 是恒真谓词
// 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) & 第 3 章 §3.1.4 Divergence (p.32) — WarpState 是 SIMT 执行的 32 线程基本单元, 含 SIMT 栈 (stack/tos)、活动掩码 (lane_valid)、寄存器堆 (rf[])、谓词堆 (pf[])
// =============================================================================
struct WarpState {
    bool valid = false;             // 该 warp 槽是否有效 (有活跃 warp)
    int nsm_id = -1;               // 所在 NSM 编号
    int cta_slot = -1;             // 所属 CTA 槽位
    int warp_id_in_cta = 0;        // 在 CTA 内的 warp 编号
    int slot = 0;                  // 在 NSM 中的 warp 槽位

    std::vector<StackEntry> stack; // SIMT 栈 (最大深度 SIMT_STACK_MAX=32)
    int tos = 0;                   // 栈顶索引 (Top Of Stack)

    uint32_t lane_valid = 0;       // 永久 lane 有效性 (部分 warp 不足 32 线程)
    bool exit_done = false;        // 是否已执行 EXIT
    int bar_wait = -1;             // 正在等待的屏障 ID (-1=无)

    // ---- 每线程寄存器堆: rf[lane * 256 + reg] ----
    // 教材引用: 第 3 章 §3.1.2 Register File Organization (p.27) — 每 warp 独立寄存器窗口 (32 lane × 256 寄存器), 对应教材"每 warp 私有寄存器堆"组织
    std::vector<uint32_t> rf;      // 32 × 256 = 8192 个 32 位寄存器
    // ---- 每线程谓词: pf[lane * 7 + p] ----
    std::vector<uint8_t> pf;       // 32 × 7 = 224 个谓词位
    // ---- 每线程链接寄存器 (CALL/RET 用) ----
    std::array<uint32_t, 32> lr{};

    // ---- 每线程本地内存基址 (flat 地址) ----
    std::array<uint64_t, 32> lmem_base{};

    uint64_t param_base = 0;       // 参数区 flat 基址 (const_mem bank 0)
    uint64_t smem_flat_base = 0;   // 共享内存 flat 基址

    // ---- CTA 坐标信息 (warp 内统一, 来自 CTAState) ----
    uint32_t ctaid_x=0, ctaid_y=0, ctaid_z=0;
    uint32_t ntid_x=0, ntid_y=0, ntid_z=0;
    uint32_t nctaid_x=0, nctaid_y=0, nctaid_z=0;
    uint8_t smid=0;                // NSM 编号 (用于 S2R SR_SMID)
    uint8_t warpid=0;              // warp 编号 (用于 S2R SR_WARPID)

    uint64_t clocklo = 0;          // 时钟计数器低 32 位 (S2R SR_CLOCKLO)

    // 初始化 warp: 分配寄存器堆/谓词, 设置 lane 有效性
    // 教材引用: 第 3 章 §3.2.1 Register Allocator (p.33) — init() 分配每 warp 的寄存器/谓词堆并设置 lane 有效性 (不足 32 时低位有效), 对应教材寄存器分配器
    void init(int lane_count) {
        rf.assign(32 * 256, 0);     // 分配 8192 个寄存器, 清零
        pf.assign(32 * 7, 0);       // 分配 224 个谓词, 清零
        lr.fill(0);                 // 链接寄存器清零
        lmem_base.fill(0);         // 本地内存基址清零
        // 设置永久 lane 有效性 (不足 32 时低位有效)
        lane_valid = (lane_count >= 32) ? 0xFFFFFFFFu : ((1u << lane_count) - 1);
    }

    // ---- 寄存器访问 (可变引用) ----
    // RZ (255) 返回 dummy_zero (写丢弃, 读 0)
    // 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — reg() 处理 RZ 寄存器 (读=0/写=丢弃) 与 PT 恒真谓词, 对应教材 SASS 特殊寄存器约定
    uint32_t& reg(int lane, int r) {
        // 1.1 F08: 宽指令的 RZ 高字也丢弃, 不越过 RF 边界。
        if (r >= RZ || r < 0) return dummy_zero;
        return rf[r * 32 + lane];
    }
    // ---- 寄存器访问 (只读) ----
    uint32_t reg(int lane, int r) const {
        if (r >= RZ || r < 0) return 0;
        return rf[r * 32 + lane];
    }
    // ---- 谓词读取 ----
    // PT (7) 是恒真谓词, 始终返回 true
    // 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) — pred()/set_pred() 实现每 lane 谓词读写, 用于 SIMT 谓词保护分支 (@P0/@!P3); PT 恒真不可写
    bool pred(int lane, int p) const {
        if (p == 7) return true; // PT
        return pf[lane * 7 + p] != 0;
    }
    // ---- 谓词写入 ----
    // PT (7) 不可写 (恒真)
    void set_pred(int lane, int p, bool v) {
        if (p == 7) return; // PT is constant
        pf[lane * 7 + p] = v ? 1 : 0;
    }

    static uint32_t dummy_zero;    // RZ 的写丢弃目标
};

// =============================================================================
// NSM 状态 (03 册 §3)
// -----------------------------------------------------------------------------
// NSM (NutShellGPU SIMT Machine) = 一个 SIMT 核心
// 每个 NSM 独立调度和执行 warp, 拥有自己的寄存器堆/共享内存/L1 缓存
//
// 资源管理:
//   - cur_warps:  当前已分配的 warp 数 (上限 64)
//   - cur_ctas:   当前已分配的 CTA 数 (上限 16)
//   - cur_regs:   当前已分配的寄存器数
//   - cur_smem:   当前已分配的共享内存字节数
// 教材引用: 第 3 章 §3.1 One-Loop Approximation (p.22) & 第 3 章 §3.1.3 Warp Scheduling (p.31) — NSMState 描述一个 SIMT 核心的资源占用 (warp/CTA/寄存器/共享内存槽), 是 round-robin 调度对象
// =============================================================================
struct NSMState {
    int id = -1;                                    // NSM 编号
    std::array<WarpState, MAX_WARPS_PER_SM> warps;  // 64 个 warp 槽
    std::array<CTAState, MAX_CTAS_PER_SM> ctas;     // 16 个 CTA 槽
    int cur_warps = 0;     // 当前 warp 占用数
    int cur_ctas = 0;      // 当前 CTA 占用数
    int cur_regs = 0;      // 当前寄存器占用数
    int cur_smem = 0;      // 当前共享内存占用字节数
};

// =============================================================================
// 错误信息 (05 册 §6)
// -----------------------------------------------------------------------------
// 记录模拟器运行中的错误位置和详情, 便于调试
// 教材引用: 第 5 章 §5.3 Validation (p.129) — ErrorInfo 记录错误位置 (sm/cta/warp/lane/pc/addr), 服务于模拟器验证流程
// =============================================================================
struct ErrorInfo {
    int code = ERR_OK;             // 错误码 (ErrCode 枚举)
    int sm = -1, cta = -1, warp = -1, lane = -1; // 错误位置
    uint64_t pc = 0;               // 出错指令的 PC
    uint64_t addr = 0;             // 出错访存地址
    std::string msg;               // 可读错误描述
};

// =============================================================================
// 跟踪条目 (调试用)
// -----------------------------------------------------------------------------
// 当 trace_enabled=true 时, 每条指令执行后记录一条跟踪
// 用于指令级调试和验证
// 教材引用: 第 5 章 §5.4 Methodology (p.131) — TraceEntry 是指令级跟踪日志条目, 用于断言驱动测试方法学定位错误
// =============================================================================
struct TraceEntry {
    int sm = -1, cta = -1, warp = -1; // 执行位置
    uint64_t pc = 0;                   // 指令 PC
    uint8_t op = 0;                   // 操作码
    uint32_t exec_mask = 0;           // 执行时的活动掩码
    bool has_stack = false;           // 是否包含 SIMT 栈快照
    std::vector<StackEntry> stack_snapshot; // SIMT 栈快照
};

// =============================================================================
// SIMT 栈快照 (t_diverge 测试用, 03 册 §4)
// -----------------------------------------------------------------------------
// 在关键 SIMT 栈操作 (SSY/BRA/EXIT/POP) 前后记录栈状态
// 用于验证分支发散/重汇聚逻辑的正确性
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) & 第 5 章 §5.3 Validation (p.129) — StackSnap 在 SSY/BRA/POP 等关键点记录 SIMT 栈状态, 用于验证发散/重汇聚逻辑
// =============================================================================
struct StackSnap {
    uint64_t pc;                    // 触发快照的指令 PC
    std::string event;              // 事件类型: "SSY"/"BRA"/"BRX"/"EXIT"/"POP"
    std::vector<StackEntry> stack;  // 栈快照 (复制当前完整栈)
    uint32_t exec_mask;             // 执行掩码
};

// =============================================================================
// FuncSim - 功能模拟器主类 (05 册 §M2)
// -----------------------------------------------------------------------------
// 功能模拟器实现完整的 GPU 功能语义, 但不建模时序
// 核心流程: init → load_code → launch → sync
//
// 执行模型:
//   sync() 循环调用 step_one_round(), 每 round 轮转所有 NSM
//   step_one_round() 对每个 NSM 的每个 warp 调用 step_warp()
//   step_warp() 取指→解码→执行 (execute_inst)
//   execute_inst() 按操作码分派执行, 处理 SIMT 栈/访存/屏障
//
// 内存模型:
//   global_mem:  全局+本地内存 (PageMem, 稀疏页式)
//   const_mem:   常量内存 (连续数组, 64 KiB 常量 + 4 KiB 参数)
//   shared_mem:  共享内存 (每 CTA 独立, 在 CTAState 中)
// 教材引用: 第 3 章 §3.1 One-Loop Approximation (p.22) & 第 2 章 §2.1.3 Memory Model (p.13) — FuncSim 是功能模拟器主类, 实现单循环调度 (step_one_round) 与主机 API (memcpy_h2d/launch), 对应教材主机-设备数据传输与单循环模型
// =============================================================================
class FuncSim {
public:
    Config cfg;                     // 全局配置

    // ---- 内存空间 ----
    PageMem global_mem;             // 全局+本地内存 (flat 地址空间)
    std::vector<uint8_t> const_mem; // 常量内存 (64 KiB 常量 + 4 KiB 参数)

    // ---- NSM 集合 ----
    std::vector<NSMState> nsms;     // 所有 NSM (默认 16 个)

    // ---- 代码段 ----
    std::vector<uint8_t> code;      // 全局代码段 (二进制机器码)
    uint64_t code_base = 0;         // 代码段基址 (PC 从此开始)

    // ---- 模块 ----
    std::vector<Module> modules;    // 已加载的模块

    // ---- 内核启动状态 ----
    int active_module = 0;
    int next_cta = 0;
    uint32_t dynamic_smem = 0;
    std::vector<Inst> decoded_code;
    std::vector<int> decoded_errors;
    bool dispatch_pending();
    int active_kernel = -1;         // 当前活跃内核索引
    uint32_t grid_x=0, grid_y=0, grid_z=0;       // Grid 维度
    uint32_t block_x=0, block_y=0, block_z=0;   // Block 维度 (CTA 维度)
    int total_ctas = 0;             // 总 CTA 数
    int completed_ctas = 0;        // 已完成 CTA 数
    std::vector<uint8_t> params;    // 内核参数 (复制到 const_mem)

    // ---- 统计信息 ----
    uint64_t inst_count = 0;        // 已执行指令数
    uint64_t divergent_branches = 0; // 发散分支数
    int max_stack_depth = 0;        // SIMT 栈最大深度
    ErrorInfo error;                // 错误信息

    // ---- 跟踪 ----
    bool trace_enabled = false;      // 是否启用指令跟踪
    std::vector<TraceEntry> trace;  // 跟踪日志

    // ---- SIMT 栈快照 (测试用) ----
    bool snap_enabled = false;       // 是否启用栈快照
    std::vector<StackSnap> snaps;    // 栈快照集合

    // ---- 全局内存分配器 ----
    uint64_t alloc_ptr = 0x10000;    // 分配指针 (从 64 KiB 开始)

    // 每线程本地内存窗口大小 (512 KiB)
    static constexpr uint64_t LOCAL_WINDOW = 0x80000ULL;

    // ================== Host API (主机端接口) ==================

    // 教材引用: 第 2 章 §2.1.3 Memory Model (p.13) — 主机 API (init/malloc/memcpy_h2d/memcpy_d2h/set_params) 模拟 cudaMalloc/cudaMemcpy, 实现主机-设备数据传输与参数传递
    // 初始化模拟器 (分配 NSM, 清零状态)
    void init();
    // 在全局内存中分配 n 字节, 返回地址 (bump 分配)
    uint64_t malloc(size_t n);
    // 释放地址 (功能模拟中无操作, 不回收)
    void free_addr(uint64_t addr);
    // 主机→设备 内存拷贝
    void memcpy_h2d(uint64_t dst, const void* src, size_t n);
    // 设备→设备 内存拷贝
    void memcpy_d2d(uint64_t dst, uint64_t src, size_t n);
    // 设备→主机 内存拷贝
    void memcpy_d2h(void* dst, uint64_t src, size_t n);
    // 设备内存填充
    void memset_d(uint64_t addr, uint8_t val, size_t n);
    // 设置内核参数 (复制到 const_mem 参数区)
    void set_params(const void* data, size_t n);

    // 从 .ntas 二进制文件加载模块
    int load_module_binary(const std::string& path);
    // 从指令字数组加载代码 (测试和演示用)
    void load_code(const std::vector<uint64_t>& words, const KernelInfo& ki);

    // 启动内核 (按索引)
    // 教材引用: 第 2 章 §2.1 Programming Model (p.10) — launch 按 grid (gx,gy,gz) × block (dx,dy,dz) 启动内核, 派发 CTA 到 NSM, 对应教材 grid/CTA/warp/thread 层次
    bool launch(int kernel_idx, uint32_t gx, uint32_t gy, uint32_t gz,
                uint32_t dx, uint32_t dy, uint32_t dz,
                uint32_t dyn_smem, const std::vector<uint8_t>& params, int module_idx = 0);
    // 启动内核 (按名称)
    bool launch_by_name(const std::string& name,
                        uint32_t gx, uint32_t gy, uint32_t gz,
                        uint32_t dx, uint32_t dy, uint32_t dz,
                        uint32_t dyn_smem, const std::vector<uint8_t>& params);

    // 同步: 运行直到所有 CTA 完成 (或出错/死锁)
    // 教材引用: 第 3 章 §3.1.3 Warp Scheduling (p.31) — sync/step_one_round 实现 round-robin 轮转调度, 每 round 遍历所有 NSM 的所有 warp 推进一条指令
    int sync();
    // 执行一轮 (轮转所有 NSM 的所有 warp)
    bool step_one_round();

    // 主机端读写设备内存
    uint32_t read_u32(uint64_t addr);
    void write_u32(uint64_t addr, uint32_t val);
    float read_f32(uint64_t addr);
    void write_f32(uint64_t addr, float val);
    void read_bytes(uint64_t addr, void* dst, size_t n);
    void write_bytes(uint64_t addr, const void* src, size_t n);

    // 查询内核是否完成
    bool is_done() const;
    // 是否有错误
    bool has_error() const { return error.code != ERR_OK; }

    // 跟踪/快照 控制
    void enable_trace(bool en) { trace_enabled = en; }
    void enable_snaps(bool en) { snap_enabled = en; }
    const std::vector<TraceEntry>& get_trace() const { return trace; }
    const std::vector<StackSnap>& get_snaps() const { return snaps; }

    // ================== 内部接口 ==================

    // CTA 调度: 将 CTA 分配到指定 NSM (资源检查+分配)
    bool dispatch_cta(int linear_cta, int kernel_idx, int nsm_id);

    // Warp 执行: 取指→解码→执行一条指令
    int step_warp(NSMState& sm, int warp_slot);
    // 指令执行: 按操作码分派执行语义
    int execute_inst(WarpState& w, const Inst& ins, uint32_t exec_mask);

    // ---- SIMT 栈操作 (03 册 §4) ----
    // 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — ssy/bra/brx/pop_reconverge/do_exit 实现 SIMT 栈的压栈/弹栈/重汇聚逻辑 (教材图 3.3 SSY+BRA+POP 流程)
    // SSY: 设置重汇聚点, 压入新栈帧
    void ssy(WarpState& w, uint64_t target);
    // BRA: 条件分支, 可能发散 (压入 taken/fall 子帧)
    int  bra(WarpState& w, uint64_t target, uint32_t guard_mask);
    // BRX: 间接分支, 多路发散 (每 lane 目标不同)
    int  brx(WarpState& w, const std::array<uint64_t,32>& targets, int n_paths);
    // POP: 到达重汇聚点, 弹出栈帧, 合并掩码
    void pop_reconverge(WarpState& w, uint64_t pc, uint64_t newpc);
    // EXIT: 清除已退出 lane, 全部退出则标记 warp 完成
    int  do_exit(WarpState& w, uint32_t exec_mask);

    // ---- 屏障同步 (03 册 §10) ----
    // 教材引用: 第 3 章 §3.1 One-Loop Approximation (p.22) — bar_sync 实现 CTA 内 warp 间屏障等待与释放 (BAR.SYNC/ARRIVE/WAIT), 单循环模型的关键同步原语
    int bar_sync(WarpState& w, uint8_t id, uint32_t exec_mask, uint8_t sub=0);

    // ---- 访存操作 (04 册 §3, 按地址空间路由) ----
    // 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — load_mem/store_mem/atom_op 按地址空间路由 (global/shared/local/const), 对应教材 L1/shared/constant 多级内存系统
    // 加载: 从 addr 读取一个值到 out_val
    int load_mem(WarpState& w, const Inst& ins, int lane, uint64_t addr, uint32_t& out_val);
    // 存储: 将 val 写入 addr
    int store_mem(WarpState& w, const Inst& ins, int lane, uint64_t addr, uint32_t val);
    // 原子操作: 读-改-写, 返回旧值
    int atom_op(WarpState& w, const Inst& ins, int lane, uint64_t addr, uint32_t arg, uint32_t cmp, uint32_t& old);

    // ---- Flat 地址解码 (04 册 §2) ----
    // 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — decode_flat 将 64 位 flat 地址分解为 (地址空间, 偏移), 路由到 global/shared/local 内存, 对应教材统一虚拟地址空间
    // 将 64 位 flat 地址分解为 (地址空间, 偏移量)
    int decode_flat(uint64_t flat, uint8_t& space, uint64_t& offset, int lane, WarpState& w);

    // ---- 取指 (02 册 §2) ----
    // 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — fetch_inst 从代码段读取并解码一条 64 位指令, 是编码/解码字段提取的入口
    // 从代码段读取并解码一条 64 位指令
    uint64_t fetch_inst(uint64_t pc, Inst& ins);

    // ---- 特殊寄存器读取 ----
    // 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — get_sr 读取特殊寄存器 (SR_TID/SR_CTAID/SR_LANEID 等), 对应教材 SASS 特殊寄存器机制
    // 获取 lane 的特殊寄存器值 (TID/CTAID/SMID 等)
    uint32_t get_sr(WarpState& w, int lane, uint8_t sr, uint32_t* hi = nullptr);

    // ---- 死锁检测 ----
    // 检查是否所有 warp 都在等待屏障且无 warp 可推进
    bool check_deadlock();
};

} // namespace ntisa
