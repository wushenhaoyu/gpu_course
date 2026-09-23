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
// func_sim.cpp - 功能模拟器实现 (S2, 无时序模型)
// =============================================================================
// 教材引用: 第 3 章 §3.1 One-Loop Approximation (p.22) — 本文件整体实现功能模拟器模型, 采用"单循环"调度, 每条指令对所有活跃 lane 立即执行, 不建模流水线时序
// =============================================================================
// 本文件实现 NutShellGPU 的功能级模拟器, 是整个模拟器的核心
// 功能模拟器不建模时序: 每条指令立即执行, 访存立即完成
// 重点关注功能正确性: 指令语义 / SIMT 栈 / 内存系统 / 屏障同步
//
// 文件结构 (按功能分区):
//   1. 配置解析 (INI 解析器) - load_ini_string / load_ini
//   2. 分页内存实现 - PageMem 方法
//   3. 主机 API - init / malloc / memcpy / load_code
//   4. 内核启动与 CTA 调度 - launch / dispatch_cta
//   5. 取指与执行循环 - fetch_inst / sync / step_one_round / step_warp
//   6. SIMT 栈操作 ★ - ssy / bra / brx / pop_reconverge / do_exit
//   7. 屏障同步 ★ - bar_sync
//   8. 特殊寄存器 - get_sr (TID/CTAID/SMID 等)
//   9. Flat 地址解码 ★ - decode_flat
//  10. 访存操作 ★ - load_mem / store_mem / atom_op
//  11. 指令执行分发 ★ - execute_inst (按操作码分派所有指令语义)
//
// ★ = GPU 教学关键概念, 注释最为详细
//
// 参考: NutShellGPU_spec 01-05 册 (具体章节在各函数注释中标注)
// =============================================================================
#include "func_sim.hpp"
#include <cfenv>
#pragma STDC FENV_ACCESS ON
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace ntisa {

// WarpState::dummy_zero: RZ (零寄存器) 的写丢弃目标
// 当指令向 RZ 写入时, 实际写入此变量 (然后被丢弃), 读 RZ 返回 0
// 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — RZ 寄存器是 SASS ISA 中的特殊零寄存器, 读恒为 0, 写丢弃, 此变量实现该语义
uint32_t WarpState::dummy_zero = 0;

// =============================================================================
// 配置 INI 解析器 (00 册 §8)
// -----------------------------------------------------------------------------
// 从 INI 格式文件/字符串加载配置参数, 覆盖默认值
// INI 格式: [section] / key = value / ; 注释
// 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — 配置项 sm_mem/l2/dram 段对应 GPU 内存层次结构 (L1/shared/const/L2/DRAM) 的容量与组织参数
// =============================================================================
// 去除字符串首尾空白
static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

void Config::load_ini_string(const std::string& content) {
    std::istringstream iss(content);
    std::string line, section;
    while (std::getline(iss, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == ';' || t[0] == '#') continue;
        if (t[0] == '[' && t.back() == ']') {
            section = t.substr(1, t.size() - 2);
            continue;
        }
        size_t eq = t.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(t.substr(0, eq));
        std::string val = trim(t.substr(eq + 1));
        // Remove trailing comment
        size_t sc = val.find(';');
        if (sc != std::string::npos) val = trim(val.substr(0, sc));

        auto geti = [&](int& dst) { dst = std::stoi(val); };
        auto getd = [&](double& dst) { dst = std::stod(val); };

        if (section == "timing_v11") {
            if(key=="core_clock_ghz")getd(core_clock_ghz);
            else if(key=="move_latency")geti(move_latency);
            else if(key=="integer_latency")geti(integer_latency);
            else if(key=="imul_latency")geti(imul_latency);
            else if(key=="idiv_latency")geti(idiv_latency);
            else if(key=="shuffle_latency")geti(shuffle_latency);
            else if(key=="rf_read_ports_per_bank")geti(rf_read_ports_per_bank);
            else if(key=="writeback_ports")geti(writeback_ports);
            else if(key=="execution_slots")geti(execution_slots);
            else if(key=="lsu_slots")geti(lsu_slots);
            else if(key=="atomic_latency")geti(atomic_latency);
        } else if (section == "gpgpu") {
            if (key == "gpgpu_n_clusters") geti(n_clusters);
            else if (key == "gpgpu_n_cores_per_cluster") geti(n_cores_per_cluster);
            else if (key == "gpgpu_n_mem") geti(n_mem);
            else if (key == "gpgpu_warp_size") geti(warp_size);
            else if (key == "gpgpu_max_warps_per_core") geti(max_warps_per_core);
            else if (key == "gpgpu_max_ctas_per_core") geti(max_ctas_per_core);
            else if (key == "gpgpu_max_threads_per_cta") geti(max_threads_per_cta);
        } else if (section == "sm_mem") {
            if (key == "smem_size_bytes") geti(smem_size);
            else if (key == "l1d_size_bytes") geti(l1d_size);
            else if (key == "smem_num_banks") geti(smem_num_banks);
            else if (key == "smem_bank_width_bytes") geti(smem_bank_width);
            else if (key == "l1d_line_bytes") geti(l1d_line);
            else if (key == "l1d_sector_bytes") geti(l1d_sector);
            else if (key == "l1d_assoc") geti(l1d_assoc);
            else if (key == "l1d_mshr_entries") geti(l1d_mshr);
            else if (key == "l1d_write_buffer_entries") geti(l1d_wdb);
            else if (key == "icache_size_bytes") geti(icache_size);
            else if (key == "const_cache_size_bytes") geti(const_cache_size);
            else if (key == "barrier_slots") geti(barrier_slots);
        } else if (section == "sm_pipeline") {
            if (key == "sp_lanes") geti(sp_lanes);
            else if (key == "sp_latency") geti(sp_latency);
            else if (key == "sp_init_interval") geti(sp_interval);
            else if (key == "sfu_lanes") geti(sfu_lanes);
            else if (key == "sfu_latency") geti(sfu_latency);
            else if (key == "sfu_init_interval") geti(sfu_interval);
            else if (key == "dp_lanes") geti(dp_lanes);
            else if (key == "dp_latency") geti(dp_latency);
            else if (key == "dp_init_interval") geti(dp_interval);
            else if (key == "branch_latency") geti(branch_latency);
            else if (key == "num_collector_units") geti(num_collector_units);
            else if (key == "operand_slots_per_collector") geti(operand_slots_per_collector);
            else if (key == "rf_logical_banks") geti(rf_logical_banks);
            else if (key == "rf_physical_entries") geti(rf_physical_entries);
            else if (key == "ibuffer_entries_per_warp") geti(ibuffer_entries_per_warp);
            else if (key == "scoreboard_entries_per_warp") geti(scoreboard_entries_per_warp);
            else if (key == "icache_mshrs") geti(icache_mshrs);
        } else if (section == "latency_ns") {
            if (key == "smem_hit") getd(lat_smem_hit);
            else if (key == "l1_hit") getd(lat_l1_hit);
            else if (key == "l2_hit") getd(lat_l2_hit);
            else if (key == "dram_row_hit") getd(lat_dram_row_hit);
            else if (key == "barrier_resume") getd(lat_barrier_resume);
            else if (key == "pcie_base_ns") getd(lat_pcie_base);
            else if (key == "pcie_per4k_ns") getd(lat_pcie_per4k);
        } else if (section == "noc") {
            if (key == "flit_bytes") geti(noc_flit_bytes);
            else if (key == "req_vc_entries") geti(noc_req_vc);
            else if (key == "rsp_vc_entries") geti(noc_rsp_vc);
            else if (key == "router_pipe_cycles") geti(noc_router_pipe);
            else if (key == "link_cycles") geti(noc_link_cycles);
            else if (key == "interleave_bytes") geti(noc_interleave);
        } else if (section == "l2") {
            if (key == "slice_size_bytes") geti(l2_slice_size);
            else if (key == "slice_assoc") geti(l2_assoc);
            else if (key == "line_bytes") geti(l2_line);
            else if (key == "sector_bytes") geti(l2_sector);
            else if (key == "mshr_per_slice") geti(l2_mshr);
            else if (key == "rop_atom_cache_bytes") geti(l2_rop_atom_cache);
        } else if (section == "dram") {
            if (key == "channels") geti(dram_channels);
            else if (key == "banks_per_channel") geti(dram_banks_per_channel);
            else if (key == "channel_width_bits") geti(dram_channel_width);
            else if (key == "data_rate_gtps") getd(dram_data_rate);
            else if (key == "burst_atom_bytes") geti(dram_burst_atom);
            else if (key == "tCK") getd(tCK);
            else if (key == "tCAS") getd(tCAS);
            else if (key == "tRCD") getd(tRCD);
            else if (key == "tRP") getd(tRP);
            else if (key == "tRAS") getd(tRAS);
            else if (key == "tWR") getd(tWR);
            else if (key == "tRC") getd(tRC);
            else if (key == "write_drain_threshold") geti(dram_write_drain);
        }
    }
}

void Config::load_ini(const std::string& path) {
    std::ifstream f(path);
    if (!f) return;
    std::stringstream ss;
    ss << f.rdbuf();
    load_ini_string(ss.str());
}

// =============================================================================
// 分页内存实现 PageMem (04 册 §2)
// -----------------------------------------------------------------------------
// 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — PageMem 模拟 global memory/DRAM 分区, 用按需分页方式支持稀疏 64 位地址空间, 对应 GPU 显存
// -----------------------------------------------------------------------------
// PageMem 用"按需分页"方式模拟 GPU 显存 (global memory):
//   - 整个 64 位地址空间按 4 KiB 分页 (PAGE_BITS=12, PAGE_SIZE=4096)
//   - 用 std::unordered_map<uint32_t, std::vector<uint8_t>> 存储已用页面
//   - 只在首次访问某页时才分配 4 KiB 内存 (按需分配, 节省内存)
//   - 小端字节序 (Little Endian, 与 x86/ARM/GPU 一致)
//
// 教学要点:
//   - 真实 GPU 的显存是连续物理内存, 这里用分页模拟是为了支持稀疏地址空间
//   - 每个页面 4 KiB, 对应 GPU L2 缓存行的整数倍
//   - 多字节访问 (r16/r32/r64) 通过组合多个 r8 实现, 不要求对齐
//   - 真实 GPU 的 global memory 访问有 coalescing 优化 (见 load_mem), 这里简化
// =============================================================================

// 获取指定页面的引用 (不存在则按需分配)
// 参数: pi = 页号 (addr >> 12)
// 返回: 该页面的字节向量引用
// 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — 按需分页分配对应 DRAM 分区的按需物理页分配, 首次访问时分配 4 KiB 页
std::vector<uint8_t>& PageMem::get_page(uint64_t pi) {
    if(recent_page && recent_key==pi) return *recent_page;
    auto it = pages.find(pi);
    if (it == pages.end()) {
        // 首次访问此页: 分配 4 KiB 并清零, 加入 map
        it = pages.emplace(pi, std::vector<uint8_t>(PAGE_SIZE, 0)).first;
    }
    recent_key=pi;recent_page=&it->second;
    return it->second;
}

// 读 1 字节 (基础读操作, 其他宽度基于此实现)
// a = 64 位地址, 拆分为页号 pi 和页内偏移 off
// 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — global memory 基础读操作, 多字节访问通过组合 r8 实现, 真实 GPU 有 L2/L1 缓存层次, 功能模拟器直接访问
uint8_t PageMem::r8(uint64_t a) {
    uint64_t pi = (uint64_t)(a >> PAGE_BITS);   // 页号 = 地址高 52 位
    uint32_t off = (uint32_t)(a & PAGE_MASK);   // 页内偏移 = 地址低 12 位
    return get_page(pi)[off];
}

// 读 2 字节 (小端: 低字节在低地址)
uint16_t PageMem::r16(uint64_t a) {
    if ((a & PAGE_MASK) <= PAGE_SIZE-2) {
        auto& page = get_page((uint64_t)(a >> PAGE_BITS));
        uint16_t value = 0;
        for (int i=0;i<2;++i) value |= (uint16_t)page[(a & PAGE_MASK)+i] << (8*i);
        return value;
    }

    return (uint16_t)r8(a) | ((uint16_t)r8(a+1) << 8);
}

// 读 4 字节 (小端组合 4 次 r8)
uint32_t PageMem::r32(uint64_t a) {
    if ((a & PAGE_MASK) <= PAGE_SIZE-4) {
        auto& page = get_page((uint64_t)(a >> PAGE_BITS));
        uint32_t value = 0;
        for (int i=0;i<4;++i) value |= (uint32_t)page[(a & PAGE_MASK)+i] << (8*i);
        return value;
    }

    return (uint32_t)r8(a) | ((uint32_t)r8(a+1) << 8) |
           ((uint32_t)r8(a+2) << 16) | ((uint32_t)r8(a+3) << 24);
}

// 读 8 字节 (组合两次 r32, 低 32 位在前)
uint64_t PageMem::r64(uint64_t a) {
    if ((a & PAGE_MASK) <= PAGE_SIZE-8) {
        auto& page = get_page((uint64_t)(a >> PAGE_BITS));
        uint64_t value = 0;
        for (int i=0;i<8;++i) value |= (uint64_t)page[(a & PAGE_MASK)+i] << (8*i);
        return value;
    }

    uint64_t lo = r32(a);
    uint64_t hi = r32(a+4);
    return lo | (hi << 32);
}

// 写 1 字节 (基础写操作)
// 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — global memory 基础写操作, 真实 GPU 写入需经过 L2/ROP, 功能模拟器直接写页
void PageMem::w8(uint64_t a, uint8_t v) {
    uint64_t pi = (uint64_t)(a >> PAGE_BITS);
    uint32_t off = (uint32_t)(a & PAGE_MASK);
    get_page(pi)[off] = v;
}

// 写 2 字节 (小端拆分)
void PageMem::w16(uint64_t a, uint16_t v) {
    if ((a & PAGE_MASK) <= PAGE_SIZE-2) {
        auto& page = get_page((uint64_t)(a >> PAGE_BITS));
        for (int i=0;i<2;++i) page[(a & PAGE_MASK)+i] = (uint8_t)(v >> (8*i));
        return;
    }

    w8(a, (uint8_t)(v & 0xFF));
    w8(a+1, (uint8_t)(v >> 8));
}

// 写 4 字节 (拆分为 4 次 w8)
void PageMem::w32(uint64_t a, uint32_t v) {
    if ((a & PAGE_MASK) <= PAGE_SIZE-4) {
        auto& page = get_page((uint64_t)(a >> PAGE_BITS));
        for (int i=0;i<4;++i) page[(a & PAGE_MASK)+i] = (uint8_t)(v >> (8*i));
        return;
    }

    w8(a, (uint8_t)(v & 0xFF));
    w8(a+1, (uint8_t)((v >> 8) & 0xFF));
    w8(a+2, (uint8_t)((v >> 16) & 0xFF));
    w8(a+3, (uint8_t)((v >> 24) & 0xFF));
}

// 写 8 字节 (拆分为低 32 位和高 32 位)
void PageMem::w64(uint64_t a, uint64_t v) {
    if ((a & PAGE_MASK) <= PAGE_SIZE-8) {
        auto& page = get_page((uint64_t)(a >> PAGE_BITS));
        for (int i=0;i<8;++i) page[(a & PAGE_MASK)+i] = (uint8_t)(v >> (8*i));
        return;
    }

    w32(a, (uint32_t)(v & 0xFFFFFFFF));
    w32(a+4, (uint32_t)(v >> 32));
}

// 批量读取 n 字节到 dst (用于 memcpy_d2h)
// 教材引用: 第 2 章 §2.1.3 Memory Model (p.13) — 批量读对应主机-设备数据传输 cudaMemcpy D2H, 功能模拟器不建模 PCIe 延迟, 立即完成
void PageMem::read_bytes(uint64_t a, void* dst, size_t n) {
    auto* p = (uint8_t*)dst;
    while (n) {
        size_t off = a & PAGE_MASK, count = std::min(n, (size_t)PAGE_SIZE-off);
        std::memcpy(p, get_page((uint64_t)(a >> PAGE_BITS)).data()+off, count);
        p+=count; a+=count; n-=count;
    }
}

// 批量写入 n 字节 (用于 memcpy_h2d)
// 教材引用: 第 2 章 §2.1.3 Memory Model (p.13) — 批量写对应主机-设备数据传输 cudaMemcpy H2D, 功能模拟器不建模 PCIe 延迟, 立即完成
void PageMem::write_bytes(uint64_t a, const void* src, size_t n) {
    auto* p = (const uint8_t*)src;
    while (n) {
        size_t off = a & PAGE_MASK, count = std::min(n, (size_t)PAGE_SIZE-off);
        std::memcpy(get_page((uint64_t)(a >> PAGE_BITS)).data()+off, p, count);
        p+=count; a+=count; n-=count;
    }
}

// 批量填充 val (用于 memset_d, 初始化显存)
void PageMem::memset(uint64_t a, uint8_t val, size_t n) {
    for (size_t i = 0; i < n; i++) w8(a + i, val);
}

// =============================================================================
// 主机 API (Host API) - 模拟 CPU 端对 GPU 的控制 (01 册 §3, 05 册 §3)
// -----------------------------------------------------------------------------
// 教材引用: 第 2 章 §2.1.3 Memory Model (p.13) — 主机 API 模拟 CUDA 主机-设备数据传输 cudaMemcpy 与 cudaMalloc, 对应教材内存模型中主机端对设备内存空间的管理
// -----------------------------------------------------------------------------
// 这些函数模拟主机 (CPU) 端调用 GPU 的 API, 对应真实 CUDA 的:
//   - cudaMalloc          → malloc       (在显存分配缓冲区)
//   - cudaFree            → free_addr    (释放, 此处为空操作)
//   - cudaMemcpy H2D      → memcpy_h2d   (主机→设备数据传输)
//   - cudaMemcpy D2H      → memcpy_d2h   (设备→主机数据传输)
//   - cudaMemcpy D2D      → memcpy_d2d   (设备→设备数据传输)
//   - cudaMemset          → memset_d     (显存填充)
//   - cudaLaunchKernel    → launch       (启动内核)
//
// 教学要点:
//   - 真实 GPU 通过 PCIe 总线传输数据, 有延迟 (lat_pcie_base)
//   - 功能模拟器不建模 PCIe 延迟, 传输立即完成
//   - 周期级模拟器 (cycle_sim) 会建模 PCIe 传输时间
// =============================================================================

// 初始化模拟器: 创建所有 NSM (Streaming Multiprocessor)
// 调用时机: 模拟器启动时调用一次
// 教材引用: 第 1 章 §1.2 GPU Hardware Basics (p.2) — init 创建所有 NSM (SM), 对应教材中 GPU 硬件基础: 多个 SM 组成集群, 每 SM 含 warp 槽/CTA 槽/寄存器堆
void FuncSim::init() {
    int n = cfg.n_nsm();           // NSM 总数 = n_clusters × n_cores_per_cluster (默认 16)
    nsms.resize(n);                // 创建 n 个 NSM 状态
    for (int i = 0; i < n; i++) {
        nsms[i].id = i;            // NSM 编号 (0 ~ n-1)
        // 初始化 warp 槽位编号 (0 ~ MAX_WARPS_PER_SM-1)
        for (int j = 0; j < MAX_WARPS_PER_SM; j++)
            nsms[i].warps[j].slot = j;
        // 初始化 CTA 槽位编号 (0 ~ MAX_CTAS_PER_SM-1)
        for (int j = 0; j < MAX_CTAS_PER_SM; j++)
            nsms[i].ctas[j].slot = j;
    }
    // 常量内存: 64 个 bank × 64 KiB + 4 KiB 参数区 = 68 KiB
    // bank 0 的前 4 KiB 用于内核参数 (param area)
    const_mem.assign(64 * 1024 + 4 * 1024, 0);
}

// 在显存分配 n 字节 (bump allocator, 简单的线性分配器)
// 返回分配的起始地址, 8 字节对齐
// 教材引用: 第 2 章 §2.1.3 Memory Model (p.13) — malloc 对应 cudaMalloc, 在设备 global memory 分配缓冲区, 功能模拟器用线性 bump allocator 简化
uint64_t FuncSim::malloc(size_t n) {
    uint64_t addr = alloc_ptr;
    alloc_ptr += (n + 7) & ~size_t(7); // 8-byte align (向上取整到 8 的倍数)
    return addr;
}

// 释放内存 (空操作: bump allocator 不支持回收)
// 真实 GPU 有复杂的显存分配器, 这里简化
void FuncSim::free_addr(uint64_t addr) {
    (void)addr; // simple bump allocator, no free
}

// 主机→设备 数据拷贝 (模拟 cudaMemcpy H2D)
// dst = 设备地址, src = 主机指针, n = 字节数
// 教材引用: 第 2 章 §2.1.3 Memory Model (p.13) — memcpy_h2d 对应 cudaMemcpy H2D, 跨主机-设备内存空间数据传输, 功能模拟器不建模 PCIe 延迟
void FuncSim::memcpy_h2d(uint64_t dst, const void* src, size_t n) {
    global_mem.write_bytes(dst, src, n);
}

// 设备→主机 数据拷贝 (模拟 cudaMemcpy D2H)
// dst = 主机指针, src = 设备地址, n = 字节数
void FuncSim::memcpy_d2h(void* dst, uint64_t src, size_t n) {
    global_mem.read_bytes(src, dst, n);
}

// 设备→设备 数据拷贝 (模拟 cudaMemcpy D2D)
// 先读到临时缓冲, 再写入目标 (源和目标可能重叠)
void FuncSim::memcpy_d2d(uint64_t dst, uint64_t src, size_t n) {
    std::vector<uint8_t> buf(n);
    global_mem.read_bytes(src, buf.data(), n);
    global_mem.write_bytes(dst, buf.data(), n);
}

// 显存填充 (模拟 cudaMemset)
void FuncSim::memset_d(uint64_t addr, uint8_t val, size_t n) {
    global_mem.memset(addr, val, n);
}

// 设置内核参数 (复制到 const_mem bank 0 的参数区)
// 参数通过常量内存传递给内核, 内核用 LDC 指令读取
// 教材引用: 第 4 章 §4.1.1 Constant Memory (p.68) — 内核参数通过常量内存 bank 0 传递, 对应教材中常量内存的专用缓存与只读广播机制
void FuncSim::set_params(const void* data, size_t n) {
    params.assign((const uint8_t*)data, (const uint8_t*)data + n);
    // Also copy to const_mem bank 0 (param area)
    if (n <= 4096) {
        std::memcpy(const_mem.data(), data, n);
    }
}

// -----------------------------------------------------------------------------
// load_module_binary: 从文件加载 NTAS1 内核二进制 (01 册 §5)
// -----------------------------------------------------------------------------
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — 解析内核二进制容器格式, 对应教材中 64 位指令编码与内核代码段加载
// -----------------------------------------------------------------------------
// 解析 NTAS1 容器格式 (类似 CUDA 的 cubin/elf):
//   [0..4]   魔数 "NTAS1" (5 字节)
//   [16..17] 版本号
//   [18..19] 内核数量
//   [20..]   内核描述符数组 + 代码段
//
// 每个内核描述符 (KernelInfo) 包含:
//   name:        内核名 (如 "saxpy")
//   code_offset: 代码在代码段中的偏移
//   code_size:   代码大小 (字节)
//   regs_u32:    每 warp 需要的 32 位寄存器数
//   regs_u64:    每 warp 需要的 64 位寄存器对数
//   static_smem: 静态共享内存需求
//   param_size:  参数区大小
//   bar_slots:   屏障槽数需求
//   flags:       标志位
// -----------------------------------------------------------------------------
int FuncSim::load_module_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return -1;
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    // Parse NTAS1 container
    if (buf.size() < 20) return -1;
    if (std::memcmp(buf.data(), "NTAS1", 5) != 0) return -1;
    uint16_t version = *(uint16_t*)(buf.data() + 16);
    uint16_t n_kernels = *(uint16_t*)(buf.data() + 18);
    (void)version;
    size_t off = 20;
    Module m;
    // 逐个解析内核描述符
    for (int k = 0; k < n_kernels; k++) {
        KernelInfo ki;
        uint16_t name_len = *(uint16_t*)(buf.data() + off); off += 2;
        ki.name.assign((char*)buf.data() + off, name_len); off += name_len;
        ki.code_offset = *(uint64_t*)(buf.data() + off); off += 8;
        ki.code_size = *(uint64_t*)(buf.data() + off); off += 8;
        ki.regs_u32 = *(uint16_t*)(buf.data() + off); off += 2;
        ki.regs_u64 = *(uint16_t*)(buf.data() + off); off += 2;
        ki.static_smem = *(uint32_t*)(buf.data() + off); off += 4;
        ki.param_size = *(uint16_t*)(buf.data() + off); off += 2;
        ki.bar_slots = *(uint8_t*)(buf.data() + off); off += 1;
        ki.flags = *(uint32_t*)(buf.data() + off); off += 4;
        m.kernels.push_back(ki);
    }
    // Copy code segment
    m.code.assign(buf.begin() + off, buf.end());
    modules.push_back(std::move(m));
    return (int)modules.size() - 1;
}

// -----------------------------------------------------------------------------
// load_code: 直接从 64 位指令字数组加载代码 (测试用)
// -----------------------------------------------------------------------------
// 用于单元测试: 直接传入编码好的指令字数组, 无需文件 I/O
// 参数:
//   words: 64 位指令字数组 (每条指令 8 字节)
//   ki: 内核信息 (寄存器需求/共享内存需求等)
// -----------------------------------------------------------------------------
void FuncSim::load_code(const std::vector<uint64_t>& words, const KernelInfo& ki) {
    Module m;
    m.kernels.push_back(ki);
    m.code.resize(words.size() * 8);
    // 将每个 64 位指令字序列化为 8 字节
    for (size_t i = 0; i < words.size(); i++) {
        std::memcpy(m.code.data() + i * 8, &words[i], 8);
    }
    modules.push_back(std::move(m));
}

// =============================================================================
// 内核启动与 CTA 调度 (01 册 §3, 03 册 §2) ★★ GPU 调度核心概念
// -----------------------------------------------------------------------------
// 教材引用: 第 2 章 §2.1 Programming Model (p.10) — Grid/CTA/warp/thread 层次调度, CTA=block 是资源分配单元, warp 是调度单元
// -----------------------------------------------------------------------------
// 内核启动流程:
//   1. 主机设置 grid 维度 (gx,gy,gz) 和 block 维度 (dx,dy,dz)
//   2. 将 grid 划分为 gx×gy×gz 个 CTA (Cooperative Thread Array = block)
//   3. 将 CTA 轮询分配到各 NSM (SM), 每个 NSM 可同时运行多个 CTA
//   4. 每个 CTA 内的线程按 32 个一组划分为 warp
//   5. warp 由 NSM 的 SIMT 调度器轮询执行
//
// 术语对照:
//   Grid  = 整个内核启动的所有线程
//   CTA   = Cooperative Thread Array (= CUDA block), 可共享共享内存/屏障
//   Warp  = 32 个线程的 SIMT 执行单元
//   Lane  = warp 内的线程编号 (0-31)
//
// 教学要点:
//   - CTA 是资源分配的基本单元: 占用寄存器/共享内存/屏障槽
//   - Warp 是指令调度的基本单元: 每 cycle 调度一个 warp 执行
//   - 同一 CTA 的 warp 可通过共享内存和屏障通信
//   - 不同 CTA 之间不能直接通信 (需通过 global memory + atomic)
// =============================================================================

// -----------------------------------------------------------------------------
// launch: 启动内核 (01 册 §3.2)
// -----------------------------------------------------------------------------
// 教材引用: 第 2 章 §2.1 Programming Model (p.10) — launch 设置 grid/block 维度并调度 CTA, 对应 CUDA 编程模型中线程索引 i=blockIdx.x*blockDim.x+threadIdx.x 的层次
// -----------------------------------------------------------------------------
// 参数:
//   kernel_idx: 内核索引 (在 modules 中)
//   gx,gy,gz:   grid 维度 (CTA 总数 = gx×gy×gz)
//   dx,dy,dz:   block 维度 (线程总数 = dx×dy×dz)
//   dyn_smem:   动态共享内存需求 (每个 CTA)
//   lp:         内核参数 (传入 const_mem bank 0)
// -----------------------------------------------------------------------------
// =============================================================================
// 1.1 修复说明 F01/F02: 按模块身份启动, 使用非零入口, 拒绝非法启动资源。
// 不以相同入口地址或名称猜测当前模块; 入口必须位于模块代码范围内。
// =============================================================================
bool FuncSim::launch(int kernel_idx, uint32_t gx, uint32_t gy, uint32_t gz,
                     uint32_t dx, uint32_t dy, uint32_t dz,
                     uint32_t dyn_smem, const std::vector<uint8_t>& lp, int module_idx) {
    if (!is_done()) return false; // Do not corrupt an in-flight launch.
    error = ErrorInfo{};
    auto fail = [&]() { error.code=ERR_LAUNCH; error.msg="Invalid launch or resources"; return false; };
    if (module_idx<0 || module_idx>=(int)modules.size() || kernel_idx<0 || kernel_idx>=(int)modules[module_idx].kernels.size()) return fail();
    const auto& k=modules[module_idx].kernels[kernel_idx];
    uint64_t b=uint64_t(dx)*dy*dz, g=uint64_t(gx)*gy*gz;
    uint64_t nr=std::max(1,int(k.regs_u32)+2*int(k.regs_u64));
    if (!gx||!gy||!gz||!dx||!dy||!dz||g>0x7fffffffULL || b>(uint64_t)cfg.max_threads_per_cta ||
        (b+31)/32>(uint64_t)cfg.max_warps_per_core || nr>255 || nr*((b+31)/32)*32>(uint64_t)cfg.rf_physical_entries ||
        uint64_t(k.static_smem)+dyn_smem>(uint64_t)cfg.smem_size || k.bar_slots>cfg.barrier_slots ||
        lp.size()>4096 || lp.size()<k.param_size || k.code_offset%8 || k.code_offset>=modules[module_idx].code.size()) return fail();
    dynamic_smem=dyn_smem;
    active_module = module_idx;
    code = modules.at(active_module).code;
    decoded_code.resize(code.size()/8); decoded_errors.resize(decoded_code.size());
    for(size_t i=0;i<decoded_code.size();++i){uint64_t word;std::memcpy(&word,code.data()+i*8,8);decoded_errors[i]=decode(word,i*8,decoded_code[i]);}
    std::fill(const_mem.begin(),const_mem.begin()+4096,0);
    next_cta = 0;
    active_kernel = kernel_idx;
    grid_x = gx; grid_y = gy; grid_z = gz;    // 记录 grid 维度
    block_x = dx; block_y = dy; block_z = dz;  // 记录 block 维度
    params = lp;
    total_ctas = (int)gx * (int)gy * (int)gz;  // CTA 总数
    completed_ctas = 0;                         // 已完成 CTA 数 (用于判断内核是否结束)

    // Copy params to const_mem bank 0
    if (!lp.empty() && lp.size() <= 4096) {
        std::memcpy(const_mem.data(), lp.data(), lp.size());
    }

    // 将 CTA 轮询分配到各 NSM (round-robin)
    // 真实 GPU 使用更复杂的调度策略 (考虑资源占用率)
    dispatch_pending();
    return true;
}

// 通过内核名启动 (查找内核索引后调用 launch)
// =============================================================================
// 1.1 修复说明 F03: CTA 退休后继续补发未驻留网格, 不截断大 grid。
// 待发 CTA 是未完成工作; 暂无可运行 warp 不等同于整个 kernel 完成。
// =============================================================================
bool FuncSim::dispatch_pending() {
    bool progress = false;
    while (next_cta < total_ctas) {
        bool dispatched = false;
        for (int j = 0; j < cfg.n_nsm(); ++j) {
            if (dispatch_cta(next_cta, active_kernel, (next_cta+j)%cfg.n_nsm())) {
                ++next_cta; dispatched = progress = true; break;
            }
        }
        if (!dispatched) break;
    }
    return progress;
}

bool FuncSim::launch_by_name(const std::string& name,
                             uint32_t gx, uint32_t gy, uint32_t gz,
                             uint32_t dx, uint32_t dy, uint32_t dz,
                             uint32_t dyn_smem, const std::vector<uint8_t>& lp) {
    for (size_t m = 0; m < modules.size(); m++) {
        for (size_t k = 0; k < modules[m].kernels.size(); k++) {
            if (modules[m].kernels[k].name == name) {
                // launch owns module selection after validation.
                return launch((int)k, gx, gy, gz, dx, dy, dz, dyn_smem, lp, (int)m);
            }
        }
    }
    return false;
}

// -----------------------------------------------------------------------------
// dispatch_cta: 将一个 CTA 分配到指定 NSM (03 册 §2.2) ★★ 资源分配核心
// -----------------------------------------------------------------------------
// 教材引用: 第 3 章 §3.2.1 Register Allocator (p.33) — dispatch_cta 检查 warp/CTA/共享内存资源约束并分配寄存器堆, 对应教材中寄存器分配器按 CTA 资源需求分配
// -----------------------------------------------------------------------------
// 将 CTA (线程块) 分配到 NSM 上运行, 分配时检查资源约束:
//   - warp 槽位: 每 NSM 最多 max_warps_per_core (64) 个 warp
//   - CTA 槽位: 每 NSM 最多 max_ctas_per_core (16) 个 CTA
//   - 共享内存: 每 NSM 共享 smem_size (48 KiB) 的共享内存
//
// CTA 分配后做的事:
//   1. 计算 CTA 坐标 (ctaid_x/y/z) 从 linear_cta 解码
//   2. 将 block 内线程划分为 warp (每 32 线程一个 warp)
//   3. 为每个 warp 分配寄存器堆/谓词/本地内存基址
//   4. 初始化 SIMT 栈 (栈底 = {EXIT_SENTINEL, entry=0, mask=lane_valid})
//   5. 设置 CTA 层面的坐标信息 (ntid/nctaid 等, 供 S2R 读取)
//
// 教学要点:
//   - 最后一个 warp 可能不足 32 线程 (如 block 100 线程 = 3 warp + 4 线程)
//   - lane_valid 标记哪些 lane 有效 (不足 32 时高位为 0)
//   - 共享内存基址 = FLAT_SHARED_BASE + slot * smem_size (每 CTA 独立)
//   - 本地内存基址 = FLAT_LOCAL_BASE + (warp_slot*32+lane) * LOCAL_WINDOW
// -----------------------------------------------------------------------------
// =============================================================================
// 1.1 修复说明 F04 资源记账
// -----------------------------------------------------------------------------
// 驻留限制同时计入 warp、CTA、寄存器、静态加动态共享内存。
// CTA 保存实际分配量, 回收必须按同一数量扣除。
// =============================================================================
bool FuncSim::dispatch_cta(int linear_cta, int kernel_idx, int nsm_id) {
    NSMState& sm = nsms[nsm_id];
    // 查找空闲 CTA 槽位
    int cta_slot = -1;
    for (int i = 0; i < cfg.max_ctas_per_core; i++) {
        if (!sm.ctas[i].running) { cta_slot = i; break; }
    }
    if (cta_slot < 0) return false;  // 无空闲 CTA 槽位

    Module& mod = modules[active_module]; // assume code is already loaded
    KernelInfo& ki = mod.kernels[kernel_idx];

    uint32_t B = block_x * block_y * block_z;  // block 内线程总数
    int W = (B + WARP_SZ - 1) / WARP_SZ;        // warp 数 (向上取整)
    int R = std::max((int)ki.regs_u32 + 2*(int)ki.regs_u64, 1);      // 每线程寄存器数
    int smem_need = (int)ki.static_smem + (int)dynamic_smem;
    int bar_need = ki.bar_slots;                 // 屏障槽数需求

    // 资源检查 (00 册 §8.1): warp 数/CTA 数/共享内存 不超限
    if (sm.cur_warps + W > cfg.max_warps_per_core) return false;
    if (sm.cur_ctas + 1 > cfg.max_ctas_per_core) return false;
    if (sm.cur_smem + smem_need > cfg.smem_size || sm.cur_regs + R*W*32 > cfg.rf_physical_entries) return false;

    // 设置 CTA 状态
    CTAState& cta = sm.ctas[cta_slot];
    cta.running = true;
    cta.nsm_id = nsm_id;
    cta.slot = cta_slot;
    cta.kernel_idx = kernel_idx;
    cta.num_warps = W;
    cta.regs_per_thread = R;
    cta.allocated_smem = smem_need;
    cta.linear_cta = linear_cta;
    // 共享内存 flat 基址: 每 CTA 独立, 按 slot 分配
    cta.smem_flat_base = FLAT_SHARED_BASE + (uint64_t)cta_slot * cfg.smem_size;

    // 从线性 CTA 编号解码 3D 坐标 (ctaid_x/y/z)
    // 坐标顺序: x 变化最快, y 次之, z 最慢 (与 CUDA 一致)
    cta.ctaid_x = linear_cta % grid_x;
    cta.ctaid_y = (linear_cta / grid_x) % grid_y;
    cta.ctaid_z = linear_cta / ((int)grid_x * (int)grid_y);
    // block 维度 (每个 CTA 内的线程数 = ntid_x * ntid_y * ntid_z)
    cta.ntid_x = block_x; cta.ntid_y = block_y; cta.ntid_z = block_z;
    // grid 维度 (CTA 数 = nctaid_x * nctaid_y * nctaid_z)
    cta.nctaid_x = grid_x; cta.nctaid_y = grid_y; cta.nctaid_z = grid_z;
    cta.shared_mem.assign(smem_need, 0);  // 分配共享内存, 清零
    cta.bar_slots_needed = bar_need;
    // 初始化 16 个屏障槽
    for (int i = 0; i < 16; i++) {
        cta.barriers[i] = BarrierState{};
        cta.barriers[i].active = false;
        cta.barriers[i].arrived = 0;
        cta.barriers[i].expected = 0;
        cta.barriers[i].waiters.clear();
    }

    // 创建 warp
    cta.warp_slots.clear();
    int base_slot = -1;
    for (int w = 0; w < W; w++) {
        // 查找空闲 warp 槽位
        int ws = -1;
        for (int i = 0; i < cfg.max_warps_per_core; i++) {
            if (!sm.warps[i].valid) { ws = i; break; }
        }
        if (ws < 0) return false; // shouldn't happen after check
        WarpState& warp = sm.warps[ws];
        warp.valid = true;
        warp.nsm_id = nsm_id;
        warp.cta_slot = cta_slot;
        warp.warp_id_in_cta = w;    // warp 在 CTA 内的编号
        warp.slot = ws;             // warp 在 NSM 中的全局槽位
        warp.bar_wait = -1;         // 未等待屏障
        warp.exit_done = false;     // 未执行 EXIT
        warp.smid = (uint8_t)nsm_id;   // 用于 S2R SR_SMID
        warp.warpid = (uint8_t)w;      // 用于 S2R SR_WARPID
        warp.clocklo = 0;              // 时钟计数器清零

        // 计算 lane 有效性: 最后一个 warp 可能不足 32 线程
        // 例: block 100 线程 → warp 0/1/2 各 32 线程, warp 3 只有 4 线程
        int lanes_in_warp = std::min(WARP_SZ, (int)B - w * WARP_SZ);
        warp.init(lanes_in_warp);  // 分配寄存器堆, 设置 lane_valid

        // 设置每线程的本地内存基址 (flat 地址空间)
        // 本地内存 = 线程私有数据, 存放在 global memory 中
        for (int lane = 0; lane < WARP_SZ; lane++) {
            int tid = w * WARP_SZ + lane;  // 线程在 CTA 内的全局编号
            if (tid >= (int)B) break;     // 超出 block 线程数, 跳过
            warp.lmem_base[lane] = FLAT_LOCAL_BASE +
                (uint64_t)((nsm_id * MAX_WARPS_PER_SM + ws) * WARP_SZ + lane) * LOCAL_WINDOW;
        }

        // CTA 层面的坐标信息 (warp 内所有 lane 共享)
        warp.ctaid_x = cta.ctaid_x; warp.ctaid_y = cta.ctaid_y; warp.ctaid_z = cta.ctaid_z;
        warp.ntid_x = cta.ntid_x; warp.ntid_y = cta.ntid_y; warp.ntid_z = cta.ntid_z;
        warp.nctaid_x = cta.nctaid_x; warp.nctaid_y = cta.nctaid_y; warp.nctaid_z = cta.nctaid_z;
        warp.param_base = 0; // params at const_mem[0], flat base = FLAT_CONST_BASE
        warp.smem_flat_base = cta.smem_flat_base;  // 共享内存基址

        // 初始化 SIMT 栈: 栈底 = {EXIT_SENTINEL, entry=0, mask=lane_valid}
        // EXIT_SENTINEL 表示到达此 PC 时 warp 退出
        // entry=0 表示代码从偏移 0 开始执行
        // mask=lane_valid 表示所有有效 lane 都活跃
        warp.stack.clear();
        StackEntry entry;
        entry.rpc = EXIT_SENTINEL;
        entry.nextpc = ki.code_offset;
        entry.mask = warp.lane_valid;
        entry.dflag = false;
        warp.stack.push_back(entry);
        warp.tos = 0;  // 栈顶指向唯一的栈帧

        cta.warp_slots.push_back(ws);
        if (base_slot < 0) base_slot = ws;
    }

    // 更新 NSM 资源占用
    sm.cur_warps += W;
    sm.cur_ctas += 1;
    sm.cur_smem += smem_need;
    sm.cur_regs += R*W*32;
    return true;
}

// -----------------------------------------------------------------------------
// fetch_inst: 取指 (03 册 §6.1 F 阶段)
// -----------------------------------------------------------------------------
// 教材引用: 第 2 章 §2.2.2 Instruction Encoding (p.18) — fetch_inst 读取 64 位指令字并解码, 对应教材中 64 位指令编码格式与往返一致性
// -----------------------------------------------------------------------------
// 从代码段读取 8 字节 (一条指令), 解码为 Inst 结构
// 参数:
//   pc: 程序计数器 (字节地址, 8 字节对齐)
//   ins: 输出, 解码后的指令
// 返回: 64 位原始指令字 (用于 trace)
// -----------------------------------------------------------------------------
// =============================================================================
// 1.1 修复说明 F05 取指边界
// -----------------------------------------------------------------------------
// 取指验证模块范围和 8 字节对齐; 错误地址返回明确错误, 不静默继续。
// =============================================================================
uint64_t FuncSim::fetch_inst(uint64_t pc, Inst& ins) {
    // 从代码段读取 8 字节
    if ((pc & 7) || pc > code.size() || code.size()-pc < 8) {
        error.code = (pc & 7) ? ERR_MISALIGNED_PC : ERR_ADDR_OUT_OF_RANGE;
        error.pc=pc; error.msg="Invalid instruction address";
        // PC 超出代码段范围: 可能是未定义的 EXIT 或代码损坏
        return 0;
    }
    uint64_t word;
    std::memcpy(&word, code.data() + pc, 8);
    // 解码: 64 位指令字 → Inst 结构 (操作码/寄存器/立即数等)
    int err;
    if(pc/8<decoded_code.size()){ins=decoded_code[pc/8];err=decoded_errors[pc/8];}
    else err = decode(word, pc, ins);
    if (err != ERR_OK) {
        error.code = err;
        error.pc = pc;
        error.msg = "decode error";
    }
    return word;
}

// =============================================================================
// 执行循环与调度 (03 册 §2-§3) ★★ warp 调度核心
// -----------------------------------------------------------------------------
// 教材引用: 第 3 章 §3.1 One-Loop Approximation (p.22) — 执行循环实现"单循环"调度模型, sync 驱动所有 warp 执行直到完成, 每轮每 warp 推进一条指令
// -----------------------------------------------------------------------------
// sync() 是主机端阻塞调用, 驱动整个 GPU 执行直到所有 CTA 完成:
//   1. step_one_round(): 遍历所有 NSM, 每个可运行 warp 执行一条指令
//   2. 检查 CTA 完成情况: 所有 warp EXIT 后释放资源
//   3. 若无进展且非完成: 可能死锁, 报错退出
//
// 调度策略 (功能模拟器):
//   - 轮询调度: 遍历所有 NSM 的所有 warp 槽, 各执行一条指令
//   - 不建模流水线时序: 每条指令立即完成
//   - 真实 GPU 的调度器更复杂: 有 warp 调度策略 (round-robin/GTO 等)
//
// 教学要点:
//   - warp 是调度的最小单元: 要么 32 lane 同时执行, 要么不执行
//   - 在屏障上等待的 warp (bar_wait >= 0) 不会被调度
//   - 已 EXIT 的 warp (exit_done) 不会被调度
//   - 死锁检测: 所有 warp 都在等屏障, 无 warp 可前进 → 死锁
// =============================================================================

// sync: 驱动 GPU 执行直到所有 CTA 完成 (主机端阻塞调用)
// 教材引用: 第 3 章 §3.1 One-Loop Approximation (p.22) — sync 是单循环功能模型的主驱动, 反复调用 step_one_round 直到所有 CTA 完成, 不建模时序
int FuncSim::sync() {
    int max_rounds = 10000000;  // 防止无限循环的安全阀
    while (!is_done() && !has_error()) {
        if (--max_rounds <= 0) {
            error.code = ERR_DEADLOCK;
            error.msg = "sync timeout";
            return error.code;
        }
        bool any_progress = step_one_round();  // 执行一轮 (所有 warp 各一条指令)
        if (!any_progress && !is_done()) {
            error.code = ERR_DEADLOCK;
            error.msg = "No progress before all CTAs completed";
            return error.code;
        }
    }
    return error.code;
}

// step_one_round: 执行一轮调度 (所有 NSM 的所有可运行 warp 各执行一条指令)
// 返回: 是否有任何 warp 取得了进展
// 教材引用: 第 3 章 §3.1.3 Warp Scheduling (p.31) — step_one_round 遍历所有 warp 各执行一条指令, 对应教材中 round-robin 轮询调度策略
bool FuncSim::step_one_round() {
    bool any_progress = false;
    // 遍历所有 NSM
    for (int sm = 0; sm < (int)nsms.size(); sm++) {
        NSMState& nsm = nsms[sm];
        if(nsm.cur_ctas==0)continue;
        // 遍历该 NSM 的所有 warp 槽
        for (int w = 0; w < cfg.max_warps_per_core; w++) {
            // 跳过无效/已退出/等屏障的 warp
            if (!nsm.warps[w].valid || nsm.warps[w].exit_done) continue;
            if (nsm.warps[w].bar_wait >= 0) continue;  // 在屏障上等待
            int err = step_warp(nsm, w);  // 执行一条指令
            if (err != ERR_OK) {
                error.code = err;
                error.sm = sm;
                error.warp = w;
                return false;
            }
            any_progress = true;
        }
    }
    // 检查 CTA 完成情况: 所有 warp EXIT → 释放资源
    for (int sm = 0; sm < (int)nsms.size(); sm++) {
        NSMState& nsm = nsms[sm];
        if(nsm.cur_ctas==0)continue;
        for (int c = 0; c < cfg.max_ctas_per_core; c++) {
            if (!nsm.ctas[c].running) continue;
            CTAState& cta = nsm.ctas[c];
            // 检查该 CTA 的所有 warp 是否都已 EXIT
            bool all_done = true;
            for (int ws : cta.warp_slots) {
                if (!nsm.warps[ws].exit_done) { all_done = false; break; }
            }
            if (all_done) {
                // 释放资源: warp 槽位/寄存器/共享内存
                for (int ws : cta.warp_slots) {
                    nsm.warps[ws].valid = false;  // 回收 warp 槽位
                }
                nsm.cur_warps -= cta.num_warps;   // 减少 warp 占用数
                nsm.cur_ctas -= 1;                 // 减少 CTA 占用数
                nsm.cur_smem -= cta.allocated_smem;
                nsm.cur_regs -= cta.regs_per_thread*cta.num_warps*32;
                cta.running = false;
                completed_ctas++;  // 已完成 CTA 计数器
            }
        }
    }
    any_progress = dispatch_pending() || any_progress;
    return any_progress;
}

// check_deadlock: 死锁检测
// 若有运行中的 CTA 但没有 warp 可前进 (都在等屏障) → 死锁
bool FuncSim::check_deadlock() {
    // 检查是否有可前进的 warp
    for (int sm = 0; sm < (int)nsms.size(); sm++) {
        for (int w = 0; w < cfg.max_warps_per_core; w++) {
            WarpState& warp = nsms[sm].warps[w];
            if (warp.valid && !warp.exit_done) {
                if (warp.bar_wait >= 0) continue; // waiting on barrier
                return false; // can progress
            }
        }
    }
    // 无可前进的 warp. 若有运行中的 CTA → 死锁
    for (int sm = 0; sm < (int)nsms.size(); sm++) {
        for (int c = 0; c < cfg.max_ctas_per_core; c++) {
            if (nsms[sm].ctas[c].running) return true;  // 死锁!
        }
    }
    return false;
}

// is_done: 所有 CTA 是否都已完成
bool FuncSim::is_done() const {
    return completed_ctas >= total_ctas;
}

// =============================================================================
// SIMT 栈操作 (03 册 §4) ★★★ 最核心的 GPU 微结构概念
// -----------------------------------------------------------------------------
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — SIMT 栈管理分支发散与重汇聚, 对应教材中 post-dominator 重汇聚点与栈帧机制
// -----------------------------------------------------------------------------
// SIMT (Single Instruction Multiple Thread) 执行模型中, 32 个线程组成一个 warp,
// 共享同一个 PC. 当遇到条件分支 (BRA) 时, 不同线程可能走向不同路径,
// 这称为"分支发散" (branch divergence).
//
// NutShellGPU 使用"SIMT 栈" (也叫 Post-Dominator 栈) 管理发散步径的重汇聚:
//   - SSY 指令: 在分支前设置"重汇聚点" (reconvergence point), 压入栈帧
//   - BRA 指令: 按谓词掩码判断是否发散:
//       全部走同一路径 → 不发散, 原地更新 nextpc
//       部分走不同路径 → 发散! 压入两个子栈帧 (taken + fall-through)
//   - 到达重汇聚点 (rpc) 时, 弹出栈帧, 合并掩码 → 所有线程重汇聚
//
// 栈帧结构 (StackEntry):
//   rpc:    重汇聚点 PC (到达此 PC 时弹出)
//   nextpc: 当前要执行的下一条指令 PC
//   mask:   32 位活动掩码 (哪些 lane 在此路径上活跃)
//   dflag:  发散标志 (true=此帧由发散产生, 需用 R2 规则弹出)
//
// 弹出规则 (在 pop_reconverge 中实现):
//   R1: 子路径到达 join 点 (newpc == tos.rpc && !tos.dflag) → 直接弹出
//   R2: 父帧执行了 join 指令 (pc == tos.rpc && tos.dflag) → 弹出父帧
// =============================================================================

// -----------------------------------------------------------------------------
// SSY: 设置重汇聚点 (03 册 §4.2)
// -----------------------------------------------------------------------------
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — SSY 在条件分支前压入重汇聚点栈帧, 对应教材中分支前的 post-dominator 标记
// -----------------------------------------------------------------------------
// 在条件分支前调用, 压入一个新栈帧, 记录重汇聚点
//
// 参数:
//   w: warp 状态
//   target: 重汇聚点 PC (通常是分支后的 join 点)
//
// 操作:
//   1. 检查栈是否已满 (SIMT_STACK_MAX=32)
//   2. 创建新栈帧: rpc=target, nextpc=PC+8 (继续顺序执行), mask=继承TOS, dflag=false
//   3. 压入栈, 更新 TOS 指针
//   4. 若启用快照, 记录 SSY 事件
// -----------------------------------------------------------------------------
void FuncSim::ssy(WarpState& w, uint64_t target) {
    if (w.stack.size() >= SIMT_STACK_MAX) {
        error.code = ERR_STACK_OVERFLOW;   // 栈溢出: 嵌套发散超过 32 层
        return;
    }
    StackEntry& tos = w.stack[w.tos];
    StackEntry ne;
    ne.rpc = target;                         // 重汇聚点 = SSY 的目标
    ne.nextpc = w.stack[w.tos].nextpc + 8;   // 下一条指令 = PC+8 (顺序执行)
    ne.mask = tos.mask;                       // 继承当前活动掩码
    ne.dflag = false;                         // 非发散产生
    w.stack.push_back(ne);                    // 压入栈
    w.tos = (int)w.stack.size() - 1;         // 更新栈顶指针
    // 快照记录 (测试发散逻辑用)
    if (snap_enabled) {
        StackSnap s;
        s.pc = w.stack[w.tos].rpc;
        s.event = "SSY";
        s.stack = w.stack;
        s.exec_mask = w.stack[w.tos].mask;
        snaps.push_back(s);
    }
}

// -----------------------------------------------------------------------------
// BRA: 条件分支 (03 册 §4.3) ★ 核心发散处理逻辑
// -----------------------------------------------------------------------------
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — BRA 根据谓词掩码判断发散, 压入 taken/fall-through 子栈帧, 对应教材中分支发散与串行路径执行
// -----------------------------------------------------------------------------
// 根据谓词掩码判断 warp 是否发散, 更新 SIMT 栈
//
// 参数:
//   w: warp 状态
//   target: 分支目标 PC (taken 路径)
//   guard_mask: 谓词为真的 lane 集合 (32 位掩码)
//
// 三种情况:
//   1. 全部走分支 (taken == m): 不发散, nextpc = target
//   2. 全部不走分支 (taken == 0): 不发散, nextpc = PC+8
//   3. 部分走分支 (0 < taken < m): 发散! 需要压入两个子栈帧
//
// 发散处理:
//   - 修改父帧: nextpc = join (重汇聚点), dflag = true (标记发散)
//   - 压入 fall-through 子帧: nextpc = PC+8, mask = fall (不走的 lane)
//   - 压入 taken 子帧: nextpc = target, mask = taken (走的 lane)
//   - 栈顶 (TOS) 指向最后压入的帧 (优先执行的路径)
//
// 教学要点:
//   - 发散后, SIMT 栈使两条路径"串行"执行 (先走 taken, 再走 fall)
//   - 到达 join 点时, pop_reconverge 合并掩码, warp 重汇聚
//   - 这种机制保证: 同一 warp 内的线程最终在重汇聚点同步
// -----------------------------------------------------------------------------
int FuncSim::bra(WarpState& w, uint64_t target, uint32_t guard_mask) {
    StackEntry& tos = w.stack[w.tos];
    uint32_t m = tos.mask;               // 当前活动掩码
    uint32_t taken = m & guard_mask;     // 走分支的 lane
    uint32_t fall = m & ~guard_mask;      // 不走分支的 lane (fall-through)

    if (taken == m) {
        // 情况 1: 所有活跃 lane 都走分支 → 不发散
        tos.nextpc = target;
        return ERR_OK;
    }
    if (taken == 0) {
        // 情况 2: 没有活跃 lane 走分支 → 不发散, 顺序执行
        tos.nextpc = tos.nextpc + 8;
        return ERR_OK;
    }
    // 情况 3: 分支发散! 部分走, 部分不走
    if (tos.rpc == EXIT_SENTINEL) {
        return ERR_NO_IPDOM;   // 无重汇聚点 (栈底), 无法发散
    }
    uint64_t join = tos.rpc;              // 重汇聚点 = 父帧的 rpc
    // 修改父帧: 标记为已发散, nextpc 指向 join
    tos.nextpc = join;
    tos.dflag = true;
    // 检查栈空间 (需压入 2 个子帧)
    if (w.stack.size() + 2 > SIMT_STACK_MAX) {
        return ERR_STACK_OVERFLOW;
    }
    // 压入 fall-through 子帧 (不走分支的路径)
    StackEntry fall_entry;
    fall_entry.rpc = join;
    fall_entry.nextpc = tos.nextpc + 8 - 8; // PC+8 (fall-through 路径的起始 PC)
    // NOTE: 此处 PC+8 计算依赖调用者传入的 BRA 指令 PC
    // 实际实现在 execute_inst 的 OP_BRA 分支中处理
    (void)fall_entry; // placeholder
    divergent_branches++;                 // 统计发散分支数
    return ERR_OK;
}

// -----------------------------------------------------------------------------
// BRX: 间接分支 (03 册 §4.4)
// -----------------------------------------------------------------------------
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — BRX 每 lane 目标不同导致多路发散, 对应教材中嵌套发散与多路径重汇聚
// -----------------------------------------------------------------------------
// 每 lane 的分支目标来自寄存器 (可能各不相同), 导致多路发散
// 实际实现在 execute_inst 的 OP_BRX 分支中处理 (需要逐 lane 分组)
// 此处为存根 (stub)
// -----------------------------------------------------------------------------
int FuncSim::brx(WarpState& w, const std::array<uint64_t,32>& targets, int n_paths) {
    (void)w; (void)targets; (void)n_paths;
    return ERR_OK;
}

// =============================================================================
// step_warp: 执行一个 warp 的一条指令 (03 册 §6) ★★ 流水线核心
// -----------------------------------------------------------------------------
// 教材引用: 第 3 章 §3.1 One-Loop Approximation (p.22) — step_warp 是单循环模型的核心: 取一条指令, 计算 exec_mask, 执行, 每次推进一个 warp 一条指令
// -----------------------------------------------------------------------------
// 这是 warp 调度的核心: 取一条指令, 计算执行掩码, 执行指令
// 步骤 (对应真实 GPU 的流水线阶段):
//   1. F (Fetch):     从代码段读取 8 字节指令 (fetch_inst)
//   2. D (Decode):    解码指令, 得到操作码/寄存器/立即数 (在 fetch_inst 中)
//   3. 谓词计算:       根据谓词字段 (g/gp/gn) 计算 guard_mask (哪些 lane 执行)
//   4. I (Issue):     计算 exec_mask = TOS.mask & guard_mask & lane_valid
//   5. 执行:          调用 execute_inst 执行指令语义
//
// 谓词保护 (Guard Predicate) 机制:
//   每条指令可带谓词 @Pn 或 @!Pn, 只有谓词为真的 lane 才执行
//   - ins.g = 1: 有谓词保护
//   - ins.gp: 谓词寄存器号 (0-6, 7=PT 恒真)
//   - ins.gn: 谓词取反 (0=@Pn, 1=@!Pn)
//   - guard_mask: 谓词为真的 lane 集合
//
// 三层掩码:
//   1. lane_valid: 永久有效的 lane (warp 不足 32 线程时高位为 0)
//   2. tos.mask:   SIMT 栈帧的活动掩码 (分支发散后每路径不同)
//   3. guard_mask: 谓词掩码 (当前指令的谓词保护)
//   exec_mask = lane_valid & tos.mask & guard_mask → 实际执行的 lane
// 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) — 三层掩码交集计算 exec_mask, 对应教材中 active mask 与谓词保护机制, 决定哪些 lane 执行当前指令
// =============================================================================
int FuncSim::step_warp(NSMState& sm, int warp_slot) {
    WarpState& w = sm.warps[warp_slot];
    if (!w.valid || w.exit_done || w.bar_wait >= 0) return ERR_OK;

    StackEntry& tos = w.stack[w.tos];
    uint64_t pc = tos.nextpc;   // 从 SIMT 栈顶获取下一条指令 PC

    // F + D 阶段: 取指 + 解码
    Inst ins;
    uint64_t word = fetch_inst(pc, ins);
    if (has_error()) return error.code;
    (void)word;

    // 计算谓词保护掩码 (guard_mask)
    // 只对活跃且有效的 lane 检查谓词
    uint32_t guard_mask = 0;
    if (ins.g) {
        // 有谓词保护: 逐 lane 检查谓词
        for (int lane = 0; lane < WARP_SZ; lane++) {
            if (!(w.lane_valid & (1u << lane))) continue;   // lane 无效, 跳过
            if (!(tos.mask & (1u << lane))) continue;       // lane 在此路径不活跃, 跳过
            bool pv = w.pred(lane, ins.gp);                 // 读取谓词值
            bool enabled = ins.gn ? !pv : pv;               // gn=1 取反
            if (enabled) guard_mask |= (1u << lane);        // 加入 guard_mask
        }
    } else {
        // 无谓词保护: guard_mask = 所有活跃 lane
        guard_mask = tos.mask & w.lane_valid;
    }

    // exec_mask = 三层掩码的交集 (实际执行的 lane 集合)
    uint32_t exec_mask = tos.mask & guard_mask & w.lane_valid;

    // 统计 SIMT 栈最大深度 (性能分析用)
    if ((int)w.stack.size() > max_stack_depth) max_stack_depth = (int)w.stack.size();

    // 记录 trace (调试用): 记录每条指令的执行信息
    if (trace_enabled) {
        TraceEntry te;
        te.sm = sm.id; te.warp = warp_slot;
        te.pc = pc; te.op = ins.op;
        te.exec_mask = exec_mask;
        // 控制流指令记录 SIMT 栈快照 (用于分析发散行为)
        if (ins.op == OP_BRA || ins.op == OP_BRX || ins.op == OP_SSY ||
            ins.op == OP_EXIT || ins.op == OP_CALL || ins.op == OP_RET) {
            te.has_stack = true;
            te.stack_snapshot = w.stack;
        }
        trace.push_back(te);
    }

    inst_count++;   // 指令计数器 (统计执行了多少条指令)
    w.clocklo++;    // warp 时钟计数器 (S2R SR_CLOCKLO)

    // 执行指令 (调用 execute_inst 分发到各指令的语义实现)
    int err = execute_inst(w, ins, exec_mask);
    if (err != ERR_OK) {
        error.code = err;
        error.sm = sm.id;
        error.warp = warp_slot;
        error.pc = pc;
        return err;
    }
    return ERR_OK;
}

// =============================================================================
// SIMT 栈重汇聚与线程退出 (03 册 §4.5-§4.6) ★★★ 核心微结构
// -----------------------------------------------------------------------------
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — pop_reconverge/do_exit/bar_sync 处理重汇聚/退出/屏障, 对应教材中分支发散后的重汇聚点合并掩码
// -----------------------------------------------------------------------------
// pop_reconverge: 检查并执行 SIMT 栈弹出 (重汇聚)
// do_exit:        处理 EXIT 指令 (线程退出)
// bar_sync:       屏障同步 (BAR 指令)
// =============================================================================

// -----------------------------------------------------------------------------
// pop_reconverge: 检查 SIMT 栈弹出条件, 执行重汇聚 (03 册 §4.5)
// -----------------------------------------------------------------------------
// 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — pop_reconverge 用 R1/R2 规则检查重汇聚点弹出栈帧, 对应教材中 post-dominator 重汇聚与掩码合并
// -----------------------------------------------------------------------------
// 每条指令执行后调用, 检查是否到达重汇聚点 (rpc), 决定是否弹出栈帧
//
// 参数:
//   w: warp 状态
//   pc: 当前指令 PC (刚执行的指令地址)
//   newpc: 下一条指令 PC (指令执行后设置的)
//
// 弹出规则:
//   R1: 子路径到达 join 点 (newpc == tos.rpc && !tos.dflag) → 弹出此帧
//       说明: 非发散帧 (dflag=false) 的子路径执行到了其重汇聚点
//   R2: 父帧执行了 join 指令 (pc == tos.rpc && tos.dflag) → 弹出此帧
//       说明: 发散帧 (dflag=true) 的父路径执行到了重汇聚点
//
// 弹出后:
//   - 栈顶变为父帧, 父帧的 mask 已经在 BRA 时设为 join 的预期掩码
//   - 继续检查新栈顶是否也需要弹出 (可能连续弹出多个帧)
//   - 特殊情况: 若新栈顶 nextpc == rpc, 说明该帧还未执行, 不弹出
// -----------------------------------------------------------------------------
void FuncSim::pop_reconverge(WarpState& w, uint64_t pc, uint64_t newpc) {
    while (w.tos >= 0 && (int)w.stack.size() > 1) {
        StackEntry& tos = w.stack[w.tos];
        bool popped = false;

        // R1: 子路径到达 join 点
        // newpc == tos.rpc: 下一条指令正好是重汇聚点
        // !tos.dflag: 此帧非发散产生 (是 fall/taken 子帧)
        if (newpc == tos.rpc && !tos.dflag) {
            w.stack.pop_back();
            w.tos--;
            popped = true;
        }

        // R2: 父帧执行了 join 指令
        // pc == tos.rpc: 当前指令的 PC 正好是重汇聚点
        // tos.dflag: 此帧由发散产生 (BRA 修改的父帧, SSY 压的帧)
        if (!popped && tos.dflag && pc == tos.rpc) {
            w.stack.pop_back();
            w.tos--;
            popped = true;
        }

        if (!popped) break;  // 不满足弹出条件, 停止

        // 只剩栈底帧: 更新其 nextpc, 让 warp 继续执行 join 点后的指令(如 EXIT)
        if (w.tos == 0) {
            w.stack[0].nextpc = newpc;
            break;
        }

        // 中间帧: 不更新 nextpc!
        // 新 TOS 可能还没执行完, 保持其 nextpc 让 warp 正常执行
        // 若新 TOS 也到达 join 点(nextpc==rpc), 继续循环弹出
        if (w.stack[w.tos].nextpc != w.stack[w.tos].rpc) {
            break;  // 新 TOS 还没执行到 join 点, 停止
        }
    }
}

// -----------------------------------------------------------------------------
// do_exit: 处理 EXIT 指令 (03 册 §4.6)
// -----------------------------------------------------------------------------
// 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) — do_exit 从 active mask 与 lane_valid 中清除已退出 lane, 对应教材中 SIMT 栈与活动掩码管理
// -----------------------------------------------------------------------------
// EXIT 指令使当前 warp 的活跃 lane 退出
// 操作:
//   1. 从所有栈帧的 mask 中清除已退出的 lane
//   2. 从 lane_valid 中清除已退出的 lane
//   3. 弹出 mask 为空的栈帧
//   4. 若所有 lane 都退出 (lane_valid == 0), 标记 warp 已完成
//
// 教学要点:
//   - 不同 lane 可能在不同时间 EXIT (分支发散后)
//   - EXIT 后, 该 lane 不再参与后续指令执行
//   - 当所有 lane 都 EXIT, warp 释放资源 (在 step_one_round 中处理)
// -----------------------------------------------------------------------------
int FuncSim::do_exit(WarpState& w, uint32_t exec_mask) {
    // 从所有栈帧中清除已退出的 lane
    for (auto& se : w.stack) {
        se.mask &= ~exec_mask;
    }
    w.lane_valid &= ~exec_mask;  // 从永久 lane 有效性中清除

    // 弹出 mask 为空的栈帧
    while (w.tos >= 0 && (int)w.stack.size() > 1 && w.stack[w.tos].mask == 0) {
        w.stack.pop_back();
        w.tos--;
    }

    // 若栈为空或所有 lane 都退出, 标记 warp 完成
    if (w.stack.empty() || w.lane_valid == 0) {
        w.exit_done = true;
    }
    return ERR_OK;
}

// -----------------------------------------------------------------------------
// bar_sync: 屏障同步 (03 册 §5) ★★ 线程同步核心
// -----------------------------------------------------------------------------
// 教材引用: 第 3 章 §3.1.1 SIMT Execution Masking (p.23) — bar_sync 实现 CTA 内 warp 屏障同步, 挂起/释放 warp, 对应教材中 __syncthreads 与 SIMT 执行控制
// -----------------------------------------------------------------------------
// BAR 指令实现 CTA 内的 warp 同步:
//   - ARRIVE: warp 到达屏障, 标记其 lane 已到达
//   - WAIT:   若所有 lane 都到达, 释放所有等待的 warp; 否则挂起当前 warp
//
// 屏障状态 (BarrierState):
//   active:   屏障是否激活 (有 warp 在等待)
//   expected: 期望到达的 lane 掩码 (CTA 内所有有效 lane)
//   arrived: 已到达的 lane 掩码
//   waiters:  等待的 warp 列表 [(warp_slot, mask), ...]
//
// 教学要点:
//   - 屏障是 CTA 级别的同步: 同一 CTA 的所有 warp 必须都到达
//   - 不同 CTA 的 warp 不共享屏障
//   - warp 到达屏障后被挂起 (bar_wait = id), 调度器不再调度它
//   - 当所有 warp 到达, 释放所有等待的 warp (bar_wait = -1)
//   - 常见用途: __syncthreads() → BAR.SYNC
// -----------------------------------------------------------------------------
// =============================================================================
// 1.1 修复说明 F06 可重用屏障代次
// -----------------------------------------------------------------------------
// 每个 warp 在同一代只登记一次; 满足本代参与数后唤醒等待者并进入下一代。
// ARRIVE/WAIT/SYNC 按代次配合, 防止提前释放或跨代重复计数。
// =============================================================================
int FuncSim::bar_sync(WarpState& w,uint8_t id,uint32_t mask,uint8_t sub) {
    if(id>=16 || sub>2) return ERR_BAR_ID;
    if(mask!=w.lane_valid) return ERR_BAR_PARTIAL; // Explicitly reject divergent barriers.
    auto& cta=nsms[w.nsm_id].ctas[w.cta_slot];auto& b=cta.barriers[id];
    auto& phase=b.arrival_phase[w.warp_id_in_cta];
    if(sub!=2){
        if(phase>b.generation) return ERR_BAR_PARTIAL; // duplicate arrival
        if(!b.active){b.active=true;b.arrived=0;b.expected=0;
            for(int ws:cta.warp_slots)b.expected+=popcount32(nsms[w.nsm_id].warps[ws].lane_valid);}
        phase=b.generation+1;b.arrived+=popcount32(mask);
        if(b.arrived==b.expected){++b.generation;b.active=false;b.arrived=0;
            for(auto& waiter:b.waiters)nsms[w.nsm_id].warps[waiter.first].bar_wait=-1;
            b.waiters.clear();}
    }else if(phase==0)return ERR_BAR_PARTIAL;
    if(sub!=1 && phase>b.generation){w.bar_wait=id;b.waiters.push_back({w.slot,mask});}
    else w.bar_wait=-1;
    w.stack[w.tos].nextpc+=8;
    return ERR_OK;
}

// =============================================================================
// 特殊寄存器 (Special Registers, S2R 指令读取) (01 册 §4, 02 册 §7)
// -----------------------------------------------------------------------------
// 教材引用: 第 2 章 §2.1 Programming Model (p.10) — 特殊寄存器 TID/CTAID/NTID 对应 CUDA 编程模型中线程索引 i=blockIdx.x*blockDim.x+threadIdx.x 的坐标计算
// -----------------------------------------------------------------------------
// GPU 线程需要访问自身坐标信息 (TID/CTAID/NTID 等) 来计算数据地址
// S2R (Special Register to Register) 指令读取这些特殊寄存器
//
// 坐标系统 (与 CUDA 一致):
//   线程坐标 (TID):  lane → (tid_x, tid_y, tid_z)
//     tid_x = lane % ntid_x
//     tid_y = (lane / ntid_x) % ntid_y
//     tid_z = lane / (ntid_x * ntid_y)
//   CTA 坐标 (CTAID): 线程块在 grid 中的位置 (ctaid_x/y/z)
//   维度信息 (NTID/NCTAID): block/grid 的维度
//
// 其他特殊寄存器:
//   SR_SMID:       NSM 编号 (硬件 ID, 用于设备查询)
//   SR_LANEID:     lane 编号 (0-31, warp 内线程号)
//   SR_WARPID:     warp 编号 (CTA 内)
//   SR_CLOCKLO/HI: 时钟计数器 (低/高 32 位, 性能分析用)
//   SR_LR:         链接寄存器 (CALL/RET 用)
//   SR_PARAM_BASE: 参数区 flat 基址
//   SR_SMEM_BASE:  共享内存 flat 基址
//   SR_LMEM_BASE:  本地内存 flat 基址 (每 lane 不同)
// =============================================================================
uint32_t FuncSim::get_sr(WarpState& w, int lane, uint8_t sr, uint32_t* hi) {
    // 全局线程号 (CTA 内): warp_id_in_cta * WARP_SZ + lane
    int tid = w.warp_id_in_cta * WARP_SZ + lane;
    switch(sr) {
    // 线程坐标 (Thread ID): 从 CTA 内全局线程号解码 3D 坐标
    case SR_TID_X: return tid % w.ntid_x;                          // x 维坐标
    case SR_TID_Y: return (tid / w.ntid_x) % w.ntid_y;             // y 维坐标
    case SR_TID_Z: return tid / (w.ntid_x * w.ntid_y);             // z 维坐标
    // CTA 坐标 (Cooperative Thread Array ID): 线程块在 grid 中的位置
    case SR_CTAID_X: return w.ctaid_x;
    case SR_CTAID_Y: return w.ctaid_y;
    case SR_CTAID_Z: return w.ctaid_z;
    // Block 维度 (Number of Threads In Dimension): 每 CTA 的线程数
    case SR_NTID_X: return w.ntid_x;
    case SR_NTID_Y: return w.ntid_y;
    case SR_NTID_Z: return w.ntid_z;
    // Grid 维度 (Number of CTAs In Dimension): CTA 总数
    case SR_NCTAID_X: return w.nctaid_x;
    case SR_NCTAID_Y: return w.nctaid_y;
    case SR_NCTAID_Z: return w.nctaid_z;
    // 硬件 ID
    case SR_SMID: return w.smid;           // NSM 编号
    case SR_CLOCKLO: return (uint32_t)w.clocklo;              // 时钟低 32 位
    case SR_CLOCKHI: return (uint32_t)(w.clocklo >> 32);      // 时钟高 32 位
    case SR_LANEID: return lane;           // lane 编号 (0-31)
    case SR_WARPID: return w.warpid;       // warp 编号
    case SR_LR: return w.lr[lane];         // 链接寄存器 (CALL/RET)
    // 内存基址 (64 位, 返回低 32 位, 高 32 位通过 *hi 返回)
    case SR_PARAM_BASE:
        if (hi) *hi = (uint32_t)(w.param_base >> 32);
        return (uint32_t)w.param_base;
    case SR_SMEM_BASE:
        if (hi) *hi = (uint32_t)(w.smem_flat_base >> 32);
        return (uint32_t)w.smem_flat_base;
    case SR_LMEM_BASE:
        if (hi) *hi = (uint32_t)(w.lmem_base[lane] >> 32);
        return (uint32_t)w.lmem_base[lane];
    case SR_WARPID_IN_GRID: return w.warpid; // simplified
    case SR_GRIDID: return 0;
    default: return 0;
    }
}

// =============================================================================
// 内存系统 (04 册) ★★★ GPU 内存层次核心
// -----------------------------------------------------------------------------
// 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — 内存系统支持 global/shared/local/const 四种地址空间 + flat 统一地址, 对应教材中 L1/shared/const 内存层次
// -----------------------------------------------------------------------------
// NutShellGPU 支持 4 种地址空间 + flat 统一地址:
//   1. Global (全局内存): 显存, 所有线程可见, 容量大但延迟高
//   2. Shared (共享内存): CTA 内可见, 48 KiB/NSM, 低延迟, bank 分组
//   3. Local (本地内存): 线程私有, 存放在 global memory 中
//   4. Const (常量内存): 只读, 通过 LDC 指令访问, 有专用缓存
//
// Flat 地址 (统一地址空间, 04 册 §4):
//   将所有地址空间映射到 64 位 flat 地址空间:
//     [0, FLAT_SHARED_BASE):           Global 内存
//     [FLAT_SHARED_BASE, FLAT_LOCAL_BASE): Shared 内存 (每 CTA 独立)
//     [FLAT_LOCAL_BASE, FLAT_CONST_BASE):  Local 内存 (每线程独立)
//     [FLAT_CONST_BASE, ...):              Const 内存
//   LD/ST 的 space=AS_FLAT 时, 用 decode_flat 解码实际空间
//
// 教学要点:
//   - 真实 GPU 用 CVTA 指令在 space 地址和 flat 地址间转换
//   - 共享内存用 bank 分组 (32 bank × 4 字节), 需避免 bank 冲突
//   - Global 内存访问有 coalescing 优化 (warp 内合并连续地址)
//   - 功能模拟器不建模 bank 冲突/coalescing, 仅保证功能正确
// =============================================================================

// -----------------------------------------------------------------------------
// decode_flat: 解码 flat 地址为实际地址空间 + 偏移 (04 册 §4)
// -----------------------------------------------------------------------------
// 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — decode_flat 将统一 flat 地址解码为实际地址空间, 对应教材中 GPU 内存层次与地址空间映射
// -----------------------------------------------------------------------------
// 根据 flat 地址的区间判断属于哪个地址空间
// 参数:
//   flat: 64 位 flat 地址
//   space: 输出, 地址空间 (AS_GLOBAL/AS_SHARED/AS_LOCAL/AS_CONST)
//   offset: 输出, 在该地址空间内的偏移
//   lane: 线程编号 (local/shared 地址依赖 lane)
//   w: warp 状态 (提供 smem_flat_base/lmem_base)
// -----------------------------------------------------------------------------
int FuncSim::decode_flat(uint64_t flat, uint8_t& space, uint64_t& offset, int lane, WarpState& w) {
    if (flat < FLAT_SHARED_BASE) {
        // Global 内存: flat 地址直接就是 global 地址
        space = AS_GLOBAL;
        offset = flat;
    } else if (flat < FLAT_LOCAL_BASE) {
        // Shared 内存: 减去 CTA 的共享内存基址得到 CTA 内偏移
        space = AS_SHARED;
        offset = flat - w.smem_flat_base;
    } else if (flat < FLAT_CONST_BASE) {
        // Local 内存: 减去 FLAT_LOCAL_BASE 得到全局偏移
        // 注意: 真实实现中 local 地址还需加上每线程的 lmem_base
        space = AS_LOCAL;
        // 1.1 F13: 与 CVTA 使用的每 lane 基址互逆, 不重复叠加 lane 偏移。
        offset = flat - w.lmem_base[lane];
    } else {
        // Const 内存: 减去 FLAT_CONST_BASE 得到常量区偏移
        space = AS_CONST;
        offset = flat - FLAT_CONST_BASE;
    }
    return ERR_OK;
}

// -----------------------------------------------------------------------------
// load_mem: 从指定地址空间加载数据 (04 册 §3) ★ 内存访问核心
// -----------------------------------------------------------------------------
// 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — load_mem 按 global/shared/local/const 地址空间加载, 对应教材中 L1/shared/const 缓存层次
// -----------------------------------------------------------------------------
// 根据 ins.mem_space() 指定的地址空间读取数据
// 支持 4 种地址空间 + flat 地址 (自动解码)
//
// 参数:
//   w: warp 状态 (提供 smem_base/lmem_base)
//   ins: 指令 (包含 space/width 信息)
//   lane: 线程编号 (local/shared 地址依赖 lane)
//   addr: 访存地址
//   out_val: 输出, 读取的数据 (32 位)
//
// 数据宽度 (width_bytes):
//   W_U8/W_S8  → 1 字节
//   W_U16/W_S16 → 2 字节
//   W_U32/W_S32 → 4 字节
//   W_U64/W_F64 → 8 字节 (返回低 32 位, 高 32 位由调用者处理)
//   W_U128     → 16 字节 (4 个寄存器)
//
// 符号扩展 (extend_load):
//   有符号加载 (S8/S16/S32) 时, 扩展到 32 位
// -----------------------------------------------------------------------------
// 1.1 修复 F13: flat 解码后的 offset 不能被原 flat 地址覆盖;
// 宽加载逐字复用统一的边界检查, 避免 local/const 高字丢失。
int FuncSim::load_mem(WarpState& w,const Inst& ins,int lane,uint64_t addr,uint32_t& out_val){
    uint8_t space=ins.mem_space();uint64_t offset=addr;
    if(space==AS_FLAT){int e=decode_flat(addr,space,offset,lane,w);if(e)return e;}
    unsigned size=width_bytes(ins.mem_width());uint64_t value=0;
    if(space==AS_GLOBAL||space==AS_LOCAL){
        uint64_t base=offset+(space==AS_LOCAL?w.lmem_base[lane]:0);
        for(unsigned j=0;j<std::min(size,4u);++j)value|=uint64_t(global_mem.r8(base+j))<<(8*j);
    }else if(space==AS_SHARED||space==AS_CONST){
        auto& data=space==AS_SHARED?nsms[w.nsm_id].ctas[w.cta_slot].shared_mem:const_mem;
        if(offset>data.size()||size>data.size()-offset)return ERR_ADDR_OUT_OF_RANGE;
        for(unsigned j=0;j<std::min(size,4u);++j)value|=uint64_t(data[offset+j])<<(8*j);
    }else return ERR_ADDR_OUT_OF_RANGE;
    out_val=extend_load(uint32_t(value),ins.mem_width());return ERR_OK;
}

// -----------------------------------------------------------------------------
// store_mem: 向指定地址空间存储数据 (04 册 §3)
// -----------------------------------------------------------------------------
// 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — store_mem 按地址空间写入, const 只读, 对应教材中 L1 写策略与只读常量内存
// -----------------------------------------------------------------------------
// 参数和逻辑与 load_mem 类似, 但方向相反 (写而非读)
// 注意: Const 内存只读, 写入返回 ERR_BAD_ENCODING
// 64 位写入: val 是低 32 位, 高 32 位从 ins.d+1 寄存器读取
// 128 位写入: 写 4 个寄存器 (d, d+1, d+2, d+3)
// -----------------------------------------------------------------------------
// 1.1 修复 F13: 统一各地址空间的宽写入, 使用字节访问避免宿主未对齐 UB。
int FuncSim::store_mem(WarpState& w,const Inst& ins,int lane,uint64_t addr,uint32_t val){
    uint8_t space=ins.mem_space();uint64_t offset=addr;
    if(space==AS_FLAT){int e=decode_flat(addr,space,offset,lane,w);if(e)return e;}
    unsigned size=width_bytes(ins.mem_width());
    if(space==AS_CONST)return ERR_BAD_ENCODING;
    if(space==AS_SHARED){auto& data=nsms[w.nsm_id].ctas[w.cta_slot].shared_mem;
        if(offset>data.size()||size>data.size()-offset)return ERR_ADDR_OUT_OF_RANGE;}
    else if(space!=AS_GLOBAL&&space!=AS_LOCAL)return ERR_ADDR_OUT_OF_RANGE;
    uint64_t base=offset+(space==AS_LOCAL?w.lmem_base[lane]:0);
    const auto& read=static_cast<const WarpState&>(w);
    for(unsigned j=0;j<size;++j){uint32_t v=j<4?val:(ins.d==RZ?0:read.reg(lane,ins.d+j/4));uint8_t byte=v>>(8*(j%4));
        if(space==AS_SHARED)nsms[w.nsm_id].ctas[w.cta_slot].shared_mem[offset+j]=byte;
        else global_mem.w8(base+j,byte);
    }return ERR_OK;
}

// -----------------------------------------------------------------------------
// atom_op: 原子操作 (04 册 §3.5) ★ 线程间同步
// -----------------------------------------------------------------------------
// 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — atom_op 实现读-改-写原子操作, 真实 GPU 在 L2/ROP 层完成, 对应教材中内存分区单元的原子操作
// -----------------------------------------------------------------------------
// 原子操作: 读-改-写 (Read-Modify-Write) 不可分割, 用于线程间同步
// 支持 12 种原子操作:
//   AOP_EXCH: 交换 (d ← old, mem ← arg)
//   AOP_ADD:  原子加 (d ← old, mem ← old + arg)
//   AOP_MIN/MAX: 有符号最小/最大
//   AOP_UMIN/UMAX: 无符号最小/最大
//   AOP_INC/DEC: 自增/自减 (模 arg)
//   AOP_CAS: 比较交换 (if old==cmp then mem←arg)
//   AOP_AND/OR/XOR: 位运算
//
// 参数:
//   arg: 操作数 (要加的值/要交换的值等)
//   cmp: 比较值 (仅 CAS 用)
//   old: 输出, 操作前的旧值 (返回给寄存器)
//
// 教学要点:
//   - 原子操作保证: 读和写之间无其他线程插入
//   - 真实 GPU 的 global 原子操作在 L2/ROP 层完成
//   - shared 内存的原子操作在共享内存 bank 中串行化
//   - 功能模拟器简化: 直接读-改-写, 不建模并发
// -----------------------------------------------------------------------------
int FuncSim::atom_op(WarpState& w, const Inst& ins, int lane, uint64_t addr,
                     uint32_t arg, uint32_t cmp, uint32_t& old) {
    uint8_t space = ins.mem_space();
    uint8_t op = ins.y & 0xF;          // 原子操作码 (AOP_*)
    uint8_t width = ins.mem_width();
    unsigned sz = width_bytes(width);
    // 1.1 修复 F14: 当前原子 ALU 实现为 32 位, 不静默接受其他宽度。
    if(sz!=4)return ERR_BAD_ENCODING;

    // 读写辅助函数 (复用 load_mem/store_mem)
    auto write_atom = [&](uint64_t a, uint32_t v) -> int {
        return store_mem(w, ins, lane, a, v);
    };

    // 原子操作: 读旧值 → 计算 → 写新值
    uint32_t cur=0;
    int read_error=load_mem(w,ins,lane,addr,cur);
    if(read_error!=ERR_OK)return read_error;
    old = cur;                        // 旧值返回给寄存器
    uint32_t newval = cur;
    switch(op) {
    case AOP_EXCH: newval = arg; break;                                    // 交换
    case AOP_ADD: newval = cur + arg; break;                               // 原子加
    case AOP_MIN: newval = (int32_t)cur < (int32_t)arg ? cur : arg; break; // 有符号最小
    case AOP_MAX: newval = (int32_t)cur > (int32_t)arg ? cur : arg; break; // 有符号最大
    case AOP_UMIN: newval = cur < arg ? cur : arg; break;                  // 无符号最小
    case AOP_UMAX: newval = cur > arg ? cur : arg; break;                  // 无符号最大
    case AOP_INC: newval = (cur >= arg) ? 0 : cur + 1; break;             // 自增 (模 arg)
    case AOP_DEC: newval = (cur == 0 || cur > arg) ? arg : cur - 1; break; // 自减
    case AOP_CAS: newval = (cur == cmp) ? arg : cur; break;               // 比较交换
    case AOP_AND: newval = cur & arg; break;                               // 位与
    case AOP_OR: newval = cur | arg; break;                                // 位或
    case AOP_XOR: newval = cur ^ arg; break;                               // 位异或
    default: return ERR_BAD_ENCODING;
    }
    return write_atom(addr,newval); // F14: 写入失败不能吞掉错误
}

// =============================================================================
// execute_inst: 指令执行分发 (02 册 §3-§8, 03 册 §6) ★★★ 核心执行引擎
// -----------------------------------------------------------------------------
// 教材引用: 第 2 章 §2.2 GPU Instruction Set Architectures (p.14) — execute_inst 是 SASS ISA 指令执行分发, 按 OP_* 操作码分派所有指令语义, 对应教材中指令集架构总览
// -----------------------------------------------------------------------------
// 这是功能模拟器的核心: 一个巨大的 switch 语句, 按操作码分发到各指令的语义实现
// 每条指令的语义:
//   1. 对所有活跃 lane (exec_mask & (1<<l)) 执行操作
//   2. 读取源寄存器 (R(l, reg)), 计算结果, 写入目的寄存器 (W(l, reg, val))
//   3. 控制流指令 (BRA/SSY/EXIT 等) 额外更新 SIMT 栈和 PC
//
// 指令分组:
//   1. 数据搬运: NOP/MOV/MOV32I/S2R/LDC/CVTA/SELP
//   2. 整数 ALU: IADD/IADD3/IMAD/IMUL/ISUB/IMNMX/LOP/SHL/SHR/SAR/...
//   3. 浮点 ALU: FADD/FSUB/FMUL/FFMA/FMNMX/FSET/FABS/FNEG/FRND/RCP/RSQ/MUFU/F2F/XCVT
//   4. 谓词比较: SETP/SETPI
//   5. 内存操作: LD/LDU/ST/LD128/ST128/ATOM/RED/PREFETCH
//   6. Warp 原语: SHFL/VOTE/PRMT (warp 内线程通信)
//   7. 控制流: SSY/BRA/BRX/CALL/RET/EXIT/BAR/MEMBAR
//
// SIMT 执行模型:
//   - 每条指令对 warp 内所有活跃 lane 同时执行 (SIMD)
//   - exec_mask 决定哪些 lane 执行
//   - 寄存器堆按 lane 分组: rf[lane * 256 + reg]
//   - 谓词堆按 lane 分组: pf[lane * 7 + p]
//
// 教学要点:
//   - SIMT = Single Instruction Multiple Thread, 32 lane 共享一条指令
//   - 与 SIMD 的区别: SIMT 允许分支发散 (不同 lane 走不同路径)
//   - R/W lambda 简化了寄存器访问, 自动处理 RZ (零寄存器)
//   - 控制流指令更新 newpc, 非控制流指令默认 newpc = pc + 8
// =============================================================================
// =============================================================================
// 1.1 修复说明 F07/F08 浮点、RZ 与跨 lane 语义
// -----------------------------------------------------------------------------
// FADD/FSUB/FMUL 以浮点舍入模式计算, 不把浮点结果舍入为整数; FFMA 保留融合语义。
// 读取 RZ 必须经过 const 访问返回零; SHFL 在写目的前快照源 lane, 避免源目的重叠污染。
// =============================================================================
int FuncSim::execute_inst(WarpState& w, const Inst& ins, uint32_t exec_mask) {
    uint64_t pc = w.stack[w.tos].nextpc;  // 当前 PC
    uint64_t newpc = pc + 8;              // 默认下一条 PC = PC + 8 (顺序执行)

    // 寄存器访问辅助函数 (lambda)
    // R(l, r): 读取 lane l 的寄存器 r
    // W(l, r, v): 写入 lane l 的寄存器 r
    // RZ (255) 特殊处理: 读返回 0, 写丢弃
    auto R = [&](int lane, int r) -> uint32_t { return static_cast<const WarpState&>(w).reg(lane, r); };
    auto W = [&](int lane, int r, uint32_t v) { w.reg(lane, r) = v; };

    switch(ins.op) {
    // =========================================================================
    // 1. 数据搬运指令 (02 册 §4)
    // =========================================================================
    // 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 数据搬运指令 MOV/MOV32I/S2R/LDC/CVTA/SELP, 对应教材中 RZ 寄存器与 SASS 指令格式
    case OP_NOP: break;  // 空操作

    // MOV: 寄存器间搬运 d ← a
    case OP_MOV: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l))
            W(l, ins.d, R(l, ins.a));
        break;
    }
    // MOV32I: 加载 32 位立即数 d ← imm32
    // X[0]=1 时为 64 位: d:d+1 ← sign_extend(imm32)
    case OP_MOV32I: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            W(l, ins.d, (uint32_t)ins.imm32);
            if (ins.x & 1) { // 64-bit: sign extend to pair
                W(l, ins.d+1, (uint32_t)((int32_t)ins.imm32 >> 31));
            }
        }
        break;
    }
    // S2R: 特殊寄存器读取 d ← SR[X[5:0]] (01 册 §4)
    // 读取 TID/CTAID/SMID/LANEID 等坐标信息
    case OP_S2R: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t hi;
            uint32_t lo = get_sr(w, l, ins.x & 0x3F, &hi);
            W(l, ins.d, lo);
            if (sr_is_u64(ins.x & 0x3F)) {  // 64 位特殊寄存器
                W(l, ins.d+1, hi);
            }
        }
        break;
    }
    // LDC: 常量内存加载 d ← const_mem[bank][simm16] (04 册 §3.4)
    // 常量内存有 64 个 bank, 每 bank 64 KiB
    // bank 0 前 4 KiB 是内核参数区
    case OP_LDC: {
        uint8_t bank = ins.ldc_bank();
        int16_t off = ins.simm16;
        uint64_t addr = (uint64_t)bank * 65536 + (uint16_t)(int16_t)off;
        // For bank 0 (param), address is just the offset
        if (bank == 0) addr = (uint16_t)(int16_t)off;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            if (ins.ldc_width() == 1) { // U64
                if (addr + 8 > const_mem.size()) return ERR_ADDR_OUT_OF_RANGE;
                uint64_t v = *(uint64_t*)(const_mem.data() + addr);
                W(l, ins.d, (uint32_t)(v & 0xFFFFFFFF));
                W(l, ins.d+1, (uint32_t)(v >> 32));
            } else {
                if (addr + 4 > const_mem.size()) return ERR_ADDR_OUT_OF_RANGE;
                W(l, ins.d, *(uint32_t*)(const_mem.data() + addr));
            }
        }
        break;
    }
    // CVTA: 地址空间转换 (04 册 §4)
    // space → flat 或 flat → space
    // X[2:0] = 源地址空间, X[3] = 反向标志
    case OP_CVTA: {
        uint8_t src_space = ins.x & 0x7;
        bool reverse = (ins.x >> 3) & 1;  // 1=flat→space, 0=space→flat
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint64_t addr = (uint64_t)R(l, ins.a) | ((uint64_t)R(l, ins.a+1) << 32);
            uint64_t result;
            if (!reverse) {
                // space → flat: 加上基址
                switch(src_space) {
                case AS_GLOBAL: result = addr; break;                          // global: flat=offset
                case AS_SHARED: result = w.smem_flat_base + addr; break;       // shared: +CTA基址
                case AS_LOCAL: result = w.lmem_base[l] + addr; break;          // local: +线程基址
                case AS_CONST: result = FLAT_CONST_BASE + addr; break;          // const: +const基址
                default: result = addr; break;
                }
            } else {
                // flat → space: 减去基址
                switch(src_space) {
                case AS_GLOBAL: result = addr; break;
                case AS_SHARED: result = addr - w.smem_flat_base; break;
                case AS_LOCAL: result = addr - w.lmem_base[l]; break;
                case AS_CONST: result = addr - FLAT_CONST_BASE; break;
                default: result = addr; break;
                }
            }
            W(l, ins.d, (uint32_t)(result & 0xFFFFFFFF));
            W(l, ins.d+1, (uint32_t)(result >> 32));
        }
        break;
    }
    // SELP: 谓词选择 d ← P ? a : c (02 册 §4)
    case OP_SELP: {
        uint8_t p = ins.y & 0x7;      // 谓词寄存器号
        bool neg = (ins.y >> 3) & 1;   // 取反标志
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            bool pv = w.pred(l, p);
            if (neg) pv = !pv;
            W(l, ins.d, pv ? R(l, ins.a) : R(l, ins.c));  // 谓词为真→a, 假→c
        }
        break;
    }

    // =========================================================================
    // 2. 整数 ALU 指令 (02 册 §5)
    // =========================================================================
    // 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — 整数 ALU 指令 IADD/IMAD/IMNMX/LOP/SHF 等, 对应教材中 IMNMX 与整数运算单元
    // 整数运算在 SP (Single Precision) 单元执行
    // 支持 32 位和 64 位操作 (通过 X 字段区分)
    // IADD: 整数加法 d ← a + c (X[2:0]==6 为 64 位)
    case OP_IADD: {
        bool is_u64 = (ins.x & 0x7) == 6;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            if (is_u64) {
                uint64_t a = R(l, ins.a) | ((uint64_t)R(l, ins.a+1) << 32);
                uint64_t c = R(l, ins.c) | ((uint64_t)R(l, ins.c+1) << 32);
                uint64_t r = a + c;
                W(l, ins.d, (uint32_t)(r & 0xFFFFFFFF));
                W(l, ins.d+1, (uint32_t)(r >> 32));
            } else {
                W(l, ins.d, R(l, ins.a) + R(l, ins.c));
            }
        }
        break;
    }
    case OP_IADD3: {
        // d ← a + c + imm8 (bits 15:8)
        uint32_t imm = ins.y; // Y field is the third operand (imm8) per spec note
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l))
            W(l, ins.d, R(l, ins.a) + R(l, ins.c) + imm);
        break;
    }
    case OP_IMAD: {
        // d ← a*c + d_old
        uint8_t mode = (ins.x >> 3) & 0x7; // 0 LO, 1 HI, 2 WIDE
        bool is_signed = (ins.x & 1) == 1;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            if (mode == 2) { // WIDE: write pair
                uint64_t a = is_signed ? (int64_t)(int32_t)R(l, ins.a) : (uint64_t)R(l, ins.a);
                uint64_t c = is_signed ? (int64_t)(int32_t)R(l, ins.c) : (uint64_t)R(l, ins.c);
                uint64_t d_old = R(l, ins.d) | ((uint64_t)R(l, ins.d+1) << 32);
                uint64_t r = a * c + d_old;
                W(l, ins.d, (uint32_t)(r & 0xFFFFFFFF));
                W(l, ins.d+1, (uint32_t)(r >> 32));
            } else if (mode == 0) { // LO
                uint32_t a = R(l, ins.a), c = R(l, ins.c), d_old = R(l, ins.d);
                if (is_signed) {
                    uint64_t r = (int64_t)(int32_t)a * (int64_t)(int32_t)c;
                    W(l, ins.d, (uint32_t)(r + d_old));
                } else {
                    uint64_t r = (uint64_t)a * (uint64_t)c;
                    W(l, ins.d, (uint32_t)(r + d_old));
                }
            } else { // HI
                uint32_t a = R(l, ins.a), c = R(l, ins.c), d_old = R(l, ins.d);
                uint64_t r;
                if (is_signed) r = (int64_t)(int32_t)a * (int64_t)(int32_t)c;
                else r = (uint64_t)a * (uint64_t)c;
                W(l, ins.d, (uint32_t)(r >> 32) + d_old);
            }
        }
        break;
    }
    case OP_IMUL: {
        uint8_t mode = (ins.x >> 3) & 0x7;
        bool is_signed = (ins.x & 1) == 1;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t a = R(l, ins.a), c = R(l, ins.c);
            uint64_t r;
            if (is_signed) r = (int64_t)(int32_t)a * (int64_t)(int32_t)c;
            else r = (uint64_t)a * (uint64_t)c;
            if (mode == 2) { // WIDE
                W(l, ins.d, (uint32_t)(r & 0xFFFFFFFF));
                W(l, ins.d+1, (uint32_t)(r >> 32));
            } else if (mode == 0) { // LO
                W(l, ins.d, (uint32_t)(r & 0xFFFFFFFF));
            } else { // HI
                W(l, ins.d, (uint32_t)(r >> 32));
            }
        }
        break;
    }
    case OP_ISUB: {
        bool is_u64 = (ins.x & 0x7) == 6;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            if (is_u64) {
                uint64_t a = R(l, ins.a) | ((uint64_t)R(l, ins.a+1) << 32);
                uint64_t c = R(l, ins.c) | ((uint64_t)R(l, ins.c+1) << 32);
                uint64_t r = a - c;
                W(l, ins.d, (uint32_t)(r & 0xFFFFFFFF));
                W(l, ins.d+1, (uint32_t)(r >> 32));
            } else {
                W(l, ins.d, R(l, ins.a) - R(l, ins.c));
            }
        }
        break;
    }
    case OP_IMNMX: {
        bool is_max = (ins.x & 1) == 1;
        bool is_unsigned = (ins.x >> 1) & 1;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t a = R(l, ins.a), c = R(l, ins.c);
            if (is_max) {
                if (is_unsigned) W(l, ins.d, (a > c) ? a : c);
                else W(l, ins.d, ((int32_t)a > (int32_t)c) ? a : c);
            } else {
                if (is_unsigned) W(l, ins.d, (a < c) ? a : c);
                else W(l, ins.d, ((int32_t)a < (int32_t)c) ? a : c);
            }
        }
        break;
    }
    case OP_LOP: {
        uint8_t op = ins.x & 0x3;
        bool imm = (ins.x >> 2) & 1;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t a = R(l, ins.a);
            uint32_t c = imm ? (uint32_t)ins.c : R(l, ins.c);
            switch(op) {
            case 0: W(l, ins.d, a & c); break;
            case 1: W(l, ins.d, a | c); break;
            case 2: W(l, ins.d, a ^ c); break;
            case 3: W(l, ins.d, a & ~c); break;
            }
        }
        break;
    }
    case OP_SHL: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l))
            W(l, ins.d, R(l, ins.a) << (R(l, ins.c) & 31));
        break;
    }
    case OP_SHR: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l))
            W(l, ins.d, R(l, ins.a) >> (R(l, ins.c) & 31));
        break;
    }
    case OP_SAR: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l))
            W(l, ins.d, (uint32_t)((int32_t)R(l, ins.a) >> (R(l, ins.c) & 31)));
        break;
    }
    case OP_POPC: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l))
            W(l, ins.d, popcount32(R(l, ins.a)));
        break;
    }
    case OP_BREV: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l))
            W(l, ins.d, bitrev32(R(l, ins.a)));
        break;
    }
    case OP_IABS: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            int32_t a = (int32_t)R(l, ins.a);
            W(l, ins.d, (a < 0) ? (uint32_t)(-a) : (uint32_t)a);
        }
        break;
    }
    case OP_INEG: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l))
            W(l, ins.d, (uint32_t)(-(int32_t)R(l, ins.a)));
        break;
    }
    case OP_IDIV: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            int32_t a = (int32_t)R(l, ins.a);
            int32_t c = (int32_t)R(l, ins.c);
            W(l, ins.d, (c == 0) ? 0 : (uint32_t)(a / c));
        }
        break;
    }
    case OP_IREM: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            int32_t a = (int32_t)R(l, ins.a);
            int32_t c = (int32_t)R(l, ins.c);
            W(l, ins.d, (c == 0) ? (uint32_t)a : (uint32_t)(a % c));
        }
        break;
    }
    case OP_BMSK: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t s = R(l, ins.a) & 31;
            W(l, ins.d, (s == 0) ? 0u : ((1u << s) - 1));
        }
        break;
    }
    case OP_FIND: {
        uint8_t mode = ins.x & 0x3;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t a = R(l, ins.a);
            switch(mode) {
            case 0: W(l, ins.d, find_msb(a)); break;
            case 1: W(l, ins.d, clz32(a)); break;
            case 2: W(l, ins.d, ctz32(a)); break;
            default: W(l, ins.d, find_msb(a)); break;
            }
        }
        break;
    }
    case OP_LOP3: {
        // LUT8(a, c, imm8) - 8-bit LUT with 3 inputs
        uint8_t lut = (uint8_t)(ins.imm32 & 0xFF);
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t a = R(l, ins.a), c = R(l, ins.c);
            uint32_t result = 0;
            for (int bit = 0; bit < 32; bit++) {
                uint8_t idx = ((a >> bit) & 1) | (((c >> bit) & 1) << 1);
                // Third input is 0 per spec, so idx is 0-3
                if (lut & (1 << idx)) result |= (1u << bit);
            }
            W(l, ins.d, result);
        }
        break;
    }
    case OP_BFE: {
        // Bit field extract: a=source, c=pos, d_old=len
        bool is_signed = ins.x & 1;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t src = R(l, ins.a);
            uint32_t pos = R(l, ins.c) & 31;
            uint32_t len = R(l, ins.d) & 31; // d_old is len
            uint32_t mask = (len == 0) ? 0 : ((1u << len) - 1);
            uint32_t extracted = (src >> pos) & mask;
            if (is_signed && len > 0 && (extracted >> (len-1))) {
                extracted |= ~mask; // sign extend
            }
            W(l, ins.d, extracted);
        }
        break;
    }
    case OP_BFI: {
        // Bit field insert: insert a's low bits into d_old
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t src = R(l, ins.a);
            uint32_t pos = R(l, ins.c) & 31;
            uint32_t len = R(l, ins.d) & 31; // d_old is len
            uint32_t mask = (len == 0) ? 0 : (((1u << len) - 1) << pos);
            uint32_t old = R(l, ins.d);
            W(l, ins.d, (old & ~mask) | ((src << pos) & mask));
        }
        break;
    }
    case OP_SHF: {
        // Funnel shift: 64-bit (c:a) shifted by X[7:2]&63, take 32 bits
        bool left = ins.x & 1;
        bool is_unsigned = (ins.x >> 1) & 1;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint64_t val = (uint64_t)R(l, ins.a) | ((uint64_t)R(l, ins.c) << 32);
            uint32_t sh = (ins.x >> 2) & 0x3F;
            uint64_t r;
            if (left) r = val << sh;
            else r = is_unsigned ? (val >> sh) : ((int64_t)val >> sh);
            W(l, ins.d, (uint32_t)(r & 0xFFFFFFFF));
        }
        break;
    }

    // =========================================================================
    // 3. 浮点 ALU 指令 (02 册 §6)
    // =========================================================================
    // 教材引用: 第 2 章 §2.2.3 Fused Multiply-Add (p.20) — 浮点 ALU 指令 FADD/FMUL/FFMA/FMNMX, 对应教材中 FFMA 单次舍入与 FMNMX 指令
    // 浮点运算在 SP 单元 (F32) 或 DP 单元 (F64) 执行
    // IEEE 754 单精度 (32 位) / 双精度 (64 位)
    // round_fp32: 根据指令的舍入模式 (ins.m) 舍入结果
    // FADD: 浮点加法 d ← a + c (IEEE 754)
    case OP_FADD: {
        const int prior_round=std::fegetround();
        const int modes[]={FE_TONEAREST,FE_TOWARDZERO,FE_DOWNWARD,FE_UPWARD};
        if(prior_round!=modes[ins.m & 3]) std::fesetround(modes[ins.m & 3]);
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            float a, c;
            uint32_t ab = R(l, ins.a), cb = R(l, ins.c);
            std::memcpy(&a, &ab, 4); std::memcpy(&c, &cb, 4);
            float r = a + c;
            W(l, ins.d, *(uint32_t*)&r);
        }
        if(prior_round!=modes[ins.m & 3]) std::fesetround(prior_round);
        break;
    }
    case OP_FSUB: {
        const int prior_round=std::fegetround();
        const int modes[]={FE_TONEAREST,FE_TOWARDZERO,FE_DOWNWARD,FE_UPWARD};
        if(prior_round!=modes[ins.m & 3]) std::fesetround(modes[ins.m & 3]);
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            float a, c;
            uint32_t ab = R(l, ins.a), cb = R(l, ins.c);
            std::memcpy(&a, &ab, 4); std::memcpy(&c, &cb, 4);
            float r = a - c;
            W(l, ins.d, *(uint32_t*)&r);
        }
        if(prior_round!=modes[ins.m & 3]) std::fesetround(prior_round);
        break;
    }
    case OP_FMUL: {
        const int prior_round=std::fegetround();
        const int modes[]={FE_TONEAREST,FE_TOWARDZERO,FE_DOWNWARD,FE_UPWARD};
        if(prior_round!=modes[ins.m & 3]) std::fesetround(modes[ins.m & 3]);
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            float a, c;
            uint32_t ab = R(l, ins.a), cb = R(l, ins.c);
            std::memcpy(&a, &ab, 4); std::memcpy(&c, &cb, 4);
            float r = a * c;
            W(l, ins.d, *(uint32_t*)&r);
        }
        if(prior_round!=modes[ins.m & 3]) std::fesetround(prior_round);
        break;
    }
    case OP_FFMA: {
        const int prior_round=std::fegetround();
        const int modes[]={FE_TONEAREST,FE_TOWARDZERO,FE_DOWNWARD,FE_UPWARD};
        if(prior_round!=modes[ins.m & 3]) std::fesetround(modes[ins.m & 3]);
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            float a, c, d;
            uint32_t ab = R(l, ins.a), cb = R(l, ins.c), db = R(l, ins.d);
            std::memcpy(&a, &ab, 4); std::memcpy(&c, &cb, 4); std::memcpy(&d, &db, 4);
            float r = (float)std::fma(a, c, d);
            (void)ins.m;
            W(l, ins.d, *(uint32_t*)&r);
        }
        if(prior_round!=modes[ins.m & 3]) std::fesetround(prior_round);
        break;
    }
    case OP_FMNMX: {
        bool is_max = ins.x & 1;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            float a, c;
            uint32_t ab = R(l, ins.a), cb = R(l, ins.c);
            std::memcpy(&a, &ab, 4); std::memcpy(&c, &cb, 4);
            if (std::isnan(a) || std::isnan(c)) {
                // NaN propagation
                uint32_t nan = 0x7FC00000;
                W(l, ins.d, nan);
            } else {
                float r = is_max ? std::max(a, c) : std::min(a, c);
                // -0/+0: take non-negative
                if (a == 0.0f && c == 0.0f && !std::signbit(a)) r = a;
                else if (a == 0.0f && c == 0.0f && !std::signbit(c)) r = c;
                W(l, ins.d, *(uint32_t*)&r);
            }
        }
        break;
    }
    case OP_FSET: {
        uint8_t code = ins.cmp_code();
        uint8_t combine = ins.y & 0x3;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t a = R(l, ins.a), c = R(l, ins.c);
            bool result = fp_cmp(a, c, code);
            uint32_t r = result ? 0xFFFFFFFFu : 0;
            uint32_t old = R(l, ins.d);
            switch(combine) {
            case 0: r = r & old; break;
            case 1: r = r | old; break;
            case 2: r = r ^ old; break;
            case 3: break; // SET (direct)
            }
            W(l, ins.d, r);
        }
        break;
    }
    case OP_FABS: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t a = R(l, ins.a);
            W(l, ins.d, a & 0x7FFFFFFF);
        }
        break;
    }
    case OP_FNEG: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t a = R(l, ins.a);
            W(l, ins.d, a ^ 0x80000000);
        }
        break;
    }
    case OP_FRND: {
        uint8_t mode = ins.x & 0x3;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            float a;
            uint32_t ab = R(l, ins.a);
            std::memcpy(&a, &ab, 4);
            float r;
            switch(mode) {
            case 0: r = std::floor(a); break;
            case 1: r = std::ceil(a); break;
            case 2: r = std::trunc(a); break;
            case 3: r = std::nearbyint(a); break;
            default: r = a; break;
            }
            W(l, ins.d, *(uint32_t*)&r);
        }
        break;
    }
    case OP_RCP: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            float a;
            uint32_t ab = R(l, ins.a);
            std::memcpy(&a, &ab, 4);
            float r = 1.0f / a;
            W(l, ins.d, *(uint32_t*)&r);
        }
        break;
    }
    case OP_RSQ: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            float a;
            uint32_t ab = R(l, ins.a);
            std::memcpy(&a, &ab, 4);
            float r = 1.0f / std::sqrt(a);
            W(l, ins.d, *(uint32_t*)&r);
        }
        break;
    }
    case OP_MUFU: {
        uint8_t func = ins.x & 0x3;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            float a;
            uint32_t ab = R(l, ins.a);
            std::memcpy(&a, &ab, 4);
            float r;
            switch(func) {
            case 0: r = std::sin(a); break;
            case 1: r = std::cos(a); break;
            case 2: r = std::pow(2.0f, a); break;
            case 3: r = (a > 0) ? std::log2(a) : -std::numeric_limits<float>::infinity(); break;
            default: r = 0; break;
            }
            W(l, ins.d, *(uint32_t*)&r);
        }
        break;
    }
    case OP_F2F: {
        uint8_t cvt = ins.x & 0x7;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            switch(cvt) {
            case 0: { // F16→F32
                uint16_t h = (uint16_t)R(l, ins.a);
                // Simplified F16→F32
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                uint32_t bits;
                if (exp == 0) {
                    if (mant == 0) bits = sign << 31;
                    else { // subnormal
                        int e = -1;
                        while (!(mant & 0x400)) { mant <<= 1; e--; }
                        mant &= 0x3FF;
                        bits = (sign << 31) | ((127 - 15 + e) << 23) | (mant << 13);
                    }
                } else if (exp == 0x1F) {
                    bits = (sign << 31) | 0x7F800000 | (mant << 13);
                } else {
                    bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                W(l, ins.d, bits);
                break;
            }
            case 1: { // F32→F16 (RN)
                uint32_t f = R(l, ins.a);
                uint32_t sign = (f >> 31) & 1;
                int32_t exp = (int32_t)((f >> 23) & 0xFF) - 127;
                uint32_t mant = f & 0x7FFFFF;
                uint16_t h;
                if (exp > 15) h = (uint16_t)((sign << 15) | 0x7C00); // inf
                else if (exp < -14) h = (uint16_t)(sign << 15); // zero (simplified)
                else h = (uint16_t)((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
                W(l, ins.d, h);
                break;
            }
            case 2: { // F32→F64
                float f;
                uint32_t fb = R(l, ins.a);
                std::memcpy(&f, &fb, 4);
                double d = (double)f;
                uint64_t dbits;
                std::memcpy(&dbits, &d, 8);
                W(l, ins.d, (uint32_t)(dbits & 0xFFFFFFFF));
                W(l, ins.d+1, (uint32_t)(dbits >> 32));
                break;
            }
            case 3: { // F64→F32
                uint64_t dbits = (uint64_t)R(l, ins.a) | ((uint64_t)R(l, ins.a+1) << 32);
                double d;
                std::memcpy(&d, &dbits, 8);
                float f = (float)d;
                uint32_t fb;
                std::memcpy(&fb, &f, 4);
                W(l, ins.d, fb);
                break;
            }
            }
        }
        break;
    }
    case OP_XCVT: {
        uint8_t src = ins.x & 0x7;
        uint8_t dst = (ins.x >> 4) & 0x7;
        bool sat = (ins.m >> 2) & 1;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            // Simplified conversion
            if (src == 5 && dst == 0) { // F32→S32
                float f; uint32_t fb = R(l, ins.a);
                std::memcpy(&f, &fb, 4);
                int32_t i = (int32_t)std::trunc(f);
                if (sat) i = std::max(-0x7FFFFFFF, std::min(0x7FFFFFFF, i));
                W(l, ins.d, (uint32_t)i);
            } else if (src == 0 && dst == 5) { // S32→F32
                int32_t i = (int32_t)R(l, ins.a);
                float f = (float)i;
                W(l, ins.d, *(uint32_t*)&f);
            } else { // fallback
                W(l, ins.d, R(l, ins.a));
            }
        }
        break;
    }

    // =========================================================================
    // 4. 谓词比较指令 (02 册 §7)
    // =========================================================================
    // 教材引用: 第 2 章 §2.2.1 NVIDIA GPU Instruction Set Architectures (p.14) — SETP/SETPI 比较并设置谓词寄存器, 对应教材中 SETP 指令与谓词保护机制
    // 比较结果写入谓词寄存器 (P0-P6), 用于后续的条件执行
    // SETP: 比较两个寄存器, 设置谓词 pd
    //   type: 0=S32, 1=U32, 2=F32 (比较类型)
    //   code: EQ/NE/LT/LE/GT/GE (比较条件)
    //   combine: AND/OR/XOR/SET (与旧谓词的组合方式)
    case OP_SETP: {
        uint8_t code = ins.cmp_code();
        uint8_t type = ins.cmp_type();
        uint8_t combine = ins.pred_combine();
        uint8_t pd = ins.d & 0x7;
        uint8_t ps = ins.setp_src_pred();
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t a = R(l, ins.a), c = R(l, ins.c);
            bool result;
            if (type == 2) { // F32
                result = fp_cmp(a, c, code);
            } else {
                bool is_unsigned = (type >= 1); // U32 or U64
                result = int_cmp(a, c, code, is_unsigned);
            }
            bool old = w.pred(l, ps);
            bool combined = result;
            switch(combine) {
            case 0: combined = result && old; break;
            case 1: combined = result || old; break;
            case 2: combined = result != old; break;
            case 3: combined = result; break;
            }
            w.set_pred(l, pd, combined);
        }
        break;
    }
    case OP_SETPI: {
        uint8_t code = ins.cmp_code();
        uint8_t type = ins.cmp_type();
        uint8_t pd = ins.d & 0x7;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t a = R(l, ins.a);
            uint32_t c = (uint32_t)ins.imm32;
            bool result;
            if (type == 2) { // F32
                result = fp_cmp(a, c, code);
            } else {
                result = int_cmp(a, c, code, type >= 1);
            }
            w.set_pred(l, pd, result);
        }
        break;
    }

    // =========================================================================
    // 5. 内存操作指令 (02 册 §8, 04 册 §3)
    // =========================================================================
    // 教材引用: 第 4 章 §4.1 First-Level Memory Structures (p.67) — LD/ST/ATOM 内存操作指令, 对应教材中 L1/shared/const 内存层次与地址空间访问
    // LD/ST: 加载/存储, 支持多种地址空间和宽度
    // LD 地址计算: [Ra + Rc] (M 格式) 或 [Ra + simm16] (MI 格式)
    // 64 位加载读两个寄存器 (d, d+1), 128 位加载读四个 (d..d+3)
    // 1.1 修复 F13: LD/ST 的 8/16 字节路径共用地址空间转换和边界检查。
    // 在写目的寄存器前快照地址; LD 的目的与地址重叠不会破坏高字读取。
    case OP_LD: case OP_LDU: case OP_ST: case OP_LD128: case OP_ST128: {
        Inst access=ins;
        bool forced128=ins.op==OP_LD128||ins.op==OP_ST128;
        if(forced128)access.x=(access.x&0x87)|(W_U128<<3);
        unsigned bytes=width_bytes(access.mem_width());bool store=ins.op==OP_ST||ins.op==OP_ST128;
        if(bytes>4&&ins.d!=RZ&&int(ins.d)+int(bytes/4)>255)return ERR_UNALIGNED_REG;
        for(int l=0;l<32;++l)if(exec_mask&(1u<<l)){
            uint64_t addr=R(l,ins.a);
            if(bytes>4&&ins.a!=RZ)addr|=uint64_t(R(l,ins.a+1))<<32;
            if(ins.kind==FK_M){uint64_t off=R(l,ins.c);
                if(bytes>4&&ins.c!=RZ)off|=uint64_t(R(l,ins.c+1))<<32;addr+=off;
            }else addr+=int64_t(ins.simm16);
            if(store){int e=store_mem(w,access,l,addr,R(l,ins.d));if(e)return e;}
            else{
                uint32_t values[4]={};int e=load_mem(w,access,l,addr,values[0]);if(e)return e;
                Inst word=access;word.x=(word.x&0x87)|(W_U32<<3);
                for(unsigned j=1;j<bytes/4;++j){e=load_mem(w,word,l,addr+4*j,values[j]);if(e)return e;}
                for(unsigned j=0;j<std::max(1u,bytes/4);++j)if(ins.d!=RZ)W(l,ins.d+j,values[j]);
            }
        }break;
    }
    case OP_ATOM: case OP_RED: {
        uint8_t op = ins.y & 0xF;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint64_t addr = R(l, ins.a);
            if (ins.kind == FK_M) addr += R(l, ins.c);
            uint32_t arg = R(l, ins.c);
            uint32_t cmp = (op == AOP_CAS) ? R(l, ins.b) : 0;
            uint32_t old;
            int e = atom_op(w, ins, l, addr, arg, cmp, old);
            if (e != ERR_OK) return e;
            if (ins.op == OP_ATOM) {
                W(l, ins.d, old);
            }
        }
        break;
    }
    case OP_PREFETCH:
        // No-op in functional sim
        break;

    // =========================================================================
    // 6. Warp 原语 (02 册 §9) ★ warp 内线程通信
    // =========================================================================
    // 教材引用: 第 2 章 §2.1.2 Threading Model (p.12) — SHFL/VOTE/PRMT warp 原语, 对应教材中 SIMT 32 线程/warp 与 warp 内线程通信
    // SHFL: warp 内寄存器洗牌 (shuffle), lane 间交换数据
    //   模式: IDX(按索引)/UP(向上)/DOWN(向下)/BFLY(蝴蝶)
    //   identity: 源 lane 无效时是否保留原值
    // VOTE: warp 内投票, 全/任/同/选票
    // PRMT: 字节重排 (permute), 从 8 字节中选择 4 字节
    case OP_SHFL: {
        uint32_t sources[WARP_SZ], indices[WARP_SZ];
        for (int l=0; l<WARP_SZ; ++l) { sources[l]=R(l,ins.a); indices[l]=R(l,ins.c); }
        uint8_t mode = ins.x & 0x3;
        bool identity = (ins.x >> 2) & 1;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t src = sources[l];
            uint32_t idx = indices[l] & 31;
            uint32_t result;
            switch(mode) {
            case 0: // IDX
                if (exec_mask & (1u << idx)) result = sources[idx];
                else result = identity ? src : sources[l];
                break;
            case 1: // UP
                result = sources[(l + idx) & 31];
                if (!(exec_mask & (1u << ((l + idx) & 31)))) result = identity ? src : sources[l];
                break;
            case 2: // DOWN
                result = sources[(l - idx) & 31];
                if (!(exec_mask & (1u << ((l - idx) & 31)))) result = identity ? src : sources[l];
                break;
            case 3: // BFLY
                result = sources[l ^ idx];
                if (!(exec_mask & (1u << (l ^ idx)))) result = identity ? src : sources[l];
                break;
            }
            W(l, ins.d, result);
        }
        break;
    }
    case OP_VOTE: {
        uint8_t mode = ins.x & 0x3;
        uint8_t p = ins.y & 0x7;
        // Compute vote across active lanes
        bool all = true, any = false;
        uint32_t ballot = 0;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            bool pv = w.pred(l, p);
            if (!pv) all = false;
            if (pv) { any = true; ballot |= (1u << l); }
        }
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            switch(mode) {
            case 0: W(l, ins.d, all ? 1 : 0); break;
            case 1: W(l, ins.d, any ? 1 : 0); break;
            case 2: {
                // EQ: all lanes have same value
                bool first = (exec_mask & 1) ? w.pred(0, p) : w.pred(ctz32(exec_mask), p);
                bool eq = true;
                for (int l2 = 0; l2 < WARP_SZ; l2++) if (exec_mask & (1u<<l2)) {
                    if (w.pred(l2, p) != first) { eq = false; break; }
                }
                W(l, ins.d, eq ? 1 : 0);
                break;
            }
            case 3: W(l, ins.d, ballot); break;
            }
        }
        break;
    }
    case OP_PRMT: {
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint32_t a = R(l, ins.a);
            uint32_t c = R(l, ins.c);
            uint8_t bytes[8] = {
                (uint8_t)(a & 0xFF), (uint8_t)((a >> 8) & 0xFF),
                (uint8_t)((a >> 16) & 0xFF), (uint8_t)((a >> 24) & 0xFF),
                (uint8_t)(c & 0xFF), (uint8_t)((c >> 8) & 0xFF),
                (uint8_t)((c >> 16) & 0xFF), (uint8_t)((c >> 24) & 0xFF)
            };
            uint32_t result = 0;
            for (int b = 0; b < 4; b++) {
                uint8_t sel = (c >> (b * 8)) & 0x1F;
                uint8_t val = (sel < 8) ? bytes[sel] : ((sel & 0x10) ? 0xFF : 0);
                result |= (uint32_t)val << (b * 8);
            }
            W(l, ins.d, result);
        }
        break;
    }

    // =========================================================================
    // 7. 控制流指令 (02 册 §10, 03 册 §4) ★★★ SIMT 栈核心
    // =========================================================================
    // 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — 控制流指令 SSY/BRA/BRX/CALL/RET/EXIT/BAR, 对应教材中分支发散与 SIMT 栈重汇聚机制
    // SSY: 设置重汇聚点, 压入 SIMT 栈帧 (分支前调用)
    // BRA: 条件分支, 处理分支发散 (见前面的 bra 函数注释)
    // BRX: 间接分支, 每 lane 目标不同, 多路发散
    // CALL/RET: 函数调用/返回 (用 LR 寄存器保存返回地址)
    // EXIT: 线程退出 (见 do_exit 函数)
    // BAR: 屏障同步 (见 bar_sync 函数)
    // =========================================================================

    // SSY: 设置重汇聚点 (03 册 §4.2)
    // 压入新栈帧: {rpc=target, nextpc=PC+8, mask=TOS.mask, dflag=false}
    case OP_SSY: {
        uint64_t target = branch_target(ins, pc);
        if (target & 7) return ERR_MISALIGNED_PC;
        // Push {rpc=target, nextpc=PC+8, mask=TOS.mask, dflag=false}
        StackEntry& tos = w.stack[w.tos];
        StackEntry ne;
        ne.rpc = target;
        ne.nextpc = pc + 8;
        ne.mask = tos.mask;
        ne.dflag = false;
        w.stack.push_back(ne);
        w.tos = (int)w.stack.size() - 1;
        if ((int)w.stack.size() > max_stack_depth) max_stack_depth = (int)w.stack.size();
        newpc = pc + 8;
        if (snap_enabled) {
            StackSnap s; s.pc = pc; s.event = "SSY";
            s.stack = w.stack; s.exec_mask = exec_mask;
            snaps.push_back(s);
        }
        break;
    }
    case OP_BRA: {
        uint64_t target = branch_target(ins, pc);
        if (target & 7) return ERR_MISALIGNED_PC;
        StackEntry& tos = w.stack[w.tos];
        uint32_t m = tos.mask;
        uint32_t taken = m & exec_mask; // exec_mask already includes guard
        uint32_t fall = m & ~exec_mask;

        if (taken == m) {
            // All take branch
            tos.nextpc = target;
            newpc = target;
        } else if (taken == 0) {
            // None take branch
            tos.nextpc = pc + 8;
            newpc = pc + 8;
        } else {
            // Divergence
            if (tos.rpc == EXIT_SENTINEL) return ERR_NO_IPDOM;
            uint64_t join = tos.rpc;
            tos.nextpc = join;
            tos.dflag = true;
            // Push fall and taken
            if (w.stack.size() + 2 > SIMT_STACK_MAX) return ERR_STACK_OVERFLOW;
            StackEntry fe, te;
            fe.rpc = join; fe.nextpc = pc + 8; fe.mask = fall; fe.dflag = false;
            te.rpc = join; te.nextpc = target; te.mask = taken; te.dflag = false;
            w.stack.push_back(fe);
            w.stack.push_back(te);
            // Swap if taken popcount < fall popcount (majority first)
            if (popcount32(taken) < popcount32(fall)) {
                std::swap(w.stack[w.stack.size()-1], w.stack[w.stack.size()-2]);
            }
            w.tos = (int)w.stack.size() - 1;
            if ((int)w.stack.size() > max_stack_depth) max_stack_depth = (int)w.stack.size();
            divergent_branches++;
            newpc = w.stack[w.tos].nextpc;
            if (snap_enabled) {
                StackSnap s; s.pc = pc; s.event = "BRA";
                s.stack = w.stack; s.exec_mask = exec_mask;
                snaps.push_back(s);
            }
        }
        break;
    }
    case OP_BRX: {
        // Indirect branch: target = R[a] per lane
        // Group by target, push paths sorted by target address
        std::vector<std::pair<uint64_t, uint32_t>> paths; // (target, mask)
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            uint64_t t = R(l, ins.a);
            if (t & 7) return ERR_MISALIGNED_PC;
            bool found = false;
            for (auto& [tgt, msk] : paths) {
                if (tgt == t) { msk |= (1u << l); found = true; break; }
            }
            if (!found) paths.push_back({t, 1u << l});
        }
        // Sort by target address ascending
        std::sort(paths.begin(), paths.end());

        if (paths.size() == 1) {
            // Uniform branch
            w.stack[w.tos].nextpc = paths[0].first;
            newpc = paths[0].first;
        } else {
            // Divergent
            StackEntry& tos = w.stack[w.tos];
            if (tos.rpc == EXIT_SENTINEL) return ERR_NO_IPDOM;
            uint64_t join = tos.rpc;
            tos.nextpc = join;
            tos.dflag = true;
            if (w.stack.size() + paths.size() > SIMT_STACK_MAX) return ERR_STACK_OVERFLOW;
            for (auto& [tgt, msk] : paths) {
                StackEntry e;
                e.rpc = join; e.nextpc = tgt; e.mask = msk; e.dflag = false;
                w.stack.push_back(e);
            }
            // Swap for majority: popcount of TOS < popcount of TOS-1
            // Apply iteratively for the newly pushed entries
            // For simplicity, just swap adjacent if needed (one pass)
            // Per spec: compare TOS with its next entry
            for (int i = (int)w.stack.size() - 1; i > w.tos + 1; i--) {
                if (popcount32(w.stack[i].mask) < popcount32(w.stack[i-1].mask)) {
                    std::swap(w.stack[i], w.stack[i-1]);
                } else break;
            }
            w.tos = (int)w.stack.size() - 1;
            if ((int)w.stack.size() > max_stack_depth) max_stack_depth = (int)w.stack.size();
            divergent_branches++;
            newpc = w.stack[w.tos].nextpc;
            if (snap_enabled) {
                StackSnap s; s.pc = pc; s.event = "BRX";
                s.stack = w.stack; s.exec_mask = exec_mask;
                snaps.push_back(s);
            }
        }
        break;
    }
    case OP_CALL: {
        uint64_t target = branch_target(ins, pc);
        if (target & 7) return ERR_MISALIGNED_PC;
        for (int l = 0; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            w.lr[l] = (uint32_t)(pc + 8);
        }
        w.stack[w.tos].nextpc = target;
        newpc = target;
        break;
    }
    case OP_RET: {
        // Per-lane target from LR
        // Simplified: uniform return (all active lanes have same LR)
        uint64_t target = w.lr[0];
        for (int l = 1; l < WARP_SZ; l++) if (exec_mask & (1u<<l)) {
            if (w.lr[l] != w.lr[0]) {
                // Divergent return - treat as BRX with LR
                // For simplicity, just use lane 0's LR
            }
        }
        w.stack[w.tos].nextpc = target;
        newpc = target;
        break;
    }
    case OP_EXIT: {
        do_exit(w, exec_mask);
        if (snap_enabled) {
            StackSnap s; s.pc = pc; s.event = "EXIT";
            s.stack = w.stack; s.exec_mask = exec_mask;
            snaps.push_back(s);
        }
        newpc = w.exit_done ? 0 : w.stack[w.tos].nextpc;
        break;
    }
    // BAR: 屏障同步 (03 册 §5)
    // sub = 0: SYNC (ARRIVE + WAIT, 最常用, 对应 __syncthreads)
    // sub = 1: ARRIVE (仅到达, 不等待)
    // sub = 2: WAIT (仅等待, 不标记到达)
    // expected = CTA 内所有有效 lane 的并集
    case OP_BAR: {
        int err=bar_sync(w,ins.x&15,exec_mask,(ins.x>>4)&3);
        if(err!=ERR_OK)return err;
        newpc=pc+8;
        break;
    }
    case OP_MEMBAR: {
        // Functional sim: no-op (all memory is immediately consistent)
        newpc = pc + 8;
        break;
    }
    case OP_YIELD: case OP_BRKPT: case OP_TRAP:
        newpc = pc + 8;
        break;

    default:
        return ERR_BAD_OPCODE;
    }

    // =========================================================================
    // PC 更新与 SIMT 栈重汇聚 (03 册 §4, §6.6)
    // =========================================================================
    // 教材引用: 第 3 章 §3.1.4 Divergence (p.32) — 指令后更新 TOS.nextpc 并调用 pop_reconverge 检查重汇聚, 对应教材中 SIMT 栈重汇聚点弹出规则
    // 非控制流指令: newpc = pc + 8 (已设置)
    // 控制流指令: newpc 已在对应 case 中设置
    // 更新 TOS.nextpc = newpc (下一条要执行的指令 PC)
    if (newpc != pc || ins.op == OP_NOP || ins.op == OP_MOV || ins.op == OP_MOV32I ||
        ins.op == OP_S2R || ins.op == OP_LDC || ins.op == OP_CVTA ||
        ins.op == OP_SELP || ins.op == OP_IADD || ins.op == OP_IADD3 ||
        ins.op == OP_IMAD || ins.op == OP_IMUL || ins.op == OP_ISUB ||
        ins.op == OP_IMNMX || ins.op == OP_LOP || ins.op == OP_SHF ||
        ins.op == OP_SHL || ins.op == OP_SHR || ins.op == OP_SAR ||
        ins.op == OP_BFE || ins.op == OP_BFI || ins.op == OP_POPC ||
        ins.op == OP_BREV || ins.op == OP_IABS || ins.op == OP_INEG ||
        ins.op == OP_IDIV || ins.op == OP_IREM || ins.op == OP_LOP3 ||
        ins.op == OP_BMSK || ins.op == OP_FIND ||
        ins.op == OP_FADD || ins.op == OP_FSUB || ins.op == OP_FMUL ||
        ins.op == OP_FFMA || ins.op == OP_FMNMX || ins.op == OP_FSET ||
        ins.op == OP_F2F || ins.op == OP_XCVT || ins.op == OP_RCP ||
        ins.op == OP_RSQ || ins.op == OP_MUFU || ins.op == OP_FRND ||
        ins.op == OP_FABS || ins.op == OP_FNEG || ins.op == OP_FCMP ||
        ins.op == OP_DADD || ins.op == OP_SETP || ins.op == OP_SETPI ||
        ins.op == OP_PLOP || ins.op == OP_PSET2 ||
        ins.op == OP_LD || ins.op == OP_ST || ins.op == OP_LDU ||
        ins.op == OP_ATOM || ins.op == OP_RED || ins.op == OP_PREFETCH ||
        ins.op == OP_LD128 || ins.op == OP_ST128 ||
        ins.op == OP_SHFL || ins.op == OP_VOTE || ins.op == OP_PRMT ||
        ins.op == OP_YIELD || ins.op == OP_BRKPT || ins.op == OP_TRAP ||
        ins.op == OP_MEMBAR || ins.op == OP_BAR) {
        // Advance PC if it wasn't a control instruction that set newpc
        // Control instructions (BRA, BRX, CALL, RET, SSY, EXIT) already set newpc
        // For non-control: newpc = pc + 8 (already set)
        // For BAR: newpc was set based on barrier logic
        w.stack[w.tos].nextpc = newpc;
    }

    // Pop reconvergence (for non-EXIT control instructions)
    if (ins.op != OP_EXIT && ins.op != OP_SSY && ins.op != OP_BRA &&
        ins.op != OP_BRX && ins.op != OP_CALL && ins.op != OP_RET &&
        ins.op != OP_BAR) {
        pop_reconverge(w, pc, newpc);
    } else if (ins.op == OP_BRA || ins.op == OP_BRX || ins.op == OP_CALL || ins.op == OP_RET) {
        pop_reconverge(w, pc, newpc);
    }

    return ERR_OK;
}

// =============================================================================
// 主机端内存访问辅助函数 (05 册 §3)
// -----------------------------------------------------------------------------
// 教材引用: 第 2 章 §2.1.3 Memory Model (p.13) — 主机端辅助函数直接读写 GPU 显存, 对应教材中主机-设备内存模型与数据传输
// -----------------------------------------------------------------------------
// 这些函数供主机端代码 (测试/main) 直接读写 GPU 显存
// 提供类型安全的接口 (read_u32/write_u32/read_f32/write_f32)
// 内部委托给 PageMem 的 r32/w32 等方法
// =============================================================================
uint32_t FuncSim::read_u32(uint64_t addr) { return global_mem.r32(addr); }
void FuncSim::write_u32(uint64_t addr, uint32_t val) { global_mem.w32(addr, val); }
// 读 32 位浮点数 (先读 uint32, 再 memcpy 为 float)
float FuncSim::read_f32(uint64_t addr) {
    uint32_t v = global_mem.r32(addr);
    float f; std::memcpy(&f, &v, 4); return f;
}
void FuncSim::write_f32(uint64_t addr, float val) {
    uint32_t v; std::memcpy(&v, &val, 4);
    global_mem.w32(addr, v);
}
void FuncSim::read_bytes(uint64_t addr, void* dst, size_t n) {
    global_mem.read_bytes(addr, dst, n);
}
void FuncSim::write_bytes(uint64_t addr, const void* src, size_t n) {
    global_mem.write_bytes(addr, src, n);
}

} // namespace ntisa
