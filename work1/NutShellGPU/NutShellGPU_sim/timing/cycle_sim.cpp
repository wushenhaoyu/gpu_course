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
// cycle_sim.cpp - 周期级模拟器实现 (S3 阶段) ★★★ 时序建模核心
// -----------------------------------------------------------------------------
// 本文件实现 CycleSim 类的所有方法, 是 NutShellGPU 周期级模拟器的核心:
//   1. 性能统计序列化 (to_json): 输出 JSON 格式的指标
//   2. 地址映射 (addr_to_mp / addr_to_l2slice / dram_addr_decode)
//   3. 初始化 (init): 配置所有组件
//   4. 主循环 (run / tick): 时间驱动模拟
//   5. 流水线阶段 (fetch/decode/issue/collector/execute/writeback_tick)
//   6. 内存系统 (l1d/noc/l2/dram_tick)
//   7. CTA 回收 (cta_retire_tick)
//   8. 统计收集 (collect_stats / output_stats_json)
//
// 设计哲学 (与功能模拟器的区别):
//   - 功能模拟器只关心 "结果是否正确" (架构状态转换)
//   - 周期模拟器额外关心 "用时多少周期" (微结构时序)
//   - 复用 FuncSim 维护架构状态, 叠加时序状态 (流水线/缓存/记分牌)
//   - 每个 tick 推进一个时钟周期, 所有组件同步前进
//
// 教学要点:
//   - 时间驱动 vs 事件驱动: 前者每周期检查所有组件, 简单但低效
//   - 顺序发射: warp 按 round-robin 顺序发射, 不乱序
//   - 简化模型: L1D/L2/NoC 在此版本中部分简化 (留作扩展)
// =============================================================================
#include "cycle_sim.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace ntisa {

// =============================================================================
// CycStats::to_json: 性能统计序列化为 JSON (05 册 §4) ★ 性能输出
// -----------------------------------------------------------------------------
// 将所有累计的性能指标序列化为 JSON 字符串, 便于工具解析与可视化:
//   - cycles: 总周期数
//   - instructions: 总指令数
//   - ipc: 指令/周期 (核心性能指标)
//   - simd_efficiency: SIMD 效率 (反映发散损失)
//   - occupancy_avg: 平均活跃 warp 数 (占用率)
//   - stall_breakdown: 停顿原因分解 (定位瓶颈)
//   - inst_issued: 各指令的发射次数 (热点分析)
//   - replays: 各类重演次数 (bank 冲突/未命中)
//   - l1/l2/dram 缓存与内存统计
//   - noc_flits_sent: NoC 流量统计
//
// 输出示例:
// {
//   "schema": "NutShellGPU-stats-1.1",
//   "cycles": 1234,
//   "instructions": 567,
//   "ipc": 0.4599,
//   ...
// }
// =============================================================================
// 教材引用: 第 5 章 §5.1 Analyzing a GPU Microarchitecture (p.119) — to_json 序列化性能指标 (IPC/SIMD 效率/停顿分解/DRAM 命中) 为 JSON 格式, 供性能分析工具解析
std::string CycStats::to_json() const {
    std::ostringstream os;
    os << std::fixed << std::setprecision(4);
    os << "{\n";
    os << "  \"schema\": \"NutShellGPU-stats-1.1\",\n";
    os << "  \"cycles\": " << total_cycles << ",\n";
    // 模拟秒数由周期和配置频率换算; 与程序的实际运行时间分开记录。
    os << "  \"simulated_seconds\": " << std::scientific << std::setprecision(17)
       << (double(total_cycles) / (core_clock_ghz * 1e9)) << ",\n";
    os << std::fixed << std::setprecision(4);
    os << "  \"instructions\": " << instructions << ",\n";
    os << "  \"ipc\": " << ipc() << ",\n";
    os << "  \"simd_efficiency\": " << simd_efficiency() << ",\n";
    os << "  \"occupancy_avg\": " << (double)occupancy_avg_warps << ",\n";
    os << "  \"issue_cycles\": " << issue_cycles << ",\n";
    os << "  \"stall_cycles\": " << stall_cycles << ",\n";
    os << "  \"stall_breakdown\": {\n";
    os << "    \"no_collector\": " << stall_no_collector << ",\n";
    os << "    \"sb_wait\": " << stall_sb_wait << ",\n";
    os << "    \"barrier_wait\": " << stall_barrier_wait << ",\n";
    os << "    \"lsu_full\": " << stall_lsu_full << ",\n";
    os << "    \"icache_miss\": " << stall_icache_miss << ",\n";
    os << "    \"no_eligible_warp\": " << stall_no_eligible << "\n";
    os << "  },\n";
    os << "  \"inst_issued\": {\n";
    bool first = true;
    for (const auto& [op, cnt] : inst_issued) {
        if (!first) os << ",\n";
        os << "    \"" << op_mnemonic((uint8_t)op) << "\": " << cnt;
        first = false;
    }
    os << "\n  },\n";
    os << "  \"simt_divergent_branch\": " << simt_divergent_branch << ",\n";
    os << "  \"simt_stack_maxdepth\": " << simt_stack_maxdepth << ",\n";
    os << "  \"replays\": {\n";
    os << "    \"smem_conflict\": " << replays_smem_conflict << ",\n";
    os << "    \"l1_miss\": " << replays_l1_miss << ",\n";
    os << "    \"prt_full\": " << replays_prt_full << ",\n";
    os << "    \"assoc_stall\": " << replays_assoc_stall << ",\n";
    os << "    \"wdb_full\": " << replays_wdb_full << "\n";
    os << "  },\n";
    os << "  \"rf_conflict_stalls\": " << rf_conflict_stalls << ",\n";
    os << "  \"writeback_stalls\": " << writeback_stalls << ",\n";
    os << "  \"pipeline_full_stalls\": " << pipeline_full_stalls << ",\n";
    os << "  \"l1_hits\": " << l1_hits << ",\n";
    os << "  \"l1_pending_merges\": " << l1_pending_merges << ",\n";
    os << "  \"memory_sectors\": " << memory_sectors << ",\n";
    os << "  \"serial_dependency_stalls\": " << serial_dependency_stalls << ",\n";
    os << "  \"unsupported_timing_ops\": " << unsupported_timing_ops << ",\n";
    os << "  \"core_clock_ghz\": " << core_clock_ghz << ",\n";
    os << "  \"rf_bank_conflict_cycles\": " << rf_bank_conflict_cycles << ",\n";
    os << "  \"sp_busy_cycles\": " << sp_busy_cycles << ",\n";
    os << "  \"sfu_busy_cycles\": " << sfu_busy_cycles << ",\n";
    os << "  \"dp_busy_cycles\": " << dp_busy_cycles << ",\n";
    os << "  \"smem_accesses\": " << smem_accesses << ",\n";
    os << "  \"smem_conflict_replays\": " << smem_conflict_replays << ",\n";
    os << "  \"l1_access_lines\": " << l1_access_lines << ",\n";
    os << "  \"l1_miss_sectors\": " << l1_miss_sectors << ",\n";
    os << "  \"l1_assoc_stall\": " << l1_assoc_stall << ",\n";
    os << "  \"prt_full\": " << prt_full << ",\n";
    os << "  \"wdb_full\": " << wdb_full << ",\n";
    os << "  \"l2_accesses\": " << l2_accesses << ",\n";
    os << "  \"l2_hits\": " << l2_hits << ",\n";
    os << "  \"l2_misses\": " << l2_misses << ",\n";
    os << "  \"dram_reads\": " << dram_reads << ",\n";
    os << "  \"dram_writes\": " << dram_writes << ",\n";
    os << "  \"dram_row_hits\": " << dram_row_hits << ",\n";
    os << "  \"dram_row_misses\": " << dram_row_misses << ",\n";
    os << "  \"dram_bank_busy_cycles\": " << dram_bank_busy_cycles << ",\n";
    os << "  \"dram_bus_busy_cycles\": " << dram_bus_busy_cycles << ",\n";
    os << "  \"bytes_to_dram\": " << bytes_to_dram << ",\n";
    os << "  \"bytes_from_dram\": " << bytes_from_dram << ",\n";
    os << "  \"dram_effective_bw_GBps\": " << (dram_effective_bw_gbps() * core_clock_ghz) << ",\n";
    os << "  \"row_buffer_hit_rate\": " << row_buffer_hit_rate() << ",\n";
    os << "  \"noc_flits_sent\": " << noc_flits_sent << ",\n";
    os << "  \"noc_stall_cycles\": " << noc_stall_cycles << "\n";
    os << "}\n";
    return os.str();
}

// =============================================================================
// 地址映射函数 (04 册 §7) ★ 地址解码
// -----------------------------------------------------------------------------
// GPU 内存地址需要映射到具体的硬件资源:
//   - addr_to_mp:       地址 → 内存分区 (MP, Memory Partition)
//   - addr_to_l2slice:  地址 → L2 切片 (每 MP 2 切片)
//   - dram_addr_decode: 地址 → (channel, bank, row, col) DRAM 物理坐标
//
// 映射策略:
//   - 用 XOR 哈希减少连续地址冲突 (相比简单取模)
//   - 256B 块为单位 (b256 = addr >> 8): 因为 NoC 路由粒度是 256B
//   - DRAM 地址解码模拟真实 DDR4 物理映射
// =============================================================================

// addr_to_mp: 地址 → 内存分区编号 (0 ~ n_mem-1)
// 使用 XOR 折叠减少冲突: 多个 6 位段异或后取模
// 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — addr_to_mp 用 XOR 折叠哈希将地址映射到内存分区, 减少连续地址冲突
int CycleSim::addr_to_mp(uint64_t addr) const {
    uint32_t b256 = (uint32_t)(addr >> 8);  // 256B 块编号
    uint32_t fold = (b256 & 0x3F) ^ ((b256 >> 6) & 0x3F) ^ ((b256 >> 12) & 0x3F) ^ ((b256 >> 18) & 0x3F);
    return fold % cfg.n_mem;
}

// addr_to_l2slice: 地址 → L2 切片编号 (0 或 1, 每 MP 2 切片)
// 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — addr_to_l2slice 将地址映射到 L2 切片 (每 MP 2 切片), 地址交错实现并行访问
int CycleSim::addr_to_l2slice(uint64_t addr) const {
    uint32_t b256 = (uint32_t)(addr >> 8);
    (void)addr_to_mp(addr); // 完整模型用于路由决策
    return ((addr >> 7) ^ (addr >> 14) ^ (b256 & 0x8)) & 1;
}

// dram_addr_decode: 地址 → (channel, bank, row, col) DRAM 物理坐标
// 模拟 DDR4 地址位映射 (简化版):
//   - channel: 由 addr_to_mp 决定 (XOR 哈希)
//   - col:    A[11:5] (列地址, 7 位, 128B 粒度)
//   - bank:   XOR 哈希减少行冲突
//   - row:    A[32:15] (行地址, 18 位)
// 教材引用: 第 4 章 §4.3 Memory Partition Unit (p.75) — dram_addr_decode 模拟 DDR4 地址位映射: channel/bank/row/col, bank 用 XOR 哈希减少行冲突
void CycleSim::dram_addr_decode(uint64_t addr, int& channel, int& bank, int& row, int& col) const {
    channel = addr_to_mp(addr);
    uint32_t a = (uint32_t)addr;
    col = (a >> 5) & 0x7F;                                              // A[11:5]
    bank = ((a >> 12) & 0x7) ^ ((a >> 16) & 0x7) ^ (channel & 0x7);    // XOR 减少冲突
    bank &= 7;
    row = (a >> 15) & 0x1FFFF;                                          // 简化: A[32:15]
}

// =============================================================================
// 1.1 周期引擎 ★ 六阶段流水线与顺序依赖互锁
// -----------------------------------------------------------------------------
// 时序顺序: 到期写回 -> 执行状态 -> 操作数读 -> 取指/译码/发射 -> CTA 回收。
// 发射和收集不修改架构寄存器/内存/PC; 只有写回提交功能语义。
// 每 warp 最多一条未提交指令, 因此 RAW/WAW/WAR/谓词/寄存器对依赖
// 均不会提前读取。此规则保守, 不模拟同 warp 独立指令重叠。
// 不同 warp 可同时位于 collector、SP/SFU/DP/BRU/LSU 中。
// 延时均为可配置的模拟值。参考: spec/06_1.1时序与修复说明.md。
// =============================================================================
void CycleSim::validate_timing_config() const {
    auto positive=[](int x){if(x<1)throw std::invalid_argument("timing parameter must be positive");};
    for(int x:{cfg.move_latency,cfg.integer_latency,cfg.imul_latency,cfg.idiv_latency,
        cfg.shuffle_latency,cfg.sp_latency,cfg.sfu_latency,cfg.dp_latency,cfg.branch_latency,
        cfg.sp_interval,cfg.sfu_interval,cfg.dp_interval,cfg.rf_logical_banks,
        cfg.rf_read_ports_per_bank,cfg.writeback_ports,cfg.execution_slots,cfg.lsu_slots,
        cfg.n_mem,cfg.dram_channels,cfg.dram_banks_per_channel,cfg.noc_flit_bytes,
        cfg.l1d_mshr,cfg.l2_mshr,cfg.smem_num_banks,cfg.smem_bank_width,cfg.atomic_latency,
        cfg.sp_lanes,cfg.sfu_lanes,cfg.dp_lanes,cfg.dram_burst_atom})positive(x);
    if(!std::isfinite(cfg.core_clock_ghz)||cfg.core_clock_ghz<=0 ||
       cfg.num_collector_units<1||cfg.num_collector_units>8 || cfg.max_warps_per_core<1||
       cfg.max_warps_per_core>MAX_WARPS_PER_SM || cfg.max_ctas_per_core<1||cfg.max_ctas_per_core>16||
       cfg.n_nsm()<1||cfg.warp_size!=32||cfg.n_nsm()!=func.cfg.n_nsm())
        throw std::invalid_argument("unsupported timing topology/configuration");
    for(double v:{cfg.lat_smem_hit,cfg.lat_l1_hit,cfg.lat_l2_hit,cfg.lat_dram_row_hit,
                  cfg.lat_barrier_resume,cfg.tRP,cfg.tRCD})
        if(!std::isfinite(v)||v<0)throw std::invalid_argument("invalid simulated latency");
}

void CycleSim::init() {
    validate_timing_config();func.init();stats=CycStats{};stats.core_clock_ghz=cfg.core_clock_ghz;
    occupancy_sum=0;nsm_cyc.clear();nsm_cyc.resize(func.cfg.n_nsm());
    for(size_t i=0;i<nsm_cyc.size();++i){auto& c=nsm_cyc[i];
        c.sp_pipe.init(cfg.sp_latency,std::max(cfg.sp_interval,(32+cfg.sp_lanes-1)/cfg.sp_lanes),cfg.execution_slots);
        c.sfu_pipe.init(cfg.sfu_latency,std::max(cfg.sfu_interval,(32+cfg.sfu_lanes-1)/cfg.sfu_lanes),cfg.execution_slots);
        c.dp_pipe.init(cfg.dp_latency,std::max(cfg.dp_interval,(32+cfg.dp_lanes-1)/cfg.dp_lanes),cfg.execution_slots);
        c.bru_pipe.init(cfg.branch_latency,1,cfg.execution_slots);
        c.lsu_pipe.init(1,1,cfg.lsu_slots);
        c.timing_l1.init(cfg.l1d_size,cfg.l1d_assoc,cfg.l1d_line,cfg.l1d_sector);
        c.mshr_ready.assign(cfg.l1d_mshr,0);
        for(int w=0;w<MAX_WARPS_PER_SM;++w)c.cyc_warps[w].func_warp=&func.nsms[i].warps[w];
    }
    timing_l2.assign(cfg.n_mem*2,TimingCache{});
    l2_mshr_ready.assign(cfg.n_mem*2,std::vector<uint64_t>(cfg.l2_mshr,0));
    for(auto& c:timing_l2)c.init(cfg.l2_slice_size,cfg.l2_assoc,cfg.l2_line,cfg.l2_sector);
    noc_ready.assign(cfg.n_mem,0);dram_bus_ready.assign(cfg.dram_channels,0);
    dram_bank_ready.assign(cfg.dram_channels*cfg.dram_banks_per_channel,0);
    dram_open_row.assign(dram_bank_ready.size(),~0ULL);
}

int CycleSim::run(uint64_t max_cycles){
    // 1.1: 预算是本次调用的增量; 0 表示不限。超预算明确失败, 可继续运行。
    uint64_t begin=stats.total_cycles;
    while(!func.is_done()&&!func.has_error() && (!max_cycles||stats.total_cycles-begin<max_cycles))tick();
    collect_stats();
    if(func.has_error())return func.error.code;
    return func.is_done()?ERR_OK:ERR_DEADLOCK;
}

void CycleSim::tick(){
    ++stats.total_cycles;
    for(size_t i=0;i<func.nsms.size();++i){auto& s=func.nsms[i];auto& c=nsm_cyc[i];
        writeback_tick(s,c);if(func.has_error())return;
        execute_tick(s,c);collector_tick(s,c);fetch_tick(s,c);if(func.has_error())return;
        decode_tick(s,c);issue_tick(s,c);
    }
    cta_retire_tick();func.dispatch_pending();
    uint64_t active=0;for(auto& s:func.nsms)active+=s.cur_warps;
    occupancy_sum+=active;
    stats.occupancy_avg_warps=double(occupancy_sum)/(stats.total_cycles*func.nsms.size());
}

void CycleSim::fetch_tick(NSMState& sm,NSMCycState& cyc){
    for(int i=0;i<cfg.max_warps_per_core;++i){int w=(cyc.fetch_cursor+i)%cfg.max_warps_per_core;
        auto& warp=sm.warps[w];auto& cw=cyc.cyc_warps[w];
        if(!warp.valid||warp.exit_done)continue;
        if(warp.bar_wait>=0){++stats.stall_barrier_wait;continue;}
        if(cw.pending_commit){++stats.serial_dependency_stalls;++stats.stall_sb_wait;continue;}
        if(!cw.ibuf_empty())continue;
        auto& h=cw.ibuf[0];h.pc=warp.stack[warp.tos].nextpc;
        func.fetch_inst(h.pc,h.inst);if(func.has_error())return;
        h.valid=true;h.mem_pending=false;cyc.fetch_cursor=(w+1)%cfg.max_warps_per_core;break;
    }
}
void CycleSim::decode_tick(NSMState&,NSMCycState&){
    // NTAS 解码在 fetch_inst 完成; 源寄存器集合在 issue 阶段建立。
}
static ExecPipe& select_pipe(CycleSim::NSMCycState& c,ExecUnit u){
    switch(u){case EU_SP:return c.sp_pipe;case EU_SFU:return c.sfu_pipe;
    case EU_DP:return c.dp_pipe;case EU_BRU:return c.bru_pipe;default:return c.lsu_pipe;}
}
void CycleSim::issue_tick(NSMState& sm,NSMCycState& cyc){
    for(int i=0;i<cfg.max_warps_per_core;++i){int w=(cyc.issue_cursor+i)%cfg.max_warps_per_core;
        auto& warp=sm.warps[w];auto& cw=cyc.cyc_warps[w];
        if(!warp.valid||warp.exit_done||warp.bar_wait>=0||cw.pending_commit||cw.ibuf_empty())continue;
        CollectorUnit* cu=nullptr;
        for(int j=0;j<cfg.num_collector_units;++j)if(!cyc.collectors[j].valid){cu=&cyc.collectors[j];break;}
        if(!cu){++stats.stall_no_collector;++stats.stall_cycles;return;}
        auto h=cw.ibuf_head();auto& ins=h.inst;*cu=CollectorUnit{};
        cu->valid=true;cu->warp=cu->warp_slot=w;cu->pc=h.pc;cu->op=ins.op;cu->inst=ins;cu->unit=op_unit(ins.op);
        uint32_t mask=warp.stack[warp.tos].mask&warp.lane_valid;
        if(ins.g&&ins.gp!=7)for(int l=0;l<32;++l)if(mask&(1u<<l)){
            bool p=warp.pred(l,ins.gp);if(ins.gn)p=!p;if(!p)mask&=~(1u<<l);
        }
        cu->exec_mask=mask;
        if(mask)for(int r:timing_sources(ins)){
            if(cu->n_ops>=int(cu->ops.size()))throw std::logic_error("operand metadata overflow");
            auto& o=cu->ops[cu->n_ops++];o.regnum=r;o.is_pred=false;o.ready=false;
        }
        // 谓词已在互锁解除后读取; 单独的谓词寄存器端口不计入 GPR bank。
        cw.pending_commit=true;cw.ibuf[0].valid=cw.ibuf[1].valid=false;
        ++stats.instructions;++stats.inst_issued[ins.op];stats.active_lane_sum+=popcount32(mask);
        ++stats.issue_cycles;cyc.issue_cursor=(w+1)%cfg.max_warps_per_core;return;
    }
    ++stats.stall_no_eligible;++stats.stall_cycles;
}

// =============================================================================
// 操作数收集 ★ 每 NSM 共享 RF bank 读端口 + 轮询公平仲裁
// -----------------------------------------------------------------------------
// bank = reg % rf_logical_banks。相同源在 collector 内只读一次。
// 冲突时分多个周期读取; 跨 collector 也竞争端口。读完后仍须等待执行槽。
// 当前 warp 互锁使源值从发射到提交保持不变, 无需复制整个寄存器堆。
// =============================================================================
void CycleSim::collector_tick(NSMState& sm,NSMCycState& cyc){
    std::vector<int> ports(cfg.rf_logical_banks,cfg.rf_read_ports_per_bank);bool conflict=false;
    for(int jj=0;jj<cfg.num_collector_units;++jj){int j=(cyc.collector_cursor+jj)%cfg.num_collector_units;
        auto& cu=cyc.collectors[j];if(!cu.valid)continue;bool ready=true;
        for(int k=0;k<cu.n_ops;++k){auto& o=cu.ops[k];if(o.ready)continue;
            int b=o.regnum%cfg.rf_logical_banks;
            if(ports[b]){--ports[b];o.ready=true;}
            else{ready=false;conflict=true;++stats.rf_conflict_stalls;}
        }
        if(!ready)continue;
        auto& p=select_pipe(cyc,cu.unit);
        if(p.busy_until>stats.total_cycles){++stats.pipeline_full_stalls;continue;}
        auto it=std::find_if(p.slots.begin(),p.slots.end(),[](const ExecPipe::Slot& s){return !s.busy;});
        if(it==p.slots.end()){++stats.pipeline_full_stalls;if(cu.unit==EU_LSU)++stats.stall_lsu_full;continue;}
        *it=ExecPipe::Slot{};it->busy=true;it->warp=it->warp_slot=cu.warp_slot;
        it->pc=cu.pc;it->inst=cu.inst;it->exec_mask=cu.exec_mask;
        it->ready_cycle=stats.total_cycles+execution_latency(cfg,cu.inst);
        if(cu.unit==EU_LSU&&cu.inst.op!=OP_LDC&&cu.inst.op!=OP_CVTA)
            it->ready_cycle=memory_ready(sm,cyc,cu.inst,cu.exec_mask,cu.warp_slot);
        it->remaining=int(std::min<uint64_t>(it->ready_cycle-stats.total_cycles,0x7fffffff));
        p.busy_until=stats.total_cycles+p.interval;cu.valid=false;
    }
    if(conflict)++stats.rf_bank_conflict_cycles;
    cyc.collector_cursor=(cyc.collector_cursor+1)%cfg.num_collector_units;
}
void CycleSim::execute_tick(NSMState&,NSMCycState& c){
    auto count=[&](ExecPipe& p){bool busy=false;for(auto& s:p.slots)if(s.busy){
        busy=true;s.remaining=int(std::min<uint64_t>(s.ready_cycle>stats.total_cycles?s.ready_cycle-stats.total_cycles:0,0x7fffffff));
    }return busy;};
    stats.sp_busy_cycles+=count(c.sp_pipe);stats.sfu_busy_cycles+=count(c.sfu_pipe);stats.dp_busy_cycles+=count(c.dp_pipe);
    count(c.bru_pipe);count(c.lsu_pipe);
}

// =============================================================================
// 写回 ★ 到期提交 + 有限写回端口
// -----------------------------------------------------------------------------
// 按 ready_cycle 选最早完成者, 同周期固定流水线顺序打破平局。
// 执行功能语义后才释放互锁, 因此 store/BAR/EXIT 不可能越过旧指令。
// 每条 warp 指令占用一个统一提交端口; 不是逐 lane RF 写端口模拟。
// =============================================================================
void CycleSim::writeback_tick(NSMState& sm,NSMCycState& c){
    for(int port=0;port<cfg.writeback_ports;++port){ExecPipe::Slot* best=nullptr;
        for(auto* p:{&c.sp_pipe,&c.sfu_pipe,&c.dp_pipe,&c.bru_pipe,&c.lsu_pipe})for(auto& s:p->slots)
            if(s.busy&&s.ready_cycle<=stats.total_cycles&&(!best||s.ready_cycle<best->ready_cycle))best=&s;
        if(!best)break;auto& w=sm.warps[best->warp_slot];
        w.clocklo=stats.total_cycles;
        int err=func.execute_inst(w,best->inst,best->exec_mask);
        if(err){func.error.code=err;func.error.sm=sm.id;func.error.warp=best->warp_slot;return;}
        ++func.inst_count;c.cyc_warps[best->warp_slot].pending_commit=false;best->busy=false;
    }
    for(auto* p:{&c.sp_pipe,&c.sfu_pipe,&c.dp_pipe,&c.bru_pipe,&c.lsu_pipe})for(auto& s:p->slots)
        if(s.busy&&s.ready_cycle<=stats.total_cycles)++stats.writeback_stalls;
}

// =============================================================================
// 访存时序 ★ 合并/共享 bank/L1-L2/有限 MSHR/NoC 带宽/DRAM 行状态
// -----------------------------------------------------------------------------
// 请求使用预约时间戳推进, 不是功能阶段直接完成访存。实际值在写回时读取
// 或写入。全局 store 采用写穿并绕过缓存分配; LDU/CG 绕过 L1。
// 同地址原子仍由功能层单次提交保障原子性, 额外模拟原子执行延时。
// =============================================================================
uint64_t CycleSim::memory_ready(NSMState& sm,NSMCycState& c,const Inst& ins,uint32_t mask,int ws){
    uint64_t now=stats.total_cycles,done=now+1;if(!mask)return done;
    auto& w=sm.warps[ws];const auto& rw=static_cast<const WarpState&>(w);
    bool store=ins.op==OP_ST||ins.op==OP_ST128,atom=ins.op==OP_ATOM||ins.op==OP_RED;
    bool wide128=ins.op==OP_LD128||ins.op==OP_ST128;
    int bytes=wide128?16:width_bytes(ins.mem_width());
    std::set<uint64_t> sectors;std::vector<std::set<uint64_t>> banks(cfg.smem_num_banks);
    bool shared=false;std::set<uint64_t> constants;
    for(int l=0;l<32;++l)if(mask&(1u<<l)){
        uint64_t addr=rw.reg(l,ins.a);
        bool pair=!atom&&(wide128||(ins.mem_width()==W_U64||ins.mem_width()==W_F64||ins.mem_width()==W_U128));
        if(pair&&ins.a!=RZ)addr|=uint64_t(rw.reg(l,ins.a+1))<<32;
        if(ins.kind==FK_M){uint64_t off=rw.reg(l,ins.c);if(pair&&ins.c!=RZ)off|=uint64_t(rw.reg(l,ins.c+1))<<32;addr+=off;}
        else if(!atom)addr+=int64_t(ins.simm16);
        uint8_t space=ins.mem_space();
        if(space==AS_FLAT){uint64_t offset;func.decode_flat(addr,space,offset,l,w);addr=offset;}
        if(space==AS_SHARED){shared=true;
            for(uint64_t a=addr/cfg.smem_bank_width;a<=(addr+bytes-1)/cfg.smem_bank_width;++a)
                banks[a%cfg.smem_num_banks].insert((store||atom)?a*32+l:a);
        }else if(space==AS_CONST){constants.insert(addr/4);}
        else{
            // LOCAL 独立于 GLOBAL 的 tag 命名空间, lane 的基址参与合并。
            if(space==AS_LOCAL)addr=(uint64_t(1)<<63)+w.lmem_base[l]+addr;
            for(uint64_t s=addr/cfg.l1d_sector;s<=(addr+bytes-1)/cfg.l1d_sector;++s)sectors.insert(s*cfg.l1d_sector);
        }
    }
    if(shared){size_t rounds=1;for(auto& b:banks)rounds=std::max(rounds,b.size());
        ++stats.smem_accesses;stats.smem_conflict_replays+=rounds-1;stats.replays_smem_conflict+=rounds-1;
        done=std::max(done,now+ns_cycles(cfg,cfg.lat_smem_hit)+rounds-1);
    }
    if(!constants.empty())done=std::max(done,now+ns_cycles(cfg,cfg.lat_l1_hit)+constants.size()-1);
    auto reserve_mshr=[&](std::vector<uint64_t>& r,uint64_t& begin){
        auto it=std::min_element(r.begin(),r.end());
        if(*it>begin){begin=*it;++stats.prt_full;}return it;
    };
    for(uint64_t addr:sectors){++stats.memory_sectors;++stats.l1_access_lines;
        uint64_t start=now;bool bypass=store||atom||ins.op==OP_LDU||(ins.y&7)==CH_CG||(ins.y&7)==CH_CV;
        TimingCache::Probe l1{};
        if(!bypass){l1=c.timing_l1.probe(addr,start);start=l1.start;
            if(l1.ready){if(l1.ready<=now)++stats.l1_hits;else ++stats.l1_pending_merges;
                done=std::max(done,std::max(now+ns_cycles(cfg,cfg.lat_l1_hit),l1.ready));continue;}}
        ++stats.l1_miss_sectors;auto mshr=reserve_mshr(c.mshr_ready,start);
        int mp=addr_to_mp(addr),sl=mp*2+addr_to_l2slice(addr);
        int flits=(cfg.l1d_sector+cfg.noc_flit_bytes-1)/cfg.noc_flit_bytes;
        uint64_t request=std::max(start,noc_ready[mp]);stats.noc_stall_cycles+=request-start;
        noc_ready[mp]=request+flits;stats.noc_flits_sent+=flits*2;
        uint64_t arrive=request+flits+cfg.noc_router_pipe+cfg.noc_link_cycles;
        ++stats.l2_accesses;TimingCache::Probe l2{};bool nocache=store||atom||(ins.y&7)==CH_CV;
        if(!nocache){l2=timing_l2[sl].probe(addr,arrive);arrive=l2.start;}
        uint64_t ready;
        if(!nocache&&l2.ready){++stats.l2_hits;ready=std::max(arrive+ns_cycles(cfg,cfg.lat_l2_hit),l2.ready);}
        else{
            ++stats.l2_misses;auto lm=reserve_mshr(l2_mshr_ready[sl],arrive);
            int ch=mp%cfg.dram_channels;
            int bank=((addr>>12)^(addr>>16)^ch)%cfg.dram_banks_per_channel;
            int bi=ch*cfg.dram_banks_per_channel+bank;uint64_t row=addr>>15;
            uint64_t begin=std::max(arrive+ns_cycles(cfg,cfg.lat_l2_hit),dram_bank_ready[bi]);
            bool hit=dram_open_row[bi]==row;
            if(hit)++stats.dram_row_hits;else ++stats.dram_row_misses;
            int latency=ns_cycles(cfg,cfg.lat_dram_row_hit)+(hit?0:ns_cycles(cfg,cfg.tRP+cfg.tRCD));
            uint64_t transfer=std::max(begin+latency,dram_bus_ready[ch]);
            int bursts=(cfg.l1d_sector+cfg.dram_burst_atom-1)/cfg.dram_burst_atom;
            ready=transfer+bursts;dram_bus_ready[ch]=ready;dram_bank_ready[bi]=ready;dram_open_row[bi]=row;
            stats.dram_bank_busy_cycles+=ready-begin;stats.dram_bus_busy_cycles+=bursts;
            if(store||atom){++stats.dram_writes;stats.bytes_to_dram+=cfg.l1d_sector;}
            if(!store){++stats.dram_reads;stats.bytes_from_dram+=cfg.l1d_sector;}
            *lm=ready;if(!nocache)l2.line->ready[l2.sector]=ready;
        }
        uint64_t response=std::max(ready,noc_ready[mp]);noc_ready[mp]=response+flits;
        ready=response+flits+cfg.noc_router_pipe+cfg.noc_link_cycles;
        if(atom)ready+=cfg.atomic_latency*popcount32(mask);
        *mshr=ready;if(!bypass)l1.line->ready[l1.sector]=ready;done=std::max(done,ready);
    }
    return done;
}

// 时间预约已在 memory_ready 完成; 不再运行旧的未连接网络/DRAM 占位循环。
void CycleSim::l1d_tick(NSMState&,NSMCycState&){}
void CycleSim::noc_tick(){}
void CycleSim::l2_tick(){}
void CycleSim::dram_tick(){}

// =============================================================================
// CycleSim::cta_retire_tick: CTA 回收 (02 册 §5) ★ 资源释放
// -----------------------------------------------------------------------------
// 检查所有 NSM 中的 CTA, 若其所有 warp 都已 EXIT:
//   1. 释放 warp 资源 (warp.valid = false)
//   2. 减少当前 warp 计数 (nsm.cur_warps)
//   3. 减少当前 CTA 计数 (nsm.cur_ctas)
//   4. 减少共享内存占用 (nsm.cur_smem)
//   5. 标记 CTA 为非运行 (cta.running = false)
//   6. 增加 func.completed_ctas 计数
//
// 教学要点:
//   - CTA (Cooperative Thread Array) 是 GPU 调度单元
//   - CTA 内所有 warp 必须都 EXIT 才算完成
//   - 释放资源后, 新 CTA 才能调度到该 NSM
// =============================================================================
// 教材引用: 第 3 章 §3.1.3 Warp Scheduling (p.31) — cta_retire_tick CTA 回收, 所有 warp EXIT 后释放资源 (warp 槽/共享内存), 影响 NSM 驻留 warp 数与占用率
// =============================================================================
// 1.1 修复说明 F10 CTA 安全回收
// -----------------------------------------------------------------------------
// 所有 warp EXIT 且 collector/执行槽均已排空后才能复用 warp 槽。
// 同时清空周期状态并扣除实际共享内存与寄存器占用。
// =============================================================================
void CycleSim::cta_retire_tick() {
    for (int sm = 0; sm < (int)func.nsms.size(); sm++) {
        NSMState& nsm = func.nsms[sm];
        for (int c = 0; c < cfg.max_ctas_per_core; c++) {
            if (!nsm.ctas[c].running) continue;
            CTAState& cta = nsm.ctas[c];
            // 检查所有 warp 是否都已 EXIT
            bool all_done = true;
            for (int ws : cta.warp_slots) {
                if (!nsm.warps[ws].exit_done) { all_done = false; break; }
            }
            auto& cyc=nsm_cyc[sm];
            if(all_done) for(int ws:cta.warp_slots){
                for(auto& cu:cyc.collectors) if(cu.valid && cu.warp_slot==ws) all_done=false;
                for(auto* pipe:{&cyc.sp_pipe,&cyc.sfu_pipe,&cyc.dp_pipe,&cyc.bru_pipe,&cyc.lsu_pipe})
                    for(auto& slot:pipe->slots) if(slot.busy && slot.warp_slot==ws) all_done=false;
            }
            if (all_done) {
                for(int ws:cta.warp_slots){cyc.cyc_warps[ws]=CycWarpState{};cyc.cyc_warps[ws].func_warp=&nsm.warps[ws];}
                // 释放资源
                for (int ws : cta.warp_slots) {
                    nsm.warps[ws].valid = false;
                }
                nsm.cur_warps -= cta.num_warps;
                nsm.cur_ctas -= 1;
                nsm.cur_smem -= cta.allocated_smem;
                nsm.cur_regs -= cta.regs_per_thread*cta.num_warps*32;
                cta.running = false;
                func.completed_ctas++;
            }
        }
    }
}

// =============================================================================
// CycleSim::collect_stats: 收集最终统计 (05 册 §4) ★ 统计汇总
// -----------------------------------------------------------------------------
// 在 run() 结束时调用, 汇总功能模拟器中的统计到 CycStats:
//   - simt_divergent_branch: 发散分支数 (来自 func.divergent_branches)
//   - simt_stack_maxdepth: SIMT 栈最大深度 (来自 func.max_stack_depth)
//   - occupancy_avg_warps: 平均活跃 warp 数 (遍历所有 NSM 统计)
//
// 占用率计算:
//   - 遍历所有 NSM, 统计 valid 且未 exit_done 的 warp 数
//   - 取平均 (total_warps / n_nsm)
// =============================================================================
// 教材引用: 第 5 章 §5.1 Analyzing a GPU Microarchitecture (p.119) — collect_stats 汇总功能模拟器统计到 CycStats: 发散分支/栈深度/平均活跃 warp 数 (占用率)
void CycleSim::collect_stats() {
    // 复制功能模拟器中的统计
    stats.simt_divergent_branch = func.divergent_branches;
    stats.simt_stack_maxdepth = func.max_stack_depth;

    // Keep the time-integrated occupancy collected by tick().
}

// =============================================================================
// CycleSim::output_stats_json: 输出 JSON 文件 (05 册 §4)
// -----------------------------------------------------------------------------
// 将统计写入指定路径的 JSON 文件, 便于后续分析工具读取
// =============================================================================
// 教材引用: 第 5 章 §5.1 Analyzing a GPU Microarchitecture (p.119) — output_stats_json 将性能统计写入 JSON 文件, 供后续分析工具读取进行 GPU 微架构瓶颈分析
void CycleSim::output_stats_json(const std::string& path) {
    std::ofstream f(path);
    if (f) {
        f << stats.to_json();
    }
}

} // namespace ntisa
