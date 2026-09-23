// =============================================================================
// NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
// Version 1.10
// Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
// Improved by: Tonghui Ming
// References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
// September 2026
// =============================================================================

// =============================================================================
// NutShellGPU 1.1: 本地教学扩展, 待课程教师审查 (非原作者发布)
// timing_model.hpp - 可配置的执行时序与有限容量缓存预约模型
// -----------------------------------------------------------------------------
// 保留 NTAS1 / warp32 / NSM / 五类执行单元。延时为可配置的模拟值。
// 时间戳均为 NSM 周期, ns 经 core_clock_ghz 换算。
// 缓存仅保存 tag 与完成时间; 数据唯一保存在 FuncSim, 避免两份数据失配。
// =============================================================================
#pragma once
#include "../func/func_sim.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace ntisa {
// =============================================================================
// TimingCache: 组相联、按扇区预约、LRU 替换 ★ 缓存就绪时间
// -----------------------------------------------------------------------------
// ready > now 表示在途填充, 不能提前算作命中; 请求可合并等待该时间。
// 被替换行尚未完成时, 新请求的 start 顺延, 不无限增加物理缓存容量。
// 本模型采用保守预约顺序, 不实现逐 flit 路由或 FR-FCFS 重排序。
// =============================================================================
struct TimingCache {
    struct Line { uint64_t tag=~0ULL, used=0; std::vector<uint64_t> ready; };
    int line_bytes=128, sector_bytes=32;
    uint64_t stamp=0;
    std::vector<std::vector<Line>> sets;
    void init(int bytes,int assoc,int line,int sector) {
        if(bytes<assoc*line || assoc<1 || line<sector || sector<1 || line%sector)
            throw std::invalid_argument("invalid timing cache geometry");
        line_bytes=line;sector_bytes=sector;stamp=0;
        sets.assign(bytes/(assoc*line),std::vector<Line>(assoc));
        for(auto& s:sets) for(auto& l:s) l.ready.assign(line/sector,0);
    }
    struct Probe { Line* line; int sector; uint64_t start,ready; };
    Probe probe(uint64_t addr,uint64_t now) {
        uint64_t tag=addr/line_bytes;auto& set=sets[tag%sets.size()];
        Line* chosen=nullptr;
        for(auto& l:set) if(l.tag==tag){chosen=&l;break;}
        if(!chosen){
            chosen=&*std::min_element(set.begin(),set.end(),[](const Line&a,const Line&b){return a.used<b.used;});
            for(auto t:chosen->ready)now=std::max(now,t);
            chosen->tag=tag;std::fill(chosen->ready.begin(),chosen->ready.end(),0);
        }
        chosen->used=++stamp;int sector=(addr%line_bytes)/sector_bytes;
        return {chosen,sector,now,chosen->ready[sector]};
    }
};

// 按功能语义列出寄存器读集合, 相同寄存器在同一 collector 内广播。
// 冷门复合指令采用保守读集合; 正确性还由每 warp 顺序提交互锁保证。
inline std::vector<int> timing_sources(const Inst& i) {
    std::set<int> s;auto add=[&](int r){if(r>=0&&r<RZ)s.insert(r);};
    switch(i.op){
    case OP_NOP:case OP_MOV32I:case OP_S2R:case OP_BRA:case OP_SSY:
    case OP_EXIT:case OP_BAR:case OP_MEMBAR:case OP_RET:case OP_CALL:
    case OP_YIELD:case OP_TRAP:case OP_BRKPT:break;
    case OP_LDC: add(i.a);break;
    case OP_MOV:case OP_RCP:case OP_RSQ:case OP_MUFU:case OP_FABS:
    case OP_FNEG:case OP_IABS:case OP_INEG:case OP_POPC:case OP_BREV:
    case OP_F2F:case OP_XCVT:case OP_FRND:case OP_BRX:
        add(i.a);break;
    case OP_CVTA:add(i.a);if(i.a!=RZ)add(i.a+1);break;
    default:
        add(i.a);if(i.kind==FK_R||i.kind==FK_M)add(i.c);
        if(i.op==OP_FFMA||i.op==OP_IMAD||i.op==OP_IADD3||i.op==OP_BFI||i.op==OP_SHF||i.op==OP_PRMT)add(i.d);
        if(i.op==OP_ST||i.op==OP_ST128){
            int n=i.op==OP_ST128?4:(width_bytes(i.mem_width())+3)/4;
            for(int j=0;j<n;j++)add(i.d+j);
        }
        if(i.op==OP_ATOM||i.op==OP_RED){add(i.c);if((i.y&15)==AOP_CAS)add(i.b);}
        if(i.op==OP_DADD){add(i.a+1);add(i.c+1);}
        if((i.kind==FK_M||i.kind==FK_MI)&&(i.op==OP_LD128||i.op==OP_ST128||i.mem_width()==W_U64||i.mem_width()==W_F64||i.mem_width()==W_U128)){
            add(i.a+1);if(i.kind==FK_M&&i.c!=RZ)add(i.c+1);
        }
    }
    return {s.begin(),s.end()};
}

inline int ns_cycles(const Config& c,double ns){return std::max(1,int(std::ceil(ns*c.core_clock_ghz)));}
inline int execution_latency(const Config& c,const Inst& i){
    switch(i.op){
    case OP_MOV:case OP_MOV32I:case OP_NOP:case OP_S2R:return c.move_latency;
    case OP_IMUL:case OP_IMAD:return c.imul_latency;
    case OP_IDIV:case OP_IREM:return c.idiv_latency;
    case OP_FADD:case OP_FSUB:case OP_FMUL:case OP_FFMA:return c.sp_latency;
    case OP_SHFL:case OP_VOTE:return c.shuffle_latency;
    case OP_LDC:return ns_cycles(c,c.lat_l1_hit);
    case OP_BAR:return c.branch_latency+ns_cycles(c,c.lat_barrier_resume);
    default: switch(op_unit(i.op)){
        case EU_SFU:return c.sfu_latency;
        case EU_DP:return c.dp_latency;
        case EU_BRU:return c.branch_latency;
        default:return c.integer_latency;
    }}
}
} // namespace ntisa
