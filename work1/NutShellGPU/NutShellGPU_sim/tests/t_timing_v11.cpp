// =============================================================================
// NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
// Version 1.10
// Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
// Improved by: Tonghui Ming
// References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
// September 2026
// =============================================================================

// =============================================================================
// NutShellGPU 1.1: 时序机制回归测试 (本地扩展)
// -----------------------------------------------------------------------------
// 验证延时是否真正影响提交、依赖与资源竞争; 延时为模拟值。
// 不测量宿主机执行速度。测试直接检查架构结果与模拟周期/统计。
// =============================================================================
#include "timing/cycle_sim.hpp"
#include <iostream>
#include <memory>
using namespace ntisa;
uint64_t op(int o,int d=0,int a=0,int c=0,int x=0,int y=0){return encode(o,0,0,0,0,FK_R,d,a,c,x,y,0,0,0,0);}
uint64_t mem(int o,int d,int a,int space=AS_GLOBAL,int width=W_U32,int pred=0){return encode(o,pred,0,0,0,FK_MI,d,a,0,(width<<3)|space,0,0,0,0,0);}
using Ptr=std::unique_ptr<CycleSim>;
Ptr create(const std::vector<uint64_t>& code,Config cfg=Config{},int threads=32,int smem=0){
    auto s=std::make_unique<CycleSim>();cfg.n_clusters=cfg.n_cores_per_cluster=1;
    s->cfg=s->func.cfg=cfg;s->init();KernelInfo k;k.name="t";k.regs_u32=32;
    s->func.load_code(code,k);if(!s->func.launch_by_name("t",1,1,1,threads,1,1,smem,{}))throw std::runtime_error("launch");return s;
}
int main(){int failed=0;auto ck=[&](bool v,const char* name){std::cout<<(v?"PASS ":"FAIL ")<<name<<"\n";failed+=!v;};
    Config c;c.idiv_latency=40;
    auto a=create({op(OP_IDIV,2,0,1),op(OP_IADD,3,2,1),op(OP_EXIT)},c);
    auto& w=a->func.nsms[0].warps[0];for(int l=0;l<32;++l){w.reg(l,0)=20;w.reg(l,1)=2;}
    a->tick();a->tick();for(int i=0;i<39;++i)a->tick();
    ck(w.reg(0,2)==0&&w.reg(0,3)==0,"no arithmetic result before ready cycle");
    ck(a->run(10000)==0&&w.reg(0,2)==10&&w.reg(0,3)==12,"RAW consumer waits for committed producer");
    ck(a->stats.serial_dependency_stalls>0,"dependency stalls recorded");
    auto arithmetic=[&](int ins){auto s=create({op(ins,2,0,1),op(OP_EXIT)},c);auto& w=s->func.nsms[0].warps[0];w.reg(0,1)=1;s->run(10000);return s->stats.total_cycles;};
    ck(arithmetic(OP_IDIV)>arithmetic(OP_IADD),"instruction-specific latency changes cycles");
    auto rf=[&](int r){auto s=create({op(OP_IADD,2,0,r),op(OP_EXIT)});s->run(10000);return std::make_pair(s->stats.total_cycles,s->stats.rf_bank_conflict_cycles);};
    auto conflict=rf(4),clear=rf(1),broadcast=rf(0);
    ck(conflict.first>clear.first&&conflict.second>0,"same RF bank serializes reads");
    ck(broadcast.second==0,"duplicate source register broadcasts");
    auto dep=create({op(OP_IADD,2,0,1),op(OP_IMUL,2,2,1),op(OP_IADD,0,2,1),op(OP_EXIT)});
    auto& d=dep->func.nsms[0].warps[0];d.reg(0,0)=5;d.reg(0,1)=3;
    ck(dep->run(10000)==0&&d.reg(0,0)==27&&d.reg(0,2)==24,"WAW and WAR preserve program order");
    auto fma=create({op(OP_FFMA,2,0,1),op(OP_EXIT)});auto& fw=fma->func.nsms[0].warps[0];
    fw.reg(0,0)=0x40000000;fw.reg(0,1)=0x40400000;fw.reg(0,2)=0x3f800000;
    ck(fma->run(10000)==0&&fw.reg(0,2)==0x40e00000,"FFMA reads old destination accumulator");
    auto store=create({mem(OP_ST,4,0),op(OP_EXIT)});store->func.nsms[0].warps[0].reg(0,4)=77;
    for(int i=0;i<30;++i)store->tick();ck(store->func.read_u32(0)==0,"store invisible before memory completion");
    // 写回前观测后丢弃本用例; 正式写后读使用各 lane 不同地址。
    auto ordered=create({mem(OP_ST,4,0),mem(OP_LD,5,0),op(OP_EXIT)});
    for(int l=0;l<32;++l){ordered->func.nsms[0].warps[0].reg(l,0)=l*4;ordered->func.nsms[0].warps[0].reg(l,4)=77;}
    ck(ordered->run(100000)==0&&ordered->func.nsms[0].warps[0].reg(0,5)==77,"store-load order and memory visibility");
    auto cached=create({mem(OP_LD,4,0),mem(OP_LD,5,0),op(OP_EXIT)});cached->func.write_u32(0,99);
    ck(cached->run(100000)==0&&cached->stats.l1_hits>0&&cached->func.nsms[0].warps[0].reg(0,5)==99,"cache hit and load correctness");
    auto coalesce=[&](int stride){auto s=create({mem(OP_LD,4,0),op(OP_EXIT)});for(int l=0;l<32;++l)s->func.nsms[0].warps[0].reg(l,0)=l*stride;s->run(1000000);return s->stats.memory_sectors;};
    ck(coalesce(4)==4&&coalesce(128)==32,"global coalescing counts requested sectors");
    auto shared=[&](int stride){auto s=create({mem(OP_LD,4,0,AS_SHARED),op(OP_EXIT)},Config{},32,8192);for(int l=0;l<32;++l)s->func.nsms[0].warps[0].reg(l,0)=l*stride;s->run(10000);return std::make_pair(s->stats.total_cycles,s->stats.smem_conflict_replays);};
    auto sc=shared(128),sn=shared(4),sb=shared(0);
    ck(sc.first>sn.first&&sc.second==31,"shared bank conflicts add serialization");
    ck(sb.second==0,"shared same-word read broadcasts");
    auto bar=create({mem(OP_ST,4,0,AS_SHARED),op(OP_BAR),mem(OP_LD,5,1,AS_SHARED),op(OP_EXIT)},Config{},64,256);
    for(int wi=0;wi<2;++wi)for(int l=0;l<32;++l){auto& w=bar->func.nsms[0].warps[wi];w.reg(l,0)=(wi*32+l)*4;w.reg(l,1)=((1-wi)*32+l)*4;w.reg(l,4)=100+wi;}
    ck(bar->run(100000)==0&&bar->func.nsms[0].warps[0].reg(0,5)==101&&bar->func.nsms[0].warps[1].reg(0,5)==100,"barrier waits for prior stores across warps");
    auto pred=create({mem(OP_LD,4,0,AS_GLOBAL,W_U32,1),op(OP_EXIT)});
    ck(pred->run(10000)==0&&pred->stats.memory_sectors==0,"disabled predicate generates no memory traffic");
    auto timeout=create({op(OP_IDIV,2,0,1),op(OP_EXIT)},c);timeout->func.nsms[0].warps[0].reg(0,1)=1;
    ck(timeout->run(1)==ERR_DEADLOCK&&!timeout->func.is_done(),"cycle budget cannot report false success");
    ck(timeout->run(10000)==0,"resume after incremental cycle budget");
    auto wide=create({mem(OP_LD,4,0,AS_GLOBAL,W_U64),op(OP_IADD,6,5,255),op(OP_EXIT)});
    wide->func.global_mem.w64(0,0x1234567800000099ULL);
    ck(wide->run(100000)==0&&wide->func.nsms[0].warps[0].reg(0,6)==0x12345678,"high word dependency retained until writeback");
    auto flat=create({mem(OP_ST,4,0,AS_FLAT,W_U64),mem(OP_LD,6,0,AS_FLAT,W_U64),op(OP_EXIT)},Config{},1,64);
    auto& fl=flat->func.nsms[0].warps[0];uint64_t fa=fl.smem_flat_base;
    fl.reg(0,0)=uint32_t(fa);fl.reg(0,1)=fa>>32;fl.reg(0,4)=123;fl.reg(0,5)=456;
    ck(flat->run(10000)==0&&fl.reg(0,6)==123&&fl.reg(0,7)==456,"flat shared decoded offset and wide high word");
    auto localflat=create({mem(OP_ST,4,0,AS_FLAT,W_U64),mem(OP_LD,6,0,AS_FLAT,W_U64),op(OP_EXIT)},Config{},2);
    auto& lf=localflat->func.nsms[0].warps[0];
    for(int l=0;l<2;++l){auto a=lf.lmem_base[l];lf.reg(l,0)=uint32_t(a);lf.reg(l,1)=a>>32;lf.reg(l,4)=100+l;lf.reg(l,5)=200+l;}
    ck(localflat->run(100000)==0&&lf.reg(1,6)==101&&lf.reg(1,7)==201,"flat local address uses the owning lane base");
    CycleSim private_mem;private_mem.cfg.n_clusters=1;private_mem.cfg.n_cores_per_cluster=2;private_mem.cfg.max_ctas_per_core=1;private_mem.func.cfg=private_mem.cfg;private_mem.init();
    KernelInfo pk;pk.name="private";pk.regs_u32=4;private_mem.func.load_code({op(OP_EXIT)},pk);
    private_mem.func.launch_by_name("private",2,1,1,32,1,1,0,{});
    ck(private_mem.func.nsms[0].warps[0].lmem_base[0]!=private_mem.func.nsms[1].warps[0].lmem_base[0],"local memory addresses distinct across NSMs");
    auto rzwide=create({encode(OP_MOV32I,0,0,0,0,FK_I,RZ,0,0,1,0,0,-1,0,0),op(OP_EXIT)},Config{},1);
    ck(rzwide->run(10000)==0&&static_cast<const WarpState&>(rzwide->func.nsms[0].warps[0]).reg(0,RZ+1)==0,"wide RZ writes do not overflow register file");
    auto quad=create({mem(OP_LD,4,0,AS_LOCAL,W_U128),op(OP_EXIT)},Config{},1);
    auto& qw=quad->func.nsms[0].warps[0];for(int j=0;j<4;++j)quad->func.global_mem.w32(qw.lmem_base[0]+j*4,101+j);
    ck(quad->run(100000)==0&&qw.reg(0,4)==101&&qw.reg(0,7)==104,"local 128-bit load reads all four words");
    auto bound=create({mem(OP_LD,4,0,AS_SHARED,W_U128),op(OP_EXIT)},Config{},1,8);
    ck(bound->run(10000)==ERR_ADDR_OUT_OF_RANGE,"wide shared load rejects out of bounds");
    Config ini;ini.load_ini_string("[timing_v11]\nidiv_latency=55\ncore_clock_ghz=2\n");
    ck(ini.idiv_latency==55&&ns_cycles(ini,3.5)==7,"INI simulated latency and ns conversion");
    auto slots=[&](int capacity){Config c;c.idiv_latency=40;c.execution_slots=capacity;
        auto s=create({op(OP_IDIV,2,0,1),op(OP_EXIT)},c,256);
        for(auto& w:s->func.nsms[0].warps)if(w.valid)for(int l=0;l<32;++l){w.reg(l,0)=42;w.reg(l,1)=2;}
        int err=s->run(10000);return std::make_pair(s->stats.total_cycles,err==0&&s->func.nsms[0].warps[7].reg(0,2)==21);};
    auto serial=slots(1),pipelined=slots(16);
    ck(serial.second&&pipelined.second&&pipelined.first<serial.first,"multiple warps overlap in independent execution slots");
    auto wb=[&](int ports){Config c;c.writeback_ports=ports;auto s=create({op(OP_EXIT)},c,64);
        for(int wi=0;wi<2;++wi){auto& slot=s->nsm_cyc[0].sp_pipe.slots[wi];slot.busy=true;slot.warp_slot=wi;
            slot.ready_cycle=1;slot.exec_mask=0xffffffff;decode(op(OP_IADD,2,0,1),0,slot.inst);
            s->nsm_cyc[0].cyc_warps[wi].pending_commit=true;s->func.nsms[0].warps[wi].reg(0,0)=10;}
        s->tick();int n=0;for(int wi=0;wi<2;++wi)n+=s->func.nsms[0].warps[wi].reg(0,2)==10;
        return std::make_pair(n,s->stats.writeback_stalls);};
    ck(wb(1).first==1&&wb(1).second==1&&wb(2).first==2,"writeback port contention delays ready results");
    Config scarce;scarce.l1d_mshr=1;auto miss=create({mem(OP_LD,4,0),op(OP_EXIT)},scarce);
    for(int l=0;l<32;++l)miss->func.nsms[0].warps[0].reg(l,0)=l*128;
    ck(miss->run(1000000)==0&&miss->stats.prt_full>0,"finite MSHR pool creates backpressure");
    ck(miss->stats.dram_row_hits>0&&miss->stats.dram_row_misses>0,"DRAM row hit and miss both modeled");
    auto branch=[](int op,int delta,int pred=0){return encode(op,pred,0,0,0,FK_B,0,0,0,0,0,0,0,delta,0);};
    auto imm=[](int d,int v){return encode(OP_MOV32I,0,0,0,0,FK_I,d,0,0,0,0,0,v,0,0);};
    auto dv=create({op(OP_S2R,0,0,0,SR_LANEID),op(OP_LOP,0,0,1,4),op(OP_SETP,0,0,255,1,0x83),
        branch(OP_SSY,5),branch(OP_BRA,3,1),imm(1,0xaa),branch(OP_BRA,2),imm(1,0xbb),op(OP_EXIT)});
    bool diverged=dv->run(10000)==0;for(int l=0;l<32;++l)diverged&=dv->func.nsms[0].warps[0].reg(l,1)==uint32_t(l%2?0xbb:0xaa);
    ck(diverged&&dv->func.divergent_branches==1,"cycle pipeline preserves SIMT divergence and reconvergence");
    auto cv=create({op(OP_CVTA,2,0),op(OP_EXIT)},Config{},1);
    ck(cv->run(10000)==0&&cv->stats.memory_sectors==0,"CVTA address conversion produces no memory request");
    auto atom=create({op(OP_ATOM,4,0,1,(W_U32<<3)|AS_SHARED,AOP_ADD),op(OP_EXIT)},Config{},32,4);
    for(int l=0;l<32;++l)atom->func.nsms[0].warps[0].reg(l,1)=1;
    ck(atom->run(100000)==0&&atom->func.nsms[0].warps[0].reg(31,4)==31,"atomic add serializes updates across lanes");
    auto badatom=create({op(OP_ATOM,4,0,1,(W_U32<<3)|AS_SHARED,AOP_ADD),op(OP_EXIT)},Config{},1,4);
    badatom->func.nsms[0].warps[0].reg(0,0)=4;
    ck(badatom->run(100000)==ERR_ADDR_OUT_OF_RANGE,"atomic load error propagates");
    bool rejected=false;try{Config bad;bad.num_collector_units=9;create({op(OP_EXIT)},bad);}catch(const std::invalid_argument&){rejected=true;}
    ck(rejected,"invalid fixed-array configuration rejected");
    std::cout<<"FAILED "<<failed<<"\n";return failed?1:0;
}
