// =============================================================================
// NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
// Version 1.10
// Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
// Improved by: Tonghui Ming
// References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
// September 2026
// =============================================================================

// =============================================================================
// NutShellGPU 1.1: 既有正确性修复回归测试 (本地扩展)
// 保留修复用例; 功能结果与资源生命周期均为断言对象。
// =============================================================================
#include "func/func_sim.hpp"
#include "timing/cycle_sim.hpp"
#include <iostream>
#include <cstring>
using namespace ntisa;
uint64_t e(int op,int d=0,int a=0,int c=0,int x=0,int m=0){return encode(op,0,0,0,m,FK_R,d,a,c,x,0,0,0,0,0);}
uint64_t imm(int d,uint32_t v){return encode(OP_MOV32I,0,0,0,0,FK_I,d,0,0,0,0,0,v,0,0);}
uint32_t bits(float x){uint32_t u;std::memcpy(&u,&x,4);return u;}
int main(){int fails=0;auto ck=[&](bool ok,const char* s){std::cout<<(ok?"PASS ":"FAIL ")<<s<<"\n";fails+=!ok;};
 PageMem memory;memory.w32(4,11);memory.w32((1ULL<<44)+4,22);ck(memory.r32(4)==11&&memory.r32((1ULL<<44)+4)==22,"64-bit page address does not alias");PageMem copied=memory;copied.w32(4,33);ck(memory.r32(4)==11&&copied.r32(4)==33,"page cache safe deep copy");memory.clear();ck(memory.r32(4)==0,"page cache invalidation on clear");
 FuncSim f;f.cfg.n_clusters=1;f.cfg.n_cores_per_cluster=1;f.init();KernelInfo k;k.name="numeric";k.regs_u32=8;
 f.load_code({imm(0,bits(.25f)),imm(1,bits(.5f)),e(OP_FADD,2,0,1),e(OP_FMUL,3,0,1),imm(255,99),e(OP_MOV,4,255),e(OP_EXIT)},k);
 ck(f.launch_by_name(k.name,1,1,1,32,1,1,0,{})&&f.sync()==0,"numeric launch");auto& w=f.nsms[0].warps[0];
 ck(w.reg(0,2)==bits(.75f),"FADD preserves fractional part");ck(w.reg(0,3)==bits(.125f),"FMUL preserves fractional part");ck(w.reg(0,4)==0,"RZ writes discarded");
 for(int m=0;m<4;++m){k.name="round"+std::to_string(m);f.load_code({imm(0,bits(1.f)),imm(1,bits(0x1p-24f)),e(OP_FADD,2,0,1,0,m),e(OP_EXIT)},k);f.launch_by_name(k.name,1,1,1,1,1,1,0,{});ck(f.sync()==0 && f.nsms[0].warps[0].reg(0,2)==(m==3?bits(1.f)+1:bits(1.f)),"directed FP32 rounding");}
 k.name="entry";k.code_offset=8;f.load_code({imm(0,7),e(OP_EXIT)},k);ck(f.launch_by_name(k.name,1,1,1,32,1,1,20000,{})&&f.nsms[0].cur_smem==20000,"dynamic shared allocation");
 ck(f.sync()==0&&f.nsms[0].warps[0].reg(0,0)==0,"nonzero entry");ck(f.nsms[0].cur_smem==0&&f.nsms[0].cur_regs==0,"resource reclamation");
 ck(!f.launch_by_name(k.name,1,1,1,32,1,1,50000,{})&&f.error.code==ERR_LAUNCH,"reject excessive shared");
 ck(!f.launch_by_name(k.name,0,1,1,32,1,1,0,{}),"reject zero grid");
 f.cfg.rf_physical_entries=256;ck(f.launch_by_name(k.name,5,1,1,32,1,1,0,{})&&f.next_cta==1,"register residency limit");ck(f.sync()==0&&f.completed_ctas==5,"register limited CTA refill");
 Inst i;f.fetch_inst(3,i);ck(f.error.code==ERR_MISALIGNED_PC,"bad PC error");
 CycleSim cy;cy.cfg.n_clusters=1;cy.cfg.n_cores_per_cluster=1;cy.func.cfg=cy.cfg;cy.func.cfg.max_ctas_per_core=1;cy.init();k.name="repeat";k.code_offset=0;cy.func.load_code({imm(0,13),e(OP_EXIT)},k);
 for(int j=0;j<2;++j){auto before=cy.stats.total_cycles;ck(cy.func.launch_by_name(k.name,5,1,1,32,1,1,30000,{})&&cy.run(1000)==0&&cy.func.completed_ctas==5,"cycle CTA refill/relaunch");ck(cy.stats.total_cycles>before&&cy.func.nsms[0].cur_smem==0&&cy.func.nsms[0].cur_regs==0,"cycle resource accounting");}
 ck(cy.stats.occupancy_avg_warps>0,"time averaged occupancy");
 FuncSim bar;bar.cfg.n_clusters=1;bar.cfg.n_cores_per_cluster=1;bar.init();k.name="bar";bar.load_code({e(OP_BAR),e(OP_EXIT)},k);bar.launch_by_name("bar",1,1,1,64,1,1,0,{});
 auto& sm=bar.nsms[0];bar.step_warp(sm,0);ck(sm.warps[0].bar_wait==0,"first warp cannot pass two-warp barrier");bar.step_warp(sm,1);ck(sm.warps[0].bar_wait==-1,"second warp releases barrier");ck(bar.sync()==0,"two warp barrier finishes");
 bar.load_code({e(OP_BAR,0,0,0,0x10),e(OP_BAR,0,0,0,0x20),e(OP_BAR),e(OP_EXIT)},k);bar.launch(0,1,1,1,64,1,1,0,{},1);ck(bar.sync()==0,"ARRIVE WAIT followed by SYNC generation");
 return fails?1:0;}
