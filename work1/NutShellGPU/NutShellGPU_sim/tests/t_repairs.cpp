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
#include "isa/ntas_enc.hpp"
#include <chrono>
#include <iostream>
#include <cstring>
using namespace ntisa;
int main(){
 int failed=0;
 auto check=[&](bool ok,const char* name){std::cout<<(ok?"PASS ":"FAIL ")<<name<<"\n";failed+=!ok;};
 PageMem mem;
 for(uint64_t a=4080;a<4110;a++){
  uint64_t v=0xF123456789ABCDEFULL;mem.w64(a,v);check(mem.r64(a)==v,"unaligned/cross-page u64");
 }
 std::vector<uint8_t> bytes(10000),back(10000);for(int i=0;i<10000;i++)bytes[i]=(uint8_t)(i*13);
 mem.write_bytes(4090,bytes.data(),bytes.size());mem.read_bytes(4090,back.data(),back.size());check(bytes==back,"cross-page bulk roundtrip");
 check(mem.r32(0x90000000)==0,"lazy zero memory");
 auto ex=encode(OP_EXIT,0,0,0,0,FK_R,0,0,0,0,0,0,0,0,0);
 FuncSim sim;sim.cfg.n_clusters=1;sim.cfg.n_cores_per_cluster=1;sim.cfg.max_ctas_per_core=1;sim.init();
 KernelInfo k;k.name="first";k.regs_u32=4;sim.load_code({ex},k);
 sim.launch_by_name("first",5,1,1,32,1,1,0,{});int err=sim.sync();
 check(err==ERR_OK&&sim.completed_ctas==5,"CTA refill beyond residency");
 // Reset after intentionally exposing baseline incomplete launch.
 FuncSim multi;multi.init();multi.load_code({ex},k);k.name="second";k.static_smem=4096;multi.load_code({ex},k);
 multi.launch_by_name("second",1,1,1,32,1,1,0,{});
 check(multi.nsms[0].cur_smem==4096,"named second module resource metadata");multi.sync();
 std::vector<uint64_t> words={
  encode(OP_S2R,0,0,0,0,FK_R,0,0,0,SR_TID_X,0,0,0,0,0),
  encode(OP_MOV32I,0,0,0,0,FK_I,1,0,0,0,0,0,1,0,0),
  encode(OP_SHFL,0,0,0,0,FK_R,0,0,1,3,0,0,0,0,0),ex};
 FuncSim shuffle;shuffle.init();k.name="shuffle";k.static_smem=0;shuffle.load_code(words,k);
 shuffle.launch_by_name("shuffle",1,1,1,32,1,1,0,{});shuffle.sync();
 bool ok=true;for(int lane=0;lane<32;lane++)ok &= shuffle.nsms[0].warps[0].reg(lane,0)==(unsigned)(lane^1);
 check(ok,"SHFL BFLY source destination alias");
 return failed?1:0;
}
