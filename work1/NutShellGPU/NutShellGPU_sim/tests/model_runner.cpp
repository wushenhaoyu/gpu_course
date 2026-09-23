// =============================================================================
// NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
// Version 1.10
// Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
// Improved by: Tonghui Ming
// References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
// September 2026
// =============================================================================

// =============================================================================
// NutShellGPU 1.1: 多结构模型周期级验证入口
// -----------------------------------------------------------------------------
// 输入/权重/NTAS1 由测试夹具提供; 卷积、池化、全连接、残差相加和 argmax
// 均在虚拟 GPU 执行。主机只装载数据、启动内核、导出结果与统计。
// 参数: fixture_directory output_directory [config.ini]
// =============================================================================
#include "timing/cycle_sim.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
using namespace ntisa;
namespace fs=std::filesystem;
std::vector<char> read(const fs::path& p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("missing "+p.string());return {(std::istreambuf_iterator<char>(f)),{}};}
int main(int argc,char** argv){try{
    if(argc<3)throw std::runtime_error("model_runner fixture_dir output_dir [config.ini]");
    fs::path root=argv[1],out=argv[2];fs::create_directories(out);
    std::ifstream meta(root/"program.txt");uint64_t size,input,logits,pred;int samples,input_count,classes;
    if(!(meta>>size>>input>>logits>>pred>>samples>>input_count>>classes))throw std::runtime_error("bad metadata");
    struct Layer{std::string name;int gx,gy,dx,dy;uint64_t out;int count;};std::vector<Layer> layers;Layer l;
    while(meta>>l.name>>l.gx>>l.gy>>l.dx>>l.dy>>l.out>>l.count)layers.push_back(l);
    CycleSim sim;if(argc>3)sim.cfg.load_ini(argv[3]);sim.func.cfg=sim.cfg;sim.init();
    auto memory=read(root/"memory.bin"),images=read(root/"input.bin");
    if(memory.size()!=size||images.size()!=size_t(samples)*input_count*4)throw std::runtime_error("fixture size mismatch");
    sim.func.memcpy_h2d(0,memory.data(),memory.size());
    for(auto& l:layers){auto raw=read(root/(l.name+".ntas"));if(raw.empty()||raw.size()%8)throw std::runtime_error("bad NTAS size");
        std::vector<uint64_t> code(raw.size()/8);std::memcpy(code.data(),raw.data(),raw.size());KernelInfo k;k.name=l.name;k.regs_u32=32;sim.func.load_code(code,k);}
    std::ofstream csv(out/"layers.csv");csv<<"sample,layer,cycles,instructions,completed_ctas,total_ctas\n";
    for(int sample=0;sample<samples;++sample){sim.func.memcpy_h2d(input,images.data()+size_t(sample)*input_count*4,input_count*4);
        for(auto& l:layers){auto before=sim.stats.total_cycles,ins=sim.func.inst_count;
            if(!sim.func.launch_by_name(l.name,l.gx,l.gy,1,l.dx,l.dy,1,0,{}))throw std::runtime_error("launch "+l.name);
            int err=sim.run(500000000);if(err||!sim.func.is_done())throw std::runtime_error("execution "+l.name+" code="+std::to_string(err)+" "+sim.func.error.msg);
            csv<<sample<<","<<l.name<<","<<sim.stats.total_cycles-before<<","<<sim.func.inst_count-ins<<","<<sim.func.completed_ctas<<","<<sim.func.total_ctas<<"\n";csv.flush();
            std::vector<char> values(l.count*4);sim.func.memcpy_d2h(values.data(),l.out,values.size());
            std::ofstream f(out/(std::to_string(sample)+"_"+l.name+".bin"),std::ios::binary);f.write(values.data(),values.size());
            std::cout<<"DONE "<<sample<<" "<<l.name<<" cycles="<<sim.stats.total_cycles-before<<std::endl;
        }
    }
    sim.output_stats_json((out/"stats.json").string());return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
