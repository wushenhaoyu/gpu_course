// =============================================================================
// NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
// Version 1.10
// Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
// Improved by: Tonghui Ming
// References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
// September 2026
// =============================================================================

// =============================================================================
// 模拟时间序列化测试: 周期/频率换算与纳秒级数值保留。
// =============================================================================
#include "timing/cycle_sim.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace ntisa;
int main(){
    int passed=0;
    for(double ghz:{0.5,1.0,2.0})for(uint64_t cycles:{uint64_t(0),uint64_t(1),uint64_t(123456789)}){
        CycStats s;s.core_clock_ghz=ghz;s.total_cycles=cycles;
        auto json=s.to_json();auto key=json.find("\"simulated_seconds\":");
        if(key==std::string::npos)throw std::runtime_error("missing simulated_seconds");
        double actual=std::stod(json.substr(json.find(':',key)+1));
        double expected=double(cycles)/(ghz*1e9);
        if(std::abs(actual-expected)>std::abs(expected)*1e-14+1e-20)
            throw std::runtime_error("simulated time conversion mismatch");
        ++passed;
    }
    std::cout<<"PASS "<<passed<<" simulated time checks\n";
}
