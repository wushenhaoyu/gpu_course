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
// NutShellGPU 硬件模拟器 - 公共宏定义头文件
// NutShellGPU_defines.svh
// -----------------------------------------------------------------------------
// 作用：本文件是 NutShellGPU 硬件模拟器（周期级 RTL）中所有 SystemVerilog
//       模块共享的通用宏定义头。每个 .sv 模块通过 `include "NutShellGPU_defines.svh"
//       在文件首引入本头，从而获得统一的编译开关与共享声明。
//
// 对应规格册：依据 05 册《模拟器实现契约》§2 "公共接口"。
//   - 05 册 §2 规定所有模块共用同一份解码器与同一份 NTAS1 语义参考实现，
//     因此公共头需要保证跨模块类型/宏的一致性。
//
// 设计要点：
//   1. 通过 `ifndef / `define / `endif 包裹实现 include guard（多重包含保护），
//      保证同一编译单元中即使多个 .sv 文件都 `include 本头，宏也只展开一次。
//   2. 真正的共享类型（参数、枚举、结构体、解码函数）定义在 NutShellGPU_pkg.sv
//      的 package NutShellGPU_pkg 中，每个模块用 `import NutShellGPU_pkg::*;` 引入；
//      本头文件目前仅作占位，用于预留未来跨模块共享的 compiler directive。
//   3. 这种 "package + include guard 头" 的组合是 SystemVerilog 推荐做法：
//      package 提供类型作用域隔离，include 头提供预处理宏共享。
// =============================================================================

`ifndef NutShellGPU_DEFINES_SVH
`define NutShellGPU_DEFINES_SVH

// NutShellGPU_pkg 包通过下述 import 形式在各模块中引入：
//   import NutShellGPU_pkg::*;
// 本头当前作为占位，预留给未来跨模块共享的 compiler directive（如编译开关、
// 调试宏等）。基线 RTL 不在此声明任何业务宏，以避免污染 package 命名空间。

`endif // NutShellGPU_DEFINES_SVH
