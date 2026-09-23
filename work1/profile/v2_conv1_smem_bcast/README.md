# conv1 v2 (lane = 输出通道 oc + shared memory)

本轮**没有单独的 REPORT.md** —— v2 是 conv1 优化链的中间站, 完整分析与 A/B 见
../v3_conv1_coalesced/REPORT.md 的第 1 节 (v1 → v2)。

一句话结论: v2 把 global load 的 sector 浪费修干净了 (bytes/sector 12.14 → 32,
global-ld sector 923,465,296 → 1,944,626, L1/TEX 95.30% → 50.92%), 但把瓶颈搬到了
global store —— lane = oc 让 32 个 lane 写到 32 个间隔 576 字节的通道平面上,
store sector 5,760,000 → 46,080,000 (8× 放大), L2 吞吐 7.77% → 72.61%。
净收益只有 1.34× (SM 活跃周期 2,244,668 → 1,675,868), 不是事前估的 ~2×。

本目录保留该轮的原始采集件 (reports/*.ncu-rep、analysis/stall_hotspots_v2.txt 等),
供逐位复核。
