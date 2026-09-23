# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.10
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================

# =============================================================================
# NutShellGPU 1.1: 模型数值检查
# -----------------------------------------------------------------------------
# 逐层比较独立参考张量; 检查由虚拟 GPU 计算的 argmax。
# 数值容差同时使用绝对/相对误差, 分类索引必须完全一致。
# =============================================================================
"""Check every tensor and the device argmax from a model_runner execution."""
import argparse,json
from pathlib import Path
import numpy as np

def check(root,out):
    root,out=Path(root),Path(out);expected=np.load(root/'expected.npz');layers=[]
    for name in expected.files:
        ref=expected[name]
        actual=np.stack([np.fromfile(out/f'{i}_{name}.bin',dtype='<f4') for i in range(len(ref))])
        np.testing.assert_allclose(actual,ref,atol=1e-4,rtol=1e-4,err_msg=name)
        layers.append(dict(layer=name,max_abs_error=float(np.max(np.abs(actual-ref))),elements=int(ref.size)))
    ref=np.load(root/'predictions.npy')
    actual=np.asarray([np.fromfile(out/f'{i}_argmax.bin',dtype='<u4')[0] for i in range(len(ref))])
    np.testing.assert_array_equal(actual,ref)
    result=dict(passed=True,samples=len(ref),layers=layers,predictions=actual.tolist())
    (out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fixture_directory');p.add_argument('output_directory');a=p.parse_args()
    print(json.dumps(check(a.fixture_directory,a.output_directory),indent=2))
