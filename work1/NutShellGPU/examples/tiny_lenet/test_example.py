# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.10
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================

"""用独立 NumPy 前向结果检查 CUDA 与虚拟 GPU, 并记录两种时间。"""
from pathlib import Path
import argparse,json,subprocess,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parent

def reference():
    meta=json.loads((ROOT/'assets/model.json').read_text());raw=(ROOT/'assets/weights.bin').read_bytes()
    weights={v['name']:np.frombuffer(raw,dtype='<f4',count=v['nbytes']//4,offset=v['offset_bytes']).reshape(v['shape']) for v in meta['tensors']}
    x=np.fromfile(ROOT/'assets/input.bin',dtype='<f4').reshape(-1,1,28,28);values={}
    for number in (1,2):
        w,b=weights[f'conv{number}.weight'],weights[f'conv{number}.bias'];n,ci,h,_=x.shape
        padded=np.pad(x,((0,0),(0,0),(1,1),(1,1)));out=np.broadcast_to(b[None,:,None,None],(n,len(b),h,h)).copy()
        # 卷积采用互相关语义; 保持 NCHW 与 OIHW 的索引约定。
        for co in range(len(b)):
            for c in range(ci):
                for ky in range(3):
                    for kx in range(3):out[:,co]+=padded[:,c,ky:ky+h,kx:kx+h]*w[co,c,ky,kx]
        values[f'conv{number}']=out;out=np.maximum(out,0);values[f'relu{number}']=out
        x=out.reshape(n,len(b),h//2,2,h//2,2).max(axis=(3,5));values[f'pool{number}']=x
    flat=x.reshape(len(x),-1);out=np.broadcast_to(weights['fc.bias'],(len(x),10)).copy()
    for i in range(flat.shape[1]):out+=flat[:,i,None]*weights['fc.weight'][None,:,i]
    values['logits']=out;return values

def run(command,out):
    out.mkdir(parents=True,exist_ok=True)
    # 计时只包围子进程; 独立参考计算与结果比较不计入程序运行时间。
    with (out/'process.log').open('w') as f:
        start=time.perf_counter();p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT);seconds=time.perf_counter()-start
    (out/'process_time.json').write_text(json.dumps(dict(program_seconds=seconds,exit_code=p.returncode),indent=2))
    p.check_returncode();return seconds

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['cuda','sim','both'],default='both')
    p.add_argument('--runner',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.backend in ('sim','both') and a.runner is None:p.error('--runner is required for simulator execution')
    refs=reference();labels=np.fromfile(ROOT/'assets/labels.bin',dtype='u1');expected=refs['logits'].argmax(1);results={}
    a.out.mkdir(parents=True,exist_ok=True)
    if a.backend in ('cuda','both'):
        out=(a.out/'cuda').resolve();seconds=run([sys.executable,'-X','utf8',str(ROOT/'run_cuda.py'),'--assets',str(ROOT/'assets'),'--out',str(out)],out)
        errors={}
        for name,ref in refs.items():
            actual=np.fromfile(out/(name+'.bin'),dtype='<f4').reshape(ref.shape)
            np.testing.assert_allclose(actual,ref,atol=1e-4,rtol=1e-4,err_msg=name);errors[name]=float(np.abs(actual-ref).max())
        pred=np.fromfile(out/'predictions.bin',dtype='<i4');np.testing.assert_array_equal(pred,expected)
        results['cuda']=dict(program_seconds=seconds,predictions=pred.tolist(),accuracy=float(np.mean(pred==labels)),max_layer_errors=errors)
    if a.backend in ('sim','both'):
        out=(a.out/'sim').resolve();seconds=run([str(a.runner.resolve()),str(ROOT/'fixture'),str(out)],out)
        mapping={'00_conv':'relu1','01_pool':'pool1','02_conv':'relu2','03_pool':'pool2','04_fc':'logits'};errors={}
        # 虚拟 GPU 的空间张量含一圈零边框; 比较前去掉边框。
        for layer,name in mapping.items():
            ref=refs[name];shape=ref.shape[1:]
            if len(shape)==3:shape=(shape[0],shape[1]+2,shape[2]+2)
            actual=np.stack([np.fromfile(out/f'{i}_{layer}.bin',dtype='<f4').reshape(shape) for i in range(len(labels))])
            if actual.ndim==4:actual=actual[:,:,1:-1,1:-1]
            np.testing.assert_allclose(actual,ref,atol=1e-4,rtol=1e-4,err_msg=layer);errors[layer]=float(np.abs(actual-ref).max())
        pred=np.array([np.fromfile(out/f'{i}_argmax.bin',dtype='<u4')[0] for i in range(len(labels))]);np.testing.assert_array_equal(pred,expected)
        stats=json.loads((out/'stats.json').read_text());seconds_sim=stats['cycles']/(stats['core_clock_ghz']*1e9)
        assert np.isclose(stats['simulated_seconds'],seconds_sim,rtol=1e-12,atol=1e-15)
        results['sim']=dict(program_seconds=seconds,simulated_seconds=stats['simulated_seconds'],cycles=stats['cycles'],core_clock_ghz=stats['core_clock_ghz'],predictions=pred.tolist(),accuracy=float(np.mean(pred==labels)),max_layer_errors=errors)
    results['labels']=labels.tolist();results['samples']=len(labels)
    (a.out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':main()
