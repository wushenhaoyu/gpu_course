# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.10
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================

"""运行随包 TinyLeNet CUDA 样例, 保存各层结果供测试脚本比较。"""
from pathlib import Path
import argparse,json
import numpy as np
from cuda_api import CUDA

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);meta=json.loads((a.assets/'model.json').read_text())
    raw=(a.assets/'weights.bin').read_bytes()
    weights={v['name']:np.frombuffer(raw,dtype='<f4',count=v['nbytes']//4,offset=v['offset_bytes']).reshape(v['shape']).copy() for v in meta['tensors']}
    x=np.fromfile(a.assets/'input.bin',dtype='<f4').reshape(-1,1,28,28);n=len(x)
    c=CUDA()
    try:
        print('DEVICE',c.name,'SAMPLES',n,flush=True)
        c.compile((Path(__file__).with_name('kernels.cu')).read_text(encoding='utf-8'));src=c.upload(x)
        def step(kernel,name,shape,args):
            dst=c.malloc(int(np.prod(shape))*4);c.launch(kernel,int(np.prod(shape)),args(dst))
            c.download(dst,np.empty(shape,'f4')).tofile(a.out/(name+'.bin'))
            print('DONE',name,'shape=',shape,flush=True)
            return dst
        for number,ci,co,h in [(1,1,8,28),(2,8,16,14)]:
            w=c.upload(weights[f'conv{number}.weight']);b=c.upload(weights[f'conv{number}.bias'])
            shape=(n,co,h,h);count=int(np.prod(shape))
            dst=step('conv',f'conv{number}',shape,lambda out:[src,w,b,out,n,ci,co,h])
            src=step('relu',f'relu{number}',shape,lambda out:[dst,out,count])
            src=step('pool',f'pool{number}',(n,co,h//2,h//2),lambda out:[src,out,n,co,h])
        w=c.upload(weights['fc.weight']);b=c.upload(weights['fc.bias'])
        logits=step('dense','logits',(n,10),lambda out:[src,w,b,out,n,784,10])
        pred=c.malloc(n*4);c.launch('classify',n,[logits,pred,n]);predictions=c.download(pred,np.empty(n,'i4'));predictions.tofile(a.out/'predictions.bin')
        print('PREDICTIONS',predictions.tolist(),flush=True)
        (a.out/'device.json').write_text(json.dumps(dict(device=c.name,samples=n),indent=2))
    finally:c.close()

if __name__=='__main__':main()
