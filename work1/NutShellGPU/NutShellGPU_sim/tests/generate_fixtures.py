# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.10
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================

# NutShellGPU 1.1: 构造残差/深度可分离 NTAS1 夹具及独立参考。
# PyTorch 只用于参考输出, 不在模拟器进程中执行。
"""Generate pure NTAS1 model fixtures and independent PyTorch CPU references."""
from pathlib import Path
import json,struct,hashlib,re
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'NutShellGPU_sim/tests/fixtures';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
OP={k:int(v,16) for k,v in re.findall(r'OP_(\w+)\s*=\s*(0x[0-9A-Fa-f]+)',(ROOT/'NutShellGPU_sim/isa/ntisa.hpp').read_text(encoding='utf-8'))}
class Asm:
 def __init__(self):self.code=[]
 def op(self,o,d=0,a=0,c=0,x=0,y=0,imm=None,offset=None,pred=0):
  word=(OP[o]<<57)|(pred<<56)|(d<<40)
  if imm is not None:word|=(int(imm)&0xffffffff)<<8
  elif offset is not None:
   assert -32768<=offset<32768
   word|=(a<<32)|(x<<24)|(y<<16)|(offset&65535)
  else:word|=(a<<32)|(c<<24)|(x<<16)|(y<<8)
  self.code.append(word)
 def imm(self,d,v):self.op('MOV32I',d,imm=v)
 def sr(self,d,v):self.op('S2R',d,x=v)
 def addscale(self,d,a,n):self.imm(30,n);self.op('IMAD',d,a,30)
 def ld(self,d,p,off=0):self.op('LD',d,p,x=0x38,offset=off)
 def st(self,d,p,off=0):self.op('ST',d,p,x=0x38,offset=off)
 def relu(self,r):self.imm(29,0);self.op('FMNMX',r,r,29,1)
class Builder:
 def __init__(self,name,x):
  self.out=OUT/name;self.out.mkdir(exist_ok=True);self.memory=bytearray(256);self.layers=[];self.expected={}
  self.x=torch.from_numpy(x.copy());self.channels=x.shape[1];self.h=x.shape[2];self.spatial=True
  self.src=self.alloc(self.channels*(self.h+2)**2*4);self.input=self.src
  self.input_array=F.pad(self.x,(1,1,1,1)).numpy().astype('<f4');self.parameters=0
 def alloc(self,n,data=None):
  pos=(len(self.memory)+255)//256*256;self.memory.extend(bytes(pos+n-len(self.memory)))
  if data is not None:self.memory[pos:pos+n]=data
  return pos
 def tensor(self,a):
  a=np.asarray(a,dtype='<f4');self.parameters+=a.size;return self.alloc(a.nbytes,a.tobytes())
 def install(self,name,a,gx,gy,dx,dy,dst,count,expected):
  a.op('EXIT');(self.out/(name+'.ntas')).write_bytes(struct.pack('<'+'Q'*len(a.code),*a.code))
  self.layers.append([name,gx,gy,dx,dy,dst,count]);self.expected[name]=np.asarray(expected,dtype='<f4').reshape(len(self.x),-1)
 def conv(self,name,weight,bias,relu=True,groups=1):
  w=self.tensor(weight);b=self.tensor(bias);co,ci,k,_=weight.shape;h=self.h;p=h+2
  assert k in (1,3) and (groups==1 or groups==self.channels==co)
  dst=self.alloc(co*p*p*4);a=Asm();a.sr(0,0);a.sr(1,8);a.sr(2,9)
  a.imm(6,self.src+(p+1)*4 if k==1 else self.src);a.addscale(6,1,p*4);a.addscale(6,0,4)
  if groups!=1:a.addscale(6,2,p*p*4)
  a.imm(7,w);a.addscale(7,2,ci*k*k*4);a.imm(8,b);a.addscale(8,2,4);a.ld(16,8)
  for c in range(ci):
   for y in range(k):
    for x in range(k):a.ld(10,6,(y*p+x)*4);a.ld(11,7,(c*k*k+y*k+x)*4);a.op('FFMA',16,10,11)
   if c+1<ci:a.imm(12,p*p*4);a.op('IADD',6,6,12)
  if relu:a.relu(16)
  a.imm(9,dst+(p+1)*4);a.addscale(9,2,p*p*4);a.addscale(9,1,p*4);a.addscale(9,0,4);a.st(16,9)
  self.x=F.conv2d(self.x,torch.from_numpy(weight),torch.from_numpy(bias),padding=k//2,groups=groups)
  if relu:self.x=F.relu(self.x)
  self.install(name,a,h,co,h,1,dst,co*p*p,F.pad(self.x,(1,1,1,1)).numpy());self.src=dst;self.channels=co
 def pool(self,name):
  h=self.h;p=h+2;q=h//2+2;dst=self.alloc(self.channels*q*q*4);a=Asm();a.sr(0,0);a.sr(1,8);a.sr(2,9)
  a.imm(6,self.src+(p+1)*4);a.addscale(6,0,8);a.addscale(6,1,p*8);a.addscale(6,2,p*p*4);a.ld(4,6)
  for off in (4,p*4,p*4+4):a.ld(5,6,off);a.op('FMNMX',4,4,5,1)
  a.imm(7,dst+(q+1)*4);a.addscale(7,0,4);a.addscale(7,1,q*4);a.addscale(7,2,q*q*4);a.st(4,7)
  self.x=F.max_pool2d(self.x,2);self.install(name,a,h//2,self.channels,h//2,1,dst,self.channels*q*q,F.pad(self.x,(1,1,1,1)).numpy());self.src=dst;self.h=h//2
 def gap(self,name):
  h=self.h;p=h+2;dst=self.alloc(self.channels*4);a=Asm();a.sr(0,0);a.imm(2,self.src+(p+1)*4);a.addscale(2,0,p*p*4);a.imm(4,0)
  for y in range(h):
   for x in range(h):a.ld(5,2,(y*p+x)*4);a.op('FADD',4,4,5)
  a.imm(5,struct.unpack('<I',struct.pack('<f',1/(h*h)))[0]);a.op('FMUL',4,4,5);a.imm(6,dst);a.addscale(6,0,4);a.st(4,6)
  self.x=self.x.mean((2,3));self.install(name,a,1,1,self.channels,1,dst,self.channels,self.x.numpy());self.src=dst;self.spatial=False
 def fc(self,name,weight,bias,relu):
  co,ci=weight.shape;wp=self.tensor(weight);bp=self.tensor(bias);dst=self.alloc(co*4);a=Asm();a.sr(0,0)
  a.imm(2,wp);a.addscale(2,0,ci*4);a.imm(3,bp);a.addscale(3,0,4);a.ld(4,3);a.imm(6,self.src)
  for j in range(ci):
   if self.spatial:
    h=self.h;p=h+2;c=j//(h*h);y=j//h%h;x=j%h;offset=(c*p*p+(y+1)*p+x+1)*4
   else:offset=j*4
   # MI signed offset范围不足时, 使用绝对地址载入。
   if offset<32768:a.ld(5,6,offset)
   else:a.imm(8,self.src+offset);a.ld(5,8)
   a.ld(7,2,j*4);a.op('FFMA',4,5,7)
  if relu:a.relu(4)
  a.imm(8,dst);a.addscale(8,0,4);a.st(4,8)
  self.x=F.linear(self.x.flatten(1),torch.from_numpy(weight),torch.from_numpy(bias))
  if relu:self.x=F.relu(self.x)
  self.install(name,a,1,1,co,1,dst,co,self.x.numpy());self.src=dst;self.spatial=False
 def residual(self,name,other,reference):
  h=self.h;p=h+2;dst=self.alloc(self.channels*p*p*4);a=Asm();a.sr(0,0);a.sr(1,8);a.sr(2,9)
  a.imm(3,(p+1)*4);a.addscale(3,0,4);a.addscale(3,1,p*4);a.addscale(3,2,p*p*4)
  a.imm(6,self.src);a.op('IADD',6,6,3);a.ld(4,6);a.imm(6,other);a.op('IADD',6,6,3);a.ld(5,6);a.op('FADD',4,4,5);a.relu(4)
  a.imm(6,dst);a.op('IADD',6,6,3);a.st(4,6);self.x=F.relu(self.x+reference)
  self.install(name,a,h,self.channels,h,1,dst,self.channels*p*p,F.pad(self.x,(1,1,1,1)).numpy());self.src=dst
 def finish(self,info):
  logits=self.src;classes=self.x.shape[1];pred=self.alloc(4);a=Asm();a.imm(0,logits);a.ld(1,0);a.imm(2,0)
  for j in range(1,classes):a.ld(3,0,j*4);a.op('SETP',0,3,1,0x24,3);a.op('MOV',1,3,pred=1);a.imm(4,j);a.op('MOV',2,4,pred=1)
  a.imm(5,pred);a.st(2,5);a.op('EXIT');name='argmax';(self.out/(name+'.ntas')).write_bytes(struct.pack('<'+'Q'*len(a.code),*a.code));self.layers.append([name,1,1,1,1,pred,1])
  np.save(self.out/'predictions.npy',self.x.argmax(1).numpy());np.savez(self.out/'expected.npz',**self.expected)
  (self.out/'memory.bin').write_bytes(self.memory);self.input_array.tofile(self.out/'input.bin')
  head=[len(self.memory),self.input,logits,pred,len(self.x),int(self.input_array[0].size),classes]
  (self.out/'program.txt').write_text(' '.join(map(str,head))+'\n'+'\n'.join(' '.join(map(str,l)) for l in self.layers)+'\n')
  info.update(parameters=self.parameters,reference='PyTorch CPU FP32; all layers checked',input_sha256=hashlib.sha256(self.input_array.tobytes()).hexdigest())
  (self.out/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print(self.out.name,info['parameters'],flush=True)

def main():
 rng=np.random.default_rng(20260917)
 for name,channels,h in [('residual_small',4,8),('depthwise_medium',16,14)]:
  b=Builder(name,rng.normal(size=(3,1,h,h)).astype('f4'))
  def conv(n,ci,co,k=3,groups=1):b.conv(n,rng.normal(0,.1,(co,ci//groups,k,k)).astype('f4'),rng.normal(0,.03,co).astype('f4'),True,groups)
  conv('00_conv',1,channels);skip,ref=b.src,b.x.clone()
  conv('01_conv',channels,channels,groups=channels if 'depthwise' in name else 1)
  if 'residual' in name:b.residual('02_add',skip,ref)
  else:conv('02_pointwise',channels,channels,1)
  b.pool('03_pool');b.gap('04_gap');b.fc('05_fc',rng.normal(0,.1,(10,channels)).astype('f4'),np.zeros(10,'f4'),False)
  b.finish(dict(model=name,synthetic_seed=20260917,trained=False))
if __name__=='__main__':main()
