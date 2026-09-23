// =============================================================================
// NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
// Version 1.10
// Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
// Improved by: Tonghui Ming
// References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
// September 2026
// =============================================================================

// =============================================================================
// TinyLeNet: NCHW FP32 前向计算
// -----------------------------------------------------------------------------
// 卷积权重为 OIHW, 卷积步幅 1/补零 1; 池化窗口 2x2/步幅 2。
// 每个线程计算一个输出元素, 分类在十个 logits 中选择首个最大值。
// =============================================================================
extern "C" __global__ void conv(const float* x,const float* w,const float* b,
 float* y,int N,int CI,int CO,int H){
 int t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t>=N*CO*H*H)return;
 int ox=t%H,oy=t/H%H,co=t/(H*H)%CO,n=t/(CO*H*H);
 float sum=b[co];
 for(int ci=0;ci<CI;++ci)
  for(int ky=0;ky<3;++ky)
   for(int kx=0;kx<3;++kx){
    int iy=oy+ky-1,ix=ox+kx-1;
    if(iy>=0&&iy<H&&ix>=0&&ix<H)
     sum+=x[((n*CI+ci)*H+iy)*H+ix]*w[((co*CI+ci)*3+ky)*3+kx];
   }
 y[t]=sum;
}
extern "C" __global__ void relu(const float* x,float* y,int count){
 int t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t<count)y[t]=x[t]>0?x[t]:0;
}
extern "C" __global__ void pool(const float* x,float* y,int N,int C,int H){
 int t=blockIdx.x*blockDim.x+threadIdx.x,Q=H/2;
 if(t>=N*C*Q*Q)return;
 int ox=t%Q,oy=t/Q%Q,c=t/(Q*Q)%C,n=t/(C*Q*Q);
 int base=((n*C+c)*H+oy*2)*H+ox*2;
 float best=x[base];
 for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
  if(x[base+dy*H+dx]>best)best=x[base+dy*H+dx];
 y[t]=best;
}
extern "C" __global__ void dense(const float* x,const float* w,const float* b,
 float* y,int N,int CI,int CO){
 int t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t>=N*CO)return;
 int n=t/CO,co=t%CO;float sum=b[co];
 for(int ci=0;ci<CI;++ci)sum+=x[n*CI+ci]*w[co*CI+ci];
 y[t]=sum;
}
extern "C" __global__ void classify(const float* x,int* y,int N){
 int n=blockIdx.x*blockDim.x+threadIdx.x;
 if(n>=N)return;
 int best=0;for(int c=1;c<10;++c)if(x[n*10+c]>x[n*10+best])best=c;
 y[n]=best;
}
