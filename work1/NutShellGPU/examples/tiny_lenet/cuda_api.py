# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.10
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================

"""CUDA Driver/NVRTC 接口。NVRTC_LIBRARY 可指定编译器动态库的位置。"""
import ctypes as C
import ctypes.util
import os
from pathlib import Path

P=C.c_void_p; I=C.c_int; U=C.c_uint64; S=C.c_size_t

class CUDA:
    def __init__(self):
        windows=os.name=='nt'
        lib=os.environ.get('NVRTC_LIBRARY')
        if not lib and windows and os.environ.get('CUDA_PATH'):
            candidates=list((Path(os.environ['CUDA_PATH'])/'bin').glob('nvrtc64_*.dll'))
            if candidates:lib=str(sorted(candidates)[-1])
        if not lib:lib=ctypes.util.find_library('nvrtc')
        if not lib:raise RuntimeError('Set NVRTC_LIBRARY to the CUDA Toolkit NVRTC library')
        self.search_path=None
        if windows and Path(lib).is_file():
            directory=Path(lib).resolve().parent
            self.search_path=os.add_dll_directory(str(directory))
            builtins=sorted(directory.glob('nvrtc-builtins64_*.dll'))
            if builtins:self.builtins=C.CDLL(str(builtins[-1]))
        self.nv=C.CDLL(lib)
        self.d=C.WinDLL('nvcuda.dll') if windows else C.CDLL('libcuda.so.1')
        self.call('cuInit',I(0));device=I();self.call('cuDeviceGet',C.byref(device),I(0))
        name=C.create_string_buffer(256);self.call('cuDeviceGetName',name,I(256),device);self.name=name.value.decode()
        major,minor=I(),I();self.call('cuDeviceGetAttribute',C.byref(major),I(75),device);self.call('cuDeviceGetAttribute',C.byref(minor),I(76),device)
        self.arch=f'compute_{major.value}{minor.value}'
        self.context=P();self.call('cuCtxCreate_v2',C.byref(self.context),I(0),device)
        self.stream=P();self.call('cuStreamCreate',C.byref(self.stream),I(0))

    def call(self,name,*args):
        code=getattr(self.d,name)(*args)
        if code:raise RuntimeError(f'{name}: CUDA error {code}')

    def compile(self,source):
        prog=P();code=self.nv.nvrtcCreateProgram(C.byref(prog),source.encode(),b'kernels.cu',I(0),None,None)
        if code:raise RuntimeError(f'nvrtcCreateProgram: {code}')
        options=[f'--gpu-architecture={self.arch}'.encode(),b'--std=c++14',b'--fmad=false']
        args=(C.c_char_p*len(options))(*options)
        code=self.nv.nvrtcCompileProgram(prog,I(len(options)),args)
        size=S();self.nv.nvrtcGetProgramLogSize(prog,C.byref(size));log=C.create_string_buffer(size.value);self.nv.nvrtcGetProgramLog(prog,log)
        if code:raise RuntimeError(log.value.decode())
        self.nv.nvrtcGetPTXSize(prog,C.byref(size));ptx=C.create_string_buffer(size.value)
        code=self.nv.nvrtcGetPTX(prog,ptx)
        self.nv.nvrtcDestroyProgram(C.byref(prog))
        if code:raise RuntimeError(f'nvrtcGetPTX: {code}')
        self.module=P();self.call('cuModuleLoadData',C.byref(self.module),ptx)

    def malloc(self,n):
        p=U();self.call('cuMemAlloc_v2',C.byref(p),S(n));return p

    def upload(self,a):
        p=self.malloc(a.nbytes);self.call('cuMemcpyHtoD_v2',p,P(a.ctypes.data),S(a.nbytes));return p

    def download(self,p,a):
        self.call('cuCtxSynchronize');self.call('cuMemcpyDtoH_v2',P(a.ctypes.data),p,S(a.nbytes));return a

    def launch(self,name,count,args):
        f=P();self.call('cuModuleGetFunction',C.byref(f),self.module,name.encode())
        values=[v if isinstance(v,U) else I(v) for v in args]
        params=(P*len(values))(*(C.cast(C.byref(v),P) for v in values))
        self.call('cuLaunchKernel',f,I((count+127)//128),I(1),I(1),I(128),I(1),I(1),I(0),self.stream,params,P())

    def close(self):
        self.call('cuCtxDestroy_v2',self.context)
