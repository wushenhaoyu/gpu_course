# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.10
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================

"""Measure a model_runner process from start to exit, using only Python stdlib."""
from pathlib import Path
import argparse,hashlib,json,platform,statistics,subprocess,time
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--runner',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--repeats',type=int,default=3)
 a=p.parse_args()
 if a.repeats<1:p.error('repeats must be positive')
 runner=a.runner.resolve();fixture=a.fixture.resolve();a.out.mkdir(parents=True,exist_ok=True)
 header=(fixture/'program.txt').read_text().splitlines()[0].split()
 result=dict(metric='external process start-to-exit wall seconds',samples=int(header[4]),
  includes='startup, input loading, simulator initialization, computation, layer outputs, statistics, exit',
  cache_state='fresh process; OS cache not flushed',platform=platform.platform(),
  executable_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
  fixture_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in fixture.iterdir() if f.is_file()},
  measurements=[])
 for i in range(a.repeats):
  out=(a.out/('run'+str(i))).resolve();out.mkdir(exist_ok=True)
  with (out/'process.log').open('w') as log:
   begin=time.perf_counter();r=subprocess.run([str(runner),str(fixture),str(out)],stdout=log,stderr=subprocess.STDOUT)
   seconds=time.perf_counter()-begin
  result['measurements'].append(dict(seconds=seconds,exit_code=r.returncode))
  if r.returncode:
   (a.out/'runtime.json').write_text(json.dumps(result,indent=2));r.check_returncode()
  stats=json.loads((out/'stats.json').read_text())
  # 程序墙钟时间由父进程测量; 模拟秒数来自虚拟 GPU 的周期/频率。
  result['measurements'][-1].update(cycles=stats['cycles'],core_clock_ghz=stats['core_clock_ghz'],
   simulated_seconds=stats['simulated_seconds'])
  result['median_seconds']=statistics.median(v['seconds'] for v in result['measurements'])
  (a.out/'runtime.json').write_text(json.dumps(result,indent=2))
  print(i,'program_seconds=',seconds,'simulated_seconds=',stats['simulated_seconds'],'cycles=',stats['cycles'],flush=True)
if __name__=='__main__':main()
