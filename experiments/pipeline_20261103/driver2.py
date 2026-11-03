import json,pathlib,subprocess,threading,time
root=pathlib.Path('/workspace/pipeline_20261103');samples=[];stop=threading.Event()
def monitor():
 while not stop.is_set():
  item={'time':time.time()}
  for p in pathlib.Path('/sys/class/drm').glob('card*/device/*'):
   if p.name not in ('pp_dpm_sclk','pp_dpm_mclk','gpu_busy_percent','mem_busy_percent'):continue
   try:item[str(p)]=p.read_text().strip()
   except OSError:pass
  samples.append(item);stop.wait(.2)
def run(args,name):
 print('START',name,flush=True)
 with (root/(name+'.log')).open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=root)
 print('DONE',name,flush=True)
t=threading.Thread(target=monitor,daemon=True);t.start()
try:
 for name in ['memory_probe','mixed_probe']:
  run(['/opt/dtk/bin/hipcc','-O3','--offload-arch=gfx938','-std=c++17',name+'.cpp','-o',name],name+'_compile')
  run([str(root/name)],name)
 run(['python3','-B','inspect_execution.py'],'inspect_execution')
finally:
 stop.set();t.join();(root/'telemetry2.json').write_text(json.dumps(samples,indent=2)+'\n')
