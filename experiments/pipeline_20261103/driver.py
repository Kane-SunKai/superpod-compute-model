"""Sequential GPU jobs; monitor only, never change clock or power settings."""
import collections,json,pathlib,subprocess,threading,time
root=pathlib.Path('/workspace/pipeline_20261103')
stop=threading.Event();samples=[]
def monitor():
    while not stop.is_set():
        item={'time':time.time()}
        for p in pathlib.Path('/sys/class/drm').glob('card*/device/*'):
            if p.name not in ('pp_dpm_sclk','pp_dpm_mclk','gpu_busy_percent','mem_busy_percent'):continue
            try:item[str(p)]=p.read_text().strip()
            except OSError:pass
        samples.append(item);stop.wait(.2)
t=threading.Thread(target=monitor,daemon=True);t.start()
def run(args,name):
    print('START',name,flush=True)
    with (root/(name+'.log')).open('w') as f:
        subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=root)
    print('DONE',name,flush=True)
try:
    run(['python3','-B','measure_graph.py','--plan','calibration_plan.json','--output',str(root/'calibration_measure.json')],'calibration')
    cc=['/opt/dtk/bin/hipcc','-O3','--offload-arch=gfx938','-std=c++17']
    for name in ['mmac_probe','lds_probe']:
        run(cc+[name+'.cpp','-o',name],name+'_compile')
        for i in (1,2):run([str(root/name)],name+'_round'+str(i))
finally:
    stop.set();t.join();(root/'telemetry.json').write_text(json.dumps(samples,indent=2)+'\n')
