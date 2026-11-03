"""Measure frozen validation, then separately inspect dispatch. No model imported."""
import json,pathlib,subprocess,threading,time
r=pathlib.Path(__file__).resolve().parent
stop=threading.Event();samples=[]
def monitor():
 while not stop.is_set():
    d={'unix':time.time()}
    for path in pathlib.Path('/sys/class/drm').glob('card*/device/*'):
        if path.name not in ('pp_dpm_sclk','pp_dpm_mclk','gpu_busy_percent','mem_busy_percent'):continue
        try:d[str(path)]=path.read_text().strip()
        except OSError:pass
    samples.append(d);stop.wait(.5)
t=threading.Thread(target=monitor,daemon=True);t.start()
try:
 subprocess.run(['python3','-B','measure_graph.py','--plan','frozen_plan.json','--output',str(r/'validation_measure.json')],check=True,cwd=r)
 subprocess.run(['python3','-B','validation_metadata.py'],check=True,cwd=r)
finally:
 stop.set();t.join();(r/'telemetry_validation.json').write_text(json.dumps(samples,indent=2)+'\n')
