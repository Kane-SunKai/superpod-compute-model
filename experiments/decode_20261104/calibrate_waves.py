"""Development-only calibration after fixed-implementation tail-wave probes."""
import dataclasses,json,pathlib,statistics,numpy as np
from resident_decode import evaluate
from superpod_compute.contracts import Hardware,Policy
r=pathlib.Path(__file__).resolve().parent
base=json.loads((r/'extended_profile.json').read_text());h=Hardware(**base['hardware']);p=Policy(**base['policy'])
meta=[];measured={}
for prefix in ('development','validation','extension','acceptance'):
 meta.extend(json.loads((r/(prefix+'_metadata.json')).read_text()))
 measured.update({x['id']:x for x in json.loads((r/(prefix+'_measure.json')).read_text())['cases']})
best=None
for launch_us in [3.121244690369221,5.0]:
 pp=dataclasses.replace(p,device_service_s=launch_us*1e-6)
 for bw in np.arange(1.72,1.941,.01):
  for lds in np.arange(56.,64.01,.5):
   predictions=[]
   for x in meta:
    schedule={**base['schedules'].get('x'.join(map(str,x['tile'])),{}),'task_waves':x.get('prefetch_global',1)==1}
    predictions.append(evaluate(x['shape'],x['tile'],h,pp,float(bw*1e12),float(lds),schedule))
   errors=[abs(a['duration_us']/measured[x['id']]['measured_graph_us']-1) for a,x in zip(predictions,meta)]
   score=(sum(e>.1 for e in errors),statistics.mean(e*e for e in errors),max(errors))
   if best is None or score<best[0]:best=(score,float(bw*1e12),float(lds),pp,predictions)
score,bw,lds,p,predictions=best
profile={**base,'version':'decode-waves-final-candidate','hardware':h.to_dict(),'policy':p.to_dict(),'input_read_Bps':bw,'shared_read_Bpc':lds,'library':meta,'respect_prefetch_waves':True,
 'calibration':dict(n=len(meta),scope='Development only; local dispatch metadata contains no timings; fixed-index task sweep identifies tail-wave scheduling',data=['development','validation','extension','acceptance'])}
(r/'wave_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
rows=[]
for a,x in zip(predictions,meta):
 t=measured[x['id']]['measured_graph_us'];err=100*(a['duration_us']/t-1)
 rows.append(dict(id=x['id'],shape=x['shape'],measured_us=t,predicted=a,error_pct=err))
 if abs(err)>10:print('FAIL',x['shape'],x['tile'],round(t,2),round(a['duration_us'],2),round(err,2))
(r/'wave_development_results.json').write_text(json.dumps(rows,indent=2)+'\n')
print('BEST',score,'HBM',bw,'LDS',lds,'launch',p.device_service_s,'MAPE',statistics.mean(abs(x['error_pct']) for x in rows))
