"""Calibrate on independent data, freeze parameters, then evaluate acceptance.
Explicit MT/K-step is additional input for D/V2/V3, never a dimension-only claim.
"""
import collections,dataclasses,hashlib,json,pathlib,statistics,sys,time
r=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(r.parents[1]))
from superpod_compute.contracts import Hardware,Policy,GemmRequest
from superpod_compute.backends import make_backend
from superpod_compute.operators import GemmModel
from experimental_model import ReadServiceModel
plan=json.loads((r/'calibration_plan.json').read_text());h0=Hardware(**plan['hardware']);p0=Policy(**plan['policy']);op=GemmModel()
def loadlog(name):return [json.loads(x) for x in (r/name).read_text().splitlines() if x.startswith('{')]
mem=loadlog('memory_saturation.log');assert len(mem)==24 and all(x['correct'] for x in mem)
fits=[]
for rd in (1,2):
 for blocks in (576,1152,2304):
  rs=sorted([x for x in mem if x['round']==rd and x['blocks']==blocks],key=lambda x:x['read_bytes'])
  xs=[x['read_bytes'] for x in rs];ys=[x['median_us']*1e-6 for x in rs];xm=statistics.mean(xs);ym=statistics.mean(ys)
  slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
  fits.append(dict(round=rd,blocks=blocks,bandwidth_Bps=1/slope,intercept_us=(ym-slope*xm)*1e6))
bw=statistics.median(x['bandwidth_Bps'] for x in fits)
# Read-only microbenchmark calibrates input reads only. Output HBM service stays
# at the original 1.72 TB/s; do not infer a write rate from read measurements.
h2=h0
meta=json.loads((r/'fixed_dispatch_all.json').read_text());cal=[x for x in meta if x.get('role')=='calibration'];acc=[x for x in meta if x.get('role')=='acceptance']
cm={x['id']:x['measured_graph_us'] for x in json.loads((r/'calibration_measure.json').read_text())['cases']}
work=r.parents[2]
am={x['id']:x['measured_graph_us'] for x in json.loads((work/'graph50-recheck-20261103/measurements.json').read_text())['cases']}
base={x['id']:x['best'] for x in json.loads((work/'llmcompass-method-validation-20261103/predictions.json').read_text())['cases']}
base.update({x['id']:x['best'] for x in json.loads((r/'calibration_predictions.json').read_text())['cases']})
def evaluate(case,h,eta):
 if not case['supported']:return None
 req=GemmRequest(*case['shape']);p=dataclasses.replace(p0,matrix_efficiency=eta,bm_values=(case['bm'],),bn_values=(case['bn'],),bk_values=(case['bk'],),add_shape_candidate=False,calibration_hardware_sha256='')
 be=make_backend('auto',h,eta);best=None;read_op=ReadServiceModel(bw)
 for mapping in read_op.candidates(req,h,p):
  v=read_op.evaluate(req,h,p,be,mapping)
  if v is not None and (best is None or op.rank_key(v)<op.rank_key(best)):best=v
 return best
training=[x for x in cal if x['group']=='prefill' and x['supported']]
grid=[];t=time.perf_counter()
for i in range(60,101):
 eta=i/100;errors={}
 for x in training:
  val=evaluate(x,h2,eta);errors[x['id']]=abs(val['duration_us']/cm[x['id']]-1)
 grid.append(dict(eta=eta,mape=statistics.mean(errors.values()),per_case_absolute_relative_error=errors))
 if i%10==0:print('calibration efficiency',eta,'MAPE',grid[-1]['mape'],flush=True)
selected=min(grid,key=lambda x:(x['mape'],-x['eta']))
loo={x['id']:min(grid,key=lambda g:(statistics.mean(e for key,e in g['per_case_absolute_relative_error'].items() if key!=x['id']),-g['eta']))['eta'] for x in training}
frozen=dict(hbm_bandwidth_Bps=bw,bandwidth_fits=fits,matrix_efficiency=selected['eta'],calibration_grid=grid,leave_one_out_efficiency=loo,
  calibration_ids=[x['id'] for x in training],
  caveats=['input-read bandwidth from independent read microbenchmark; output-write rate unchanged',
           'efficiency is effective GEMM matrix service calibrated on 7 unseen shapes, not a replacement of physical MMAC 8 cycles',
           'explicit MT and K-step are additional observed inputs; GSU != 1 unsupported',
           'no acceptance latency used in calibration objective'])
(r/'frozen_calibration.json').write_text(json.dumps(frozen,indent=2)+'\n')
print('FROZEN bandwidth',bw,'eta',selected['eta'],'LOO',loo,flush=True)
rows=[]
for x in cal+acc:
 measured=(cm if x['role']=='calibration' else am)[x['id']]
 versions={'V0':base[x['id']],'V1':base[x['id']],'D_observed_MT':x.get('best'),'V2_read_bw':evaluate(x,h2,1.),'V3_effective_compute':evaluate(x,h2,selected['eta'])}
 rec={key:x[key] for key in ('id','group','shape','role')};rec.update(measured_us=measured,supported_execution_plan=x['supported'],versions={})
 for name,v in versions.items():
  if v is None:rec['versions'][name]=None;continue
  err=v['duration_us']/measured-1
  rec['versions'][name]=dict(predicted_us=v['duration_us'],signed_error_pct=err*100,absolute_error_pct=abs(err)*100,pass10=abs(err)<=.1,best=v)
 rows.append(rec)
 print(x['id'],{k:round(v['absolute_error_pct'],2) if v else None for k,v in rec['versions'].items()},flush=True)
 (r/'variant_results.json').write_text(json.dumps(dict(completed=False,parameters_sha256=hashlib.sha256((r/'frozen_calibration.json').read_bytes()).hexdigest(),cases=rows),indent=2)+'\n')
summary={}
for role in ('calibration','acceptance'):
 summary[role]={}
 for version in rows[0]['versions']:
  selectedrows=[x for x in rows if x['role']==role];valid=[x for x in selectedrows if x['versions'][version] is not None]
  summary[role][version]=dict(total=len(selectedrows),supported=len(valid),pass10=sum(x['versions'][version]['pass10'] for x in valid),mape=statistics.mean(x['versions'][version]['absolute_error_pct'] for x in valid))
result=dict(completed=True,parameters_sha256=hashlib.sha256((r/'frozen_calibration.json').read_bytes()).hexdigest(),wall_s=time.perf_counter()-t,summary=summary,cases=rows)
(r/'variant_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
