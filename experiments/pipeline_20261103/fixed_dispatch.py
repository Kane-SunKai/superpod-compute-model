"""Diagnostic: observed MT/K-step -> existing analytical model. No time fitting.
Tensile output ij is transpose of row-major C; MT first axis maps to N.
This is an optional-execution-plan experiment, NOT dimension-only prediction.
"""
import dataclasses,json,pathlib,re,sys,time
r=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(r.parents[1]))
from superpod_compute.contracts import Hardware,Policy,GemmRequest
from superpod_compute.operators import GemmModel
from superpod_compute.backends import make_backend
plan=json.loads((r/'calibration_plan.json').read_text());h=Hardware(**plan['hardware']);p=Policy(**plan['policy']);op=GemmModel();backend=make_backend('auto',h)
meta=json.loads((r/('metadata_all.json' if (r/'metadata_all.json').exists() else 'execution_metadata.json')).read_text())
results=[]
for case in meta:
 names=[x['name'] for x in case['kernels']]
 if len(names)!=1 or not re.search(r'_MT(\d+)x(\d+)x(\d+)_',names[0]):
  results.append({**case,'supported':False});continue
 name=names[0];bn,bm,bk=map(int,re.search(r'_MT(\d+)x(\d+)x(\d+)_',name).groups())
 gsu=int(re.search(r'_GSU(\d+)_',name).group(1))
 if gsu!=1:
  results.append({**case,'supported':False,'reason':'split-K not represented'});continue
 req=GemmRequest(*case['shape'])
 fixed=dataclasses.replace(p,bm_values=(bm,),bn_values=(bn,),bk_values=(bk,),add_shape_candidate=False)
 best=None;count=0;start=time.perf_counter()
 for mapping in op.candidates(req,h,fixed):
  value=op.evaluate(req,h,p,backend,mapping)
  if value is not None:
   count+=1
   if best is None or op.rank_key(value)<op.rank_key(best):best=value
 results.append({**case,'supported':best is not None,'bm':bm,'bn':bn,'bk':bk,'best':best,'feasible':count,'search_s':time.perf_counter()-start})
 print(case['shape'],bm,bn,bk,round(best['duration_us'],3) if best else 'infeasible',flush=True)
(r/('fixed_dispatch_all.json' if len(meta)>8 else 'fixed_dispatch_initial.json')).write_text(json.dumps(results,indent=2)+'\n')
