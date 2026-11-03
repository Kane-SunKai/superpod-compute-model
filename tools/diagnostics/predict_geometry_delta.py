"""Exact new optimum by evaluating only the newly admitted CU geometries.

The baseline contains the old exact optimum. All old mappings retain the same
cost, so min(old optimum, all added mappings) is the exact expanded optimum.
"""
import argparse,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from math import isqrt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from superpod_compute import GemmRequest,Hardware,Policy,Mapping,evaluate_mapping
from superpod_compute.backends import make_backend
from superpod_compute.operators import GemmModel,ceildiv

def legacy(tm,tn,cores):
    f=[]
    for d in range(1,isqrt(cores)+1):
        if cores%d==0:f.extend([(d,cores//d),(cores//d,d)])
    f.extend((g,g) for g in (1,2,4,8));r=set()
    for m,n in f:
        m=max(1,min(m,tm));n=max(1,min(n,tn,cores//m))
        if m*n<=cores:r.add((m,n))
    return r

def work(row,base):
    h=Hardware(**base['hardware']);p=Policy(**base['policy']);model=GemmModel()
    req=GemmRequest(*row['shape'],rhs_storage='nk-contiguous-transposed')
    backend=make_backend('auto',h,p.matrix_efficiency)
    prior=Mapping(**row['best']['mapping'])
    best=model.evaluate(req,h,p,backend,prior)
    if abs(best['duration_us']-row['model_us'])>1e-6:raise ValueError('Old mapping cost changed')
    old={};added=valid=0;start=time.monotonic()
    for m in model.candidates(req,h,p):
        key=(m.bm,m.bn)
        if key not in old:old[key]=legacy(ceildiv(req.M,m.bm),ceildiv(req.N,m.bn),h.cores)
        if (m.gm,m.gn) in old[key]:continue
        added+=1
        if not model.capacity(m,h)['valid']:continue
        actual=model.evaluate(req,h,p,backend,m);valid+=1
        if actual and model.rank_key(actual)<model.rank_key(best):best=actual
    return dict(id=row['id'],group=row['group'],shape=row['shape'],model_us=best['duration_us'],
                old_model_us=row['model_us'],best=best,added_candidates=added,added_feasible=valid,
                wall_seconds=time.monotonic()-start)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();old=json.loads(args.baseline.read_text())
    frozen=json.loads((ROOT/'results/regular50_20261103/frozen_plan.json').read_text())['baseline']
    data=dict(completed=False,cases=[],scope='Exact expanded candidate optimum; original costs unchanged')
    if args.output.exists():data=json.loads(args.output.read_text())
    done={r['id'] for r in data['cases']}
    def save():
        temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(args.output)
    save()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(work,row,frozen) for row in old['cases'] if row['id'] not in done]
        for f in as_completed(fs):
            r=f.result();data['cases'].append(r);save()
            print(len(data['cases']),r['id'],round(r['old_model_us'],3),'->',round(r['model_us'],3),r['added_candidates'],flush=True)
    data['completed']=True;save()

if __name__=='__main__':main()
