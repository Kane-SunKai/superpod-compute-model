"""Evaluate frozen shape-only baseline; no reading GPU measurements or fitting."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from superpod_compute import GemmRequest,Hardware,Policy,predict

def work(row,baseline):
    begin=time.monotonic()
    h=Hardware(**baseline['hardware'])
    p=Policy(**baseline['policy'])
    result=predict(GemmRequest(*row['shape'],rhs_storage=baseline.get('rhs_storage','kn-contiguous')),h,policy=p)
    return dict(id=row['id'],group=row['group'],shape=row['shape'],
                model_us=result['best']['duration_us'],best=result['best'],
                searched_candidates=result['searched_candidates'],
                feasible_candidates=result['feasible_candidates'],wall_seconds=time.monotonic()-begin)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args()
    plan=json.loads(args.plan.read_text())
    for name,value in plan['baseline']['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value:
            raise RuntimeError('Frozen model source changed: '+name)
    identity=hashlib.sha256(args.plan.read_bytes()).hexdigest()
    data=dict(completed=False,plan_sha256=identity,cases=[],scope=plan['baseline']['layout_limitation'])
    if args.output.exists():
        data=json.loads(args.output.read_text())
        if data['plan_sha256']!=identity:raise ValueError('Resume identity mismatch')
    done={r['id'] for r in data['cases']}
    def save():
        tmp=args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(args.output)
    begin=time.monotonic();save()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(work,row,plan['baseline']) for row in plan['cases'] if row['id'] not in done]
        for future in as_completed(futures):
            row=future.result();data['cases'].append(row)
            data['wall_seconds']=time.monotonic()-begin;save()
            print(len(data['cases']),row['id'],round(row['model_us'],4),'us',round(row['wall_seconds'],2),'s',flush=True)
    data['completed']=True;save()

if __name__=='__main__':main()
