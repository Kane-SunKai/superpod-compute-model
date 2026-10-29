"""Controlled V3.2 comparison: identical inputs, search space and fixed device service.

No fitting of hardware throughput, validation GEMMs or per-backend launch cost.
12 historical calibration shapes are reported separately from 60 validation shapes.
"""
import argparse,concurrent.futures,csv,json,statistics,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from superpod_compute import Hardware,Policy,GemmRequest,predict
def experiment_identity(measurements):
    paths=[ROOT/'superpod_compute'/name for name in ('contracts.py','api.py','operators.py','scheduler.py','backends.py')]
    paths += [ROOT/'superpod_compute/data'/name for name in ('bw1100.json','bw1100_mac256.json')]
    paths += [ROOT/'examples'/name for name in ('bw1100_graph_calibrated.json','bw1100_mac256_comparison.json')]
    values={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    values['measurements_sha256']=hashlib.sha256(measurements.read_bytes()).hexdigest()
    return values
def work(mode,row):
    profile='bw1100.json' if mode=='mmac' else 'bw1100_mac256.json'
    policy='bw1100_graph_calibrated.json' if mode=='mmac' else 'bw1100_mac256_comparison.json'
    h=Hardware.load(ROOT/'superpod_compute/data'/profile)
    p=Policy(**json.loads((ROOT/'examples'/policy).read_text()))
    r=predict(GemmRequest(*row['shape']),h,policy=p)
    b=r['best'];t=statistics.mean([row['graph_round1']['p50_us'],row['graph_round2']['p50_us']])
    error=100*(b['duration_us']/t-1)
    return {'id':row['id'],'group':row['group'],'role':row['role'],'shape':row['shape'],'mode':mode,
            'measured_graph_us':t,'model_us':b['duration_us'],'signed_error_percent':error,'absolute_error_percent':abs(error),
            'passes_15_percent':abs(error)<=15,'mapping':b['mapping'],'hbm_read_bytes':b['hbm_read_bytes'],
            'l2_to_core_bytes':b['l2_to_core_bytes'],'searched':r['searched_candidates'],'feasible':r['feasible_candidates'],
            'compute_lower_bound_us':b['useful_flops']/(h.core_peak_flops*h.cores)*1e6,
            'wave_count':b['wave_count'],'device_service_us':p.device_service_s*1e6}
def summary(rows):
    if not rows:return {'count':0}
    return {'count':len(rows),'passes':sum(r['passes_15_percent'] for r in rows),
            'pass_rate_percent':100*sum(r['passes_15_percent'] for r in rows)/len(rows),
            'median_absolute_percent':statistics.median(r['absolute_error_percent'] for r in rows),
            'mean_absolute_percent':statistics.mean(r['absolute_error_percent'] for r in rows)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--measurements',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();data=json.loads(args.measurements.read_text());assert data['completed']
    args.output_dir.mkdir(parents=True,exist_ok=True);target=args.output_dir/'comparison_modes.json'
    report={'completed':False,'cases':[],'fixed_device_service_us':3.121244690369221,'identity':experiment_identity(args.measurements),
            'method':'Same 72 historical Graph measurements, same exhaustive candidate set, memory services and frozen device cost. Independently optimize each backend. 12 calibration cases excluded from validation pass rate.'}
    if target.exists():
        prior=json.loads(target.read_text())
        if prior.get('method')!=report['method'] or prior.get('identity')!=report['identity']:
            ap.error('Existing report does not match source/hardware/policy/measurement fingerprints; use a new output directory')
        report['cases']=prior['cases']
        report['previous_wall_seconds']=prior.get('wall_seconds_this_run',0)
    done={(r['mode'],r['id']) for r in report['cases']};start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(work,mode,row) for row in data['cases'] for mode in ('mmac','mac256') if (mode,row['id']) not in done]
        for future in concurrent.futures.as_completed(jobs):
            r=future.result();report['cases'].append(r)
            target.write_text(json.dumps(report,indent=2)+'\n')
            print(f"{len(report['cases'])}/144 {r['mode']} {r['id']} {r['model_us']:.3f} us error={r['signed_error_percent']:.2f}%",flush=True)
    report['summaries']={}
    for mode in ('mmac','mac256'):
        rows=[r for r in report['cases'] if r['mode']==mode]
        held=[r for r in rows if r['role']=='validation']
        # Preserve the predeclared groups: decode M=1 is a regular LLM case.
        report['summaries'][mode]={'all_validation':summary(held),
             'regular_validation':summary([r for r in held if r['group']!='irregular']),
             'irregular_validation':summary([r for r in held if r['group']=='irregular']),
             'calibration_only':summary([r for r in rows if r['role']=='calibration'])}
        report['summaries'][mode]['peak_already_exceeds_measured_by_15_percent']=sum(r['compute_lower_bound_us']>1.15*r['measured_graph_us'] for r in held)
    report['group_definition']='Keep original groups: irregular is the predeclared 10-case irregular group; regular includes all 10 decode cases, even M=1.'
    report['calibration_transfer']='Existing 3.121244690369221 us device service frozen for both backends; policy fingerprints migrated explicitly. No new per-backend calibration or GEMM-fit factors.'
    report['completed']=True;report['wall_seconds_this_run']=time.monotonic()-start
    report['cases'].sort(key=lambda r:(r['id'],r['mode']))
    target.write_text(json.dumps(report,indent=2)+'\n')
    fields=['mode','id','group','role','shape','measured_graph_us','model_us','signed_error_percent','passes_15_percent','compute_lower_bound_us']
    with target.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(report['cases'])
    print(json.dumps(report['summaries'],indent=2),flush=True)
if __name__=='__main__':main()
