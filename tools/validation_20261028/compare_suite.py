"""Full declared candidate search; calibration and validation are kept separate."""
import argparse, concurrent.futures, csv, json, statistics, sys, time
from pathlib import Path

def worker(item):
    source,row,version=item
    sys.path.insert(0,source)
    from superpod_compute import Hardware,GemmRequest,Policy,predict
    hardware=Hardware.load(Path(source)/'superpod_compute/data/bw1100.json')
    policy=Policy(initial_cache_state='warm-if-fit') if version=='v2' else Policy()
    r=predict(GemmRequest(*row['shape']),hardware,policy=policy)
    b=r['best']
    t=(row['graph_round1']['p50_us']+row['graph_round2']['p50_us'])/2
    return dict(id=row['id'],group=row['group'],role=row['role'],shape=row['shape'],measured_graph_us=t,
                measured_eager_us=row['eager_event']['p50_us'],model_raw_us=b['duration_us'],
                compute_lower_bound_us=b['useful_flops']/(hardware.core_peak_flops*hardware.cores)*1e6,
                mapping=b['mapping'],hbm_read_bytes=b['hbm_read_bytes'],l2_to_core_bytes=b['l2_to_core_bytes'],
                output_bytes=b['output_bytes'],wave_count=b['wave_count'],reuse_order=b.get('reuse_order'),
                warm_residency_admitted=b.get('warm_residency_admitted',False),
                searched=r['searched_candidates'],feasible=r['feasible_candidates'],
                graph_round_drift_percent=abs(row['graph_round1']['p50_us']-row['graph_round2']['p50_us'])/t*100,
                correct=row['correctness']['passed'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--version',choices=['v1','v2'],default='v2');ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--source',type=Path);ap.add_argument('--measurements',type=Path);ap.add_argument('--output-dir',type=Path)
    args=ap.parse_args();root=Path(__file__).parent
    project=root if (root/'superpod_compute').is_dir() else root.parents[1]
    if args.version=='v1' and args.source is None:ap.error('V1 requires --source pointing to an unchanged V1 checkout')
    source=str((args.source or project).resolve())
    default_results=root if project==root else project/'results/validation_20261028'
    data=json.loads((args.measurements or default_results/'final_measurements.json').read_text());assert data['completed']
    report=dict(version=args.version,source=source,cases=[],completed=False)
    output_dir=args.output_dir or default_results;output_dir.mkdir(parents=True,exist_ok=True)
    out=output_dir/f'comparison_{args.version}.json'
    begin=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,(source,row,args.version)) for row in data['cases']]
        for f in concurrent.futures.as_completed(futures):
            row=f.result();report['cases'].append(row)
            print(row['id'],round(row['model_raw_us'],3),round(row['measured_graph_us'],3),flush=True)
            out.write_text(json.dumps(report,indent=2)+'\n')
    calibration=[x for x in report['cases'] if x['role']=='calibration']
    service=max(0,statistics.median(x['measured_graph_us']-x['model_raw_us'] for x in calibration)) if args.version=='v2' else 0
    report['device_service_us']=service
    report['calibration_method']='One nonnegative median residual from 12 predeclared calibration shapes; zero validation-shape fitting'
    for row in report['cases']:
        row['model_calibrated_us']=row['model_raw_us']+service
        row['signed_error_percent']=100*(row['model_calibrated_us']/row['measured_graph_us']-1)
        row['absolute_error_percent']=abs(row['signed_error_percent'])
        row['passes_15_percent']=row['absolute_error_percent']<=15
        row['frozen_peak_precludes_15_percent']=row['compute_lower_bound_us']>row['measured_graph_us']*1.15
    held=[x for x in report['cases'] if x['role']=='validation']
    report['validation_summary']=dict(count=len(held),passes=sum(x['passes_15_percent'] for x in held),
        mean_absolute_percent=statistics.mean(x['absolute_error_percent'] for x in held),
        median_absolute_percent=statistics.median(x['absolute_error_percent'] for x in held),
        max_absolute_percent=max(x['absolute_error_percent'] for x in held),
        frozen_peak_precludes_count=sum(x['frozen_peak_precludes_15_percent'] for x in held))
    report['cases'].sort(key=lambda x:x['id']);report['completed']=True;report['wall_seconds']=time.monotonic()-begin
    out.write_text(json.dumps(report,indent=2)+'\n')
    fields=['id','group','role','shape','measured_graph_us','measured_eager_us','model_raw_us','model_calibrated_us','signed_error_percent','passes_15_percent','compute_lower_bound_us','frozen_peak_precludes_15_percent','graph_round_drift_percent']
    with out.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(report['cases'])
    print(json.dumps(report['validation_summary']),flush=True)

if __name__=='__main__':main()
