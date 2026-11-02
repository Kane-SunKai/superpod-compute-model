"""Test response-readiness scheduling against frozen measured dependency service.

No parameter tuning. Report both the autonomous mapper and observed-macrotiling
diagnostic mode; the latter is not a shape-only deployment prediction.
"""
import argparse,concurrent.futures,hashlib,json,statistics,sys,time
from dataclasses import replace
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from superpod_compute import GemmRequest,Hardware,Policy,predict
from superpod_compute.kernel_plan import parse_tensile_kernel,evaluate_macrotiling_ablation


def work(row,metadata,mode,service):
    h=Hardware.load(ROOT/'superpod_compute/data/bw1100.json')
    p=Policy(**json.loads((ROOT/'examples/bw1100_graph_calibrated.json').read_text()))
    if mode!='observed_macro_zero':p=replace(p,input_read_latency_s=service)
    req=GemmRequest(*row['shape'])
    entry={k:row[k] for k in ('id','group','shape')}
    entry.update(mode=mode,service_ns=p.input_read_latency_s*1e9)
    entry['measured_us']=statistics.mean(row['graph_round'+str(i)]['p50_us'] for i in (1,2))
    try:
        if mode=='free_mapper_readiness':result=predict(req,h,policy=p)
        else:
            names=[name for name in metadata['kernels'] if '_MT' in name]
            if len(names)!=1:raise ValueError('Expected one matrix kernel')
            plan=parse_tensile_kernel(names[0],orientation='row-major-transposed-blas')
            result=evaluate_macrotiling_ablation(req,h,p,plan)
            entry['kernel_plan']=plan.to_dict()
        best=result['best'];entry.update(status='predicted',model_us=best['duration_us'],mapping=best['mapping'],wave_count=best['wave_count'])
        entry['error_percent']=100*(entry['model_us']/entry['measured_us']-1)
    except (ValueError,NotImplementedError) as e:
        entry.update(status='unsupported',reason=str(e))
    return entry


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args()
    measurement=args.data_dir/'measurements.json';micro=args.data_dir/'dependent_memory_results.json';meta=args.data_dir/'kernels.json'
    rows=[r for r in json.loads(measurement.read_text())['cases'] if r['layout']=='contiguous']
    metadata={r['id']:r for r in json.loads(meta.read_text())['cases']}
    service=statistics.mean(s['event_slope_ns_per_load'] for r in json.loads(micro.read_text())['rounds'] for s in r['slopes'] if s['footprint_bytes']==262144 and s['blocks']==1)*1e-9
    sources=list((ROOT/'superpod_compute').glob('*.py'))+[measurement,micro,meta]
    identity={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report={'completed':False,'cases':[],'service_ns':service*1e9,'identity':identity,
            'scope':'Independent scalar dependent-load service as fixed sensitivity proxy. Payload bandwidth separated from response readiness. Not an asserted pure L2/bulk-copy latency; no fitting to GEMM timings. Observed-macro modes require measured kernel selection and remain diagnostic.'}
    if args.output.exists():
        prior=json.loads(args.output.read_text())
        if prior['identity']!=identity:raise ValueError('Resume source/data mismatch')
        report['cases']=prior['cases']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temp=args.output.with_suffix('.json.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(args.output)
    done={(r['id'],r['mode']) for r in report['cases']}
    modes=('free_mapper_readiness','observed_macro_zero','observed_macro_readiness')
    start=time.monotonic();save()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(work,row,metadata[row['id']],mode,service) for row in rows for mode in modes if (row['id'],mode) not in done]
        for future in concurrent.futures.as_completed(futures):
            entry=future.result();report['cases'].append(entry);report['wall_seconds']=time.monotonic()-start;save()
            print(len(report['cases']),len(rows)*len(modes),entry['id'],entry['mode'],entry['status'],round(entry.get('error_percent',0),2),flush=True)
    report['completed']=True;report['summaries']={}
    for mode in modes:
        selected=[r for r in report['cases'] if r['mode']==mode]
        valid=[r for r in selected if r['status']=='predicted']
        report['summaries'][mode]={'total':len(selected),'supported':len(valid),'passes':sum(abs(r['error_percent'])<=15 for r in valid),
                                  'mean_absolute_error_percent':statistics.mean(abs(r['error_percent']) for r in valid)}
    save();print(json.dumps(report['summaries'],indent=2))


if __name__=='__main__':main()
