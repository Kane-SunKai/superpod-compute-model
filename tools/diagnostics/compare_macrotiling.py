"""Isolate observed macrotiling using the already archived local measurements.

Not a BLAS-performance predictor: dispatch metadata is provided evidence, and
the old service model still optimizes buffers/groupings for each macrotiling.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy
from superpod_compute.kernel_plan import parse_tensile_kernel, evaluate_macrotiling_ablation


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Use a fresh output to preserve experiment evidence')
    measurement_path=ROOT/'results/validation_20261028/final_measurements.json'
    metadata_path=ROOT/'results/validation_20261028_v31/kernel_metadata_final.json'
    baseline_path=ROOT/'results/validation_20261029_v32/comparison_modes.json'
    rows=json.loads(measurement_path.read_text())['cases']
    metadata={r['id']:r for r in json.loads(metadata_path.read_text())['cases']}
    baseline={r['id']:r for r in json.loads(baseline_path.read_text())['cases'] if r['mode']=='mmac'}
    hardware_path=ROOT/'superpod_compute/data/bw1100.json'
    policy_path=ROOT/'examples/bw1100_graph_calibrated.json'
    hw=Hardware.load(hardware_path);policy=Policy(**json.loads(policy_path.read_text()))
    fingerprint_paths=[measurement_path,metadata_path,baseline_path,hardware_path,policy_path,
                       ROOT/'superpod_compute/kernel_plan.py',ROOT/'superpod_compute/operators.py',
                       ROOT/'superpod_compute/backends.py',ROOT/'superpod_compute/scheduler.py']
    output={'completed':False,'cases':[],
            'identity':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in fingerprint_paths},
            'scope':'Archived validation only; observed macrotiling ablation. No new measurement export, fitting, or claim of full library timing.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for row in rows:
        if row['role']!='validation':continue
        entry={key:row[key] for key in ('id','shape','group')}
        measured=statistics.mean(row['graph_round'+str(i)]['p50_us'] for i in (1,2))
        entry.update(measured_us=measured,baseline_us=baseline[row['id']]['model_us'],
                     baseline_absolute_error_percent=baseline[row['id']]['absolute_error_percent'])
        try:
            names=[name for name in metadata[row['id']]['kernels'] if '_MT' in name]
            if len(names)!=1:raise ValueError('Expected one identifiable matrix kernel')
            plan=parse_tensile_kernel(names[0],orientation='row-major-transposed-blas')
            entry.update(kernel_plan=plan.to_dict(),coverage=plan.coverage(GemmRequest(*row['shape']),hw.cores))
            result=evaluate_macrotiling_ablation(GemmRequest(*row['shape']),hw,policy,plan)
            estimate=result['best']['duration_us']
            entry.update(status='estimated',macrotiling_us=estimate,
                         signed_error_percent=100*(estimate/measured-1),
                         absolute_error_percent=100*abs(estimate/measured-1),mapping=result['best']['mapping'],
                         wave_count=result['best']['wave_count'])
        except (ValueError,NotImplementedError) as exc:
            entry.update(status='unsupported',reason=str(exc))
        output['cases'].append(entry)
        args.output.write_text(json.dumps(output,indent=2)+'\n')
        print(entry['id'],entry['status'],round(entry.get('signed_error_percent',0),2),flush=True)
    output['completed']=True
    for group in ('all','irregular','regular'):
        subset=[r for r in output['cases'] if group=='all' or (r['group']=='irregular')==(group=='irregular')]
        valid=[r for r in subset if r['status']=='estimated']
        output[group]={'total':len(subset),'supported':len(valid),
                       'passes':sum(r['absolute_error_percent']<=15 for r in valid),
                       'baseline_passes_same_supported':sum(r['baseline_absolute_error_percent']<=15 for r in valid),
                       'mean_absolute_error_percent':statistics.mean(r['absolute_error_percent'] for r in valid)}
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('all','regular','irregular')},indent=2))


if __name__=='__main__':main()
