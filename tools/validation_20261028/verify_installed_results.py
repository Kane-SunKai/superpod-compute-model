"""Re-evaluate archived mappings against installed code, without a new search."""
from pathlib import Path
import argparse,json,math,sys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path);ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    root=(args.project or Path(__file__).resolve().parents[2]).resolve()
    sys.path.insert(0,str(root))
    from superpod_compute import Hardware,Policy,GemmRequest,Mapping,evaluate_mapping
    from superpod_compute.contracts import digest,MODEL_VERSION,SEARCH_VERSION
    results=root/'results/validation_20261028_v31';checked={}
    for version,profile in [('v3','bw1100_mmac_only_v3.json'),('v31','bw1100.json')]:
        report=json.loads((results/f'comparison_{version}.json').read_text())
        hw=Hardware.load(root/'superpod_compute/data'/profile)
        differences=[]
        for row in report['cases']:
            result=evaluate_mapping(GemmRequest(*row['shape']),Mapping(**row['mapping']),hw,policy=Policy(initial_cache_state='warm-if-fit'),trace_limit=0)
            if not math.isclose(result['duration_us'],row['model_raw_us'],rel_tol=1e-10,abs_tol=1e-9):
                differences.append({'id':row['id'],'recorded':row['model_raw_us'],'current':result['duration_us']})
        checked[version]={'checked_mappings':len(report['cases']),'differences':differences}
    hw=Hardware.load(root/'superpod_compute/data/bw1100.json')
    policy=Policy(**json.loads((root/'examples/bw1100_graph_calibrated.json').read_text()))
    report=json.loads((results/'comparison_v31.json').read_text())
    sample=next(x for x in report['cases'] if x['id']=='prefill_34')
    actual=evaluate_mapping(GemmRequest(*sample['shape']),Mapping(**sample['mapping']),hw,policy=policy,trace_limit=0)
    summary={'model_version':MODEL_VERSION,'search_version':SEARCH_VERSION,'profiles':checked,
             'calibration_fingerprint_matches':policy.calibration_hardware_sha256==digest(hw.to_dict()),
             'calibrated_regular_us':actual['duration_us'],
             'calibrated_example_matches':math.isclose(actual['duration_us'],sample['model_calibrated_us'],rel_tol=1e-10),
             'validation_summary':report['validation_summary']}
    if args.output:args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    assert all(not v['differences'] for v in checked.values())
    assert summary['calibration_fingerprint_matches'] and summary['calibrated_example_matches']

if __name__=='__main__':main()
