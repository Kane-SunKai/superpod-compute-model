"""Check all saved V3.2 mappings against current installed code, without searching."""
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from superpod_compute import Hardware,Policy,GemmRequest,Mapping,evaluate_mapping
def main():
    report=json.loads((ROOT/'results/validation_20261029_v32/comparison_modes.json').read_text())
    assert report['completed'] and len(report['cases'])==144
    profiles={}
    for mode in ('mmac','mac256'):
        hf='bw1100.json' if mode=='mmac' else 'bw1100_mac256.json'
        pf='bw1100_graph_calibrated.json' if mode=='mmac' else 'bw1100_mac256_comparison.json'
        profiles[mode]=(Hardware.load(ROOT/'superpod_compute/data'/hf),Policy(**json.loads((ROOT/'examples'/pf).read_text())))
    for row in report['cases']:
        h,p=profiles[row['mode']]
        actual=evaluate_mapping(GemmRequest(*row['shape']),Mapping(**row['mapping']),h,policy=p,trace_limit=0)
        assert math.isclose(actual['duration_us'],row['model_us'],rel_tol=1e-10,abs_tol=1e-9),row['id']
    print('Verified 144 recorded mappings against installed code, profiles and frozen-service policies.')
if __name__=='__main__':main()
