"""Model prediction for mlp_linear_2 at half size (2048x2048x8192), same policy
as the AICB operator table so the two numbers are comparable."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy, predict

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = ROOT / 'experiments/regression_100_v2_20261109/policy_no192.json'
OUT = HERE / 'half_mlp2_model.json'
CASES = [(4096, 4096, 16384), (2048, 2048, 8192)]


def main():
    hw = Hardware.load(HW)
    pol = Policy(**json.loads(POL.read_text(encoding='utf-8')))
    rows = []
    for m, n, k in CASES:
        r = predict(GemmRequest(m, n, k, 'fp16', layout='row-major-contiguous'), hw,
                    policy=pol, backend='mmac', cache_path=None)['best']
        mp = r['mapping']
        rows.append(dict(shape=[m, n, k], us=r['duration_us'],
                         tile='%dx%dx%d' % (mp['bm'], mp['bn'], mp['bk'])))
        print('%dx%dx%d  %.1f us  tile %s' % (m, n, k, r['duration_us'],
                                              rows[-1]['tile']), flush=True)
    OUT.write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
