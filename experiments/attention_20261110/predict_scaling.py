"""Model prediction for the nine scaling points, same policy as the AICB table."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy, predict

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = ROOT / 'experiments/regression_100_v2_20261109/policy_no192.json'
OUT = HERE / 'scaling_model.json'
SHAPES = [
    ('half_all', 2048, 2048, 8192),
    ('half_M', 2048, 4096, 16384),
    ('half_N', 4096, 2048, 16384),
    ('half_K', 4096, 4096, 8192),
    ('base', 4096, 4096, 16384),
    ('dbl_M', 8192, 4096, 16384),
    ('dbl_N', 4096, 8192, 16384),
    ('dbl_K', 4096, 4096, 32768),
    ('dbl_all', 8192, 8192, 32768),
]


def main():
    hw = Hardware.load(HW)
    pol = Policy(**json.loads(POL.read_text(encoding='utf-8')))
    rows = []
    for label, m, n, k in SHAPES:
        r = predict(GemmRequest(m, n, k, 'fp16', layout='row-major-contiguous'), hw,
                    policy=pol, backend='mmac', cache_path=None)['best']
        mp = r['mapping']
        rows.append(dict(label=label, shape=[m, n, k], us=r['duration_us'],
                         tile='%dx%dx%d' % (mp['bm'], mp['bn'], mp['bk']),
                         waves=r['wave_count']))
        print('%-9s %-22s %9.1f us  %-13s waves %d' % (
            label, '%dx%dx%d' % (m, n, k), r['duration_us'], rows[-1]['tile'],
            r['wave_count']), flush=True)
        OUT.write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
