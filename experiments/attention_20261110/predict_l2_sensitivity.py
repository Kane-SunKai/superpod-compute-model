"""Does the model react to L2 capacity at all?

The hardware L2 cannot be resized, so the only test of "what if L2 were bigger"
is what the model itself does when the profile's l2_bytes changes. Three shapes
whose per-workgroup slice is 8.39, 4.19 and 16.8 MB.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy, predict

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = ROOT / 'experiments/regression_100_v2_20261109/policy_no192.json'
OUT = HERE / 'l2_sensitivity_model.json'
SHAPES = [('half_K', 4096, 4096, 8192), ('base', 4096, 4096, 16384),
          ('dbl_K', 4096, 4096, 32768)]


def main():
    hw0 = Hardware.load(HW)
    pol = Policy(**json.loads(POL.read_text(encoding='utf-8')))
    rows = []
    for mib in (4, 8, 16, 32):
        hw = replace(hw0, l2_bytes=mib * 1024 * 1024)
        for label, m, n, k in SHAPES:
            r = predict(GemmRequest(m, n, k, 'fp16', layout='row-major-contiguous'), hw,
                        policy=pol, backend='mmac', cache_path=None)['best']
            mp = r['mapping']
            rows.append(dict(l2_mib=mib, label=label, shape=[m, n, k],
                             us=r['duration_us'],
                             tile='%dx%dx%d' % (mp['bm'], mp['bn'], mp['bk']),
                             waves=r['wave_count']))
            print('L2=%2d MiB  %-7s %-20s %9.1f us  %-13s' % (
                mib, label, '%dx%dx%d' % (m, n, k), r['duration_us'],
                rows[-1]['tile']), flush=True)
            OUT.write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
