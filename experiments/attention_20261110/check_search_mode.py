"""Sanity check that buffer_dims='search' still enumerates the old space."""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy, predict
from superpod_compute.operators import default_registry

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = ROOT / 'experiments/regression_100_v2_20261109/policy_no192.json'


def main():
    hw = Hardware.load(HW)
    base = Policy(**json.loads(POL.read_text(encoding='utf-8')))
    model = default_registry().get('gemm')
    req = GemmRequest(4096, 4096, 16384, 'fp16', layout='row-major-contiguous')
    for mode in ('max', 'search'):
        pol = replace(base, buffer_dims=mode)
        n = sum(1 for _ in model.candidates(req, hw, pol))
        t0 = time.monotonic()
        r = predict(req, hw, policy=pol, backend='mmac', cache_path=None)['best']
        dt = time.monotonic() - t0
        mp = r['mapping']
        print('%-7s candidates %6d   predict %6.1f s   %8.1f us   %dx%dx%d gm%dxgn%d'
              % (mode, n, dt, r['duration_us'], mp['bm'], mp['bn'], mp['bk'],
                 mp['gm'], mp['gn']))


if __name__ == '__main__':
    main()
