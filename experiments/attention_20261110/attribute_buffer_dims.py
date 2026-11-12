"""Attribute the cost of each depth dimension one at a time (marginal best)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from dataclasses import replace

from superpod_compute import GemmRequest, Hardware, Policy
from superpod_compute.backends import make_backend
from superpod_compute.operators import default_registry

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = ROOT / 'experiments/regression_100_v2_20261109/policy_no192.json'
SHAPES = [('base', 4096, 4096, 16384), ('odd', 512, 5120, 13824)]
NAMES = ['l2_buffers', 'shared_buffers', 'chunk', 'output_buffers', 'register_input_buffers']


def main():
    hw = Hardware.load(HW)
    base = Policy(**json.loads(POL.read_text(encoding='utf-8')))
    pol = replace(base, buffer_dims='search')
    model = default_registry().get('gemm')
    backend = make_backend('mmac', hw, pol.matrix_efficiency)
    print('buffer_dims=%s (full product required for a marginal)' % pol.buffer_dims)
    for label, m, n, k in SHAPES:
        req = GemmRequest(m, n, k, 'fp16', layout='row-major-contiguous')
        best = None
        marginal = [defaultdict(lambda: float('inf')) for _ in range(5)]
        for mp in model.candidates(req, hw, pol):
            if not model.capacity(mp, hw)['valid']:
                continue
            r = model.evaluate(req, hw, pol, backend, mp)
            if r is None:
                continue
            us = r['duration_us']
            best = us if best is None else min(best, us)
            values = (mp.l2_buffers, mp.shared_buffers, mp.output_chunk_bytes,
                      mp.output_buffers, mp.register_input_buffers)
            for i, v in enumerate(values):
                marginal[i][v] = min(marginal[i][v], us)
        print('\n%s %dx%dx%d  best %.1f us' % (label, m, n, k, best))
        for i, name in enumerate(NAMES):
            parts = []
            for v in sorted(marginal[i]):
                parts.append('%s=%s: %.1f (%+.1f%%)' % (
                    name, v, marginal[i][v], 100 * (marginal[i][v] / best - 1)))
            print('   ' + '   '.join(parts))


if __name__ == '__main__':
    main()
