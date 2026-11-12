"""Check that buffer_dims='max' reproduces the full-search answers, and how much faster."""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy, predict
from superpod_compute.operators import default_registry

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = ROOT / 'experiments/regression_100_v2_20261109/policy_no192.json'
SCALING = HERE / 'scaling_model.json'
STUDY = HERE / 'buffer_dims_study.json'
SHAPES = [('half_all', 2048, 2048, 8192), ('half_M', 2048, 4096, 16384),
          ('half_N', 4096, 2048, 16384), ('half_K', 4096, 4096, 8192),
          ('base', 4096, 4096, 16384), ('dbl_M', 8192, 4096, 16384),
          ('dbl_N', 4096, 8192, 16384), ('dbl_K', 4096, 4096, 32768),
          ('dbl_all', 8192, 8192, 32768),
          ('small', 1024, 1024, 128), ('odd', 512, 5120, 13824),
          ('decode', 64, 4096, 4096)]


def main():
    hw = Hardware.load(HW)
    pol = Policy(**json.loads(POL.read_text(encoding='utf-8')))
    model = default_registry().get('gemm')
    ref = {}
    for f, key in ((SCALING, 'label'), (STUDY, 'label')):
        if f.exists():
            for r in json.loads(f.read_text(encoding='utf-8')):
                ref[r[key]] = r
    print('buffer_dims=%s' % pol.buffer_dims)
    print('%-9s %-20s %6s %7s %7s %9s %9s %7s %-14s' % (
        '组', 'M×N×K', '候选', '生ms', '评ms', '本次µs', '全搜索µs', '差', '分块'))
    worst = 0.0
    for label, m, n, k in SHAPES:
        req = GemmRequest(m, n, k, 'fp16', layout='row-major-contiguous')
        t0 = time.monotonic()
        n_cand = sum(1 for _ in model.candidates(req, hw, pol))
        t_gen = 1000 * (time.monotonic() - t0)
        t1 = time.monotonic()
        r = predict(req, hw, policy=pol, backend='mmac', cache_path=None)['best']
        ms = 1000 * (time.monotonic() - t1)
        mp = r['mapping']
        old = ref.get(label, {}).get('us')
        diff = (100 * (r['duration_us'] / old - 1)) if old else None
        if diff is not None:
            worst = max(worst, abs(diff))
        print('%-9s %-20s %6d %7.0f %7.0f %9.1f %9s %7s %-14s' % (
            label, '%dx%dx%d' % (m, n, k), n_cand, t_gen, ms, r['duration_us'],
            ('%.1f' % old) if old else '—',
            ('%+.2f%%' % diff) if diff is not None else '—',
            '%dx%dx%d' % (mp['bm'], mp['bn'], mp['bk'])))
    print('\n与全量枚举的最大偏差: %.2f%%' % worst)


if __name__ == '__main__':
    main()
