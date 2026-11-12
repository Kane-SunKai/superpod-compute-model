"""Score the shipped tile-correction table against the 100 measured NT shapes.

For every shape whose library kernel used a square macro tile (the name does not
say which dimension is M, so square tiles are the unambiguous ones), evaluate the
model at exactly that tile and geometry, and compare model/measured with the
table's factor. The table was fitted before the HBM read/write split, so this
checks whether it survived the model change.
"""
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBE = ROOT.parent / 'bw_probe'
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy
from superpod_compute.backends import make_backend
from superpod_compute.operators import default_registry
from superpod_compute.tile_correction import load as load_table

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = ROOT / 'experiments/regression_100_v2_20261109/policy_no192.json'
V2 = ROOT / 'experiments/regression_100_v2_20261109'
OUT = HERE / 'tile_correction_check.json'


def main():
    hw = Hardware.load(HW)
    pol = Policy(**json.loads(POL.read_text(encoding='utf-8')))
    model = default_registry().get('gemm')
    backend = make_backend('mmac', hw, pol.matrix_efficiency)
    table = load_table('bw1100_tile_correction')['factors']
    nt = {c['id']: c for c in json.loads(
        (PROBE / 'nt100_20261111.json').read_text(encoding='utf-8'))['cases']}
    plan = {p['id']: p for p in json.loads(
        (V2 / 'plan.json').read_text(encoding='utf-8'))}

    by_tile = defaultdict(list)
    skipped = 0
    for rid, p in plan.items():
        c = nt.get(rid)
        if not c or not c.get('kernel'):
            continue
        t = re.match(r'(\d+)x(\d+)x(\d+)', c['kernel'])
        if not t:
            skipped += 1
            continue
        bm, bn, bk = (int(x) for x in t.groups())
        if bm != bn:
            skipped += 1
            continue
        m, n, k = p['shape']
        req = GemmRequest(m, n, k, 'fp16', layout='right-transposed-contiguous')
        best = None
        for mp in model.candidates(req, hw, pol):
            if (mp.bm, mp.bn, mp.bk) != (bm, bn, bk):
                continue
            if not model.capacity(mp, hw)['valid']:
                continue
            r = model.evaluate(req, hw, pol, backend, mp)
            if r is not None and (best is None or r['duration_s'] < best):
                best = r['duration_s']
        if best is None:
            skipped += 1
            continue
        by_tile[(bm, bn, bk)].append((100 * (best * 1e6 / c['default_us'] - 1), rid))

    rows = []
    print('%-13s %4s %9s %9s %9s %8s %7s %8s %8s' % (
        'tile', 'n', 'median', 'min', 'max', 'spread', 'ratio', 'table', 'drift'))
    for tile, vals in sorted(by_tile.items(), key=lambda kv: -len(kv[1])):
        xs = [v[0] for v in vals]
        med = st.median(xs)
        old = table.get(tile)
        ratio = 1 + med / 100
        rows.append(dict(tile=list(tile), n=len(xs), median_model_over_meas=round(med, 2),
                         lo=round(min(xs), 2), hi=round(max(xs), 2),
                         spread=round(max(xs) - min(xs), 2),
                         ratio=round(ratio, 4), table_factor=old,
                         drift=(round(100 * (ratio / old - 1), 1) if old else None),
                         shape_ids=[v[1] for v in vals][:8]))
        print('%-13s %4d %+8.1f%% %+8.1f%% %+8.1f%% %7.1f %7.3f %8s %8s' % (
            '%dx%dx%d' % tile, len(xs), med, min(xs), max(xs), max(xs) - min(xs),
            ratio, ('%.3f' % old) if old else '—',
            ('%+.0f%%' % (100 * (ratio / old - 1))) if old else '—'))
    print('\nratio = 模型/实测（该分块下的中位数）。table 是表里现成的系数，'
          'drift 是新比值相对表里系数的偏移。')
    print('\n%d shapes used, %d skipped (no parseable square tile)' % (
        sum(len(v) for v in by_tile.values()), skipped))
    OUT.write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
