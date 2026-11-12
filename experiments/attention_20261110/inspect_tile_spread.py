"""What drives the spread inside one macro tile?

Dumps every shape that the library ran with 256x256x32 or 256x256x16, with the
model/measured ratio, sorted so a trend against M, N or K is visible.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBE = ROOT.parent / 'bw_probe'
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy
from superpod_compute.backends import make_backend
from superpod_compute.operators import default_registry

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = ROOT / 'experiments/regression_100_v2_20261109/policy_no192.json'
V2 = ROOT / 'experiments/regression_100_v2_20261109'
WANT = [(256, 256, 32), (256, 256, 16)]


def main():
    hw = Hardware.load(HW)
    pol = Policy(**json.loads(POL.read_text(encoding='utf-8')))
    model = default_registry().get('gemm')
    backend = make_backend('mmac', hw, pol.matrix_efficiency)
    nt = {c['id']: c for c in json.loads(
        (PROBE / 'nt100_20261111.json').read_text(encoding='utf-8'))['cases']}
    plan = {p['id']: p for p in json.loads((V2 / 'plan.json').read_text(encoding='utf-8'))}

    groups = {t: [] for t in WANT}
    for rid, p in plan.items():
        c = nt.get(rid)
        if not c or not c.get('kernel'):
            continue
        t = re.match(r'(\d+)x(\d+)x(\d+)', c['kernel'])
        if not t:
            continue
        tile = tuple(int(x) for x in t.groups())
        if tile not in groups:
            continue
        m, n, k = p['shape']
        req = GemmRequest(m, n, k, 'fp16', layout='right-transposed-contiguous')
        best = None
        for mp in model.candidates(req, hw, pol):
            if (mp.bm, mp.bn, mp.bk) != tile:
                continue
            if not model.capacity(mp, hw)['valid']:
                continue
            r = model.evaluate(req, hw, pol, backend, mp)
            if r is not None and (best is None or r['duration_s'] < best):
                best = r['duration_s']
        if best is None:
            continue
        groups[tile].append((100 * (best * 1e6 / c['default_us'] - 1), rid, m, n, k))

    for tile in WANT:
        rows = sorted(groups[tile])
        print('\n%s  (%d shapes)' % ('x'.join(map(str, tile)), len(rows)))
        print('%-11s %8s %10s %10s %10s %8s %8s' % (
            'id', '模型/实测', 'M', 'N', 'K', 'WGs', '面板MB'))
        for e, rid, m, n, k in rows:
            wgs = ((m + tile[0] - 1) // tile[0]) * ((n + tile[1] - 1) // tile[1])
            print('%-11s %+7.1f%% %10d %10d %10d %8d %8.1f' % (
                rid, e, m, n, k, wgs, tile[0] * k * 2 / 2**20))


if __name__ == '__main__':
    main()
