"""Which of the five buffer dimensions actually matter, and what does fixing them cost?

For each shape this enumerates the full candidate space once and aggregates:
  best duration per (l2_buffers, shared_buffers, output_chunk, output_buffers,
  register_input_buffers) combo, and best per (bm,bn,bk) tile. The gap between the
  overall winner and the best candidate of a fixed combo is the loss you take for
  fixing it.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy
from superpod_compute.api import signature_for
from superpod_compute.backends import make_backend
from superpod_compute.operators import default_registry

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = ROOT / 'experiments/regression_100_v2_20261109/policy_no192.json'
OUT = HERE / 'buffer_dims_study.json'
SHAPES = [('base', 4096, 4096, 16384), ('small', 1024, 1024, 128),
          ('odd', 512, 5120, 13824), ('decode', 64, 4096, 4096)]


def main():
    hw = Hardware.load(HW)
    pol = Policy(**json.loads(POL.read_text(encoding='utf-8')))
    model = default_registry().get('gemm')
    backend = make_backend('mmac', hw, pol.matrix_efficiency)
    rows = []
    for label, m, n, k in SHAPES:
        req = GemmRequest(m, n, k, 'fp16', layout='row-major-contiguous')
        best = None
        by_combo = defaultdict(lambda: None)
        by_combo_map = {}
        total = 0
        for mp in model.candidates(req, hw, pol):
            total += 1
            cap = model.capacity(mp, hw)
            if not cap['valid']:
                continue
            r = model.evaluate(req, hw, pol, backend, mp)
            if r is None:
                continue
            key = (mp.l2_buffers, mp.shared_buffers, mp.output_chunk_bytes,
                   mp.output_buffers, mp.register_input_buffers)
            if best is None or r['duration_s'] < best[0]:
                best = (r['duration_s'], key, mp, r)
            if by_combo[key] is None or r['duration_s'] < by_combo[key][0]:
                by_combo[key] = (r['duration_s'], mp)
                by_combo_map[key] = mp
        combos = sorted(by_combo.items(), key=lambda kv: kv[1][0])
        rec = dict(label=label, shape=[m, n, k], candidates=total,
                   combos=len(combos),
                   best_us=round(best[0] * 1e6, 1),
                   best_combo=list(best[1]),
                   best_tile='%dx%dx%d' % (best[2].bm, best[2].bn, best[2].bk),
                   top=[dict(combo=list(kv[0]), us=round(kv[1][0] * 1e6, 1),
                             tile='%dx%dx%d' % (kv[1][1].bm, kv[1][1].bn, kv[1][1].bk))
                        for kv in combos[:8]],
                   worst_us=round(combos[-1][1][0] * 1e6, 1))
        rows.append(rec)
        print('\n%s %dx%dx%d  %d candidates, %d combos, best %.1f us  %s  combo %s'
              % (label, m, n, k, total, len(combos), rec['best_us'],
                 rec['best_tile'], rec['best_combo']), flush=True)
        for c in rec['top']:
            d = 100 * (c['us'] / rec['best_us'] - 1)
            print('    combo %-28s %9.1f us  %-13s  %+6.1f%%' % (
                c['combo'], c['us'], c['tile'], d), flush=True)
        print('    worst combo %.1f us (+%.1f%%)' % (
            rec['worst_us'], 100 * (rec['worst_us'] / rec['best_us'] - 1)), flush=True)
        OUT.write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
