"""Which low-occupancy form does the row sweep support?

Three candidates, all calibrated on the same 12 points (six row counts, two
dtypes) at length 2048:

  additive      t = launch + (K-1)*p + r*p/(cores*clock*occ)        what the model does now
  floor         t = launch + max(K*p, r*p/(cores*clock*occ))        a latency floor
  concurrency   t = launch + L*max(1, r/C)                         C rows in flight, each L long

`p` is the saturated per-row cost in cycles from the profile.
The point of the exercise is the hardware's 32 resident wavefronts per CU:
the concurrency form turns that into rows-in-flight through the kernel's block
size, so the fitted C should land near 72 * 32 / wavefronts_per_block.

Usage: python analyze_band.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_root(start):
    for p in (start, *start.parents):
        if (p / 'superpod_compute').is_dir():
            return p
    raise SystemExit('superpod_compute not found above %s' % start)


ROOT = find_root(HERE)
sys.path.insert(0, str(ROOT))
from superpod_compute import Hardware
from superpod_compute.vector_ops import VectorRequest, predict

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
PROFILE = ROOT / 'superpod_compute/data/bw1100_vector_profile.json'
BAND = Path('/workspace/bw_probe_20261105/transition_band_20261112.json')


def main():
    hw = Hardware.load(HW)
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    cases = json.loads(BAND.read_text(encoding='utf-8'))['cases']
    launch = profile['launch_s'] * 1e6

    pts = []
    for c in cases:
        req = VectorRequest('softmax', c['rows'], c['length'], c['dtype'])
        m = predict(req, hw, profile)
        pts.append(dict(rows=c['rows'], unit=c['dtype'], measured=c['measured_us'],
                        p=m['per_row_cycles'], occ=m['occupancy'],
                        throughput=m['per_row_cycles'] * c['rows'] / (hw.cores * hw.clock_hz) * 1e6 / m['occupancy']))

    def score(fn):
        errs = [abs(100 * (fn(q) / q['measured'] - 1)) for q in pts]
        return sum(errs) / len(errs), max(errs)

    # additive: pick (K-1)*p/clock, i.e. the current latency term
    best = min(((k, *score(lambda q, k=k: launch + (k - 1) * q['p'] / hw.clock_hz * 1e6 + q['throughput']))
                for k in [x / 4 for x in range(4, 41)]), key=lambda t: t[1])
    print('additive    K=%.2f  mean %.1f%%  max %.1f%%' % best)

    # floor: max(K*p/clock, throughput)
    bestf = min(((k, *score(lambda q, k=k: launch + max(k * q['p'] / hw.clock_hz * 1e6, q['throughput'])))
                 for k in [x / 4 for x in range(4, 81)]), key=lambda t: t[1])
    print('floor       K=%.2f  mean %.1f%%  max %.1f%%' % bestf)

    # concurrency: L * max(1, rows/C)
    bestc = None
    for C in range(64, 1201, 8):
        for L10 in range(60, 200):
            L = L10 / 10.0
            r = score(lambda q, C=C, L=L: launch + L * max(1.0, q['rows'] / C))
            if bestc is None or r[0] < bestc[0]:
                bestc = (C, L) + r
    print('concurrency C=%d rows, L=%.1f us  mean %.1f%%  max %.1f%%' % bestc)
    C = bestc[0]
    print('   -> rows in flight per CU = %.1f, wavefronts per block = %.1f (at 32 resident/CU)'
          % (C / hw.cores, hw.resident_wavefronts_per_core / (C / hw.cores)))

    print('\n%-6s %-5s %9s %9s %9s %9s %9s' % (
        'rows', 'dtype', '实测', 'additive', 'floor', 'concur', '现在模型'))
    for q in pts:
        a = launch + (best[0] - 1) * q['p'] / hw.clock_hz * 1e6 + q['throughput']
        f = launch + max(bestf[0] * q['p'] / hw.clock_hz * 1e6, q['throughput'])
        c = launch + bestc[1] * max(1.0, q['rows'] / C)
        m = predict(VectorRequest('softmax', q['rows'], 2048, q['unit']), hw, profile)['duration_us']
        print('%-6d %-5s %9.2f %9.2f %9.2f %9.2f %9.2f' % (
            q['rows'], q['unit'], q['measured'], a, f, c, m))


if __name__ == '__main__':
    main()
