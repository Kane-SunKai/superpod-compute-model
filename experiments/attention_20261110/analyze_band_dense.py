"""Recover the per-row cost curve from the dense sweep and score candidate forms.

For every measured point the per-row cost in CU-cycles is
    (measured - launch) * cores * clock / rows
If that were constant the operator would be a pure throughput machine. It is not:
at small row counts a block cannot hide its own latency, so the per-row cost is
higher. This prints the slowdown factor kappa = p(rows) / p(saturated) and fits
the concurrency forms against it.

Usage: python analyze_band_dense.py
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
DENSE = Path('/workspace/bw_probe_20261105/band_dense_20261112.json')


def main():
    hw = Hardware.load(HW)
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    launch = profile['launch_s'] * 1e6
    cases = json.loads(DENSE.read_text(encoding='utf-8'))['cases']

    groups = {}
    for c in cases:
        groups.setdefault((c['length'], c['dtype']), []).append(c)

    for key, pts in sorted(groups.items()):
        pts.sort(key=lambda x: x['rows'])
        tail = [p['measured_per_row_cycles'] for p in pts[-3:]]
        p_inf = sum(tail) / len(tail)
        print('\n== length=%d %s  (saturated per-row = %.0f cycles) ==' % (key[0], key[1], p_inf))
        print('%-7s %10s %10s %8s %10s' % ('rows', '周期/行', 'κ = p/p∞', 'κ·rows', '误差'))
        for p in pts:
            k = p['measured_per_row_cycles'] / p_inf
            print('%-7d %10.0f %10.2f %10.0f %+7.1f%%' % (
                p['rows'], p['measured_per_row_cycles'], k, k * p['rows'], p['err_pct']))

        # For a hard concurrency limit, κ = max(1, C/rows) so κ·rows is C until
        # it saturates. The plateau of that column is the rows-in-flight figure.
        inflight = [p['measured_per_row_cycles'] / p_inf * p['rows']
                    for p in pts if p['measured_per_row_cycles'] / p_inf > 1.15]
        if inflight:
            C = sum(inflight) / len(inflight)
            print('  κ>1.15 的点给出 C ≈ %.0f 行在飞 → 每 CU %.1f 行 → 每 block %.1f 条 wavefront'
                  % (C, C / hw.cores, hw.resident_wavefronts_per_core / (C / hw.cores)))

    # score the candidate forms on every point
    def pt(q):
        req = VectorRequest('softmax', q['rows'], q['length'], q['dtype'])
        m = predict(req, hw, profile)
        return q, m

    scored = [pt(q) for q in cases]

    def score(fn):
        e = [abs(100 * (fn(q, m) / q['measured_us'] - 1)) for q, m in scored]
        return sum(e) / len(e), max(e)

    cur = score(lambda q, m: m['duration_us'])
    print('\n当前模型                              mean %.1f%%  max %.1f%%' % cur)

    best = None
    for K10 in range(10, 121, 5):
        K = K10 / 10.0
        for Cd in (48, 64, 72, 96, 128, 192, 256, 384, 512, 768, 1024):
            for gamma in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
                def fn(q, m, K=K, C=Cd, gamma=gamma):
                    # smooth concurrency: the slowdown decays from K to 1 over C rows
                    kappa = 1.0 + (K - 1.0) * min(1.0, (C / max(q['rows'], 1)) ** gamma)
                    return launch + kappa * m['per_row_cycles'] * q['rows'] / (hw.cores * hw.clock_hz) * 1e6
                r = score(fn)
                if best is None or r[0] < best[0]:
                    best = (r[0], r[1], K, Cd, gamma)
    print('平滑并发 K=%.1f C=%d gamma=%.2f        mean %.1f%%  max %.1f%%' % (
        best[2], best[3], best[4], best[0], best[1]))


if __name__ == '__main__':
    main()
