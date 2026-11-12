"""Calibrate the low-occupancy latency factor on the row sweep, off-line.

measure_vector_validation.py saves its measurements, so the factor can be tuned
and the validation re-scored without touching the GPU again.

The row sweep is the right place to fit it: it holds the reduction length fixed
and moves only the row count, which is exactly the variable the factor is about.
Rows at or below 512 are the under-filled regime; the rest is the test of whether
the factor stays out of the way at scale.

Usage: python tune_latency.py [profile.json]
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
PROFILE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'superpod_compute/data/bw1100_vector_profile.json'
GAPS = Path('/workspace/bw_probe_20261105/vector_gaps_20261111.json')
VALID = Path('/workspace/bw_probe_20261105/vector_validation_20261111.json')

UNDER_FILLED = 512


def main():
    hw = Hardware.load(HW)
    base = json.loads(PROFILE.read_text(encoding='utf-8'))
    gaps = json.loads(GAPS.read_text(encoding='utf-8'))
    soft = [c for c in gaps if c['group'].startswith('softmax')]
    warp = [c for c in soft if c['length'] == 2048]
    gmem = [c for c in soft if c['length'] == 4096]

    print('== grid search on the row sweep (rows <= %d) ==' % UNDER_FILLED)
    best = {}
    for key, points in (('warp', warp), ('gmem', gmem)):
        rows = ''
        winner = None
        for alpha in (1.0, 1.15, 2, 3, 4, 5, 6, 7, 8, 10, 12):
            base['latency_factor'][key] = alpha
            small = [abs(100 * (predict(VectorRequest('softmax', c['rows'], c['length'], 'fp16'),
                                        hw, base)['duration_us'] / c['us'] - 1))
                     for c in points if c['rows'] <= UNDER_FILLED]
            large = [abs(100 * (predict(VectorRequest('softmax', c['rows'], c['length'], 'fp16'),
                                        hw, base)['duration_us'] / c['us'] - 1))
                     for c in points if c['rows'] > UNDER_FILLED]
            score = sum(small) / len(small)
            rows += '  %5.2f -> small %.1f%%  large %5.1f%%\n' % (
                alpha, score, sum(large) / len(large))
            if winner is None or score < winner[1]:
                winner = (alpha, score)
        print('%s\n%s  best alpha %.2f (%.1f%%)' % (key, rows, winner[0], winner[1]))
        best[key] = winner[0]
        base['latency_factor'][key] = winner[0]

    PROFILE.write_text(json.dumps(base, indent=2) + '\n')
    print('\nwrote %s with latency_factor %s' % (PROFILE, base['latency_factor']))

    if not VALID.exists():
        return
    cases = json.loads(VALID.read_text(encoding='utf-8'))['cases']
    print('\n== validation re-scored off-line (no new measurement) ==')
    print('%-20s %-6s %10s %10s %8s' % ('label', 'regime', 'measured', 'predicted', 'err'))
    errs = []
    for c in cases:
        p = predict(VectorRequest(c['kind'], c['rows'], c['length'], c['dtype']), hw, base)
        e = 100 * (p['duration_us'] / c['measured_us'] - 1)
        errs.append(abs(e))
        print('%-20s %-6s %10.1f %10.1f %+7.1f%%  %s' % (
            c['label'], p['regime'], c['measured_us'], p['duration_us'], e, c['why']))
    print('\nmean |err| %.1f%%  max %.1f%%  (%d shapes)'
          % (sum(errs) / len(errs), max(errs), len(errs)))

    for c in cases:
        p = predict(VectorRequest(c['kind'], c['rows'], c['length'], c['dtype']), hw, base)
        c['predicted_us'] = p['duration_us']
        c['err_pct'] = round(100 * (p['duration_us'] / c['measured_us'] - 1), 2)
        c['regime'] = p['regime']
    VALID.write_text(json.dumps(dict(cases=cases, achievement_applied=[],
                                     note='re-scored by tune_latency.py after the '
                                          'latency factor was set; achievement left at 1.0 '
                                          'because no group of residuals was flat'),
                                indent=1) + '\n')
    print('rewrote %s with the final predictions' % VALID)


if __name__ == '__main__':
    main()
