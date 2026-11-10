"""Fit bw1100_decode_replay.json from measured decode shapes. Reproducible.

Training data: the 2026-11-04 decode measurement sets (metadata + Graph times)
and the 2026-11-06 150-shape grid. The held-out validation shapes listed in
--holdout are removed before fitting and never enter the selection table.
"""
import argparse, json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from superpod_compute.decode_replay import class_key, service

EXP = ROOT / 'experiments'
SETS = [('decode_20261104', n) for n in ('development', 'extension', 'final', 'acceptance', 'validation')]
SETS.append(('decode_mapper_pair_20261104', 'decode_mapper_pair'))
FIXED_S = 3.121244690369221e-06   # per-GEMM device service, from the existing policy
CUS = 72


def cases(obj):
    return obj['cases'] if isinstance(obj, dict) else obj


def load_sets():
    rows = []
    for folder, stem in SETS:
        meas = cases(json.loads((EXP / folder / f'{stem}_measure.json').read_text(encoding='utf-8')))
        meta = {tuple(x['shape']): x for x in cases(json.loads(
            (EXP / folder / f'{stem}_metadata.json').read_text(encoding='utf-8')))}
        for c in meas:
            x = meta.get(tuple(c['shape']))
            if not x or not x.get('supported') or x.get('kernel_count', 1) != 1 or x.get('split_k', 1) != 1:
                continue
            t = c['measured_graph_us']
            rows.append(dict(shape=list(c['shape']), fast_us=t if isinstance(t, (int, float)) else st.mean(t),
                             slow_us=None, tile=x['tile'], prefetch_global=x['prefetch_global'],
                             workgroup=x['workgroup'], source=stem))
    return rows


def load_grid(path):
    rows = []
    for c in cases(json.loads(Path(path).read_text(encoding='utf-8'))):
        if c.get('kernel_count') != 1 or c.get('split_k', 1) != 1 or not c.get('tile'):
            continue
        rows.append(dict(shape=c['shape'], fast_us=c['fast_us'], slow_us=c['slow_us'], tile=c['tile'],
                         prefetch_global=c['prefetch_global'], workgroup=c['workgroup'], source='grid_20261106'))
    return rows


def fit(rows):
    by = defaultdict(list)
    for r in rows:
        r['class'] = class_key(r['tile'], r['prefetch_global'], r['workgroup'])
        by[r['class']].append(r)
    rates = [x * 1e9 for x in range(2, 120)]
    best = None
    for cap in [x * 1e11 for x in range(14, 23)]:
        classes, errs = {}, []
        for key, rs in by.items():
            proto = dict(tile=rs[0]['tile'], prefetch_global=rs[0]['prefetch_global'],
                         workgroup=rs[0]['workgroup'])
            def score(rate):
                prof = dict(cus=CUS, fixed_s=FIXED_S, bandwidth_cap_Bps=cap,
                            classes={key: dict(proto, r_block_Bps=rate)})
                return st.mean(abs(service(r['shape'], key, prof)['duration_s'] * 1e6 / r['fast_us'] - 1) for r in rs)
            rate = min(rates, key=score)
            ratios = [r['slow_us'] / r['fast_us'] for r in rs if r['slow_us']]
            classes[key] = dict(proto, r_block_Bps=rate, samples=len(rs),
                                slow_over_fast=max(ratios) if ratios else 1.0, train_mape=score(rate))
            errs += [classes[key]['train_mape']] * len(rs)
        if best is None or st.mean(errs) < best[0]:
            best = (st.mean(errs), cap, classes)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True, action='append', help='measured grid JSON; repeatable')
    ap.add_argument('--holdout', required=True, action='append',
                    help='JSON list of [M,N,K] never used for fitting; repeatable')
    ap.add_argument('--output', default=str(ROOT / 'superpod_compute/data/bw1100_decode_replay.json'))
    args = ap.parse_args()
    hold = {tuple(s) for h in args.holdout for s in json.loads(Path(h).read_text(encoding='utf-8'))}
    rows = load_sets() + [r for g in args.grid for r in load_grid(g)]
    uniq = {}
    for r in rows:
        if tuple(r['shape']) not in hold:
            uniq.setdefault(tuple(r['shape']), r)   # first source wins on duplicates
    rows = list(uniq.values())
    mape, cap, classes = fit(rows)
    dims = list(zip(*[r['shape'] for r in rows]))
    profile = dict(
        version='decode-replay-1.0', cus=CUS, fixed_s=FIXED_S, bandwidth_cap_Bps=cap,
        domain=dict(M=[1, 32], N=[min(dims[1]), max(dims[1])], K=[min(dims[2]), max(dims[2])]),
        classes=classes,
        table=[dict(shape=r['shape'], **{'class': r['class']}) for r in sorted(rows, key=lambda r: r['shape'])],
        provenance=dict(training_shapes=len(rows), holdout_excluded=len(hold), train_mape=mape,
                        sources=sorted({r['source'] for r in rows}),
                        note='Measured kernel class per shape (tile, prefetch depth, workgroup); '
                             'no vendor kernel names or selection database stored.'))
    Path(args.output).write_text(json.dumps(profile, indent=1) + '\n', encoding='utf-8')
    print(f'{len(rows)} shapes, {len(classes)} classes, cap {cap/1e12:.2f} TB/s, train MAPE {mape*100:.1f}%')


if __name__ == '__main__':
    main()
