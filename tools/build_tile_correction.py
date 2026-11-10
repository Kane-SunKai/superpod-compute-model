"""Build a tile correction table from fit_tile_eff output. Reproducible.

factor(tile) = median over calibration shapes of model(eta=1) / measured, where
measured is the fastest single-kernel, non-split-K variant with that tile.
Only tiles seen on at least --min-shapes calibration shapes are kept.
"""
import argparse, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', default=str(ROOT / 'experiments/tile_calib_20261107/fit.json'))
    ap.add_argument('--name', default='bw1100_tile_correction')
    ap.add_argument('--min-shapes', type=int, default=5)
    args = ap.parse_args()
    fit = json.loads(Path(args.fit).read_text(encoding='utf-8'))
    by = {}
    for r in fit['rows']:
        if 'error' not in r:
            by.setdefault(tuple(r['tile']), []).append(r['ratio_eta1'])
    tiles = [dict(tile=list(t), factor=round(st.median(v), 6), shapes=len(v),
                  spread=round(max(v) / min(v), 4))
             for t, v in sorted(by.items()) if len(v) >= args.min_shapes]
    out = dict(
        version='tile-correction-1.0',
        model_profile=dict(hardware='bw1100_mapper_mmac_hardware.json', policy='bw1100_mapper_mmac_policy.json',
                           backend='mmac', model_version='resident-k-explicit-operand-layout-3.4'),
        calibration_shapes=[c['shape'] for c in fit['cases']],
        measurement='FP16 rocBLAS gemm_ex with forced solution index; warm same-address Graph; '
                    'fastest single-kernel GSU1 variant per tile; BW1100 nmz-sh-1, 2026-11-07',
        scope='Compute-bound prefill shapes (M >= 512). Not fitted on decode; empirical per-tile lookup, '
              'does not extrapolate to tiles outside this list or to other hardware.',
        tiles=tiles)
    dest = ROOT / 'superpod_compute/data' / f'{args.name}.json'
    dest.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(f'{len(tiles)} tiles -> {dest}')


if __name__ == '__main__':
    main()
