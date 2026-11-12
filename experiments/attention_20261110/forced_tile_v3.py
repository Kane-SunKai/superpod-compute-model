"""decode_largeN_12 = 32 x 40960 x 3072, predicted at fixed tiles.

Reference measurements:
  NT layout, library default 64x64x16   1553.3 us
  NN layout, library default 256x256x16  260.4 us
  NN layout, fastest kernel 64x192x16    151.0 us

The model has no notion of layout; it only sees the tile. So forcing each tile
shows how much of the gap is tile choice and how much is the per-tile cost.
"""
import json, sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy, predict

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
PO = ROOT / 'experiments/regression_100_v2_20261109/policy_no192.json'
SHAPE = (32, 40960, 3072)
TILES = [('64x64x16  (NT default)', (64, 64, 16)),
         ('256x256x16 (NN default)', (256, 256, 16)),
         ('64x192x16 (NN fastest)', (64, 192, 16))]
MEAS = {'64x64x16  (NT default)': 1553.3,
        '256x256x16 (NN default)': 260.4,
        '64x192x16 (NN fastest)': 151.0}

hw = Hardware.load(HW)
base = Policy(**json.loads(PO.read_text(encoding='utf-8')))

r = predict(GemmRequest(*SHAPE, 'fp16', layout='row-major-contiguous'), hw, policy=base,
            backend='mmac', cache_path=None)
m = r['best']['mapping']
print('free search  %8.1f us  tile %dx%dx%d  gm=%d gn=%d  cands %d' % (
    r['best']['duration_us'], m['bm'], m['bn'], m['bk'], m['gm'], m['gn'],
    r['searched_candidates']))
print()

for label, (bm, bn, bk) in TILES:
    pol = replace(base, bm_values=(bm,), bn_values=(bn,), bk_values=(bk,), add_shape_candidate=False)
    try:
        rr = predict(GemmRequest(*SHAPE, 'fp16', layout='row-major-contiguous'), hw, policy=pol,
                     backend='mmac', cache_path=None)
        us = rr['best']['duration_us']
        mm = rr['best']['mapping']
        meas = MEAS[label]
        print('%-24s model %8.1f us  vs meas %8.1f  %+6.1f%%  waves %d gm=%d gn=%d' % (
            label, us, meas, 100 * (us / meas - 1), rr['best']['wave_count'], mm['gm'], mm['gn']))
    except ValueError as exc:
        print('%-24s ERROR %s' % (label, exc))
