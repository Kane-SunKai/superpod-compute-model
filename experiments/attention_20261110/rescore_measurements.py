"""Re-score a saved measurement file against the current profile.

The measurements do not change when the profile does, so the predicted column and
the error have to be recomputed rather than re-measured.

Usage: python rescore_measurements.py <json> [profile]
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
TARGET = Path(sys.argv[1])
PROFILE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'superpod_compute/data/bw1100_vector_profile.json'

KIND = {'decode_softmax': 'softmax', 'gelu': 'gelu', 'layernorm': 'layernorm'}
MEAS_KEY = ('measured_us',)


def main():
    data = json.loads(TARGET.read_text(encoding='utf-8'))
    hw = Hardware.load(HW)
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    updated = 0
    for c in data.get('cases', []):
        kind = c.get('kind') or KIND.get(c.get('group'))
        if not kind and 'seq' in c:
            kind = 'softmax'
        if not kind or not c.get('measured_us'):
            continue
        rows_ = c.get('rows') or (c.get('local_heads', c.get('heads')) * c['seq'])
        length_ = c.get('length') or c['seq']
        m = predict(VectorRequest(kind, rows_, length_, c['dtype']), hw, profile)
        c['model_us'] = m['duration_us']
        c['err_pct'] = round(100 * (m['duration_us'] / c['measured_us'] - 1), 2)
        c['regime'] = m['regime']
        c['efficiency'] = m['efficiency']
        c['occupancy'] = m['occupancy']
        updated += 1
    TARGET.write_text(json.dumps(data, indent=1) + '\n')
    print('re-scored %d cases in %s' % (updated, TARGET))


if __name__ == '__main__':
    main()
