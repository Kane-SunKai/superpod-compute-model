"""Why the V3 outliers are fast: compare the model's reuse credit with the
traffic a 256x256 tiling actually has to move.

Reuses the mappings already chosen by the V2 search (no candidate enumeration).
For each shape it prints, in microseconds: the MMAC floor, the model, the
measurement, and the time the same grid would need if the L2 gave no reuse at
all at the measured effective read rate.
"""
import json, math, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from superpod_compute import Hardware, Policy, GemmRequest
from superpod_compute.api import evaluate_mapping

V2 = ROOT / 'experiments/regression_100_v2_20261109'
PROBE = ROOT.parent / 'bw_probe'
HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = V2 / 'policy_no192.json'

FLAGGED = ['gpt7b_11', 'gpt7b_12', 'gpt7b_15', 'gpt13b_10', 'gpt13b_11', 'gpt13b_12']
REF = ['atten_qkv', 'atten_linear']          # AICB ops, in band
OPS = {'mlp_linear_1': (4096, 16384, 4096), 'mlp_linear_2': (4096, 4096, 16384),
       'logit_time': (4096, 50304, 4096), 'atten_qkv': (4096, 12288, 4096)}


def lib_tile(name):
    if not name:
        return None
    t = re.search(r'MT(\d+)x(\d+)x(\d+)', name)
    if t:
        return int(t.group(2)), int(t.group(1)), int(t.group(3))
    t = re.match(r'(\d+)x(\d+)x(\d+)', name)
    if t:
        return int(t.group(1)), int(t.group(2)), int(t.group(3))
    return None


def main():
    hw = Hardware.load(HW)
    pol = Policy(**json.loads(POL.read_text(encoding='utf-8')))
    peak = hw.cores * hw.arrays_per_core * (hw.instruction_m * hw.instruction_n
                                                    * hw.instruction_k) / hw.instruction_issue_cycles * hw.clock_hz
    pred = {r['id']: r for r in
            json.loads((V2 / 'predictions_bw2.json').read_text(encoding='utf-8'))}
    nt = {c['id']: c for c in
          json.loads((PROBE / 'nt100_20261111.json').read_text(encoding='utf-8'))['cases']}
    ops = {c['op']: c for c in
           json.loads((PROBE / 'aicb_ops_20261111.json').read_text(encoding='utf-8'))['cases']}
    opmodel = {r['op']: r for r in json.loads(
        (Path(__file__).resolve().parent / 'aicb_ops_model.json').read_text(encoding='utf-8'))}

    print(f'{"case":<15}{"MxNxK":<24}{"model tile":<13}{"lib tile":<13}'
          f'{"floor":>8}{"model":>8}{"meas":>8}{"model/floor":>12}{"meas/floor":>11}'
          f'{"noreuse GB":>11}{"noreuse us":>11}{"model HBM GB":>13}')
    rows = []
    for oid in FLAGGED + REF:
        if oid not in pred:
            continue
        m, n, k = pred[oid]['shape']
        mp = pred[oid]['mapping']
        meas = nt[oid]['default_us']
        lt = lib_tile(nt[oid].get('kernel'))
        rows.append((oid, m, n, k, mp, meas, lt, None))

    for oid in ('mlp_linear_1', 'mlp_linear_2', 'logit_time', 'atten_qkv'):
        m, n, k = OPS[oid]
        c = ops[oid]
        rows.append((oid, m, n, k, None, c['us'], lib_tile(c['kernels'][0] if c['kernels'] else ''), oid))

    for oid, m, n, k, mp, meas, lt, opname in rows:
        if mp is None:
            mp = opmodel[opname]['mapping'] if 'mapping' in opmodel[opname] else {
                'bm': int(opmodel[opname]['tile'].split('x')[0]),
                'bn': int(opmodel[opname]['tile'].split('x')[1]),
                'bk': int(opmodel[opname]['tile'].split('x')[2])}
            pmodel = opmodel[opname]['us']
            model_us, hbm_gb = pmodel, None
        else:
            target = pred[oid]['predicted_us']
            best = None
            for L in (1, 2):
                for S in (1, 2):
                    r = evaluate_mapping(
                        GemmRequest(m, n, k, 'fp16', layout='right-transposed-contiguous'),
                        _mapping(mp, L, S), hw, policy=pol, backend='mmac')
                    if best is None or abs(r['duration_us'] - target) < abs(best['duration_us'] - target):
                        best = r
            r = best
            model_us = r['duration_us']
            hbm_gb = r['hbm_read_bytes'] / 1e9
        floor = 2 * m * n * k / 2 / peak * 1e6
        bm, bn = (lt[0], lt[1]) if lt else (mp['bm'], mp['bn'])
        nwg = math.ceil(m / bm) * math.ceil(n / bn)
        noreuse = nwg * (bm + bn) * k * 2 / 1e9
        t_noreuse = noreuse * 1e9 / hw.hbm_bandwidth_Bps * 1e6
        print(f'{oid:<15}{"%dx%dx%d" % (m, n, k):<24}'
              f'{"%dx%dx%d" % (mp["bm"], mp["bn"], mp["bk"]):<13}'
              f'{("%dx%dx%d" % lt) if lt else "?":<13}'
              f'{floor:>8.0f}{model_us:>8.0f}{meas:>8.0f}{model_us / floor:>12.2f}{meas / floor:>11.2f}'
              f'{noreuse:>11.2f}{t_noreuse:>11.0f}'
              f'{(f"{hbm_gb:.2f}" if hbm_gb else "—"):>13}', flush=True)


def _mapping(mp, l2_buffers=2, shared_buffers=2):
    from superpod_compute.contracts import Mapping
    fields = Mapping.__dataclass_fields__ if hasattr(Mapping, '__dataclass_fields__') else {}
    values = {n: mp[n] for n in fields if n in mp}
    values.setdefault('l2_buffers', l2_buffers)
    values.setdefault('shared_buffers', shared_buffers)
    values.setdefault('operand_layout', 'engine-routed')
    return Mapping(**values)


if __name__ == '__main__':
    main()
