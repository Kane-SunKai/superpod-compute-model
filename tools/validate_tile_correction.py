"""End-to-end check of the mapper with Policy.tile_correction enabled.

Runs the public predict() entry point on the 25 prefill/GPT shapes that are
outside the calibration set, compares with the 3.4 mapper, and reports errors
against the library default and (where measured) the fastest kernel.
"""
import concurrent.futures, json, statistics as st, sys
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy, predict

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
PO = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_policy.json'
JG = ROOT / 'experiments/joint_gemm_20261105'
SWEEP = ROOT / 'experiments/factorial_20261106/model_vs_sweep.json'


def cases():
    out = {}
    for pat in ('final_prefill_validation_*.json', 'final_fresh_prefill_*.json', 'final_gpt_*.json'):
        for p in sorted(JG.glob(pat)):
            d = json.loads(p.read_text(encoding='utf-8'))
            if not isinstance(d, dict) or (d.get('batch') or 1) != 1 or d['shape'][0] <= 32:
                continue
            out[tuple(d['shape'])] = dict(id=d['id'], default=d['measured_us'], best=None, old=d['final_us'])
    for r in json.loads(SWEEP.read_text(encoding='utf-8')):
        s = tuple(r['shape'])
        if s[0] <= 32:
            continue
        e = out.setdefault(s, dict(id='sweep_%dx%dx%d' % s, default=r['default_us'],
                                   best=r['best_us'], old=r['mapper_us']))
        e['best'] = r['best_us']
    return out


def work(job):
    shape, mode = job
    h = Hardware.load(HW)
    p = Policy(**json.loads(PO.read_text(encoding='utf-8-sig')))
    if mode == 'corrected':
        p = replace(p, tile_correction='bw1100_tile_correction')
    r = predict(GemmRequest(*shape, 'fp16', layout='row-major-contiguous'), h, policy=p,
                backend='mmac', cache_path=None)
    m = r['best']['mapping']
    return shape, mode, r['best']['duration_us'], '%dx%dx%d' % (m['bm'], m['bn'], m['bk'])


def main():
    cs = cases()
    jobs = [(s, mode) for s in cs for mode in ('plain', 'corrected')]
    res = {}
    with concurrent.futures.ProcessPoolExecutor(6) as pool:
        for s, mode, us, tile in pool.map(work, jobs, chunksize=1):
            res.setdefault(s, {})[mode] = (us, tile)
    print('%-26s %-20s %8s | %-11s %8s %7s | %-11s %8s %7s' % (
        'case', 'shape', 'default', '3.4', 'pred', 'err', 'corrected', 'pred', 'err'))
    rows = []
    for s, c in sorted(cs.items(), key=lambda kv: kv[1]['id']):
        r = res[s]
        plain, corr = r['plain'], r['corrected']
        rows.append(dict(id=c['id'], shape=list(s), default=c['default'], best=c['best'], old=c['old'],
                         plain=plain[0], plain_tile=plain[1], corrected=corr[0], corrected_tile=corr[1]))
        print('%-26s %-20s %8.1f | %-11s %8.1f %+6.1f%% | %-11s %8.1f %+6.1f%%' % (
            c['id'], str(s), c['default'], plain[1], plain[0], 100 * (plain[0] / c['default'] - 1),
            corr[1], corr[0], 100 * (corr[0] / c['default'] - 1)))
    for label, key in (('3.4 mapper', 'old'), ('tile-corrected', 'corrected')):
        e = [abs(r[key] / r['default'] - 1) for r in rows]
        print('%-16s vs library default: %d/%d within 10%%, MAPE %.1f%%, max %.1f%%' % (
            label, sum(x <= .1 for x in e), len(e), 100 * st.mean(e), 100 * max(e)))
    fb = [r for r in rows if r['best']]
    for label, key in (('3.4 mapper', 'old'), ('tile-corrected', 'corrected')):
        e = [abs(r[key] / r['best'] - 1) for r in fb]
        print('%-16s vs fastest kernel (%d sweep shapes): MAPE %.1f%%' % (label, len(fb), 100 * st.mean(e)))
    (ROOT / 'experiments/tile_calib_20261107/integrated_validation.json').write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
