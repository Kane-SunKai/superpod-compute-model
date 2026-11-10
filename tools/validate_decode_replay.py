"""Score the replay model on the 24 held-out decode cases of the 3.4 workbook.

Reports kernel-selection accuracy against the library's recorded kernel and the
end-to-end error, next to the 3.4 mapper's error on the same cases.
"""
import json, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from superpod_compute.decode_replay import class_key, load, predict, service

SRC = ROOT / 'experiments/joint_gemm_20261105'


def main():
    prof = load(); rows = []
    files = sorted(SRC.glob('final_decode_validation_*.json')) + sorted(SRC.glob('final_fresh_decode_*.json'))
    for f in files:
        d = json.loads(f.read_text(encoding='utf-8')); md = (d.get('metadata') or [{}])[0]
        shape = d['shape']; meas = d['measured_us']
        r = predict(shape, prof)
        true_key = class_key(md['tile'], md['prefetch_global'], md['workgroup'])
        oracle = None
        if true_key in prof['classes']:
            oracle = service(shape, true_key, prof)['duration_s'] * 1e6
        rows.append(dict(id=d['id'], shape=shape, meas=meas, pred=r['duration_us'],
                         hit=r['kernel_class'] == true_key, oracle=oracle, old=d['final_us'],
                         sens=r.get('address_sensitive', False)))
    print('%-24s %-20s %8s %8s %7s %6s %8s %7s' % ('id', 'shape', 'meas', 'pred', 'err%', 'kernel', 'oracle%', '3.4 %'))
    for r in rows:
        o = '%+7.1f' % ((r['oracle'] / r['meas'] - 1) * 100) if r['oracle'] else '    n/a'
        print('%-24s %-20s %8.1f %8.1f %+7.1f %6s %8s %+7.1f%s' % (
            r['id'], str(r['shape']), r['meas'], r['pred'], (r['pred'] / r['meas'] - 1) * 100,
            'ok' if r['hit'] else 'miss', o, (r['old'] / r['meas'] - 1) * 100, '  addr' if r['sens'] else ''))
    report('24 held-out (3.4 workbook)', rows)
    if len(sys.argv) > 1:
        fresh = []
        for c in json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['cases']:
            if c.get('kernel_count') != 1 or not c.get('tile'):
                continue
            r = predict(c['shape'], prof)
            true_key = class_key(c['tile'], c['prefetch_global'], c['workgroup'])
            oracle = (service(c['shape'], true_key, prof)['duration_s'] * 1e6
                      if true_key in prof['classes'] else None)
            fresh.append(dict(meas=c['fast_us'], pred=r['duration_us'], hit=r['kernel_class'] == true_key,
                              oracle=oracle, old=None))
        report('12 fresh holdout (fast mode, never near any tuned shape)', fresh)


def report(label, rows):
    print('\n== %s ==' % label)
    e = [abs(r['pred'] / r['meas'] - 1) for r in rows]
    oe = [abs(r['oracle'] / r['meas'] - 1) for r in rows if r['oracle']]
    old = [abs(r['old'] / r['meas'] - 1) for r in rows if r['old'] is not None]
    print('kernel selection: %d/%d' % (sum(r['hit'] for r in rows), len(rows)))
    print('replay   : %d/%d within 10%%, MAPE %.1f%%' % (sum(x <= .1 for x in e), len(e), st.mean(e) * 100))
    if oe:
        print('oracle   : %d/%d within 10%%, MAPE %.1f%% (true kernel class given)' % (sum(x <= .1 for x in oe), len(oe), st.mean(oe) * 100))
    if old:
        print('3.4      : %d/%d within 10%%, MAPE %.1f%%' % (sum(x <= .1 for x in old), len(old), st.mean(old) * 100))


if __name__ == '__main__':
    main()
