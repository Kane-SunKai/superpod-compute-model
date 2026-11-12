"""Compare the NN ground truth with the NT one, and the model against both."""
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V2 = ROOT / 'experiments/regression_100_v2_20261109'
PROBE = ROOT.parent / 'bw_probe'


def main():
    nn = {c['id']: c['default_us'] for c in
          json.loads((V2 / 'r100v2.json').read_text(encoding='utf-8'))['cases']}
    nt = {c['id']: c['default_us'] for c in
          json.loads((PROBE / 'nt100_20261111.json').read_text(encoding='utf-8'))['cases']}
    model = {r['id']: r['predicted_us'] for r in
             json.loads((V2 / 'predictions_bw2.json').read_text(encoding='utf-8'))}
    plan = {p['id']: p for p in json.loads((V2 / 'plan.json').read_text(encoding='utf-8'))}

    rows = []
    for rid, p in plan.items():
        if rid not in nt or rid not in nn:
            continue
        rows.append(dict(id=rid, cat=p['category'], shape=p['shape'],
                         nn=nn[rid], nt=nt[rid], model=model[rid],
                         ratio=nt[rid] / nn[rid],
                         e_nn=100 * (model[rid] / nn[rid] - 1),
                         e_nt=100 * (model[rid] / nt[rid] - 1)))

    def stats(key):
        v = [r[key] for r in rows]
        a = [abs(x) for x in v]
        return sum(x <= 10 for x in a), len(a), st.mean(a), max(a)

    print('model vs NN: %d/%d within 10%%, MAPE %.1f%%, max %.1f%%' % stats('e_nn'))
    print('model vs NT: %d/%d within 10%%, MAPE %.1f%%, max %.1f%%' % stats('e_nt'))
    rs = [r['ratio'] for r in rows]
    print('NT/NN ratio: median %.3f  min %.3f  max %.3f' % (st.median(rs), min(rs), max(rs)))
    print()
    print('%-14s %-20s %9s %9s %6s %9s %8s' % ('id', 'shape', 'NN', 'NT', 'NT/NN', 'model', 'vs NT'))
    for r in sorted(rows, key=lambda r: -abs(r['e_nt']))[:15]:
        print('%-14s %-20s %9.1f %9.1f %6.2f %9.1f %+7.1f%%' % (
            r['id'], 'x'.join(map(str, r['shape'])), r['nn'], r['nt'], r['ratio'],
            r['model'], r['e_nt']))
    print()
    for cat in ('decode_normal', 'decode_largeN', 'prefill', 'gpt7b', 'gpt13b'):
        rs = [r for r in rows if r['cat'] == cat]
        if not rs:
            continue
        a = [abs(r['e_nt']) for r in rs]
        b = [abs(r['e_nn']) for r in rs]
        print('%-14s n=%2d  vs NN %2d ok MAPE %5.1f   vs NT %2d ok MAPE %5.1f   NT/NN median %.2f' % (
            cat, len(rs), sum(x <= 10 for x in b), st.mean(b),
            sum(x <= 10 for x in a), st.mean(a), st.median([r['ratio'] for r in rs])))


if __name__ == '__main__':
    main()
