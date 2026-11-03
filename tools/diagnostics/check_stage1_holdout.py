"""Evaluate the frozen V1 service profile once on the predeclared holdout."""
import json
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from superpod_compute import Hardware, Policy, GemmRequest, Mapping
from tools.diagnostics.experimental_stage_services import ServiceProfile


def main():
    out = ROOT / 'results/stage1_20261103'
    profile = ServiceProfile.load(out / 'service_profile_v1.json')
    expected = json.loads((out / 'profile_v1_regression.json').read_text())['profile_sha256']
    assert profile.sha256 == expected, 'V1 changed after regression'
    baseline = json.loads((ROOT / 'results/regular50_20261103/frozen_plan.json').read_text())['baseline']
    h, p = Hardware(**baseline['hardware']), Policy(**baseline['policy'])
    predictions = json.loads((out / 'holdout_v1_predictions.json').read_text())
    measurements = json.loads((out / 'holdout_v1_measurements.json').read_text())
    assert predictions['completed'] and measurements['completed']
    rows = []
    # Finish every prediction before accessing measured durations.
    for row in predictions['cases']:
        req = GemmRequest(*row['shape'], rhs_storage='nk-contiguous-transposed')
        result = profile.evaluate(req, Mapping(**row['best']['mapping']), h, p)
        rows.append(dict(id=row['id'], shape=row['shape'], hardware_us=row['model_us'],
                         profile_us=result['duration_us'], selection=result['service_profile']['selection']))
    for row in rows:
        records = [r for r in measurements['cases'] if r['id'] == row['id']]
        assert len(records) == 3
        times = [r['timing']['p50_us'] for r in records]
        row['measured_us'] = st.mean(times)
        row['round_spread_pct'] = 100 * (max(times) - min(times)) / st.mean(times)
        for kind in ('hardware', 'profile'):
            row[kind + '_error_pct'] = 100 * (row[kind + '_us'] / row['measured_us'] - 1)
    summary = {}
    for kind in ('hardware', 'profile'):
        errors = [abs(r[kind + '_error_pct']) for r in rows]
        summary[kind] = dict(n=len(rows), passes10=sum(e <= 10 for e in errors),
                             mape=st.mean(errors), median=st.median(errors), max_error=max(errors))
    report = dict(profile_sha256=profile.sha256, completed=True, summary=summary, cases=rows,
                  note='These holdouts become observed after this report; future model iterations need fresh holdouts.')
    (out / 'profile_v1_holdout.json').write_text(json.dumps(report, indent=2) + '\n')
    for r in rows:
        print(r['id'], r['shape'], round(r['measured_us'], 3), round(r['profile_us'], 3), round(r['profile_error_pct'], 2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
