"""Paired read-counter evidence; preserve units and do not time the profiler."""
import argparse
import hashlib
import json
from pathlib import Path

from summarize_pmc import summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--measurements', type=Path, required=True)
    ap.add_argument('--pmc-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    measured = {r['id']: r for r in json.loads(args.measurements.read_text())['cases']}
    counters = ('TA_TA_BUSY', 'TCC_EA_RDREQ', 'TCC_EA_RDREQ_32B',
                'TCC_EA1_RDREQ', 'TCC_EA1_RDREQ_32B', 'TCP_TCP_TA_DATA_STALL_CYCLES')
    cases = []
    for path in sorted(args.pmc_dir.glob('*/report.csv')):
        status = json.loads((path.parent/'status.json').read_text())
        if status['returncode']:
            raise ValueError(f'Failed acquisition: {path}')
        cases.append(summarize(path, measured[path.parent.name], counters))
    if len(cases) != 6:
        raise ValueError('Expected all six paired conditions')
    by_id = {r['id']: r for r in cases}
    pairs = []
    for k in (1023, 1024, 1025):
        a = by_id[f'k_stride_isolation_1536x1152x{k}_contiguous']
        b = by_id[f'k_stride_isolation_1536x1152x{k}_pad_a16']
        ratios = {}
        for key, value in a['counters'].items():
            ratios[key] = b['counters'][key]['median']/value['median'] if value['median'] else None
        pairs.append({'K': k, 'reference': a['id'], 'variant': b['id'],
                      'same_kernel': a['kernel'] == b['kernel'],
                      'same_geometry': a['geometry'] == b['geometry'],
                      'graph_time_ratio': b['graph_us']/a['graph_us'], 'counter_ratios': ratios})
    report = {'cases': cases, 'pairs': pairs,
              'measurement_sha256': hashlib.sha256(args.measurements.read_bytes()).hexdigest(),
              'scope': 'Three warmed isolated calls for counters; separate Graph timing. TA busy and TCC EA requests retain vendor counter units; no conversion into bandwidth, cycles per instruction, or HBM bytes.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for row in pairs:
        print(row['K'], 'time ratio', round(row['graph_time_ratio'],4),
              'TA busy ratio', round(row['counter_ratios']['TA_TA_BUSY_instance_sum'],4),
              'EA read request ratio', round(row['counter_ratios']['TCC_EA_RDREQ_instance_sum'],4))


if __name__ == '__main__':
    main()
