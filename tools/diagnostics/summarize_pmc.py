"""Join counter observations to independent Graph measurements; never time PMC.

Counters stay in their reported units. Summing TCC instances is useful for
matched comparisons, but is NOT a conversion to bytes or HBM traffic.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


SINGLE_COUNTERS = (
    'SQ_INSTS_VALU', 'SQ_INSTS_VMEM_RD', 'SQ_INSTS_VMEM_WR',
    'SQ_INSTS_LDS', 'SQ_WAIT_INST_LDS', 'SQ_LDS_BANK_CONFLICT',
)
INSTANCE_COUNTERS = ('TCC_HIT', 'TCC_MISS', 'TCP_TCP_TA_DATA_STALL_CYCLES')
GEOMETRY = ('grd', 'wgr', 'lds', 'arch_vgpr', 'accum_vgpr', 'wave_size')


def summarize(csv_path, measurement):
    with csv_path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 6:
        raise ValueError(f'{csv_path}: expected 3 warmups and 3 observations')
    observed = rows[3:]
    names = {r['KernelName'] for r in rows}
    if len(names) != 1:
        raise ValueError(f'{csv_path}: mixed kernels')
    metadata = {}
    for field in GEOMETRY:
        values = {int(r[field]) for r in rows}
        if len(values) != 1:
            raise ValueError(f'{csv_path}: changing {field}')
        metadata[field] = values.pop()
    if metadata['grd'] % metadata['wgr']:
        raise ValueError('HIP global work-items must divide by workgroup size')
    metadata['workgroups'] = metadata['grd'] // metadata['wgr']
    samples = {k: [float(r[k]) for r in observed] for k in SINGLE_COUNTERS}
    for prefix in INSTANCE_COUNTERS:
        columns = [k for k in observed[0] if k.startswith(prefix + '[')]
        if not columns:
            raise ValueError(f'Missing counter instances: {prefix}')
        samples[prefix + '_instance_sum'] = [sum(float(r[k]) for k in columns) for r in observed]
    samples['TCC_HIT_plus_MISS_instance_sum'] = [
        a + b for a, b in zip(samples['TCC_HIT_instance_sum'], samples['TCC_MISS_instance_sum'])
    ]
    counters = {k: {'median': statistics.median(v), 'min': min(v), 'max': max(v)} for k, v in samples.items()}
    return {
        'id': measurement['id'], 'shape': measurement['shape'], 'layout': measurement['layout'],
        'graph_us': statistics.mean(measurement[f'graph_round{i}']['p50_us'] for i in (1, 2)),
        'kernel': names.pop(), 'geometry': metadata, 'counters': counters,
        'rows': len(rows), 'observations': len(observed),
        'source_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--measurements', type=Path, required=True)
    ap.add_argument('--pmc-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    measured = {r['id']: r for r in json.loads(args.measurements.read_text())['cases']}
    cases = []
    for path in sorted(args.pmc_dir.glob('*/report.csv')):
        status = json.loads((path.parent / 'status.json').read_text())
        if status['returncode'] != 0:
            raise ValueError(f'Failed acquisition: {path}')
        cases.append(summarize(path, measured[path.parent.name]))
    by_id = {r['id']: r for r in cases}
    reference = 'historical_control_1536x1152x1024_contiguous'
    pairs = [(reference, r['id']) for r in cases if r['id'] != reference]
    pairs.append(('historical_irregular_1537x1153x1023_contiguous', 'stride_intervention_1537x1153x1023_pad_a16'))
    comparisons = []
    for old, new in pairs:
        a, b = by_id[old], by_id[new]
        ratios = {}
        for field in a['counters']:
            denominator = a['counters'][field]['median']
            ratios[field] = b['counters'][field]['median'] / denominator if denominator else None
        comparisons.append({
            'reference': old, 'variant': new, 'same_kernel': a['kernel'] == b['kernel'],
            'same_geometry_and_resources': a['geometry'] == b['geometry'],
            'graph_time_ratio': b['graph_us'] / a['graph_us'], 'counter_ratios': ratios,
        })
    output = {
        'scope': 'Last 3 of 6 isolated synchronous GEMMs for counters. Graph timing comes from a separate non-profiled measurement. Counters are mechanism evidence, not byte traffic, service time, or a prediction input. Cache-counter changes do not establish HBM traffic changes.',
        'measurement_sha256': hashlib.sha256(args.measurements.read_bytes()).hexdigest(),
        'cases': cases, 'comparisons': comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    for r in comparisons:
        print(r['variant'], 'vs', r['reference'], 'time', round(r['graph_time_ratio'], 4),
              'TCC hit+miss', round(r['counter_ratios']['TCC_HIT_plus_MISS_instance_sum'], 4))


if __name__ == '__main__':
    main()
