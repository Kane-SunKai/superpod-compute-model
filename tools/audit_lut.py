"""Audit bundled OS cycles, or a reference CSV passed via --source."""
import argparse
import csv
import json
from collections import defaultdict
from math import ceil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source', type=Path,
        default=Path(__file__).resolve().parents[1]
        / 'superpod_compute/data/array_cycles.csv',
    )
    args = parser.parse_args()
    keys = defaultdict(set)
    bad = []
    with args.source.open(encoding='utf-8', newline='') as stream:
        for row in csv.reader(stream):
            if len(row) < 7:
                continue
            m, n, k, ah, aw = map(int, row[:5])
            cycle = int(float(row[6]))
            keys[(m, n, k, ah, aw, row[5])].add(cycle)
            predicted = max(1, ceil(m / ah) * ceil(n / aw) * (k + ah + aw - 2) - 1)
            if row[5] != 'os' or predicted != cycle:
                bad.append([m, n, k, cycle, predicted, row[5]])
    conflicts = sum(len(values) > 1 for values in keys.values())
    print(json.dumps({
        'unique_keys': len(keys), 'conflicts': conflicts,
        'nonmatching_rows': len(bad), 'examples': bad[:10],
    }, indent=2))
    return int(bool(bad or conflicts or not keys))


if __name__ == '__main__':
    raise SystemExit(main())
