import csv
import json
from collections import defaultdict
from math import ceil
from pathlib import Path

source = Path(__file__).parents[2] / 'llmcompass-model-report/reference_runtime/systolic_array_model/look_up_table_16_16.csv'
keys = defaultdict(set)
bad = []
for row in csv.reader(source.open()):
    if len(row) < 7:
        continue
    m, n, k, ah, aw = map(int, row[:5])
    cycle = int(float(row[6]))
    keys[(m,n,k,ah,aw,row[5])].add(cycle)
    predicted = ceil(m/ah)*ceil(n/aw)*(k+ah+aw-2)-1
    if predicted != cycle:
        bad.append([m,n,k,cycle,predicted])
print(json.dumps({'unique_keys':len(keys),'conflicts':sum(len(v)>1 for v in keys.values()),'nonmatching_rows':len(bad),'examples':bad[:10]}, indent=2))
