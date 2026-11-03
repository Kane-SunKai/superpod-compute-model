import json
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'results/stage1_20261103'
d=json.loads((p/'calibration_plan.json').read_text())
seen={tuple(r['shape']) for r in d['cases']}|{tuple(r['shape']) for r in d['holdout']}
original=json.loads((p.parent/'regular50_20261103/frozen_plan.json').read_text())
seen|={tuple(r['shape']) for r in original['cases']}
rows=[]
def add(group,m,n,k):
    assert (m,n,k) not in seen;seen.add((m,n,k))
    rows.append(dict(id=f'extra_{len(rows):02d}_{m}x{n}x{k}',group=group,shape=[m,n,k],role='calibration'))
for m in [3,5,12,24,48]:
    for n in [8192,28672]:add('small_m_services',m,n,8192)
for m in [448,768,1536,3072,4352]:add('matrix_services',m,8192,8192)
for m in [768,1536,2304]:
    add('matrix_services',m,28672,8192);add('matrix_services',m,8192,28672)
for k in [192,384,576,896,1536,3072,6144,12288,24576]:add('k_services',512,768,k)
for m in [1022,1026]:add('boundary_services',m,4096,4096)
assert len(rows)==32
d['cases']=rows
d['purpose']='Independent added sizes isolate small-M dispatch, NT GEMM services, and K-tail granularity. No original fifty or frozen holdout shapes are measured.'
target=p/'service_extension_plan.json'
if target.exists():assert json.loads(target.read_text())==d
else:target.write_text(json.dumps(d,indent=2)+'\n')
print('Frozen',len(rows),'extra calibration shapes')
