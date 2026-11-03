"""Freeze disjoint development/validation data and retrieval policy."""
import json,pathlib,re,hashlib
r=pathlib.Path(__file__).resolve().parent;previous=r.parent/'pipeline_20261103';work=r.parents[2]
meta=json.loads((previous/'metadata_all.json').read_text())+json.loads((previous/'holdout_metadata.json').read_text())
dev=[]
for x in meta:
    row=dict(id=x['id'],shape=x['shape'],role='development',kernel_count=len(x['kernels']),supported=False)
    if len(x['kernels'])==1:
        name=x['kernels'][0]['name'];match=re.search(r'_MT(\d+)x(\d+)x(\d+)_',name);gsu=re.search(r'_GSU(\d+)_',name)
        if match and gsu and int(gsu.group(1))==1:
            bn,bm,bk=map(int,match.groups());row.update(supported=True,tile=[bm,bn,bk],source='observed BW1100 torch.matmul FP16 row-major dispatch')
    dev.append(row)
assert len(dev)==78 and len({tuple(x['shape']) for x in dev})==78
shapes=[('decode',1,10240,5120),('decode',4,10240,5120),('decode',16,10240,5120),('decode',32,20480,5120),('decode',2,5120,10240),('decode',8,30720,5120),
 ('prefill',512,10240,5120),('prefill',1024,10240,5120),('prefill',2048,10240,5120),('prefill',512,5120,20480),('prefill',1024,5120,20480),('prefill',768,20480,5120),('prefill',1536,20480,5120),('prefill',2048,5120,10240),
 ('square',576,576,576),('square',1152,1152,1152),('square',1792,1792,1792),('square',2560,2560,2560),
 ('long_k',640,1024,2048),('long_k',640,1024,6144),('long_k',640,1024,12288),
 ('aspect',4096,320,2048),('aspect',320,4096,2048),('aspect',1280,2560,1024)]
known={tuple(x['shape']) for x in dev}
old=json.loads((work/'compute_model/results/validation_20261029_v32/comparison_modes.json').read_text())
known.update(tuple(x['shape']) for x in old['cases'])
overlap=known&{tuple(x[1:]) for x in shapes}
if overlap:raise ValueError(f'Validation already observed: {overlap}')
base=json.loads((previous/'calibration_plan.json').read_text())
cal=json.loads((previous/'frozen_calibration.json').read_text())
plan={k:base[k] for k in ('hardware','policy','protocol')}
plan.update(cases=[dict(id=f'val_{g}_{m}x{n}x{k}',shape=[m,n,k],group=g,role='validation') for g,m,n,k in shapes],
 development_cases=dev,development_count=78,validation_count=24,
 retrieval={'metric':'sum absolute log2 ratio of M,N,K; equal weights; tie by source id','distinct_templates':5},
 frozen_cost={'matrix_efficiency':cal['matrix_efficiency'],'input_hbm_bandwidth_Bps':cal['hbm_bandwidth_Bps'],'calibration_source':'previous 22 development cases; reused without validation tuning'},
 baseline_commit='20d3cc3',data_rule='All old 44+22+12 are DEVELOPMENT here. Only the new 24 are independent validation. No validation trace used for selection.')
path=r/'frozen_plan.json'
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(plan,indent=2)+'\n');print('78 development / 24 independent validation',hashlib.sha256(path.read_bytes()).hexdigest())
print('templates',sorted(set(tuple(x['tile']) for x in dev if x['supported'])))
