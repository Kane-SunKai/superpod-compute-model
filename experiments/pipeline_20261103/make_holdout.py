import hashlib,json,pathlib
r=pathlib.Path(__file__).resolve().parent
plan=json.loads((r/'calibration_plan.json').read_text());known={tuple(x['shape']) for f in ('calibration_plan.json','acceptance_plan.json') for x in json.loads((r/f).read_text())['cases']}
shapes=[('decode',2,16384,6144),('decode',16,32768,8192),('decode',4,24576,8192),('decode',32,16384,12288),
        ('prefill',640,8192,8192),('prefill',1536,8192,8192),('prefill',1024,16384,6144),('prefill',2048,6144,16384),
        ('square',448,448,448),('square',896,896,896),('long_k',384,1024,2048),('aspect',2048,384,2048)]
assert not known&{tuple(x[1:]) for x in shapes}
plan['cases']=[dict(id=f'hold_{g}_{m}x{n}x{k}',group=g,shape=[m,n,k],role='holdout') for g,m,n,k in shapes]
plan['parameters_sha256']=hashlib.sha256((r/'frozen_calibration.json').read_bytes()).hexdigest()
path=r/'holdout_plan.json'
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(plan,indent=2)+'\n');print('12 holdout shapes frozen before measuring; parameters',plan['parameters_sha256'])
