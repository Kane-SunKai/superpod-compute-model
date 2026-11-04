"""Freeze fresh Decode families before observing GPU timings."""
import datetime,hashlib,json,pathlib
from resident_decode import predict
r=pathlib.Path(__file__).resolve().parent
plan=json.loads((r/'acceptance_plan.json').read_text())
used={tuple(x['shape']) for prefix in ('development','validation','extension','acceptance') for x in json.loads((r/(prefix+'_plan.json')).read_text())['cases']}
families=[(7168,5632),(8960,6400),(9728,6912),(13312,9216),(20480,10240),(37888,8704)]
cases=[dict(id=f'final_{m}x{n}x{k}',shape=[m,n,k],group='decode',role='independent-validation') for n,k in families for m in (1,2,4,8,16,32)]
assert not used&{tuple(x['shape']) for x in cases}
assert not {(n,k) for m,n,k in used}&set(families)
plan.update(cases=cases,frozen_shapes_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),data_separation='All 6 N/K pairs unseen; earlier 124 shapes are development. No final measurements or dispatch metadata used.')
for name in ['final_plan.json','freeze_final.json']:
 if (r/name).exists():raise FileExistsError(name)
(r/'final_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
profile=json.loads((r/'wave_profile.json').read_text())
freeze=dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),profile=profile,
 sources={str(f.relative_to(r.parents[1])):hashlib.sha256(f.read_bytes()).hexdigest() for f in [r/'resident_decode.py',r/'calibrate_waves.py',*sorted((r.parents[1]/'superpod_compute').glob('*.py'))]},
 plan_sha256=hashlib.sha256((r/'final_plan.json').read_bytes()).hexdigest(),predictions=[dict(id=x['id'],**predict(x['shape'],profile)) for x in cases])
(r/'freeze_final.json').write_text(json.dumps(freeze,indent=2)+'\n')
print('Frozen 36 cases, 6 new families, before GPU measurement:',freeze['frozen_at_utc'])
