import copy,dataclasses,json,pathlib,time
from mapper import predict,Hardware
r=pathlib.Path(__file__).resolve().parent;p=json.loads((r/'frozen_plan.json').read_text())
# Keep this functional check small; it is not a performance validation sample.
p['policy'].update(bm_values=[16],bn_values=[128],bk_values=[32],add_shape_candidate=False)
shape=[16,512,512];cache=r/'cache_checks'
t=time.perf_counter();a=predict(shape,p,cache);first=time.perf_counter()-t
t=time.perf_counter();b=predict(shape,p,cache);second=time.perf_counter()-t
assert b['cache_hit'] and a['methods']==b['methods']
h=Hardware(**p['hardware']);hh=dataclasses.replace(h,cores=36).to_dict()
c=predict(shape,p,cache,hh);assert c['signature_sha256']!=b['signature_sha256'] and not c['calibration_applied']
pp=copy.deepcopy(p);pp['retrieval']['distinct_templates']=3
d=predict(shape,pp,cache);assert d['signature_sha256']!=b['signature_sha256']
assert all(v is None or v['best']['capacity']['valid'] for v in c['methods'].values())
result=dict(first_seconds=first,cache_hit_seconds=second,exact_hit=True,hardware_change_invalidates=True,
            retrieval_change_invalidates=True,different_hardware_uses_no_BW1100_calibration=True,
            scope='Functional cache / transfer checks, not new-hardware accuracy')
(r/'cache_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
