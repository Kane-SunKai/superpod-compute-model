"""Offline controls: candidate ranking and capacity changes; no validation fit."""
import dataclasses,json,pathlib,time
from mapper import Hardware,Policy,search,retrieve,GemmModel
r=pathlib.Path(__file__).resolve().parent;p=json.loads((r/'frozen_plan.json').read_text());c=json.loads((r/'controlled_plan.json').read_text())
h=Hardware(**p['hardware']);policy=dataclasses.replace(Policy(**p['policy']),matrix_efficiency=p['frozen_cost']['matrix_efficiency'],calibration_hardware_sha256='')
bw=p['frozen_cost']['input_hbm_bandwidth_Bps'];rows=[]
for case in c['cases']:
 for tile in c['tiles']:
    value=search(case['shape'],h,policy,bw,[tile]);rows.append(dict(id=case['id'],shape=case['shape'],tile=tile,**value))
(r/'controlled_predictions.json').write_text(json.dumps(rows,indent=2)+'\n')
dev={tuple(x['shape']) for x in p['development_cases']};val={tuple(x['shape']) for x in p['cases']}
assert not dev&val
assert {tuple(x['shape']) for x in c['cases']} <= dev
assert len(val)==24 and len(dev)==78
probe_shape=[1024,8192,8192]
variants={'reference':h,'shared_16KiB':dataclasses.replace(h,shared_budget_bytes_per_core=16*1024),
          'registers_128KiB':dataclasses.replace(h,register_budget_bytes_per_core=128*1024),
          'cores_36':dataclasses.replace(h,cores=36),'L2_512KiB':dataclasses.replace(h,l2_bytes=512*1024)}
tiles=sorted({tuple(x['tile']) for x in p['development_cases'] if x['supported']})
hr=[]
for name,hw in variants.items():
 for tile in tiles:
    # Hardware-only comparison: no measured BW/eta from another target.
    pp=dataclasses.replace(policy,matrix_efficiency=1.,device_service_s=0.)
    v=search(probe_shape,hw,pp,hw.hbm_bandwidth_Bps,[tile])
    hr.append(dict(variant=name,tile=tile,feasible=v['feasible'],best=v['best']))
(r/'hardware_checks.json').write_text(json.dumps(dict(shape=probe_shape,scope='hypothetical capacity sensitivity; not measured hardware accuracy',cases=hr),indent=2)+'\n')
print('Disjointness OK; 60 controlled predictions; 5 hardware configurations x 17 templates')
