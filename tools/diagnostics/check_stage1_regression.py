import json,statistics as st,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from superpod_compute import Hardware,Policy,GemmRequest,Mapping
from tools.diagnostics.experimental_stage_services import ServiceProfile
OUT=ROOT/'results/stage1_20261103'
version=sys.argv[1] if len(sys.argv)>1 else 'v1'
profile=ServiceProfile.load(OUT/f'service_profile_{version}.json')
baseline=json.loads((ROOT/'results/regular50_20261103/frozen_plan.json').read_text())['baseline']
h=Hardware(**baseline['hardware']);p=Policy(**baseline['policy'])
base=json.loads((OUT/'regression_geometry_predictions.json').read_text());assert base['completed']
out=[]
# Predictions are complete before any acceptance duration is read below.
for row in base['cases']:
    req=GemmRequest(*row['shape'],rhs_storage='nk-contiguous-transposed')
    result=profile.evaluate(req,Mapping(**row['best']['mapping']),h,p)
    out.append(dict(id=row['id'],group=row['group'],shape=row['shape'],model_us=result['duration_us'],
                    hardware_model_us=row['model_us'],services=result['effective_software_services'],selection=result['service_profile']['selection']))
measured={r['id']:r for r in json.loads((ROOT/'results/regular50_20261103/comparison.json').read_text())['cases']}
for row in out:
    row['measured_us']=measured[row['id']]['linear_rotating']['us']
    row['error_pct']=100*(row['model_us']/row['measured_us']-1)
    print(row['id'],round(row['model_us'],3),round(row['measured_us'],3),round(row['error_pct'],2),row['selection']['sources'])
errors=[abs(r['error_pct']) for r in out]
report=dict(profile_sha256=profile.sha256,cases=out,n=len(out),passes10=sum(v<=10 for v in errors),mape=st.mean(errors),max_error=max(errors))
(OUT/f'profile_{version}_regression.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='cases'})
