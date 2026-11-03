import json,pathlib,sys,time
root=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(root.parents[1]))
from superpod_compute.api import predict
from superpod_compute.contracts import Hardware,Policy,GemmRequest
plan=json.loads((root/'calibration_plan.json').read_text());h=Hardware(**plan['hardware']);p=Policy(**plan['policy']);rows=[]
for row in plan['cases']:
 start=time.perf_counter();result=predict(GemmRequest(*row['shape']),h,policy=p)
 rows.append({**row,**result,'search_s':time.perf_counter()-start})
 (root/'calibration_predictions.json').write_text(json.dumps(dict(completed=False,cases=rows),indent=2)+'\n')
 print(row['id'],round(result['best']['duration_us'],3),'us',round(rows[-1]['search_s'],2),'s search',flush=True)
(root/'calibration_predictions.json').write_text(json.dumps(dict(completed=True,cases=rows),indent=2)+'\n')
