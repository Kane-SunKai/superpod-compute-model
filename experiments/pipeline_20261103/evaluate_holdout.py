import hashlib,json,pathlib,re,statistics,time
from predict_with_plan import predict_plan,Hardware,Policy
r=pathlib.Path(__file__).resolve().parent;plan=json.loads((r/'holdout_plan.json').read_text());cal=json.loads((r/'frozen_calibration.json').read_text())
assert hashlib.sha256((r/'frozen_calibration.json').read_bytes()).hexdigest()==plan['parameters_sha256']
h=Hardware(**plan['hardware']);p=Policy(**plan['policy']);meta=json.loads((r/'holdout_metadata.json').read_text());measure={x['id']:x for x in json.loads((r/'holdout_measure.json').read_text())['cases']};rows=[]
for case in meta:
 row={k:case[k] for k in ('id','group','shape','role')};row['measured_us']=measure[case['id']]['measured_graph_us'];ks=case['kernels']
 if len(ks)!=1 or not re.search(r'_MT(\d+)x(\d+)x(\d+)_',ks[0]['name']):row.update(supported=False,reason='multiple or unknown kernel')
 else:
  name=ks[0]['name'];bn,bm,bk=map(int,re.search(r'_MT(\d+)x(\d+)x(\d+)_',name).groups());gsu=int(re.search(r'_GSU(\d+)_',name).group(1));ex=dict(layout='row-major-contiguous',bm=bm,bn=bn,bk=bk,split_k=gsu,source=name)
  try:
   start=time.perf_counter();v=predict_plan(case['shape'],ex,h,p,cal);row.update(supported=True,result=v,wall_s=time.perf_counter()-start)
   err=v['best']['duration_us']/row['measured_us']-1;row.update(signed_error_pct=err*100,absolute_error_pct=abs(err)*100,pass10=abs(err)<=.1)
  except (NotImplementedError,ValueError) as e:row.update(supported=False,reason=str(e))
 rows.append(row);print(row['id'],round(row.get('absolute_error_pct',-1),2),row.get('reason',''),flush=True)
valid=[x for x in rows if x['supported']]
summary=dict(total=len(rows),supported=len(valid),pass10=sum(x['pass10'] for x in valid),mape=statistics.mean(x['absolute_error_pct'] for x in valid))
(r/'holdout_results.json').write_text(json.dumps(dict(parameters_sha256=plan['parameters_sha256'],summary=summary,cases=rows),indent=2)+'\n');print(summary)
