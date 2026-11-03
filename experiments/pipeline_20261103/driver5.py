import pathlib,subprocess
r=pathlib.Path('/workspace/pipeline_20261103')
text=(r/'metadata_all.py').read_text().replace("plans=[json.loads((root/'calibration_plan.json').read_text()),json.loads((root/'acceptance_plan.json').read_text())]","plans=[json.loads((root/'holdout_plan.json').read_text())]").replace("root/'metadata_all.json'","root/'holdout_metadata.json'").replace("root/'metadata_trace.json'","root/'holdout_metadata_trace.json'")
(r/'holdout_metadata.py').write_text(text)
for args,name in [(['python3','-B','measure_graph.py','--plan','holdout_plan.json','--output',str(r/'holdout_measure.json')],'holdout_measure'),(['python3','-B','holdout_metadata.py'],'holdout_metadata')]:
 print('START',name,flush=True)
 with (r/(name+'.log')).open('w') as f:subprocess.run(args,cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,flush=True)
