import pathlib,subprocess
r=pathlib.Path('/workspace/pipeline_20261103')
for name in ['metadata_all','cache_graph_events']:
 print('START',name,flush=True)
 with (r/(name+'.log')).open('w') as f:subprocess.run(['python3','-B',name+'.py'],cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,flush=True)
