"""Record only coarse library dispatch identity; no disassembly or timed fitting."""
import json,pathlib,sys
import torch
sys.path.insert(0,'/workspace');from benchmark_gemm import configure_precision
configure_precision(torch);root=pathlib.Path('/workspace/pipeline_20261103');rows=[]
plans=[json.loads((root/'calibration_plan.json').read_text()),json.loads((root/'acceptance_plan.json').read_text())]
with torch.inference_mode():
 for case in [c for p in plans for c in p['cases']]:
  m,n,k=case['shape'];a=torch.randn(m,k,device='cuda',dtype=torch.float16)*.1;b=torch.randn(k,n,device='cuda',dtype=torch.float16)*.1;c=torch.empty(m,n,device='cuda',dtype=torch.float16)
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(5):torch.matmul(a,b,out=c)
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=stream):torch.matmul(a,b,out=c)
  for _ in range(3):g.replay()
  torch.cuda.synchronize()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:g.replay();torch.cuda.synchronize()
  path=root/'metadata_trace.json';prof.export_chrome_trace(str(path));events=json.loads(path.read_text())['traceEvents']
  ks=[dict(name=e['name'],args=e.get('args',{})) for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
  rows.append({**case,'kernels':ks});(root/'metadata_all.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(len(rows),case['id'],[x['name'].split('_MT')[1].split('_SE')[0] if '_MT' in x['name'] else x['name'] for x in ks],flush=True)
  del g,a,b,c
