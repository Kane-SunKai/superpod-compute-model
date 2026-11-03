"""Coarse kernel metadata and paired cache diagnostic, separate from timing baseline."""
import json,pathlib,statistics,sys
import torch
sys.path.insert(0,'/workspace')
from benchmark_gemm import configure_precision
root=pathlib.Path('/workspace/pipeline_20261103');configure_precision(torch)
shapes=[(1,8192,8192),(1,28672,8192),(32,8192,8192),(32,28672,8192),
        (2048,28672,8192),(2048,8192,28672),(1,12288,6144),(768,6144,6144)]
rows=[]
with torch.inference_mode():
 for m,n,k in shapes:
  a=torch.randn(m,k,device='cuda',dtype=torch.float16)*.1
  b=torch.randn(k,n,device='cuda',dtype=torch.float16)*.1
  c=torch.empty(m,n,device='cuda',dtype=torch.float16)
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(30):torch.matmul(a,b,out=c)
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):torch.matmul(a,b,out=c)
  for _ in range(6):graph.replay()
  torch.cuda.synchronize()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   graph.replay();torch.cuda.synchronize()
  path=root/f'profile_{m}_{n}_{k}.json';prof.export_chrome_trace(str(path))
  events=json.loads(path.read_text())['traceEvents']
  kernels=[dict(name=e['name'],args=e.get('args',{})) for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
  rows.append(dict(shape=[m,n,k],kernels=kernels))
  print([m,n,k],[x['name'] for x in kernels],flush=True)
  del graph,a,b,c
(root/'execution_metadata.json').write_text(json.dumps(rows,indent=2)+'\n')
cache=[]
with torch.inference_mode():
 eviction=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
 for m,n,k in [(1,8192,8192),(1,28672,8192),(256,256,256),(1024,1024,1024),(1536,1536,1536)]:
  a=torch.randn(m,k,device='cuda',dtype=torch.float16)*.1;b=torch.randn(k,n,device='cuda',dtype=torch.float16)*.1;c=torch.empty(m,n,device='cuda',dtype=torch.float16)
  for _ in range(30):torch.matmul(a,b,out=c)
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):torch.matmul(a,b,out=c)
  s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  records=[]
  for r in (1,2):
   vals={'hot':[],'evict':[]}
   for _ in range(25):
    for mode in ('hot','evict'):
     if mode=='evict':eviction.fill_(1)
     else:g.replay()
     s.record();g.replay();e.record();e.synchronize();vals[mode].append(s.elapsed_time(e)*1000)
   records.append({mode:dict(p50_us=statistics.median(v),samples_us=v) for mode,v in vals.items()})
  cache.append(dict(shape=[m,n,k],rounds=records,protocol='single Graph GEMM; paired hot versus 64MiB fill before start event; diagnostic only'))
  print('cache',m,n,k,records[0]['hot']['p50_us'],records[0]['evict']['p50_us'],flush=True)
  del g,a,b,c
(root/'cache_diagnostic.json').write_text(json.dumps(cache,indent=2)+'\n')
