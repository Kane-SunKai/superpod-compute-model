"""Cache diagnostic with timing events inside replay, avoiding CPU gaps."""
import json,pathlib,statistics,sys
import torch
sys.path.insert(0,'/workspace');from benchmark_gemm import configure_precision
configure_precision(torch);root=pathlib.Path('/workspace/pipeline_20261103');rows=[]
with torch.inference_mode():
 eviction=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
 for m,n,k in [(1,8192,8192),(1,28672,8192),(256,256,256),(1024,1024,1024),(1536,1536,1536)]:
  a=torch.randn(m,k,device='cuda',dtype=torch.float16)*.1;b=torch.randn(k,n,device='cuda',dtype=torch.float16)*.1;c=torch.empty(m,n,device='cuda',dtype=torch.float16)
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   eviction.fill_(1)
   for _ in range(30):torch.matmul(a,b,out=c)
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  events=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in range(16)]
  for s,e in events:s.record();e.record()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=stream):
   for idx,(s,e) in enumerate(events):
    if idx%2:eviction.fill_(1)
    else:torch.matmul(a,b,out=c)
    s.record();torch.matmul(a,b,out=c);e.record()
  rounds=[]
  for rd in (1,2):
   for _ in range(6):g.replay()
   torch.cuda.synchronize();vals={'hot':[],'evict':[]}
   for _ in range(25):
    g.replay();torch.cuda.synchronize()
    for idx,(s,e) in enumerate(events):vals['evict' if idx%2 else 'hot'].append(s.elapsed_time(e)*1000)
   rounds.append({mode:dict(p50_us=statistics.median(v),samples_us=v) for mode,v in vals.items()})
  rows.append(dict(shape=[m,n,k],rounds=rounds,protocol='external GPU events captured INSIDE Graph, alternating hot and 64MiB eviction; diagnostic only'))
  (root/'cache_graph_events.json').write_text(json.dumps(rows,indent=2)+'\n')
  print([m,n,k],[(x['hot']['p50_us'],x['evict']['p50_us']) for x in rounds],flush=True)
  del g,a,b,c,events
