"""GPU-continuous hot vs >32MiB rotating working set; diagnostic only."""
import json,math,pathlib,statistics,sys
import torch
sys.path.insert(0,'/workspace');from benchmark_gemm import configure_precision
configure_precision(torch);r=pathlib.Path('/workspace/pipeline_20261103');rows=[]
with torch.inference_mode():
 for m,n,k in [(1,8192,8192),(1,28672,8192),(256,256,256),(1024,1024,1024),(1536,1536,1536)]:
  footprint=2*(m*k+k*n+m*n);banks=max(2,2**math.ceil(math.log2(max(1,32*1024*1024/footprint))))
  arrays=[(torch.randn(m,k,device='cuda',dtype=torch.float16)*.1,torch.randn(k,n,device='cuda',dtype=torch.float16)*.1,torch.empty(m,n,device='cuda',dtype=torch.float16)) for _ in range(banks)]
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for a,b,c in arrays:
    for _ in range(2):torch.matmul(a,b,out=c)
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();graphs={};calls=64 if 2*m*n*k>1e9 else 256
  for mode in ('hot','rotate'):
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g,stream=stream):
    for i in range(calls):
     a,b,c=arrays[0 if mode=='hot' else i%banks];torch.matmul(a,b,out=c)
   graphs[mode]=g
  s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);rounds=[]
  for rd in (1,2):
   rec={}
   for mode,g in graphs.items():
    for _ in range(6):g.replay()
    torch.cuda.synchronize();vs=[]
    for _ in range(25):s.record();g.replay();e.record();e.synchronize();vs.append(s.elapsed_time(e)*1000/calls)
    rec[mode]={'p50_us':statistics.median(vs),'samples_us':vs}
   rounds.append(rec)
  rows.append(dict(shape=[m,n,k],banks=banks,working_set_bytes=footprint*banks,graph_calls=calls,rounds=rounds))
  (r/'cache_rotation.json').write_text(json.dumps(rows,indent=2)+'\n');print([m,n,k],banks,[(x['hot']['p50_us'],x['rotate']['p50_us']) for x in rounds],flush=True)
  del graphs,g,arrays,a,b,c
