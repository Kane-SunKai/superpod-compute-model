import json,sys,pathlib,torch
sys.path.insert(0,"/workspace")
from benchmark_gemm import configure_precision
sys.path.insert(0,"/workspace/gemm_boundary_20261102")
from boundary_experiment import tensors
root=pathlib.Path("/workspace/gemm_boundary_20261102")
rows=json.loads((root/"measurements.json").read_text())["cases"]
selected=[r for r in rows if r["layout"]=="contiguous" and (r["group"].startswith("large_axis") or r["shape"] in [[1536,1152,1024],[257,4093,769],[3072,2304,1024],[130,97,270],[512,768,16384]])]
configure_precision(torch);torch.cuda.set_device(0)
output=[]
with torch.inference_mode():
 for r in selected:
  a,b,c=tensors(torch,r)
  for _ in range(5):torch.mm(a,b,out=c)
  torch.cuda.synchronize()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   torch.mm(a,b,out=c);torch.cuda.synchronize()
  path=root/(r["id"]+"_geometry_trace.json")
  prof.export_chrome_trace(str(path))
  events=json.loads(path.read_text())["traceEvents"]
  output.append({"id":r["id"],"shape":r["shape"],"kernels":[{"name":e["name"],"args":e.get("args",{})} for e in events if e.get("cat")=="kernel"]})
  del prof,a,b,c
(root/"geometry.json").write_text(json.dumps(output,indent=2))
print("geometry cases",len(output))
