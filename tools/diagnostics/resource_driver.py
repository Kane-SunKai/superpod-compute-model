import sys,json,pathlib,torch
sys.path.insert(0,"/workspace/gemm_boundary_20261102")
from boundary_experiment import tensors
rows=json.loads(pathlib.Path("/workspace/gemm_boundary_20261102/measurements.json").read_text())["cases"]
with torch.inference_mode():
 for row in rows:
  a,b,c=tensors(torch,row)
  for _ in range(2):torch.mm(a,b,out=c)
  torch.cuda.synchronize()
  del a,b,c
print("resource driver completed")
