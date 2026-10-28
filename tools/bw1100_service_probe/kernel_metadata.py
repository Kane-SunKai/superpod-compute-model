"""Collect chosen kernel names in memory; no trace files or timing fitting."""
import collections,json,pathlib,sys
import torch
sys.path.insert(0,'/workspace')
from benchmark_gemm import configure_precision
root=pathlib.Path('/workspace/compute_mmac_20261028')
cases=json.loads((root.parent/'compute_validation_20261028/final_measurements.json').read_text())['cases']
configure_precision(torch);torch.cuda.set_device(0)
output=[]
with torch.inference_mode():
    for case in cases:
        if case['role']!='validation':continue
        m,n,k=case['shape'];a=torch.randn(m,k,device='cuda',dtype=torch.float16);b=torch.randn(k,n,device='cuda',dtype=torch.float16);c=torch.empty(m,n,device='cuda',dtype=torch.float16)
        for _ in range(4):torch.mm(a,b,out=c)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
            for _ in range(2):torch.mm(a,b,out=c)
            torch.cuda.synchronize()
        kernels=collections.Counter(e.name for e in p.events() if 'CUDA' in str(e.device_type))
        output.append({'id':case['id'],'shape':case['shape'],'kernels':dict(kernels)})
        del p,a,b,c
    assert len(output)==60 and all(x['kernels'] for x in output)
    (root/'kernel_metadata_final.json').write_text(json.dumps({'cases':output,'scope':'Kernel names only; profiler timings excluded from accuracy comparisons'},indent=2)+'\n')
print('kernel metadata cases:',len(output))
