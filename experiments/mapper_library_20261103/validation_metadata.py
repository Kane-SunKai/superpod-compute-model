"""Post-measurement dispatch evidence, explicitly excluded from predictions."""
import json,pathlib,re,sys
import torch
sys.path.insert(0,'/workspace');from benchmark_gemm import configure_precision
configure_precision(torch);r=pathlib.Path(__file__).resolve().parent
assert json.loads((r/'validation_measure.json').read_text())['completed']
rows=[]
with torch.inference_mode():
 for case in json.loads((r/'frozen_plan.json').read_text())['cases']:
    m,n,k=case['shape'];a=torch.randn(m,k,device='cuda',dtype=torch.float16)*.1;b=torch.randn(k,n,device='cuda',dtype=torch.float16)*.1;c=torch.empty(m,n,device='cuda',dtype=torch.float16)
    s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):torch.matmul(a,b,out=c)
    torch.cuda.current_stream().wait_stream(s);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=s):torch.matmul(a,b,out=c)
    for _ in range(3):g.replay()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        g.replay();torch.cuda.synchronize()
    path=r/'posthoc_trace.json';prof.export_chrome_trace(str(path));events=json.loads(path.read_text())['traceEvents']
    kernels=[e['name'] for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
    row=dict(id=case['id'],shape=case['shape'],kernel_count=len(kernels),supported=False)
    if len(kernels)==1:
        mt=re.search(r'_MT(\d+)x(\d+)x(\d+)_',kernels[0]);gsu=re.search(r'_GSU(\d+)_',kernels[0])
        if mt and gsu:
            bn,bm,bk=map(int,mt.groups());row.update(tile=[bm,bn,bk],split_k=int(gsu.group(1)),supported=int(gsu.group(1))==1)
    rows.append(row);(r/'posthoc_metadata.json').write_text(json.dumps(rows,indent=2)+'\n');print('posthoc',row,flush=True)
    del g,a,b,c
