"""Development-only explicit-tile kernel experiment, no installation required."""
import hashlib,json,pathlib,random,statistics,sys,time
import torch
from env_probe import kernel
sys.path.insert(0,'/workspace')
from benchmark_gemm import configure_precision,correctness_sample
r=pathlib.Path(__file__).resolve().parent;p=json.loads((r/'controlled_plan.json').read_text())
out=r/'controlled_measure.json';assert not out.exists()
flags,warnings=configure_precision(torch);torch.manual_seed(20261103)
data=dict(completed=False,plan_sha256=hashlib.sha256((r/'controlled_plan.json').read_bytes()).hexdigest(),cases=[],
          environment=dict(torch=torch.__version__,hip=torch.version.hip,device=str(torch.cuda.get_device_properties(0)),flags=flags),warnings=warnings)
def save():
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(out)
with torch.inference_mode():
 for row in p['cases']:
    m,n,k=row['shape'];a=torch.randn(m,k,device='cuda',dtype=torch.float16)*.1;b=torch.randn(k,n,device='cuda',dtype=torch.float16)*.1;c=torch.empty(m,n,device='cuda',dtype=torch.float16)
    configs=list(p['tiles'])+[None];random.Random(row['id']).shuffle(configs)
    for tile in configs:
        item=dict(id=row['id'],shape=row['shape'],tile=tile,status='pending')
        try:
            if tile:
                bm,bn,bk=tile;grid=((m+bm-1)//bm,(n+bn-1)//bn)
                def call():return kernel[grid](a,b,c,m,n,k,bm,bn,bk,num_warps=4,num_stages=1)
                obj=call();item['resources']=dict(registers_per_thread=obj.n_regs,spills=obj.n_spills,shared_bytes=obj.metadata.shared)
            else:
                def call():return torch.matmul(a,b,out=c)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(10):call()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            check=correctness_sample(torch,a,b,c,'fp16');item['correctness']=check
            if not check['passed']:raise RuntimeError('Numerical check failed')
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                for _ in range(16):call()
            for _ in range(3):graph.replay()
            torch.cuda.synchronize();rounds=[]
            begin,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            for _ in range(2):
                values=[]
                for __ in range(10):
                    begin.record();graph.replay();end.record();end.synchronize();values.append(begin.elapsed_time(end)*1000/16)
                rounds.append(dict(p50_us=statistics.median(values),samples_us=values))
            item.update(status='ok',rounds=rounds,measured_us=statistics.mean(x['p50_us'] for x in rounds))
            del graph
        except Exception as e:
            item.update(status='error',error=str(e))
        data['cases'].append(item);save();print(row['id'],tile,item['status'],item.get('measured_us'),flush=True)
    del a,b,c
data['completed']=True;save()
