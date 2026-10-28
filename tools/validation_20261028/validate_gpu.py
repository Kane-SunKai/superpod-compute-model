"""Predeclared shapes, final summaries only; no profiler or per-iteration logs."""
import json, math, random, statistics, time
from pathlib import Path
import torch
from benchmark_gemm import configure_precision, correctness_sample

ROOT = Path('/workspace/compute_validation_20261028')

def cases():
    result = []
    def add(group, shapes, role='validation'):
        for shape in shapes:
            result.append(dict(id=f'{group}_{len(result):02d}', group=group, role=role, shape=list(shape)))
    add('calibration', [(x,x,x) for x in (8,16,24,32,48,64)] + [(1,32,32),(4,32,64),(32,4,64),(16,64,32),(64,16,32),(32,64,16)], 'calibration')
    add('square', [(x,x,x) for x in (96,128,192,256,384,512,768,1024,1536,2048,3072,4096)])
    add('decode', [(m,n,k) for m,n,k in [(1,4096,4096),(1,8192,8192),(2,4096,4096),(4,4096,4096),(8,4096,4096),(16,4096,4096),(32,4096,4096),(1,11008,4096),(8,11008,4096),(1,4096,11008)]])
    add('prefill', [(m,n,k) for m,n,k in [(3072,2304,1024),(512,4096,4096),(1024,4096,4096),(2048,4096,4096),(1024,11008,4096),(1024,4096,11008),(4096,1024,4096),(4096,4096,512)]])
    add('k_sweep', [(512,768,k) for k in (16,32,64,128,256,512,1024,4096,8192,16384)])
    add('irregular', [(130,97,270),(127,129,255),(513,767,1025),(1000,1000,1000),(1537,1153,1023),(257,4093,769),(4093,257,769),(33,8191,2047),(2047,33,8191),(65,129,257)])
    add('tall_wide', [(8192,64,1024),(64,8192,1024),(4096,128,2048),(128,4096,2048),(8192,256,128),(256,8192,128),(2048,512,8192),(512,2048,8192),(256,384,16),(384,256,16)])
    assert len({tuple(x['shape']) for x in result}) == len(result)
    return result

def stats(v):
    s=sorted(v)
    return dict(p50_us=statistics.median(s),p10_us=s[int(.1*(len(s)-1))],p90_us=s[math.ceil(.9*len(s))-1],count=len(s))

def measure(fn, count, divisor=1):
    start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record();end.record();end.synchronize()
    values=[]
    for _ in range(count):
        start.record();fn();end.record();end.synchronize()
        values.append(start.elapsed_time(end)*1000/divisor)
    return stats(values)

def main():
    torch.cuda.set_device(0);torch.manual_seed(20261028)
    configure_precision(torch)
    plan=cases()
    # Keep the predeclared order and a reproducible randomized measurement order.
    order=list(range(len(plan)));random.Random(20261028).shuffle(order)
    report=dict(protocol='FP16 C=AB; warmed repeated same-address HIP graph, event time/replay GEMMs. Eager event timing reported separately. No cold-cache claim.',
                torch=torch.__version__,hip=torch.version.hip,device=str(torch.cuda.get_device_properties(0)),
                cases=plan,completed=False,calibration_cases=12,validation_cases=len(plan)-12)
    root=ROOT.resolve();root.mkdir(exist_ok=True)
    def save():
        (root/'final_measurements.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    start_time=time.monotonic()
    with torch.inference_mode():
        for idx in order:
            row=plan[idx];m,n,k=row['shape']
            a=torch.randn(m,k,device='cuda',dtype=torch.float16)*.1
            b=torch.randn(k,n,device='cuda',dtype=torch.float16)*.1
            c=torch.empty(m,n,device='cuda',dtype=torch.float16)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(30):torch.mm(a,b,out=c)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            row['eager_event']=measure(lambda:torch.mm(a,b,out=c),20)
            batch=64 if 2*m*n*k>1e9 else 256
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):
                for _ in range(batch):torch.mm(a,b,out=c)
            for _ in range(6):g.replay()
            torch.cuda.synchronize()
            row['graph_batch']=batch
            row['graph_round1']=measure(g.replay,25,batch)
            row['graph_round2']=measure(g.replay,25,batch)
            row['correctness']=correctness_sample(torch,a,b,c,'fp16')
            if not row['correctness']['passed']:raise RuntimeError('correctness failure '+row['id'])
            print(row['id'],row['shape'],round(row['graph_round2']['p50_us'],3),flush=True)
            del g,a,b,c
            save()
    report['completed']=True;report['wall_seconds']=time.monotonic()-start_time;save()

if __name__=='__main__':main()
