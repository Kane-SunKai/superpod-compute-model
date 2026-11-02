"""One-variable GEMM experiments; measurements never used to select the plan.

Run inside the existing BW1100 container, with /root/sk mounted as /workspace.
All output paths must resolve below /workspace (or /root/sk on the host).
No package installation, clock changes, or profiler-duration accuracy claims.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time


def plan():
    rows = []
    def add(group, shape, layout='contiguous', role='diagnosis'):
        ident = group + '_' + 'x'.join(map(str, shape)) + '_' + layout
        rows.append(dict(id=ident, group=group, shape=list(shape), layout=layout, role=role))
    original = [(130,97,270),(127,129,255),(513,767,1025),(1000,1000,1000),
                (1537,1153,1023),(257,4093,769),(4093,257,769),
                (33,8191,2047),(2047,33,8191),(65,129,257)]
    for shape in original:
        add('historical_irregular', shape)
    for shape in [(3072,2304,1024),(512,768,1024),(1536,1152,1024),
                  (1024,1024,1024),(4096,4096,4096),(1,8192,8192),
                  (512,768,16384)]:
        add('historical_control', shape)
    for axis in range(3):
        for delta in [-1, 1, 8]:
            shape = [512,768,1024]; shape[axis] += delta
            add('axis_' + 'MNK'[axis], shape)
    for delta in [-32,-16,-8,-4,-2,2,4,16,32]:
        add('k_tail', [512,768,1024+delta])
    for axis in range(3):
        for delta in [-1,1]:
            shape=[1536,1152,1024];shape[axis]+=delta
            add('large_axis_'+'MNK'[axis],shape)
    # Same logical M,N,K and operation, only physical row strides change.
    for shape in [original[i] for i in (0,2,4,5,7,8)]:
        for layout in ['pad_a16','pad_b16','pad_c16','pad_abc16']:
            add('stride_intervention', shape, layout)
    # Predeclared unseen shapes: do not use these times to tune parameters.
    for shape in [(129,131,259),(511,769,1021),(1533,1159,1009),
                  (259,4091,773),(4091,259,773),(35,8189,2051),
                  (2051,35,8189),(999,1003,997),(67,133,263),
                  (513,769,1031),(777,333,2049),(256,384,33)]:
        add('heldout', shape, role='validation')
    assert len({r['id'] for r in rows}) == len(rows)
    return rows


def stats(values):
    v=sorted(values);n=len(v)
    return dict(p50_us=statistics.median(v),p10_us=v[int(.1*(n-1))],
                p90_us=v[math.ceil(.9*n)-1],samples=n)


def measure(torch, fn, count, divisor):
    start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    values=[]
    start.record();end.record();end.synchronize()
    for _ in range(count):
        start.record();fn();end.record();end.synchronize()
        values.append(start.elapsed_time(end)*1000/divisor)
    return stats(values)


def tensors(torch, row):
    m,n,k=row['shape'];layout=row['layout']
    padded = set() if layout=='contiguous' else set(layout[4:-2])
    lda=math.ceil(k/16)*16 if 'a' in padded else k
    ldb=math.ceil(n/16)*16 if 'b' in padded else n
    ldc=math.ceil(n/16)*16 if 'c' in padded else n
    a=(torch.randn(m,lda,device='cuda',dtype=torch.float16)*.1)[:,:k]
    b=(torch.randn(k,ldb,device='cuda',dtype=torch.float16)*.1)[:,:n]
    c=torch.empty(m,ldc,device='cuda',dtype=torch.float16)[:,:n]
    return a,b,c


def output_path(path):
    p=Path(path).resolve()
    if not any(p.is_relative_to(Path(root)) for root in ('/workspace','/root/sk')):
        raise ValueError('Experiment output must stay in /root/sk or its /workspace mount')
    return p


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['plan','measure','profile'],required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--exclude-holdout',action='store_true')
    args=ap.parse_args();target=output_path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    rows=plan()
    if args.exclude_holdout:rows=[r for r in rows if r['role']!='validation']
    if target.exists():raise FileExistsError('Preserve previous evidence: '+str(target))
    base=dict(plan=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              seed=20261102,completed=False,cases=[],phase=args.phase,
              protocol='FP16 torch.mm C=AB; same-address warm HIP Graph; 30 warmup, 6 graph warmup, 2 rounds x 25 Event measurements; profiler names collected separately, never used as latency')
    def save():
        temp=target.with_suffix(target.suffix+'.tmp')
        temp.write_text(json.dumps(base,indent=2,allow_nan=False)+'\n')
        temp.replace(target)
    if args.phase=='plan':base['completed']=True;save();return
    import torch
    sys.path.insert(0,'/workspace')
    from benchmark_gemm import configure_precision,correctness_sample
    configure_precision(torch);torch.cuda.set_device(0);torch.manual_seed(20261102)
    base['environment']=dict(torch=torch.__version__,hip=torch.version.hip,
                             gpu=str(torch.cuda.get_device_properties(0)))
    if not torch.cuda.is_available():raise RuntimeError('No GPU; never fall back to CPU')
    order=list(rows);random.Random(20261102).shuffle(order)
    start_time=time.monotonic();save()
    with torch.inference_mode():
        for row in order:
            a,b,c=tensors(torch,row);record=dict(row)
            record['strides']=[list(t.stride()) for t in (a,b,c)]
            record['pointer_modulo_bytes']={str(q):[t.data_ptr()%q for t in (a,b,c)] for q in (4,16,64,256)}
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(30):torch.mm(a,b,out=c)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            if args.phase=='profile':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
                    for _ in range(2):torch.mm(a,b,out=c)
                    torch.cuda.synchronize()
                record['kernels']=dict(Counter(e.name for e in p.events() if 'CUDA' in str(e.device_type)))
                del p
            else:
                m,n,k=row['shape'];batch=64 if 2*m*n*k>1e9 else 256
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g,stream=stream):
                    for _ in range(batch):torch.mm(a,b,out=c)
                for _ in range(6):g.replay()
                torch.cuda.synchronize()
                record['graph_batch']=batch
                record['graph_round1']=measure(torch,g.replay,25,batch)
                record['graph_round2']=measure(torch,g.replay,25,batch)
                record['correctness']=correctness_sample(torch,a,b,c,'fp16')
                if not record['correctness']['passed']:raise RuntimeError('Incorrect result: '+row['id'])
                del g
            base['cases'].append(record);base['wall_seconds']=time.monotonic()-start_time;save()
            value=record.get('graph_round2',{}).get('p50_us',len(record.get('kernels',{})))
            print(len(base['cases']),len(rows),row['id'],round(value,4),flush=True)
            del a,b,c
    base['completed']=True;save()


if __name__=='__main__':main()
