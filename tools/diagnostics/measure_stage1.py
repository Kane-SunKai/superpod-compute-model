"""Only AIOB-style X@W.T; profiler metadata separate from Graph timing."""
import argparse,gc,hashlib,json,math,random,statistics,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--phase',choices=['measure','profile','equivalence'],default='measure')
    args=ap.parse_args();target=Path(args.output).resolve()
    if not any(target.is_relative_to(Path(v)) for v in ['/workspace','/root/sk']):raise ValueError('Output root')
    if target.exists():raise FileExistsError(target)
    plan=json.loads(Path(args.plan).read_text());protocol=plan['protocol']
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,'/workspace')
    from benchmark_gemm import configure_precision,correctness_sample
    torch.cuda.set_device(0);torch.manual_seed(20261103)
    flags,warnings=configure_precision(torch)
    data=dict(completed=False,phase=args.phase,plan_sha256=hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol=protocol,cases=[],
              environment=dict(torch=torch.__version__,hip=torch.version.hip,device=str(torch.cuda.get_device_properties(0)),precision_flags=flags),warnings=warnings)
    target.parent.mkdir(parents=True,exist_ok=True)
    def save():
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(target)
    save();begin=time.monotonic()
    with torch.inference_mode():
        for rnd in range(protocol['rounds'] if args.phase!='profile' else 1):
            jobs=list(plan['cases']);random.Random(20261103+rnd).shuffle(jobs)
            for row in jobs:
                m,n,k=row['shape'];ib=2*(m*k+n*k)
                slots=max(protocol['minimum_ring_slots'],math.ceil(protocol['target_input_working_set_bytes']/ib)) if args.phase!='profile' else 1
                aa=[torch.randn(m,k,device='cuda',dtype=torch.float16)*.1 for _ in range(slots)]
                ww=[torch.randn(n,k,device='cuda',dtype=torch.float16)*.1 for _ in range(slots)]
                for func in (['matmul_nt','linear'] if args.phase=='equivalence' else ['matmul_nt']):
                    out=[None]*slots
                    def run(j):
                        out[j]=torch.matmul(aa[j],ww[j].t()) if func=='matmul_nt' else F.linear(aa[j],ww[j])
                    for i in range(max(30,slots)):run(i%slots)
                    torch.cuda.synchronize()
                    record=dict(id=row['id'],group=row['group'],shape=row['shape'],function=func,round=rnd+1,
                                input_strides=[list(aa[0].stride()),list(ww[0].stride())],
                                pointer_mod256=[aa[0].data_ptr()%256,ww[0].data_ptr()%256],
                                ring_slots=slots,distinct_input_bytes=ib*slots)
                    if args.phase=='profile':
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                            run(0);torch.cuda.synchronize()
                        tmptrace=target.parent/('trace_'+row['id']+'.json')
                        prof.export_chrome_trace(str(tmptrace))
                        events=json.loads(tmptrace.read_text())['traceEvents']
                        record['kernels']=[dict(name=e['name'],args=e.get('args',{})) for e in events if e.get('cat')=='kernel']
                        # Only metadata is saved in the aggregate. Preserve raw trace on server.
                        del prof
                    else:
                        calls=math.ceil(protocol['minimum_graph_calls']/slots)*slots
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        g=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g,stream=stream):
                            for i in range(calls):run(i%slots)
                        for _ in range(6):g.replay()
                        torch.cuda.synchronize()
                        checks=[correctness_sample(torch,aa[j],ww[j].t(),out[j],'fp16') for j in sorted({0,slots-1})]
                        if not all(c['passed'] for c in checks):raise RuntimeError('Correctness failed')
                        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record();end.record();end.synchronize();values=[]
                        for _ in range(protocol['event_samples']):
                            start.record();g.replay();end.record();end.synchronize()
                            values.append(start.elapsed_time(end)*1000/calls)
                        s=sorted(values)
                        record.update(graph_calls=calls,samples_us=values,correctness=checks,
                                      timing=dict(p50_us=statistics.median(s),p10_us=s[int(.1*(len(s)-1))],p90_us=s[math.ceil(.9*len(s))-1]))
                        del g
                    data['cases'].append(record);data['wall_seconds']=time.monotonic()-begin;save()
                    print(len(data['cases']),row['id'],func,round(record.get('timing',{}).get('p50_us',0),4),flush=True)
                    del out;torch.cuda.synchronize();gc.collect()
                del aa,ww;gc.collect()
    data['completed']=True;save()

if __name__=='__main__':main()
