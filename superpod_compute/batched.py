"""Independent-head batched GEMM adapter over the same tile/service model.

Each C tile belongs to exactly one batch member. A/B reuse is counted only
within that member in a wave; there is no fictitious sharing between heads.
Waves pack independent output tasks onto the configured CUs. This adapter
does not model FlashAttention fusion, masking, or tensor-layout conversion.
"""
from collections import Counter
from itertools import product
from .contracts import GemmRequest, Mapping, integer
from .operators import GemmModel, ceildiv
from .backends import make_backend
from .scheduler import operand_pipeline, output_pipeline
from .operand_layout import operand_service_time,routed_operands

def _waves(batch,m,n,bm,bn,cores):
    tasks=[(head,i,j,min(bm,m-i),min(bn,n-j)) for head in range(batch) for i in range(0,m,bm) for j in range(0,n,bn)]
    classes=Counter()
    for first in range(0,len(tasks),cores):
        group=tasks[first:first+cores]
        a={(h,i):mi for h,i,j,mi,nj in group};b={(h,j):nj for h,i,j,mi,nj in group}
        sizes=tuple(sorted(Counter(mi*nj*2 for _,_,_,mi,nj in group).items()))
        classes[(len(group),sum(a.values())+sum(b.values()),sum(mi+nj for _,_,_,mi,nj in group),max(mi+nj for _,_,_,mi,nj in group),sizes)]+=1
    return classes,len(tasks)

def _output(sizes,chunk,buffers,h,p):
    bounds=sorted({0,*[q for size,num in sizes for q in (size//chunk,ceildiv(size,chunk))]});segments=[]
    for lo,hi in zip(bounds,bounds[1:]):
        total=sum(min(chunk,max(0,size-lo*chunk))*num for size,num in sizes)
        local=max(min(chunk,max(0,size-lo*chunk)) for size,num in sizes)
        if total:
            w=local/(h.shared_write_bytes_per_cycle_per_core*h.clock_hz) if h.shared_write_bytes_per_cycle_per_core else 0
            segments.append((hi-lo,w,total/h.l2_write_bandwidth_Bps,max(total/h.hbm_bandwidth_Bps,total/h.l2_read_bandwidth_Bps)))
    return output_pipeline(segments,buffers,buffers)['duration_s']+p.epilogue_s_per_wave

def predict_batched_gemm(batch,m,n,k,hardware,policy,*,transpose_b=False):
    for label,value in [('batch',batch),('M',m),('N',n),('K',k)]:integer(value,label)
    h=hardware;p=policy;model=GemmModel();backend=make_backend('mmac',h,p.matrix_efficiency)
    footprint=2*batch*(m*k+k*n+m*n)
    if footprint>h.hbm_capacity_bytes:raise ValueError('Batched inputs/output exceed HBM')
    best=None;count=feasible=0
    # Reuse the policy's resource-constrained tile candidates; geometries are
    # replaced by a batch-aware pool of independent output tasks, not B broadcast.
    shapes=sorted({(v.bm,v.bn,v.bk,v.l2_buffers,v.shared_buffers,v.output_chunk_bytes,v.output_buffers,v.register_input_buffers,v.operand_layout)
                   for v in model.candidates(GemmRequest(m,n,k),h,p) if v.bm>=64 and v.bn>=64})
    wave_cache={};out_cache={}
    for bm,bn,bk,L,S,chunk,O,V,layout in shapes:
        count+=1;mapping=Mapping(bm,bn,bk,1,1,L,S,chunk,O,V,layout);capacity=model.capacity(mapping,h)
        if not capacity['valid']:continue
        key=(bm,bn)
        if key not in wave_cache:wave_cache[key]=_waves(batch,m,n,bm,bn,h.cores)
        classes,tasks=wave_cache[key]
        if any(L*unique*bk*2+active*min(bm*bn*2,chunk*O)>h.l2_bytes for active,unique,duplicated,local,sizes in classes):continue
        feasible+=1;duration=p.device_service_s+h.kernel_launch_s;hbytes=rbytes=outbytes=0;details=[]
        for (active,unique,duplicated,local,sizes),repeat in classes.items():
            segments=[];full,tail=divmod(k,bk);service_totals=[0.0]*4
            for number,width in [(full,bk),(int(tail>0),tail)]:
                if not number:continue
                compute=backend.estimate(bm,bn,ceildiv(width,h.instruction_k)*h.instruction_k)
                H=2*unique*width;R=2*duplicated*width
                th=max(H/h.hbm_bandwidth_Bps,H/h.l2_write_bandwidth_Bps)
                store=2*local*width/(h.shared_write_bytes_per_cycle_per_core*h.clock_hz) if p.charge_input_shared_write and h.shared_write_bytes_per_cycle_per_core else 0
                if layout=='engine-routed' and p.charge_input_shared_write:
                    if not h.shared_write_bytes_per_cycle_per_core:raise ValueError('Routed input requires Shared write bandwidth')
                    store=routed_operands(bm,bn,width,h)['shared_bytes']/(h.shared_write_bytes_per_cycle_per_core*h.clock_hz)
                tr=max(R/h.l2_read_bandwidth_Bps,store)
                ts,_=operand_service_time(mapping,width,h,p,compute)
                tc=compute.duration_s+model.panel_control_time(width,h,p)
                segments.append((number,th,tr,ts,tc));hbytes+=repeat*number*H;rbytes+=repeat*number*R
                service_totals=[a+number*b for a,b in zip(service_totals,[th,tr,ts,tc])]
            run=operand_pipeline(segments,L,S,V)
            outkey=(sizes,chunk,O)
            if outkey not in out_cache:out_cache[outkey]=_output(sizes,chunk,O,h,p)
            ot=out_cache[outkey];duration+=repeat*(run['duration_s']+ot);outbytes+=repeat*sum(size*num for size,num in sizes)
            details.append(dict(active_cores=active,count=repeat,input_s=run['duration_s'],output_s=ot,service_s=service_totals))
        result=dict(duration_us=duration*1e6,duration_s=duration,mapping=mapping.to_dict(),output_tasks=tasks,wave_count=sum(classes.values()),wave_classes=details,hbm_read_bytes=hbytes,l2_read_bytes=rbytes,output_bytes=outbytes)
        if best is None or result['duration_us']<best['duration_us']:best=result
    if best is None:raise ValueError('No batched GEMM candidate fits resources')
    assert best['output_bytes']==batch*m*n*2
    return dict(operator='batched_gemm',batch=batch,shape=[m,n,k],transpose_b=transpose_b,best=best,searched_candidates=count,feasible_candidates=feasible,
        scope='Independent heads packed across CUs; same MMAC/store/control services; reuse only within one head in a wave; no inter-wave cache retention or layout-conversion time; no attention fusion',
        minimum_batched_tile=64)
