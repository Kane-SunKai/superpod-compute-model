"""Per-wave operand/compute dependencies with explicit workgroup barriers.

H/R are the existing aggregate input services. S is a shared per-CU port;
each wave owns its operand registers and is assigned one matrix engine.
No instruction latencies or GEMM-dependent correction factors are added.
"""
from math import ceil
from .contracts import integer,positive


def wave_operand_pipeline(segments,l2_buffers,shared_buffers,register_buffers,plan,engines,
                          *,shared_port=False,trace_limit=0,read_latency_s=0.0):
    for count in (l2_buffers,shared_buffers,register_buffers):
        integer(count,'buffer count')
        if count>8:raise ValueError('At most eight buffers supported')
    integer(engines,'matrix engines');integer(trace_limit,'trace limit',0)
    positive(read_latency_s,'read latency',True)
    waves=plan.waves
    H=R=S=0.0;L=[0.0]*l2_buffers;shared=[0.0]*shared_buffers
    matrix=[0.0]*engines
    register_free=[[0.0]*waves for _ in range(register_buffers)]
    prior_group=0.0;trace=[];index=0;final_c=0.0
    for count,h,r,s,c in segments:
        integer(count,'panel count',0)
        for value in (h,r,s,c):positive(value,'service time',True)
        per_wave_s=s/waves
        per_wave_c=c/ceil(waves/engines)
        for _ in range(count):
            ls=index%l2_buffers;ss=index%shared_buffers;rs=index%register_buffers
            h0=max(H,L[ls],R if shared_port else 0.0);H=h0+h
            request=max(H,shared[ss]);ready=request+read_latency_s
            r0=max(R,ready);R=r0+r;L[ls]=R
            previous_registers=register_free[rs]
            group_release=max(previous_registers) if plan.read_waits_for_group_release else 0.0
            reads=[]
            for w in range(waves):
                start=max(S,R,previous_registers[w],group_release)
                S=start+per_wave_s;reads.append((start,S))
            shared[ss]=max(end for _,end in reads)
            barrier=max(prior_group,shared[ss] if plan.barrier_after_reads else 0.0)
            computations=[]
            for w in range(waves):
                engine=w%engines
                start=max(barrier,reads[w][1],matrix[engine])
                end=start+per_wave_c;matrix[engine]=end
                computations.append((start,end))
            register_free[rs]=[end for _,end in computations]
            prior_group=final_c=max(end for _,end in computations)
            if len(trace)<trace_limit:
                trace.append({'panel':index,'H_start_s':h0,'H_end_s':H,
                              'R_request_s':request,'R_response_ready_s':ready,'R_start_s':r0,'R_end_s':R,
                              'S_start_s':min(a for a,_ in reads),'S_end_s':shared[ss],
                              'C_start_s':min(a for a,_ in computations),'C_end_s':final_c,
                              'wave_reads':reads,'wave_compute':computations,'compute_barrier_s':barrier})
            index+=1
    return {'duration_s':final_c,'H_end_s':H,'R_end_s':R,'S_end_s':S,'C_end_s':final_c,
            'panels':index,'skipped_panels':0,'trace':trace,'stages':['HBM','L2','per-wave Shared-read','per-wave matrix'],
            'wave_count_per_workgroup':waves,'matrix_engines':engines,
            'scope':'Explicit wave dependencies; no barrier instruction service, tail instruction path or address-alignment penalty yet'}
