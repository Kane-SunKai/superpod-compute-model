"""Operator plugins supply legal mappings and work; scheduler/cache stay generic."""
from collections import Counter
from math import ceil, isqrt
from typing import Protocol
from .contracts import Mapping
from .scheduler import pipeline, pipeline_segments, operand_pipeline, output_pipeline
from .backends import mmac_operand_plan
from .operand_layout import routed_operands, operand_service_time

def ceildiv(a,b):return (a+b-1)//b
def align(a,q):return ceildiv(a,q)*q

class OperatorModel(Protocol):
    name: str
    version: str
    def validate(self,request,hardware): ...
    def special(self,request,hardware,policy): ...
    def candidates(self,request,hardware,policy): ...
    def capacity(self,mapping,hardware): ...
    def mapping_from_dict(self,value): ...
    def rank_key(self,result): ...
    def evaluate(self,request,hardware,policy,backend,mapping,trace_limit=0): ...

class OperatorRegistry:
    def __init__(self):self.models={}
    def register(self,model):
        if model.name in self.models:raise ValueError('Operator already registered: '+model.name)
        self.models[model.name]=model
    def get(self,name):
        if name not in self.models:raise NotImplementedError(f'Operator {name!r} is not implemented; register its algorithm, candidates and compute services')
        return self.models[name]

def extents(total,tile):
    full,tail=divmod(total,tile);result=[]
    if full:result.append((tile,full))
    if tail:result.append((tail,1))
    return result

class GemmModel:
    name='gemm'
    version='dense-resident-c-explicit-operands-3.4'
    def input_hbm_bandwidth(self,hardware):
        """Read service hook; default preserves the original hardware contract."""
        return hardware.hbm_bandwidth_Bps
    def input_a_hbm_bandwidth(self,hardware):
        """A-view fill rate; the flat input rate when the profile sets no split."""
        return hardware.hbm_read_a_Bps or hardware.hbm_bandwidth_Bps
    def input_shared_write_time(self,m,n,k,hardware,policy):
        """Optional input-store service hook, seconds per CU per K panel."""
        return 2*(m+n)*k/(hardware.shared_write_bytes_per_cycle_per_core*hardware.clock_hz) if policy.charge_input_shared_write and hardware.shared_write_bytes_per_cycle_per_core else 0.0
    def panel_control_time(self,k,hardware,policy):
        """Optional K-loop control service hook, seconds per CU per panel."""
        return ceildiv(k,hardware.instruction_k)*policy.k_loop_control_cycles_per_step/hardware.clock_hz
    @staticmethod
    def mapping_from_dict(value):return Mapping(**value)
    @staticmethod
    def rank_key(result):
        m=result['mapping']
        return (result['duration_s'],result['capacity']['shared_bytes'],m['shared_buffers'],m['l2_buffers'],tuple(m[k] for k in ('bm','bn','bk','gm','gn')))
    def validate(self,request,hardware):
        if request.operator!=self.name:raise ValueError('Request/operator mismatch')
        if request.M==0 or request.N==0:return
        footprint=2*(request.M*request.K+request.K*request.N+request.M*request.N)
        if footprint>hardware.hbm_capacity_bytes:raise ValueError('A+B+C exceed configured HBM capacity; out-of-core execution is not implemented')

    def special(self,request,hardware,policy):
        if request.M==0 or request.N==0:return {'status':'empty-output','duration_s':0.0,'output_bytes':0}
        if request.K==0:
            size=request.M*request.N*2
            return {'status':'zero-fill','duration_s':hardware.kernel_launch_s+size/hardware.hbm_bandwidth_Bps+(hardware.hbm_latency_s if policy.charge_hbm_latency else 0),'output_bytes':size,'scope':'ideal global-memory zero-fill; beta=0, no GEMM'}
        return None

    @staticmethod
    def capacity(mapping,hardware):
        p=mapping;gm=p.gm*p.bm;gn=p.gn*p.bn
        inputs=p.shared_buffers*(p.bm+p.bn)*p.bk*2
        routed=routed_operands(p.bm,p.bn,p.bk,hardware) if p.operand_layout=='engine-routed' else None
        if routed:inputs=p.shared_buffers*routed['shared_bytes']
        output=min(p.bm*p.bn*2,p.output_chunk_bytes*p.output_buffers)
        registers=p.bm*p.bn*4+hardware.register_reserve_bytes_per_core
        operand_registers=0
        partition_registers=[]
        if hardware.matrix_backend in ('mmac','mac-throughput') and hardware.shared_read_bytes_per_cycle_per_core:
            operand_registers=p.register_input_buffers*(routed['shared_bytes'] if routed else mmac_operand_plan(p.bm,p.bn,p.bk,hardware)[1])
            registers+=operand_registers
        if hardware.register_partitions_per_core:
            _,_,pm,pn,_=mmac_operand_plan(p.bm,p.bn,p.bk,hardware)
            fm=ceildiv(p.bm,hardware.instruction_m);fn=ceildiv(p.bn,hardware.instruction_n)
            qm=ceildiv(fm,pm);qn=ceildiv(fn,pn)
            for i in range(pm):
                for j in range(pn):
                    mi=max(0,min(qm,fm-i*qm))*hardware.instruction_m
                    ni=max(0,min(qn,fn-j*qn))*hardware.instruction_n
                    active=bool(mi and ni)
                    input_regs=routed['register_bytes_by_engine'][i*pn+j] if routed else 2*p.bk*(mi+ni)
                    value=(mi*ni*4+p.register_input_buffers*input_regs) if active else 0
                    value+=ceildiv(hardware.register_reserve_bytes_per_core,hardware.register_partitions_per_core)
                    partition_registers.append(value)
        l2=p.l2_buffers*(gm+gn)*p.bk*2+p.gm*p.gn*output
        reasons=[]
        if max(inputs,output)>hardware.shared_budget_bytes_per_core:reasons.append('shared')
        if registers>hardware.register_budget_bytes_per_core:reasons.append('registers')
        if partition_registers and max(partition_registers)>hardware.register_bytes_per_partition:reasons.append('register-partition')
        if l2>hardware.l2_bytes:reasons.append('L2')
        if p.gm*p.gn>hardware.cores:reasons.append('cores')
        result={'valid':not reasons,'rejections':reasons,'shared_bytes':max(inputs,output),'shared_input_bytes':inputs,'shared_output_bytes':output,'register_bytes':registers,'operand_register_bytes':operand_registers,'l2_bytes':l2,'parallel_cores':p.gm*p.gn}
        if partition_registers:result.update(register_bytes_per_partition=partition_registers,register_partition_limit=hardware.register_bytes_per_partition)
        return result

    def candidates(self,request,hardware,policy):
        def values(base,dim):
            return sorted(set(base)|({min(max(base),max(min(base),align(dim,hardware.matrix_k_quantum)))} if policy.add_shape_candidate else set()))
        factors=[]
        for d in range(1,isqrt(hardware.cores)+1):
            if hardware.cores%d==0:
                factors.extend([(d,hardware.cores//d),(hardware.cores//d,d)])
        factors.extend((g,g) for g in (1,2,4,8))
        for bm in values(policy.bm_values,request.M):
            for bn in values(policy.bn_values,request.N):
                tm,tn=ceildiv(request.M,bm),ceildiv(request.N,bn);geometries=set()
                for gm,gn in factors:
                    gm=max(1,min(gm,tm));gn=max(1,min(gn,tn,hardware.cores//gm))
                    if gm*gn<=hardware.cores:geometries.add((gm,gn))
                for bk in values(policy.bk_values,request.K):
                    counts=policy.register_buffer_counts if hardware.matrix_backend in ('mmac','mac-throughput') and hardware.shared_read_bytes_per_cycle_per_core else (2,)
                    for gm,gn in sorted(geometries):
                        for layout in policy.operand_layouts:
                            for combo in self.buffer_dimensions(hardware,policy,bm,bn,bk,gm,gn,layout,counts):
                                yield Mapping(bm,bn,bk,gm,gn,*combo,layout)

    def buffer_dimensions(self,hardware,policy,bm,bn,bk,gm,gn,layout,counts):
        """Pipeline-depth dimensions to enumerate for one tile and geometry.

        'max' yields at most two: the deepest and the shallowest combination that
        fit the Shared, L2 and register budgets. Enumerating the whole product
        bought nothing -- on four shapes spanning small, odd and decode tiles the
        winner was always one of those two ends, while the product is 112x the
        candidates. 'search' restores the old product.
        """
        chunks=sorted({min(x,bm*bn*2) for x in policy.output_chunk_values})
        combos=sorted(((L,S,chunk,O,V) for L in policy.buffer_counts
                       for S in policy.buffer_counts for chunk in chunks
                       for O in policy.output_buffer_counts for V in counts),
                      key=lambda c:(-c[0],-c[1],c[2],-c[3],-c[4]))
        if policy.buffer_dims=='search':return combos
        stage = (bm + bn) * bk * 2
        budget = hardware.shared_budget_bytes_per_core
        picks = []
        # Deepest and shallowest feasible combinations bracket the duration
        # ranking; keeping both preserves rank_key's minimum-resource tie-break
        # for shapes where a single K panel makes extra buffers worthless.
        for ordered in (combos, combos[::-1]):
            for combo in ordered:
                # Shared holds the input stage and the output window at once;
                # reject what cannot fit before paying for a full capacity check.
                if max(combo[1] * stage, min(bm * bn * 2, combo[2] * combo[3])) > budget:
                    continue
                if self.capacity(Mapping(bm,bn,bk,gm,gn,*combo,layout),hardware)['valid']:
                    if combo not in picks: picks.append(combo)
                    break
        return picks

    def evaluate(self,request,hardware,policy,backend,mapping,trace_limit=0):
        p=mapping;cap=self.capacity(p,hardware)
        if not cap['valid']:return None
        if request.M==0 or request.N==0 or request.K==0:raise ValueError('Use the special empty/zero-fill route')
        GM,GN=p.gm*p.bm,p.gn*p.bn
        full,tail=divmod(request.K,p.bk)
        latency=hardware.hbm_latency_s if policy.charge_hbm_latency else 0.0
        spare=hardware.l2_bytes-cap['l2_bytes']
        ab_bytes=2*(request.M+request.N)*request.K
        warm=policy.initial_cache_state=='warm-if-fit' and ab_bytes<=spare
        compute=backend.estimate(p.bm,p.bn,align(p.bk,hardware.matrix_k_quantum)) if full else None
        last=backend.estimate(p.bm,p.bn,align(tail,hardware.matrix_k_quantum)) if tail else None
        shared_port=policy.transfer_resource_mode=='shared-port'
        # Reuse is an explicit capacity-bounded residence plan, not a prediction
        # of a hardware cache replacement policy. Conservative extra window space
        # remains reserved even when an input view aliases retained data.
        def retained_k(dim):
            k=min(request.K,spare//(2*dim))
            return k if k==request.K else k//p.bk*p.bk

        def axis_classes(total,tile):
            # First inner wave populates retained inputs. Only later waves hit.
            result=[];remaining=total
            first=min(tile,remaining);result.append((first,1,False));remaining-=first
            for size,count in extents(remaining,tile):result.append((size,count,True))
            return result

        output_cache={}
        def output_plan(m,n):
            key=(m,n)
            if key in output_cache:return output_cache[key]
            sizes=[(mi*ni*2,mc*nc) for mi,mc in extents(m,p.bm) for ni,nc in extents(n,p.bn)]
            chunk=p.output_chunk_bytes
            boundaries=sorted({0,*[x for size,_ in sizes for x in (size//chunk,ceildiv(size,chunk))]})
            segments=[];byte_runs=[]
            for lo,hi in zip(boundaries,boundaries[1:]):
                amount=sum(min(chunk,max(0,size-lo*chunk))*count for size,count in sizes)
                if not amount:continue
                per_core=max(min(chunk,max(0,size-lo*chunk)) for size,_ in sizes)
                bw=hardware.shared_write_bytes_per_cycle_per_core*hardware.clock_hz
                tw=per_core/bw if bw else 0.0
                tr=amount/hardware.l2_write_bandwidth_Bps+policy.l2_transfer_latency_s
                # Drain is a streamed L2 read + HBM write; bottleneck rate, not
                # two copies of the same link service. No extra output LDS read.
                th=max(amount/hardware.hbm_write_bandwidth_Bps,amount/hardware.l2_read_bandwidth_Bps)+latency
                segments.append((hi-lo,tw,tr,th))
                byte_runs.append({'chunks':hi-lo,'bytes_per_chunk_all_cores':amount,
                                  'max_bytes_per_chunk_per_core':per_core,'register_to_shared_s':tw,
                                  'l2_write_s':tr,'l2_read_hbm_drain_s':th})
            run=output_pipeline(segments,p.output_buffers,p.output_buffers,shared_port=shared_port,trace_limit=trace_limit)
            value={'duration_s':run['duration_s']+policy.epilogue_s_per_wave,'byte_runs':byte_runs,'pipeline':run,
                   'shared_output_limit_per_core':min(p.bm*p.bn*2,p.output_chunk_bytes*p.output_buffers),
                   'l2_output_window_limit':p.gm*p.gn*min(p.bm*p.bn*2,p.output_chunk_bytes*p.output_buffers),
                   'register_to_shared_service_s':sum(num*tw for num,tw,_,_ in segments),
                   'output_shared_read_charged':False}
            output_cache[key]=value;return value

        def evaluate_order(order):
            total=hardware.kernel_launch_s+policy.device_service_s
            hbytes=rbytes=obytes=flops=wave_count=saved=0;classes=[];sources=Counter()
            # 'A': reuse one M stripe across successive N waves; 'B': transpose
            # traversal order. Do not share one operand across incompatible stripes.
            if order=='A':
                wave_specs=[(m,n,mc*nc,reused) for m,mc in extents(request.M,GM) for n,nc,reused in axis_classes(request.N,GN)]
            elif order=='B':
                wave_specs=[(m,n,mc*nc,reused) for n,nc in extents(request.N,GN) for m,mc,reused in axis_classes(request.M,GM)]
            else:
                wave_specs=[(m,n,mc*nc,False) for m,mc in extents(request.M,GM) for n,nc in extents(request.N,GN)]
            for m,n,count,reused in wave_specs:
                rm,rn=ceildiv(m,p.bm),ceildiv(n,p.bn);active=rm*rn
                keep=retained_k(m if order=='A' else n) if reused else 0
                def transfer(k,hit):
                    cold=(m+n)*k*2
                    saved_bytes=cold if warm else ((m if order=='A' else n)*k*2 if hit else 0)
                    dh=cold-saved_bytes;dr=(rn*m+rm*n)*k*2
                    # HBM fills consume L2 write bandwidth. Stream each fill at
                    # the slower endpoint rate; R separately charges L2 reads.
                    # The A view streams at its own measured rate when nothing is
                    # retained; a retained operand falls back to the flat rate.
                    if dh==cold:
                        fill=m*k*2/self.input_a_hbm_bandwidth(hardware)+n*k*2/self.input_hbm_bandwidth(hardware)
                    else:
                        fill=dh/self.input_hbm_bandwidth(hardware)
                    ht=max(fill,dh/hardware.l2_write_bandwidth_Bps)+(latency if dh else 0)
                    write=self.input_shared_write_time(min(p.bm,m),min(p.bn,n),k,hardware,policy)
                    if p.operand_layout=='engine-routed' and policy.charge_input_shared_write:
                        bw=hardware.shared_write_bytes_per_cycle_per_core*hardware.clock_hz
                        if not bw:raise ValueError('Engine-routed input requires Shared write bandwidth')
                        write=routed_operands(p.bm,p.bn,k,hardware)['shared_bytes']/bw
                    rt=max(dr/hardware.l2_read_bandwidth_Bps,write)
                    return dh,dr,ht,rt+policy.l2_transfer_latency_s,write
                hit_full=min(full,keep//p.bk);segments=[];bytes_runs=[]
                for number,k,hit,service in [(hit_full,p.bk,True,compute),(full-hit_full,p.bk,False,compute),(int(tail>0),tail,keep==request.K,last)]:
                    if not number:continue
                    dh,dr,h,r,input_write=transfer(k,hit)
                    st,layout_plan=operand_service_time(p,k,hardware,policy,service)
                    segments.append((number,h,r,st,service.duration_s+self.panel_control_time(k,hardware,policy)))
                    bytes_runs.append({'panels':number,'k_elements':k,'HBM_bytes_per_panel':dh,'L2_to_core_bytes_per_panel':dr,
                                       'Shared_to_register_bytes_per_core':layout_plan['shared_read_bytes'] if layout_plan else service.shared_read_bytes,
                                       'Shared_read_service_s_per_core':st,
                                       'Shared_input_write_service_s_per_core':input_write,
                                       'R_service_s':r,
                                       'operand_layout_plan':layout_plan,
                                       'retained_operand':('AB' if warm else order if hit else None)})
                    hbytes+=count*number*dh;rbytes+=count*number*dr;saved+=count*number*((m+n)*k*2-dh)
                if hardware.shared_read_bytes_per_cycle_per_core and hardware.matrix_backend in ('mmac','mac-throughput'):
                    run=operand_pipeline(segments,p.l2_buffers,p.shared_buffers,p.register_input_buffers,shared_port=shared_port,trace_limit=trace_limit)
                else:
                    run=pipeline_segments([(num,h,r,c) for num,h,r,_,c in segments],p.l2_buffers,p.shared_buffers,shared_port=shared_port,trace_limit=trace_limit)
                output=output_plan(m,n)
                per_wave=run['duration_s']+output['duration_s'];total+=count*per_wave
                out=m*n*2;obytes+=count*out;wave_count+=count
                flops+=count*active*2*p.bm*p.bn*(full*align(p.bk,hardware.matrix_k_quantum)+align(tail,hardware.matrix_k_quantum))
                if full:sources[compute.source]+=count*full
                if tail:sources[last.source]+=count
                classes.append({'valid_m':m,'valid_n':n,'count':count,'active_cores':active,'full_k_panels':full,'tail_k':tail,
                                'retained_k':request.K if warm else keep,'is_reuse_wave':reused,'panel_byte_runs':bytes_runs,
                                'full_panel_compute':compute.to_dict() if compute else None,'tail_panel_compute':last.to_dict() if last else None,
                                'full_panel_control_s':self.panel_control_time(p.bk,hardware,policy) if full else 0.0,
                                'tail_panel_control_s':self.panel_control_time(tail,hardware,policy) if tail else 0.0,
                                'pipeline':run,'output_stream':output,'epilogue_s':policy.epilogue_s_per_wave,
                                'write_to_l2_s':out/hardware.l2_write_bandwidth_Bps,'write_to_hbm_s':out/hardware.hbm_bandwidth_Bps,'duration_s':per_wave})
            notes=['resident FP32 C registers across all K panels; sequential output waves',
                   'C output: FP16 register->Shared write, L2 write, L2-read/HBM drain; bounded Shared and L2 slots',
                   'Shared output readback deliberately has no separate time; output conversion/layout cost remains configurable',
                   'directional L2 bandwidths; endpoint streaming uses max times; full shared-port/cache contention is not modeled',
                   'L2 prefix retention is a planned ideal residence policy, not measured hardware cache behavior',
                   'register aggregate capacity checked; compiler per-thread allocation/occupancy not modeled',
                   'optional Shared->register service uses an explicit rectangular-engine operand reuse plan; independent LDS read/write ports assumed',
                   'FP16/BF16 share 16-bit service; conversion/layout execution cost remains a separate configurable term',
                   'device_service_s is additive per GEMM, distinct from hardware.kernel_launch_s host submission setting',
                   'optional K-loop control service is separate from physical MMAC instruction issue interval; per instruction-K step',
                   'optional effective Shared-read service is a global calibration parameter, separate from nominal hardware bandwidth',
                   'matrix service selected by hardware/backend; no GEMM-shape-dependent correction']
            if p.operand_layout=='engine-routed':
                notes.extend(['Engine-routed A/B copies use explicit Shared and register packet capacity; L2 inputs are fetched once per CU, then replicated into Shared',
                              'Shared operand reads use the declared hardware bank/phase model; legacy calibrated effective-read rate is bypassed for this layout'])
            if request.layout=='right-transposed-contiguous':
                notes.append('B is a transposed view of contiguous W[N,K]; no full B transpose is charged. Producer transaction/stride efficiency is not yet modeled beyond logical byte volume')
            return {'mapping':p.to_dict(),'capacity':cap,'duration_s':total,'duration_us':total*1e6,
                    'kernel_launch_s':hardware.kernel_launch_s,'device_service_s':policy.device_service_s,
                    'initial_cache_state':policy.initial_cache_state,'warm_residency_admitted':warm,
                    'reuse_order':order,'retention_budget_bytes':spare,'hbm_bytes_saved_by_reuse':saved,
                    'wave_count':wave_count,'wave_classes':classes,'hbm_read_bytes':hbytes,'l2_to_core_bytes':rbytes,'output_bytes':obytes,
                    'useful_flops':2*request.M*request.N*request.K,'executed_tile_flops':flops,'array_sources_wave_panels':dict(sources),'scope_notes':notes}
        orders=['none']
        if policy.input_reuse and not warm:
            if request.N>GN and retained_k(min(GM,request.M)):orders.append('A')
            if request.M>GM and retained_k(min(GN,request.N)):orders.append('B')
        return min((evaluate_order(order) for order in orders),key=lambda result:result['duration_s'])


def default_registry():
    registry=OperatorRegistry();registry.register(GemmModel());return registry
