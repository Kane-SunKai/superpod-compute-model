"""Operator plugins supply legal mappings and work; scheduler/cache stay generic."""
from collections import Counter
from math import ceil, isqrt
from typing import Protocol
from .contracts import Mapping
from .scheduler import pipeline

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
    version='dense-resident-c-1.0'
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
        output=p.bm*p.bn*2
        registers=p.bm*p.bn*4+hardware.register_reserve_bytes_per_core
        l2=p.l2_buffers*(gm+gn)*p.bk*2+gm*gn*2
        reasons=[]
        if max(inputs,output)>hardware.shared_budget_bytes_per_core:reasons.append('shared')
        if registers>hardware.register_budget_bytes_per_core:reasons.append('registers')
        if l2>hardware.l2_bytes:reasons.append('L2')
        if p.gm*p.gn>hardware.cores:reasons.append('cores')
        return {'valid':not reasons,'rejections':reasons,'shared_bytes':max(inputs,output),'shared_input_bytes':inputs,'shared_output_bytes':output,'register_bytes':registers,'l2_bytes':l2,'parallel_cores':p.gm*p.gn}

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
                    for gm,gn in sorted(geometries):
                        for L in policy.buffer_counts:
                            for S in policy.buffer_counts:
                                yield Mapping(bm,bn,bk,gm,gn,L,S)

    def evaluate(self,request,hardware,policy,backend,mapping,trace_limit=0):
        p=mapping;cap=self.capacity(p,hardware)
        if not cap['valid']:return None
        if request.M==0 or request.N==0 or request.K==0:raise ValueError('Use the special empty/zero-fill route')
        GM,GN=p.gm*p.bm,p.gn*p.bn
        full,tail=divmod(request.K,p.bk)
        latency=hardware.hbm_latency_s if policy.charge_hbm_latency else 0.0
        total=hardware.kernel_launch_s;hbytes=rbytes=obytes=flops=wave_count=0;classes=[];sources=Counter()
        compute=backend.estimate(p.bm,p.bn,align(p.bk,hardware.matrix_k_quantum)) if full else None
        last=backend.estimate(p.bm,p.bn,align(tail,hardware.matrix_k_quantum)) if tail else None
        for m,mc in extents(request.M,GM):
            for n,nc in extents(request.N,GN):
                count=mc*nc;rm,rn=ceildiv(m,p.bm),ceildiv(n,p.bn);active=rm*rn
                def transfer(k):
                    dh=(m+n)*k*2;dr=(rn*m+rm*n)*k*2
                    return dh,dr,dh/hardware.hbm_bandwidth_Bps+latency,dr/hardware.l2_bandwidth_Bps+policy.l2_transfer_latency_s
                dh,dr,h,r=transfer(p.bk)
                if tail:
                    th,tr,ht,rt=transfer(tail);tail_triplet=(ht,rt,last.duration_s)
                else:th=tr=0;tail_triplet=None
                run=pipeline(h,r,compute.duration_s if compute else 0.0,full,p.l2_buffers,p.shared_buffers,tail_triplet,shared_port=policy.transfer_resource_mode=='shared-port',trace_limit=trace_limit)
                out=m*n*2;write_r=out/hardware.l2_bandwidth_Bps+policy.l2_transfer_latency_s;write_h=out/hardware.hbm_bandwidth_Bps+latency
                tail_time=policy.epilogue_s_per_wave+write_r+write_h
                per_wave=run['duration_s']+tail_time
                total+=count*per_wave;hbytes+=count*(full*dh+th);rbytes+=count*(full*dr+tr);obytes+=count*out;wave_count+=count
                flops+=count*active*2*p.bm*p.bn*(full*align(p.bk,hardware.matrix_k_quantum)+align(tail,hardware.matrix_k_quantum))
                if full:sources[compute.source]+=count*full
                if tail:sources[last.source]+=count
                classes.append({'valid_m':m,'valid_n':n,'count':count,'active_cores':active,'full_k_panels':full,'tail_k':tail,'full_panel_H_bytes':dh if full else 0,'full_panel_R_bytes':dr if full else 0,'tail_panel_H_bytes':th,'tail_panel_R_bytes':tr,'full_panel_compute':compute.to_dict() if compute else None,'tail_panel_compute':last.to_dict() if last else None,'pipeline':run,'epilogue_s':policy.epilogue_s_per_wave,'write_to_l2_s':write_r,'write_to_hbm_s':write_h,'duration_s':per_wave})
        notes=['cold input at each wave; within-wave A/B deduplication only','one resident output tile per Core; no split-K; sequential waves','FP16/BF16 share 16-bit service configuration; FP32 accumulation is capacity-accounted','no GPU instruction/occupancy or L2-bank simulation']
        if policy.epilogue_s_per_wave==0:notes.append('output conversion/layout cost is uncalibrated and set to zero')
        if not policy.charge_hbm_latency:notes.append('source IO latency is retained in profile but not charged per panel, matching LLMCompass GEMM IO formula')
        if 'own-os-wavefront-estimate' in sources:notes.append('some array shapes used the independent OS miss estimator, not SCALE-Sim or hardware measurement')
        return {'mapping':p.to_dict(),'capacity':cap,'duration_s':total,'duration_us':total*1e6,'kernel_launch_s':hardware.kernel_launch_s,'wave_count':wave_count,'wave_classes':classes,'hbm_read_bytes':hbytes,'l2_to_core_bytes':rbytes,'output_bytes':obytes,'useful_flops':2*request.M*request.N*request.K,'executed_tile_flops':flops,'array_sources_wave_panels':dict(sources),'scope_notes':notes}

def default_registry():
    registry=OperatorRegistry();registry.register(GemmModel());return registry
