"""Rejected V1 diagnostic: hardware-bound software execution services.

The analytical mapper, buffers and pipeline are unchanged. Independent GEMMs
calibrate effective stage services; these are NOT replacement hardware peaks.
No measured acceptance duration is used at prediction time. The twelve unseen
holdouts passed only 4/12 at 10%; this is NOT a production prediction mode.
"""
from dataclasses import replace
import json
import math
from pathlib import Path
from superpod_compute.contracts import digest
from superpod_compute.backends import make_backend
from superpod_compute.operators import GemmModel

FACTOR_NAMES=('matrix_scale','hbm_scale','shared_operand_scale','l2_read_scale')
IDENTITY={name:1.0 for name in FACTOR_NAMES}

class ServiceBackend:
    def __init__(self,backend,factors,clock):
        self.backend=backend;self.factors=factors;self.clock=clock
    def estimate(self,m,n,k):
        value=self.backend.estimate(m,n,k)
        cycles=math.ceil(value.cycles*self.factors['matrix_scale'])
        return replace(value,cycles=cycles,duration_s=cycles/self.clock,
                       shared_read_bytes=math.ceil(value.shared_read_bytes*self.factors['shared_operand_scale']))

def evaluate_services(request,mapping,hardware,policy,factors,*,trace_limit=0,backend='auto'):
    """Reevaluate the same planned work, changing only declared stage services."""
    base=make_backend(backend,hardware,policy.matrix_efficiency)
    effective=replace(hardware,hbm_bandwidth_Bps=hardware.hbm_bandwidth_Bps/factors['hbm_scale'],
                      l2_read_bytes_per_cycle=hardware.l2_read_bytes_per_cycle/factors['l2_read_scale'])
    wrapped=ServiceBackend(base,factors,hardware.clock_hz)
    result=GemmModel().evaluate(request,effective,policy,wrapped,mapping,trace_limit=trace_limit)
    if result is None:raise ValueError('Mapping violates physical resource budgets')
    result['effective_software_services']=dict(factors)
    result['effective_service_units']={'matrix_scale':'cycles per planned matrix work',
        'hbm_scale':'effective HBM service seconds per byte / reference',
        'shared_operand_scale':'effective operand delivery service / reference; not a measured byte counter',
        'l2_read_scale':'effective L2 read service seconds per byte / reference'}
    return result

class ServiceProfile:
    def __init__(self,value):
        self.data=value
        if value.get('schema')!='software-stage-services-1':raise ValueError('Unknown service profile schema')
        self.points=value['points']
        if not self.points:raise ValueError('Empty service profile')
        for row in self.points:
            if row.get('role')!='calibration':raise ValueError('Only independent calibration points allowed')
            for name in FACTOR_NAMES:
                f=row['factors'][name]
                if not isinstance(f,(float,int)) or not math.isfinite(f) or f<=0:raise ValueError('Invalid service factor')
            if row['factors']['matrix_scale']<1:raise ValueError('Cannot exceed independently measured matrix peak')
        self.sha256=digest(value)

    @classmethod
    def load(cls,path):return cls(json.loads(Path(path).read_text(encoding='utf-8-sig')))

    def validate(self,request,hardware):
        if digest(hardware.to_dict())!=self.data['hardware_sha256']:
            raise ValueError('Software service profile is bound to a different hardware configuration; recalibrate or omit it')
        if request.dtype!=self.data['dtype'] or request.rhs_storage!=self.data['rhs_storage']:
            raise ValueError('Software profile does not match dtype / physical weight layout')

    def select(self,request):
        m,n,k=request.M,request.N,request.K
        points=[p for p in self.points if (p['shape'][0]<=64)==(m<=64)]
        if not points:raise ValueError('No execution class coverage')
        exact=[p for p in points if p['shape']==[m,n,k]]
        if exact:return dict(exact[0]['factors']),{'method':'calibration-point','sources':[exact[0]['id']]}
        pair=[p for p in points if p['shape'][1:]==[n,k]]
        axis=0
        if len(pair)<2:
            pair=[p for p in points if p['shape'][:2]==[m,n] and p['shape'][2]%128==k%128]
            axis=2
        if len(pair)>=2:
            ordered=sorted(pair,key=lambda p:p['shape'][axis])
            value=[m,n,k][axis]
            lower=[p for p in ordered if p['shape'][axis]<=value]
            upper=[p for p in ordered if p['shape'][axis]>=value]
            lo=lower[-1] if lower else ordered[0]
            hi=upper[0] if upper else ordered[-1]
            # Extrapolate only within an observed implementation family.
            if not lower:lo,hi=ordered[:2]
            elif not upper:lo,hi=ordered[-2:]
            same_family=lo.get('family') and lo.get('family')==hi.get('family')
            if self.data.get('interpolation','linear')=='nearest' or not same_family:
                closest=min((lo,hi),key=lambda p:(abs(p['shape'][axis]-value),p['shape'][axis]))
                return dict(closest['factors']),{'method':'nearest-implementation-band','sources':[closest['id']],
                    'extrapolated':not lower or not upper}
            x0,x1=lo['shape'][axis],hi['shape'][axis]
            t=(value-x0)/(x1-x0)
            # Bounded service interpolation; never invent a faster matrix peak.
            f={name:max(.5 if name=='shared_operand_scale' else .01,
                        lo['factors'][name]+t*(hi['factors'][name]-lo['factors'][name])) for name in FACTOR_NAMES}
            f['matrix_scale']=max(1.,f['matrix_scale']);f['l2_read_scale']=max(1.,f['l2_read_scale'])
            f['hbm_scale']=max(self.data.get('minimum_hbm_scale',.86),f['hbm_scale'])
            return f,{'method':'implementation-band-service-interpolation','sources':[lo['id'],hi['id']],
                      'axis':'MNK'[axis],'extrapolated':not lower or not upper}
        def distance(p):
            return sum(w*abs(math.log2(a/b)) for a,b,w in zip((m,n,k),p['shape'],(1.,2.,2.)))
        near=min(points,key=lambda p:(distance(p),p['id']))
        return dict(near['factors']),{'method':'nearest-service-class','sources':[near['id']],
                                    'distance':distance(near),'extrapolated':True}

    def evaluate(self,request,mapping,hardware,policy,*,backend='auto',trace_limit=0):
        self.validate(request,hardware)
        factors,selection=self.select(request)
        result=evaluate_services(request,mapping,hardware,policy,factors,backend=backend,trace_limit=trace_limit)
        result['service_profile']={'sha256':self.sha256,'runtime':self.data['runtime'],
                                  'selection':selection,'scope':'hardware-bound effective software services; not portable hardware peaks'}
        return result

    def apply(self,result,request,hardware,policy,*,backend='auto'):
        from superpod_compute.contracts import Mapping
        if result['best'] is None:return result
        actual=self.evaluate(request,Mapping(**result['best']['mapping']),hardware,policy,backend=backend)
        signature={**result['signature'],'software_service_profile_sha256':self.sha256}
        return {**result,'signature':signature,'signature_sha256':digest(signature),
                'hardware_plan_best':result['best'],'best':actual,
                'top_candidates':[],
                'optimality':'hardware mapper optimum; effective software services reevaluate that plan, not a second global search',
                'validation_status':'requires held-out measurements for the declared runtime/layout'}
