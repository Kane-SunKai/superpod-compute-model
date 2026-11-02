"""Explicit wavefront ownership for a resident-C matrix workgroup.

Independent implementation: a wave owns a C rectangle; operand registers are
reused within that wave, not across all waves scheduled on the same engine.
An explicit plan is required. This module does not infer a vendor's dispatch.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import ceil

from .backends import TileEstimate
from .contracts import integer
from .operators import GemmModel


@dataclass(frozen=True)
class WavePlan:
    block_m: int
    block_n: int
    wave_m: int
    wave_n: int
    wave_size: int = 64
    shared_panel_stride_bytes: int = 0
    barrier_after_reads: bool = True
    read_waits_for_group_release: bool = False

    def __post_init__(self):
        for name,value in vars(self).items():
            if name in ('barrier_after_reads','read_waits_for_group_release'):
                if not isinstance(value,bool):raise ValueError('Barrier flags must be boolean')
            else:integer(value,name,0 if name=='shared_panel_stride_bytes' else 1)
        if self.block_m%self.wave_m or self.block_n%self.wave_n:
            raise ValueError('Explicit wave rectangles must cover the complete block')

    @property
    def partitions(self):return self.block_m//self.wave_m,self.block_n//self.wave_n

    @property
    def waves(self):
        pm,pn=self.partitions
        return pm*pn


class WaveMmacBackend:
    version='explicit-wave-mmac-experimental-1.0'

    def __init__(self,hardware,plan):
        if hardware.matrix_backend!='mmac':raise ValueError('Explicit MMAC hardware required')
        if plan.wave_m%hardware.instruction_m or plan.wave_n%hardware.instruction_n:
            raise ValueError('Wave rectangles must contain whole matrix fragments')
        self.hw=hardware;self.plan=plan
        self.estimate=lru_cache(maxsize=8192)(self._estimate)

    @property
    def signature(self):return {'backend':self.version,'wave_plan':vars(self.plan),'dispatch':'explicit supplied plan'}

    def _estimate(self,m,n,k):
        if (m,n)!=(self.plan.block_m,self.plan.block_n):raise ValueError('Macroblock differs from supplied wave plan')
        integer(k,'K')
        h=self.hw;p=self.plan;pm,pn=p.partitions
        steps=ceil(k/h.instruction_k)
        per_wave=(p.wave_m//h.instruction_m)*(p.wave_n//h.instruction_n)*steps
        # Waves are assigned round-robin to matrix engines. A fixed assignment
        # accounts for the final incomplete round, not an ideal fractional CU.
        cycles=ceil(p.waves/h.arrays_per_core)*per_wave*h.instruction_issue_cycles
        operand_bytes=p.waves*(p.wave_m+p.wave_n)*k*2
        return TileEstimate(cycles,cycles/h.clock_hz,'explicit-wave-mmac',pm,pn,(m,n,k),
                            'wave-owned C rectangles; intra-wave operand reuse',
                            per_wave*p.waves,min(p.waves,h.arrays_per_core),operand_bytes)


class WaveGemmModel(GemmModel):
    """Diagnostic plugin: use the same wave-owned operands for capacity/time.

    Buffer counts are still explicit mapping inputs. This does not claim to
    reproduce compiler register allocation, alignment, tail code, or epilogue.
    """
    version='wave-owned-resident-c-experimental-1.0'

    def __init__(self,backend):self.backend=backend

    def capacity(self,mapping,hardware):
        cap=super().capacity(mapping,hardware)
        bytes_per_panel=self.backend.estimate(mapping.bm,mapping.bn,mapping.bk).shared_read_bytes
        new_operands=mapping.register_input_buffers*bytes_per_panel
        cap['register_bytes']+=new_operands-cap['operand_register_bytes']
        cap['operand_register_bytes']=new_operands
        stride=self.backend.plan.shared_panel_stride_bytes
        if stride:
            logical_bytes=(mapping.bm+mapping.bn)*mapping.bk*2
            if stride<logical_bytes:raise ValueError('Shared panel stride cannot be smaller than its operands')
            cap['shared_input_bytes']=mapping.shared_buffers*stride
            cap['shared_bytes']=max(cap['shared_input_bytes'],cap['shared_output_bytes'])
        cap['rejections']=[r for r in cap['rejections'] if r not in ('registers','shared')]
        if cap['register_bytes']>hardware.register_budget_bytes_per_core:cap['rejections'].append('registers')
        if cap['shared_bytes']>hardware.shared_budget_bytes_per_core:cap['rejections'].append('shared')
        cap['valid']=not cap['rejections']
        cap['operand_ownership']='wavefront, not engine'
        return cap


class ScheduledWaveGemmModel(WaveGemmModel):
    version='explicit-wave-scheduling-experimental-1.0'

    def run_operand_pipeline(self,segments,mapping,hardware,**kwargs):
        from .wave_scheduler import wave_operand_pipeline
        return wave_operand_pipeline(segments,mapping.l2_buffers,mapping.shared_buffers,
                                     mapping.register_input_buffers,self.backend.plan,hardware.arrays_per_core,**kwargs)
