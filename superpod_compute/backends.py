"""Matrix service only. No HBM, L2 transfers, or kernel launch charged here.

Fast utilization branches adapted from LLMCompass (BSD-3-Clause); see
LICENSE.LLMCompass and THIRD_PARTY.md. The OS folded-wavefront miss model is
independent and explicitly labeled; it does not invoke SCALE-Sim.
"""
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
from math import ceil
from pathlib import Path
from .contracts import integer, positive

@dataclass(frozen=True)
class TileEstimate:
    cycles: int
    duration_s: float
    source: str
    array_partition_m: int
    array_partition_n: int
    subproblem: tuple

    def to_dict(self):return {**asdict(self),'subproblem':list(self.subproblem)}

class PeakBackend:
    version='peak-1.0'
    def __init__(self,hardware,efficiency=1.0):
        positive(efficiency,'efficiency')
        if efficiency>1:raise ValueError('efficiency must be <= 1')
        self.hw=hardware;self.efficiency=efficiency
        self.estimate=lru_cache(maxsize=8192)(self._estimate)
    @property
    def signature(self):return {'backend':self.version,'efficiency':self.efficiency}
    def _estimate(self,m,n,k):
        for value in (m,n,k):integer(value,'tile dimension')
        cycles=ceil(2*m*n*k/(self.hw.core_peak_flops/self.hw.clock_hz*self.efficiency))
        return TileEstimate(cycles,cycles/self.hw.clock_hz,'peak',1,1,(m,n,k))

class ArrayBackend:
    version='llmcompass-compatible-array-1.0'
    def __init__(self,hardware,efficiency=1.0,lut_path=None):
        positive(efficiency,'efficiency')
        if efficiency>1:raise ValueError('efficiency must be <= 1')
        self.hw=hardware;self.efficiency=efficiency
        self.array_cycles=lru_cache(maxsize=8192)(self._array_cycles)
        self.estimate=lru_cache(maxsize=8192)(self._estimate)
        path=Path(lut_path) if lut_path else Path(__file__).parent/'data/array_cycles.csv'
        data=path.read_bytes();self.lut_hash=hashlib.sha256(data).hexdigest();self.table={}
        for row in csv.reader(data.decode().splitlines()):
            m,n,k,h,w=map(int,row[:5]);key=(m,n,k,h,w,row[5]);value=int(row[6])
            if key in self.table and self.table[key]!=value:raise ValueError('Conflicting cycle LUT entries')
            self.table[key]=value

    @property
    def signature(self):return {'backend':self.version,'efficiency':self.efficiency,'lut_sha256':self.lut_hash,'dataflow':'os','array_split_k':1,'miss_model':'folded-os-wavefront-1.0'}

    @staticmethod
    def os_cycles(m,n,k,height,width):
        # Whole physical-array wavefront startup/drain per output fold. No memory
        # stalls. This formula agrees with every entry in the bundled source LUT;
        # equality on those entries is not a general hardware accuracy guarantee.
        folds=ceil(m/height)*ceil(n/width)
        return max(1,folds*(k+height+width-2)-1)

    def _array_cycles(self,m,n,k):
        for value in (m,n,k):integer(value,'array dimension')
        h,w=self.hw.array_rows,self.hw.array_cols;mac=self.hw.mac_per_cell_per_cycle
        volume=m*n*k/(h*w*max(h,w))
        if m>=h and n>=w:
            if volume>=128:return ceil(m*n*k/h/w/mac/.99),'llmcompass-fast-0.99'
            if volume>=64:return ceil(m*n*k/h/w/mac/.98),'llmcompass-fast-0.98'
        elif m>=h and n<w:
            if k*m/h/max(h,w)>=64:
                return ceil(m*n*k/h/w/mac/(n/w/.98)),'llmcompass-narrow-fast'
        elif m<h and n>=w:
            if k*n/w/max(h,w)>=64:
                return ceil(m*n*k/h/w/mac/(m/h/.98)),'llmcompass-narrow-fast'
        elif k/max(h,w)>=64:
            return ceil(m*n*k/h/w/mac/(m/h*n/w/.98)),'llmcompass-narrow-fast'
        key=(m,n,k,h,w,'os')
        if key in self.table:return ceil(self.table[key]/mac),'llmcompass-lut'
        # Symmetry shortcut is only used on square arrays, unlike an unguarded
        # M/N swap for an arbitrary rectangular accelerator.
        transposed=(n,m,k,h,w,'os')
        if h==w and transposed in self.table:return ceil(self.table[transposed]/mac),'llmcompass-lut-transposed'
        return ceil(self.os_cycles(m,n,k,h,w)/mac),'own-os-wavefront-estimate'

    def _estimate(self,m,n,k):
        for value in (m,n,k):integer(value,'tile dimension')
        options=[];units=self.hw.arrays_per_core
        for pm in range(1,units+1):
            if units%pm:continue
            pn=units//pm;sub=(ceil(m/pm),ceil(n/pn),k)
            cycles,source=self.array_cycles(*sub);cycles=ceil(cycles/self.efficiency)
            options.append(TileEstimate(cycles,cycles/self.hw.clock_hz,source,pm,pn,sub))
        return min(options,key=lambda x:(x.cycles,x.array_partition_m,x.array_partition_n))

def make_backend(name,hardware,efficiency=1.0,lut_path=None):
    if name=='array':return ArrayBackend(hardware,efficiency,lut_path)
    if name=='peak':return PeakBackend(hardware,efficiency)
    raise ValueError('Unknown backend: '+name)
