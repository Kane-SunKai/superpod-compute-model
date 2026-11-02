"""Byte-address geometry, separate from cache-hit and latency assumptions.

These are address counts, not measured hardware transactions. In particular,
sector overlap does not imply a cache hit or a single physical transaction.
No GEMM-time correction coefficient is inferred by this module.
"""
from collections import Counter
from dataclasses import dataclass
from math import gcd

from .contracts import integer


@dataclass(frozen=True)
class RowMajorStorage:
    rows: int
    columns: int
    row_stride_elements: int
    element_bytes: int = 2
    base_mod_256_bytes: int = 0

    def __post_init__(self):
        for name in ('rows','columns','row_stride_elements'):
            integer(getattr(self,name),name,0)
        integer(self.element_bytes,'element_bytes')
        if self.element_bytes not in (1,2,4,8,16):
            raise ValueError('Element width must be a supported power of two')
        integer(self.base_mod_256_bytes,'base_mod_256_bytes',0)
        if self.row_stride_elements<self.columns:
            raise ValueError('Row stride must cover all logical columns')
        if self.base_mod_256_bytes>=256:
            raise ValueError('Supply the pointer residue modulo 256 bytes')
        if self.base_mod_256_bytes%self.element_bytes:
            raise ValueError('Element-unaligned storage is not supported')

    @property
    def stride_bytes(self):return self.row_stride_elements*self.element_bytes

    @property
    def logical_bytes(self):return self.rows*self.columns*self.element_bytes

    @property
    def address_span_bytes(self):
        if not self.rows or not self.columns:return 0
        return (self.rows-1)*self.stride_bytes+self.columns*self.element_bytes

    def address(self,row,column):
        integer(row,'row',0);integer(column,'column',0)
        if row>=self.rows or column>=self.columns:raise ValueError('Element outside logical matrix')
        return self.base_mod_256_bytes+row*self.stride_bytes+column*self.element_bytes

    def row_start_residues(self,alignment_bytes):
        """Exact histogram with bounded work even for very large matrices."""
        integer(alignment_bytes,'alignment_bytes')
        if alignment_bytes>256 or 256%alignment_bytes:
            raise ValueError('Alignment must divide the known 256-byte modulus')
        if not self.rows or not self.columns:return {}
        period=alignment_bytes//gcd(self.stride_bytes,alignment_bytes)
        complete,tail=divmod(self.rows,period)
        result=Counter()
        for r in range(period):
            count=complete+(r<tail)
            if count:result[(self.base_mod_256_bytes+r*self.stride_bytes)%alignment_bytes]+=count
        return dict(sorted(result.items()))


def describe_read(addresses,load_bytes,*,sector_bytes=64):
    """One wave's uniform-width read; None denotes a masked inactive lane.

    Caller supplies instruction addresses, NOT just matrix dimensions. Tail
    paths with a different opcode must be separate reads. Sector size is an
    explicit analysis choice, not a claimed BW1100 hardware transaction size.
    """
    integer(load_bytes,'load_bytes')
    integer(sector_bytes,'sector_bytes')
    if load_bytes not in (2,4,8,16):raise ValueError('Supported lane widths: 2,4,8,16 bytes')
    if sector_bytes not in (32,64,128,256):raise ValueError('Choose an explicit address-sector size')
    addresses=tuple(addresses)
    if len(addresses)>64:raise ValueError('Describe at most one 64-lane wave at a time')
    active=[a for a in addresses if a is not None]
    for a in active:integer(a,'lane address',0)
    sectors=set();sum_lane_sectors=0;crossing=0
    for a in active:
        first,last=a//sector_bytes,(a+load_bytes-1)//sector_bytes
        sectors.update(range(first,last+1));sum_lane_sectors+=last-first+1;crossing+=last>first
    unique_bytes=0;end=-1
    for a in sorted(active):
        b=a+load_bytes
        unique_bytes+=max(0,b-max(a,end));end=max(end,b)
    return {'active_lanes':len(active),'load_bytes_per_lane':load_bytes,
            'requested_bytes':len(active)*load_bytes,'unique_address_bytes':unique_bytes,
            'word_unaligned_lanes':sum(a%4!=0 for a in active),
            'vector_unaligned_lanes':sum(a%load_bytes!=0 for a in active),
            'address_sector_bytes':sector_bytes,'unique_address_sectors':len(sectors),
            'sum_per_lane_address_sectors':sum_lane_sectors,'sector_crossing_lanes':crossing,
            'scope':'Address geometry only; no inferred physical transactions, cache hits or latency'}
