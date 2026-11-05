"""Small address/resource model for a wave's Shared vector-read service.

This component is optional. The existing mapper does not silently acquire
addresses from a vendor library or change its service rate. Callers must
provide the planned access addresses and a hardware phase/bank description.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import ceil

from .contracts import integer


@dataclass(frozen=True)
class SharedReadEstimate:
    cycles: int
    requested_bytes: int
    serviced_bytes: int
    phase_cycles: tuple


@dataclass(frozen=True)
class SharedReadBanks:
    bank_count: int
    bank_word_bytes: int
    wave_lanes: int
    vector_bytes: int
    phase_lanes: tuple
    words_per_bank_cycle: int = 1

    def __post_init__(self):
        for key in ('bank_count','bank_word_bytes','wave_lanes','vector_bytes','words_per_bank_cycle'):
            integer(getattr(self,key),key)
        if self.vector_bytes % self.bank_word_bytes:
            raise ValueError('Vector width must contain an integer number of bank words')
        groups=tuple(tuple(g) for g in self.phase_lanes)
        if not groups or any(not g for g in groups):
            raise ValueError('Phase groups must be nonempty')
        for group in groups:
            for lane in group:integer(lane,'phase lane',0)
        flat=[lane for group in groups for lane in group]
        if sorted(flat)!=list(range(self.wave_lanes)):
            raise ValueError('Each lane must appear in exactly one service phase')
        object.__setattr__(self,'phase_lanes',groups)

    @lru_cache(maxsize=8192)
    def estimate(self, byte_addresses):
        """byte_addresses: one planned aligned vector start per wave lane.

        Equal-word reads within a phase use broadcast. Distinct words served
        by the same bank serialize. Separate service phases are added; banks
        within a phase act concurrently. There is no instruction execution loop.
        """
        if not isinstance(byte_addresses,tuple):
            raise TypeError('Use an immutable address tuple for cached estimation')
        if len(byte_addresses)!=self.wave_lanes:
            raise ValueError('Expected one start address per wave lane')
        for addr in byte_addresses:
            integer(addr,'address',0)
            if addr%self.vector_bytes:
                raise ValueError('Vector starts must be aligned to the vector width')
        phase_cycles=[];serviced=0
        for lanes in self.phase_lanes:
            banks=[set() for _ in range(self.bank_count)]
            for lane in lanes:
                first=byte_addresses[lane]//self.bank_word_bytes
                for offset in range(self.vector_bytes//self.bank_word_bytes):
                    word=first+offset
                    banks[word%self.bank_count].add(word)
            phase_cycles.append(ceil(max(map(len,banks))/self.words_per_bank_cycle))
            serviced+=sum(map(len,banks))*self.bank_word_bytes
        return SharedReadEstimate(sum(phase_cycles),self.wave_lanes*self.vector_bytes,
                                  serviced,tuple(phase_cycles))


def cdna_b128_read_banks():
    """Public AMD b128 lane-phase pattern; BW1100 use requires validation.

    Source: AMD Composable Kernel documentation, 'Understanding AMD GPU LDS
    and Bank Conflicts', Read Access Pattern. No vendor BW1100 code is used.
    """
    pairs=((0,20),(4,16),(8,28),(12,24),(32,52),(36,48),(40,60),(44,56))
    groups=tuple(tuple(range(a,a+4))+tuple(range(b,b+4)) for a,b in pairs)
    return SharedReadBanks(32,4,64,16,groups)
