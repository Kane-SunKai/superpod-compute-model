"""Explicit engine-routed operand storage, with capacity and traffic together.

Each engine owns M/N fragments and reuses its A/B operands. The routed plan
stores one engine-local copy, in lane-major aligned vectors, in Shared. Copies
across engines are written after a CU-local input fetch, not fetched from HBM
again. This is a planned implementation family, not a library kernel decoder.
"""
from functools import lru_cache
from math import ceil
from .backends import mmac_operand_plan
from .shared_memory import SharedReadBanks


@lru_cache(maxsize=8192)
def routed_operands(m,n,k,hardware):
    h=hardware
    if h.matrix_backend not in ('mmac','mac-throughput'):
        raise ValueError('Engine-routed storage requires a matrix-instruction backend')
    # Every matrix instruction consumes a complete K quantum. Padded zeros
    # occupy Shared/register packets as well as compute issue slots.
    k=ceil(k/h.instruction_k)*h.instruction_k
    _,_,pm,pn,_=mmac_operand_plan(m,n,k,h)
    fm=ceil(m/h.instruction_m);fn=ceil(n/h.instruction_n)
    qm=ceil(fm/pm);qn=ceil(fn/pn)
    packet=h.shared_wave_lanes*h.shared_read_vector_bytes
    parts=[]
    for i in range(pm):
        for j in range(pn):
            mi=max(0,min(qm,fm-i*qm))*h.instruction_m
            ni=max(0,min(qn,fn-j*qn))*h.instruction_n
            if mi and ni:
                # Each lane stores the same number of aligned packets; the
                # final packet may carry explicitly padded operand values.
                ab=ceil(2*mi*k/packet)*packet
                bb=ceil(2*ni*k/packet)*packet
                parts.append(ab+bb)
            else:parts.append(0)
    total=sum(parts)
    cycles=None
    if h.shared_bank_count:
        banks=SharedReadBanks(h.shared_bank_count,h.shared_bank_word_bytes,
                              h.shared_wave_lanes,h.shared_read_vector_bytes,
                              h.shared_read_phase_lanes,h.shared_words_per_bank_cycle)
        addresses=tuple(lane*h.shared_read_vector_bytes for lane in range(h.shared_wave_lanes))
        cycles=banks.estimate(addresses).cycles*(total//packet)
    return dict(shared_bytes=total,register_bytes_by_engine=tuple(parts),
                shared_read_cycles=cycles,packet_bytes=packet,
                shared_read_bytes=total,
                placement='separate engine-local A/B copies; lane-major aligned vectors')


def operand_service_time(mapping,k,hardware,policy,compute):
    if mapping.operand_layout=='engine-routed':
        plan=routed_operands(mapping.bm,mapping.bn,k,hardware)
        if plan['shared_read_cycles'] is not None:
            return plan['shared_read_cycles']/hardware.clock_hz,plan
        bw=hardware.shared_read_bytes_per_cycle_per_core*hardware.clock_hz
        if not bw:raise ValueError('Routed operands need a Shared bank model or read service rate')
        return plan['shared_read_bytes']/bw,plan
    bw=(policy.effective_shared_read_bytes_per_cycle_per_core or hardware.shared_read_bytes_per_cycle_per_core)*hardware.clock_hz
    return (compute.shared_read_bytes/bw if bw and compute.shared_read_bytes else 0.0),None
