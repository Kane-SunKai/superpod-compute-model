"""Explicit execution-plan evidence for comparing GEMM model and BLAS kernels.

This adapter does NOT predict a BLAS dispatch decision. It parses a supplied
Tensile-style kernel name and diagnoses how its macrotiles cover the requested
matrix. Row-major torch.mm uses the transposed BLAS problem C^T = B^T A^T:
kernel dimension 0 is logical N, dimension 1 is logical M. Callers must state
this orientation; it must not be inferred from a kernel-name substring.

Parsed vector width is a static kernel hint, NOT a guarantee that every load
executes at that width. Boundary branches can change executed instructions.
"""
from dataclasses import dataclass, asdict, replace
from math import ceil, prod
import re

from .contracts import integer


@dataclass(frozen=True)
class KernelPlan:
    name: str
    macro_0: int
    macro_1: int
    depth_u: int
    global_split_u: int
    workgroup: tuple
    global_read_vector_width: int
    prefetch_global: int
    prefetch_local: int
    direct_to_lds: bool
    orientation: str
    provenance: str = 'supplied kernel name; not a shape-only dispatch prediction'

    def __post_init__(self):
        for field in ('macro_0','macro_1','depth_u','global_split_u','global_read_vector_width'):
            integer(getattr(self,field),field)
        if len(self.workgroup)!=3:raise ValueError('workgroup must have three axes')
        for value in self.workgroup:integer(value,'workgroup axis')
        for field in ('prefetch_global','prefetch_local'):integer(getattr(self,field),field,0)
        if self.orientation not in ('row-major-transposed-blas','logical-mn'):
            raise ValueError('Declare whether BLAS axes are transposed relative to the request')

    @property
    def bm(self):
        return self.macro_1 if self.orientation=='row-major-transposed-blas' else self.macro_0

    @property
    def bn(self):
        return self.macro_0 if self.orientation=='row-major-transposed-blas' else self.macro_1

    @property
    def threads(self):return prod(self.workgroup)

    def to_dict(self):
        return {**asdict(self),'logical_bm':self.bm,'logical_bn':self.bn,
                'threads':self.threads,'timing_claim':'none; structural execution plan only'}

    def coverage(self, request, cores):
        integer(cores,'cores')
        tm,tn=ceil(request.M/self.bm),ceil(request.N/self.bn)
        blocks=tm*tn*self.global_split_u
        full=(request.M//self.bm)*(request.N//self.bn)*self.global_split_u
        return {'tiles_m':tm,'tiles_n':tn,'output_tiles':tm*tn,
                'main_kernel_workgroups':blocks,'full_output_workgroups':full,
                'boundary_output_workgroups':blocks-full,
                'one_resident_block_per_cu_rounds':ceil(blocks/cores),
                'rounds_assumption':'one resident workgroup per CU; not measured occupancy',
                'padded_output_elements':tm*self.bm*tn*self.bn,
                'k_remainder_depth_u':request.K%self.depth_u,
                'split_k_requires_reduction':self.global_split_u>1}

    def validate_hip_grid(self, request, grid, block):
        """HIP profiler grid is global work-items, not CUDA gridDim block count.

        All three axes matter; the flattened Y dimension can include split-K.
        This check applies only to the main matrix kernel, not post-reduction.
        """
        if len(grid)!=3 or len(block)!=3:raise ValueError('Expected 3D HIP grid and block')
        for x in (*grid,*block):integer(x,'launch dimension')
        observed_items,threads=prod(grid),prod(block)
        if observed_items%threads:raise ValueError('Global work-items not divisible by block threads')
        observed=observed_items//threads
        expected=ceil(request.M/self.bm)*ceil(request.N/self.bn)*self.global_split_u
        return {'observed_workgroups':observed,'expected_workgroups':expected,
                'observed_threads':threads,'expected_threads':self.threads,
                'matches':observed==expected and threads==self.threads}


def parse_tensile_kernel(name, *, orientation):
    def extract(pattern, label):
        match=re.search(pattern,name)
        if not match:raise ValueError('Missing '+label+' in kernel name')
        return tuple(map(int,match.groups()))
    mt=extract(r'(?:^|_)MT(\d+)x(\d+)x(\d+)(?:_|$)','MT')
    wg=extract(r'(?:^|_)WG(\d+)_(\d+)_(\d+)(?:_|$)','WG')
    def scalar(tag):return extract(r'(?:^|_)'+tag+r'(\d+)(?:_|$)',tag)[0]
    return KernelPlan(name,*mt,scalar('GSU'),wg,scalar('GRVW'),scalar('PGR'),scalar('PLR'),bool(scalar('DTL')),orientation)


def evaluate_macrotiling_ablation(request, hardware, policy, plan):
    """Hold logical bm/bn/depth_u to observed kernel; reuse the old cost model.

    Remaining buffering/grouping is still optimized by the model. This isolates
    the influence of macrotiling; it is not an accurate BLAS-kernel simulator.
    Reject Split-K because dropping its reduction would manufacture a result.
    """
    from .api import predict
    if plan.global_split_u!=1:
        raise NotImplementedError('Split-K requires partial outputs and an explicit reduction model')
    constrained=replace(policy,bm_values=(plan.bm,),bn_values=(plan.bn,),
                        bk_values=(plan.depth_u,),add_shape_candidate=False)
    result=predict(request,hardware,policy=constrained)
    return {**result,'kernel_plan':plan.to_dict(),
            'comparison_scope':'macrotiling ablation only; buffering, memory alignment, tail instruction paths and occupancy are not inferred from the kernel',
            'coverage':plan.coverage(request,hardware.cores)}
