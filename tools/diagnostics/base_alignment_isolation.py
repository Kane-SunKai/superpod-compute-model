"""Keep GEMM dimensions/strides fixed; vary only A/B/C storage offsets."""
import hashlib
from pathlib import Path
import boundary_experiment

def plan():
    offsets=[(0,0,0),(2,0,0),(4,0,0),(8,0,0),(16,0,0),(0,2,0),(0,0,2)]
    sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return [dict(id=f'base_alignment_{m}x{n}x1024_a{a}_b{b}_c{c}',group='base_alignment',
                 shape=[m,n,1024],layout='contiguous-offset',base_offsets_bytes=[a,b,c],
                 role='diagnosis',construction_sha256=sha)
            for m,n in ((1536,1152),(3072,2304)) for a,b,c in offsets]

def tensors(torch,row):
    m,n,k=row['shape'];values=[]
    for index,((rows,cols),offset) in enumerate(zip(((m,k),(k,n),(m,n)),row['base_offsets_bytes'])):
        if offset%2:raise ValueError('FP16 offsets must be multiples of 2 bytes')
        size=rows*cols;skip=offset//2
        if index<2:storage=torch.randn(size+skip,device='cuda',dtype=torch.float16)*.1
        else:storage=torch.empty(size+skip,device='cuda',dtype=torch.float16)
        view=storage[skip:].view(rows,cols)
        if not view.is_contiguous() or view.stride()!=(cols,1):raise RuntimeError('Unexpected stride')
        values.append(view)
    return tuple(values)

if __name__=='__main__':
    boundary_experiment.plan=plan
    boundary_experiment.tensors=tensors
    boundary_experiment.main()
