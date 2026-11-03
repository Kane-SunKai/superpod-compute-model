import json,pathlib
import torch,triton,triton.language as tl
r=pathlib.Path('/workspace/mapper_library_20261103')
@triton.jit
def kernel(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for kb in range(tl.cdiv(K,BK)):
        aa=tl.load(A+m[:,None]*K+(kb*BK+k[None,:]),(m[:,None]<M)&(kb*BK+k[None,:]<K),0)
        bb=tl.load(B+(kb*BK+k[:,None])*N+n[None,:],(kb*BK+k[:,None]<K)&(n[None,:]<N),0)
        acc+=tl.dot(aa,bb)
    tl.store(C+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))
if __name__=='__main__':
    a=torch.randn(32,128,device='cuda',dtype=torch.float16)*.1;b=torch.randn(128,256,device='cuda',dtype=torch.float16)*.1;c=torch.empty(32,256,device='cuda',dtype=torch.float16)
    obj=kernel[(2,2)](a,b,c,32,256,128,16,128,32,num_warps=4,num_stages=1)
    torch.cuda.synchronize();ref=a.float()@b.float();err=float((c.float()-ref).abs().max());ok=torch.allclose(c.float(),ref,atol=.003,rtol=.02)
    result=dict(torch=torch.__version__,triton=triton.__version__,device=torch.cuda.get_device_name(0),correct=ok,max_abs_error=err,
                n_regs=getattr(obj,'n_regs',None),n_spills=getattr(obj,'n_spills',None),shared_bytes=getattr(obj.metadata,'shared',None))
    (r/'env_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
    assert ok
