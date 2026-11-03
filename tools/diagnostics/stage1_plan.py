"""Independent software-service calibration; original fifty never train parameters."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/stage1_20261103'
OUT.mkdir(parents=True,exist_ok=True)
cases=[]
def add(group,m,n,k):
    cases.append(dict(id=f'cal_{len(cases):02d}_{m}x{n}x{k}',group=group,
                      shape=[m,n,k],role='calibration'))
for m in [1,3,6,12,24,48]:
    for n,k in [(4096,4096),(12288,8192),(24576,8192),(32768,8192)]:add('decode',m,n,k)
for q in [320,640,896,1280,1792,2560,3584]:add('square',q,q,q)
for m in [640,1280,2560]:
    for n,k in [(8192,8192),(24576,8192),(8192,24576)]:add('prefill',m,n,k)
for k in [192,384,576,896,1536,3072,6144,12288,24576,32768]:add('long_k',640,896,k)
for v in [(6144,192,1536),(192,6144,1536),(2560,320,768),(320,2560,768),
          (1920,1408,1024),(3584,2688,1024),(1152,4096,4096),(1153,4096,4096)]:add('aspect_boundary',*v)
assert len(cases)==58
original=json.loads((ROOT/'results/regular50_20261103/frozen_plan.json').read_text())
assert not ({tuple(c['shape']) for c in cases}&{tuple(c['shape']) for c in original['cases']})
holdout=[(5,8192,4096),(12,16384,8192),(24,24576,4096),(96,6144,8192),
         (640,14336,4096),(1280,4096,14336),(1792,8192,8192),(1152,1152,1152),
         (2304,2304,2304),(640,896,1984),(1151,4096,4096),(1154,4096,4096)]
assert not ({tuple(c['shape']) for c in cases}|{tuple(c['shape']) for c in original['cases']})&set(holdout)
plan=dict(shape_order='M,N,K',cases=cases,source='AIOB/Megatron torch.matmul(X,W.t()); FP16/no bias',
          protocol=dict(rounds=3,event_samples=25,target_input_working_set_bytes=64*1024**2,
                        minimum_ring_slots=4,minimum_graph_calls=64,modes=['rotating']),
          holdout=[dict(id=f'holdout_{i:02d}',group='unseen',shape=list(v),role='heldout') for i,v in enumerate(holdout)],
          original_plan_sha256=hashlib.sha256((ROOT/'results/regular50_20261103/frozen_plan.json').read_bytes()).hexdigest(),
          sources=['https://github.com/aliyun/aicb/blob/master/workload_generator/mocked_model/training/AiobMegatron.py#L172',
                   'https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py'])
target=OUT/'calibration_plan.json'
if target.exists():assert json.loads(target.read_text())==plan
else:target.write_text(json.dumps(plan,indent=2)+'\n')
print('Frozen',len(cases),'calibration and',len(holdout),'unmeasured holdout cases')
