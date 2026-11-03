"""Freeze independent calibration cases before seeing their measurements."""
import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
WORK=ROOT.parents[2]
old=json.loads((WORK/'graph50-recheck-20261103/frozen_plan.json').read_text())
cases=[]
def add(group,m,n,k):
    cases.append(dict(id=f'cal_{group}_{m}x{n}x{k}',group=group,shape=[m,n,k],role='calibration'))
for m in (1,8,32):
    for n in (6144,12288,24576):add('decode',m,n,6144)
for k in (4096,12288):add('decode',8,12288,k)
for m,n,k in [(384,6144,6144),(768,6144,6144),(1536,6144,6144),
              (768,24576,6144),(1536,24576,6144),(768,6144,24576),(1536,6144,24576),
              (192,192,192),(320,320,320),(640,640,640),(1280,1280,1280)]:
    add('square' if m==n==k else 'prefill',m,n,k)
assert not ({tuple(c['shape']) for c in cases}&{tuple(c['shape']) for c in old['cases']})
plan={**old,'cases':cases,'stage':'independent calibration; frozen before measurement',
      'acceptance_ids':[c['id'] for c in old['cases'] if c['group']!='boundary']}
target=ROOT/'calibration_plan.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(plan,indent=2)+'\n')
print(len(cases),'calibration;',len(plan['acceptance_ids']),'acceptance;',hashlib.sha256(target.read_bytes()).hexdigest())
