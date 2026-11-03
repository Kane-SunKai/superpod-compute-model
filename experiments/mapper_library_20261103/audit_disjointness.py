"""Broader historical shape audit, performed BEFORE validation measurements."""
import hashlib,json,pathlib
r=pathlib.Path(__file__).resolve().parent;root=r.parents[1]
known=set();files=[]
def walk(x,found):
 if isinstance(x,dict):
    s=x.get('shape')
    if isinstance(s,list) and len(s)==3 and all(isinstance(a,int) for a in s):found.add(tuple(s))
    for keys in [('M','N','K'),('m','n','k')]:
        if all(isinstance(x.get(a),int) for a in keys):found.add(tuple(x[a] for a in keys))
    for q in x.values():walk(q,found)
 elif isinstance(x,list):
    for q in x:walk(q,found)
for path in (root/'results').rglob('*.json'):
 if path.stat().st_size>30000000:continue
 try:obj=json.loads(path.read_text(encoding='utf-8-sig'))
 except (ValueError,UnicodeError):continue
 shapes=set();walk(obj,shapes);known.update(shapes)
 files.append(dict(path=str(path.relative_to(root)),shape_count=len(shapes),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
plan=json.loads((r/'frozen_plan.json').read_text());known.update(tuple(x['shape']) for x in plan['development_cases'])
assert not (r/'validation_measure.json').exists()
replacements=[];choices=[1216,1344,1408,1664,1856,1920,2304,2688,2816]
for case in plan['cases']:
 if tuple(case['shape']) not in known:continue
 assert case['group']=='square'
 target=next(n for n in choices if (n,n,n) not in known and [n,n,n] not in [x['shape'] for x in plan['cases']])
 replacements.append(dict(old=case['shape'],new=[target]*3,reason='overlaps older stage1 development/holdout; replaced before measurement'))
 case.update(shape=[target]*3,id=f'val_square_{target}x{target}x{target}')
assert not known&{tuple(x['shape']) for x in plan['cases']}
(r/'initial_plan_superseded.json').write_bytes((r/'frozen_plan.json').read_bytes())
(r/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
if (r/'predictions.json').exists():(r/'predictions.json').rename(r/'initial_partial_predictions_superseded.json')
(r/'disjointness_audit.json').write_text(json.dumps(dict(files_scanned=len(files),known_unique_shapes=len(known),replacements=replacements,final_overlap=[],files=files),indent=2)+'\n')
print('files',len(files),'known shapes',len(known),'replacements',replacements,'final overlap ZERO')
