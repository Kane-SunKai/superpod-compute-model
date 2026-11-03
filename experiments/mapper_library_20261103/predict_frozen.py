import hashlib,json,pathlib,time,concurrent.futures
from mapper import predict,model_sources
r=pathlib.Path(__file__).resolve().parent
def one(case):
    plan=json.loads((r/'frozen_plan.json').read_text())
    start=time.perf_counter();result=predict(case['shape'],plan,r/'cache')
    return {**case,**result,'wall_s':time.perf_counter()-start}
if __name__=='__main__':
    plan=json.loads((r/'frozen_plan.json').read_text());out=r/'predictions.json'
    if out.exists():assert not json.loads(out.read_text())['completed'], 'Completed predictions are immutable'
    data=dict(completed=False,plan_sha256=hashlib.sha256((r/'frozen_plan.json').read_bytes()).hexdigest(),sources=model_sources(),cases=[])
    # Independent CPU searches only; GPU measurements remain sequential.
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        for result in pool.map(one,plan['cases']):
            data['cases'].append(result)
            print(result['id'],round(result['wall_s'],2),{k:round(v['best']['duration_us'],3) if v else None for k,v in result['methods'].items()},flush=True)
            out.write_text(json.dumps(data,indent=2)+'\n')
    data['completed']=True;out.write_text(json.dumps(data,indent=2)+'\n')
    (r/'prediction_seal.json').write_text(json.dumps(dict(predictions_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),plan_sha256=data['plan_sha256'],finished_unix=time.time(),count=len(data['cases'])),indent=2)+'\n')
