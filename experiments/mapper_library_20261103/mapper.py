"""Experimental transferable tile candidates; never reads measured latency."""
import dataclasses, hashlib, json, math, pathlib, sys, time
R = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(R.parents[1]))
sys.path.insert(0, str(R.parent / 'pipeline_20261103'))
from superpod_compute.contracts import Hardware, Policy, GemmRequest, digest
from superpod_compute.backends import make_backend
from superpod_compute.operators import GemmModel
from experimental_model import ReadServiceModel

def retrieve(shape, library, limit=5):
    ranked = sorted((x for x in library if x['supported']),
                    key=lambda x: (sum(abs(math.log2(a / b)) for a, b in zip(shape, x['shape'])), x['id']))
    found, seen = [], set()
    for x in ranked:
        key = tuple(x['tile'])
        if key in seen: continue
        seen.add(key)
        found.append(dict(tile=list(key), source_id=x['id'], source_shape=x['shape'],
                          distance=sum(abs(math.log2(a / b)) for a, b in zip(shape, x['shape']))))
        if len(found) == limit: break
    return found

def search(shape, hw, policy, read_bw=None, tiles=None):
    """Enumerate the existing physical capacity / pipeline model unchanged."""
    start = time.perf_counter(); op = ReadServiceModel(read_bw); req = GemmRequest(*shape)
    op.validate(req, hw); backend = make_backend('auto', hw, policy.matrix_efficiency)
    policies = [policy] if tiles is None else [dataclasses.replace(policy,
        bm_values=(t[0],), bn_values=(t[1],), bk_values=(t[2],), add_shape_candidate=False) for t in tiles]
    best = None; total = feasible = 0
    for p in policies:
        for mapping in op.candidates(req, hw, p):
            total += 1
            v = op.evaluate(req, hw, p, backend, mapping)
            if v is None: continue
            feasible += 1
            if best is None or op.rank_key(v) < op.rank_key(best): best = v
    return dict(best=best, candidates=total, feasible=feasible, search_seconds=time.perf_counter()-start)

def model_sources():
    paths = list((R.parents[1] / 'superpod_compute').glob('*.py')) + [pathlib.Path(__file__), R.parent/'pipeline_20261103/experimental_model.py']
    return {str(p.relative_to(R.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def predict(shape, plan, cache_dir=None, hardware=None):
    h = Hardware(**(hardware or plan['hardware'])); p = Policy(**plan['policy'])
    # Software tile hints can transfer; measured BW1100 service parameters cannot.
    matched = h.to_dict() == Hardware(**plan['hardware']).to_dict()
    calibrated = dataclasses.replace(p, calibration_hardware_sha256='',
        matrix_efficiency=plan['frozen_cost']['matrix_efficiency'] if matched else 1.,
        device_service_s=p.device_service_s if matched else 0.)
    bw = plan['frozen_cost']['input_hbm_bandwidth_Bps'] if matched else h.hbm_bandwidth_Bps
    signature = dict(shape=shape, hardware=h.to_dict(), policy=calibrated.to_dict(), read_bw=bw,
        library=plan['development_cases'], retrieval=plan['retrieval'], sources=model_sources(),
        layout='row-major-contiguous', dtype='fp16', version='library-mapper-1')
    key = digest(signature); path = pathlib.Path(cache_dir)/(key+'.json') if cache_dir else None
    if path and path.exists():
        result = json.loads(path.read_text()); assert result['signature_sha256'] == key
        return {**result, 'cache_hit':True}
    entries = retrieve(shape, plan['development_cases'], plan['retrieval']['distinct_templates'])
    evaluations = []
    for entry in entries:
        evaluations.append({**entry, **search(shape,h,calibrated,bw,[entry['tile']])})
    valid = [x for x in evaluations if x['best'] is not None]
    automatic = search(shape,h,calibrated,bw)
    # Distinguish library dispatch emulation from model-minimum selection.
    nearest = valid[0] if valid else automatic
    library_min = min(valid,key=lambda x:GemmModel.rank_key(x['best'])) if valid else automatic
    union = min([automatic,library_min], key=lambda x:GemmModel.rank_key(x['best']))
    original = search(shape,h,dataclasses.replace(p,calibration_hardware_sha256='')) if matched else None
    result = dict(shape=shape,signature_sha256=key,cache_hit=False,calibration_applied=matched,
                  entries=evaluations, methods=dict(original=original,automatic=automatic,
                  nearest=nearest,library_min=library_min,union=union),
                  limitations='Single-kernel Split-K=1 only; library tiles are candidates, not certified library dispatch. Different hardware uses its own bandwidth, eta=1 and no calibrated device service.')
    if path:
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--M',type=int,required=True);ap.add_argument('--N',type=int,required=True);ap.add_argument('--K',type=int,required=True)
    ap.add_argument('--hardware',type=pathlib.Path);ap.add_argument('--output',type=pathlib.Path,required=True)
    a=ap.parse_args();plan=json.loads((R/'frozen_plan.json').read_text())
    result=predict([a.M,a.N,a.K],plan,R/'cache',json.loads(a.hardware.read_text()) if a.hardware else None)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('cache_hit',result['cache_hit'],{k: v['best']['duration_us'] if v else None for k,v in result['methods'].items()})
