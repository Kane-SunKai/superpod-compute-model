"""Offline optional execution-plan prediction. No GPU calls or timing lookup."""
import argparse,dataclasses,hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parents[1]))
from superpod_compute.contracts import Hardware,Policy,GemmRequest,canonical,digest,integer
from superpod_compute.backends import make_backend
from experimental_model import ReadServiceModel

def predict_plan(shape,execution,hardware,policy,calibration):
    for field in ('bm','bn','bk','split_k'):integer(execution[field],field)
    if execution['split_k']!=1:raise NotImplementedError('Split-K and post-reduction are not implemented in this experimental path')
    if execution.get('layout')!='row-major-contiguous':raise ValueError('Execution plan layout mismatch')
    req=GemmRequest(*shape);eta=calibration['matrix_efficiency'];read_bw=calibration['hbm_bandwidth_Bps']
    p=dataclasses.replace(policy,bm_values=(execution['bm'],),bn_values=(execution['bn'],),bk_values=(execution['bk'],),
                          add_shape_candidate=False,matrix_efficiency=eta,calibration_hardware_sha256='')
    op=ReadServiceModel(read_bw);be=make_backend('auto',hardware,eta);best=None;count=0
    for mapping in op.candidates(req,hardware,p):
        v=op.evaluate(req,hardware,p,be,mapping)
        if v is None:continue
        count+=1
        if best is None or op.rank_key(v)<op.rank_key(best):best=v
    if best is None:raise ValueError('Observed tile has no feasible mapping under the hardware capacity limits')
    return dict(best=best,feasible_mappings=count,scope='observed macro-tile/K-step input; remaining buffer/group mapping optimized by model; no latency lookup',
                physical_mmac_issue_cycles=hardware.instruction_issue_cycles,effective_matrix_efficiency=eta,
                input_hbm_bandwidth_Bps=read_bw,output_hbm_bandwidth_Bps=hardware.hbm_bandwidth_Bps)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for v in ('M','N','K'):ap.add_argument('--'+v,type=int,required=True)
    ap.add_argument('--execution-plan',type=pathlib.Path,required=True)
    ap.add_argument('--calibration',type=pathlib.Path,default=ROOT/'frozen_calibration.json')
    ap.add_argument('--hardware',type=pathlib.Path,default=ROOT.parents[1]/'superpod_compute/data/bw1100.json')
    ap.add_argument('--policy',type=pathlib.Path,default=ROOT.parents[1]/'examples/bw1100_graph_calibrated.json')
    ap.add_argument('--cache-dir',type=pathlib.Path)
    ap.add_argument('--output',type=pathlib.Path,required=True)
    a=ap.parse_args();start=time.perf_counter()
    ex=json.loads(a.execution_plan.read_text(encoding='utf-8-sig'));cal=json.loads(a.calibration.read_text(encoding='utf-8-sig'))
    hw=Hardware.load(a.hardware);po=Policy(**json.loads(a.policy.read_text(encoding='utf-8-sig')))
    calibrated_hardware=Hardware(**json.loads((ROOT/'calibration_plan.json').read_text())['hardware'])
    if hw.to_dict()!=calibrated_hardware.to_dict():
        raise ValueError('This experimental calibration is specific to the frozen BW1100 hardware profile; recalibrate for other hardware')
    shapes=[a.M,a.N,a.K]
    sources={str(p.relative_to(ROOT.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT.parents[1]/'superpod_compute').glob('*.py'))+[ROOT/'experimental_model.py',pathlib.Path(__file__)]}
    signature=dict(shape=shapes,execution=ex,calibration=cal,hardware=hw.to_dict(),policy=po.to_dict(),sources=sources)
    key=digest(signature);cache=(a.cache_dir/(key+'.json')) if a.cache_dir else None
    if cache and cache.exists():
        result=json.loads(cache.read_text());assert result['signature_sha256']==key;hit=True
    else:
        result=predict_plan(shapes,ex,hw,po,cal);result.update(signature_sha256=key,shape=shapes,execution_plan=ex);hit=False
        if cache:cache.parent.mkdir(parents=True,exist_ok=True);cache.write_text(json.dumps(result,indent=2)+'\n')
    result.update(cache_hit=hit,wall_s=time.perf_counter()-start)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('shape','cache_hit','wall_s')},ensure_ascii=False),result['best']['duration_us'],'us')
if __name__=='__main__':main()
