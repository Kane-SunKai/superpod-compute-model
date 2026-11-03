import json,statistics as st,sys,time
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from superpod_compute import Hardware,Policy,GemmRequest,Mapping,predict,evaluate_mapping
from superpod_compute.kernel_plan import parse_tensile_kernel
OUT=ROOT/'results/stage1_20261103'
plan=json.loads((OUT/'calibration_plan.json').read_text())
measure=json.loads((OUT/'calibration_measurements.json').read_text())
metadata=json.loads((OUT/'calibration_metadata.json').read_text())
meta={r['id']:r for r in metadata['cases']}
base=json.loads((ROOT/'results/regular50_20261103/frozen_plan.json').read_text())['baseline']
h=Hardware(**base['hardware']);p=Policy(**base['policy'])
out=[]
for row in plan['cases']:
    names=[r for r in meta[row['id']]['kernels'] if r['name'].startswith('Cijk_')]
    actual=st.mean(r['timing']['p50_us'] for r in measure['cases'] if r['id']==row['id'])
    record=dict(row,actual_us=actual)
    try:
        assert len(names)==1
        entry=names[0]
        kp=parse_tensile_kernel(entry['name'],orientation='row-major-transposed-blas')
        grid=kp.validate_hip_grid(GemmRequest(*row['shape']),entry['args']['grid'],entry['args']['block'])
        record.update(plan=kp.to_dict(),grid=grid)
        pp=replace(p,bm_values=(kp.bm,),bn_values=(kp.bn,),bk_values=(max(16,kp.depth_u),),add_shape_candidate=False)
        best=predict(GemmRequest(*row['shape'],rhs_storage='nk-contiguous-transposed'),h,policy=pp)['best']
        record['mapping']=best['mapping'];record['variants']={'macro_only':best['duration_us']}
        mn_policy=replace(p,bm_values=(kp.bm,),bn_values=(kp.bn,),add_shape_candidate=False)
        mn=predict(GemmRequest(*row['shape'],rhs_storage='nk-contiguous-transposed'),h,policy=mn_policy)['best']
        record['mn_mapping']=mn['mapping'];record['variants']['macro_mn_only']=mn['duration_us']
        for eff in [.8,.7]:
            z=evaluate_mapping(GemmRequest(*row['shape'],rhs_storage='nk-contiguous-transposed'),Mapping(**mn['mapping']),h,
                               policy=replace(mn_policy,matrix_efficiency=eff),trace_limit=0)
            record['variants']['mn_eff_'+str(eff)]=z['duration_us']
        for label,lat,eff in [('ready_417',417e-9,1.),('compute_80',0,.8),('both',417e-9,.8)]:
            estimate=evaluate_mapping(GemmRequest(*row['shape'],rhs_storage='nk-contiguous-transposed'),Mapping(**best['mapping']),h,
                                      policy=replace(pp,input_read_latency_s=lat,matrix_efficiency=eff),trace_limit=0)
            record['variants'][label]=estimate['duration_us']
        print(row['id'],round(actual,3),grid['matches'],kp.bm,kp.bn,kp.depth_u,{k:round(v,3) for k,v in record['variants'].items()},flush=True)
    except (ValueError,AssertionError) as ex:record['error']=str(ex);print('unsupported',row['id'],str(ex),flush=True)
    out.append(record)
    (OUT/'template_ablation.json').write_text(json.dumps(out,indent=2)+'\n')
for variant in ['macro_only','macro_mn_only','mn_eff_0.8','mn_eff_0.7','ready_417','compute_80','both']:
    rr=[r for r in out if variant in r.get('variants',{})]
    e=[abs(r['variants'][variant]/r['actual_us']-1)*100 for r in rr]
    print(variant,len(e),sum(v<=10 for v in e),st.mean(e),flush=True)
