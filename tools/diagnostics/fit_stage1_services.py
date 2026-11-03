"""Calibrate stage services on independent shapes only, never acceptance times."""
import hashlib,json,math,statistics as st,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from superpod_compute import GemmRequest,Hardware,Policy,Mapping
from superpod_compute.contracts import digest
from tools.diagnostics.experimental_stage_services import IDENTITY,FACTOR_NAMES,ServiceProfile,evaluate_services
OUT=ROOT/'results/stage1_20261103'

def load_rows():
    points=[]
    for pm,mm,md in [('geometry_predictions.json','calibration_measurements.json','calibration_metadata.json'),
                     ('extension_predictions.json','service_extension_measurements.json','service_extension_metadata.json')]:
        pp=json.loads((OUT/pm).read_text());measure=json.loads((OUT/mm).read_text());metadata=json.loads((OUT/md).read_text())
        assert pp['completed'] and measure['completed'] and metadata['completed']
        meta={r['id']:r for r in metadata['cases']}
        for row in pp['cases']:
            timings=[r['timing']['p50_us'] for r in measure['cases'] if r['id']==row['id']]
            assert len(timings)==3
            kk=[r['name'] for r in meta[row['id']]['kernels'] if r['name'].startswith('Cijk_')]
            points.append(dict(row,actual_us=st.mean(timings),spread_pct=(max(timings)-min(timings))/st.mean(timings)*100,
                               family=kk[0] if len(kk)==1 else None))
    return points

def main():
    base=json.loads((ROOT/'results/regular50_20261103/frozen_plan.json').read_text())['baseline']
    h=Hardware(**base['hardware']);p=Policy(**base['policy']);data=load_rows()
    forbidden={tuple(r['shape']) for r in json.loads((ROOT/'results/regular50_20261103/frozen_plan.json').read_text())['cases']}
    forbidden|={tuple(r['shape']) for r in json.loads((OUT/'calibration_plan.json').read_text())['holdout']}
    assert not forbidden&{tuple(r['shape']) for r in data}
    points=[];diagnostics=[]
    bounds={'matrix_scale':(1.,64.),'hbm_scale':(h.hbm_bandwidth_Bps/2e12,64.),
            'shared_operand_scale':(.5,64.),'l2_read_scale':(1.,64.)}
    for row in data:
        req=GemmRequest(*row['shape'],rhs_storage='nk-contiguous-transposed');mapping=Mapping(**row['best']['mapping'])
        target=row['actual_us']
        def evaluate(f):return evaluate_services(req,mapping,h,p,f)['duration_us']
        choices=[];reference=evaluate(IDENTITY)
        for name,(lo,hi) in bounds.items():
            low={**IDENTITY,name:lo};high={**IDENTITY,name:hi}
            a,b=evaluate(low),evaluate(high)
            if a>target or b<target:continue
            for _ in range(24):
                mid=(lo+hi)/2;v=evaluate({**IDENTITY,name:mid})
                if v<target:lo=mid
                else:hi=mid
            f={**IDENTITY,name:(lo+hi)/2}
            choices.append((abs(math.log(f[name])),name,f,evaluate(f)))
        if choices:
            _,kind,fitted,training=min(choices)
        else:
            kind='unresolved';fitted=dict(IDENTITY);training=reference
        points.append(dict(id=row['id'],shape=row['shape'],role='calibration',family=row['family'],
                           primary_service=kind,factors=fitted))
        diagnostics.append(dict(id=row['id'],shape=row['shape'],measured_us=target,reference_us=reference,
                                reconstruction_us=training,primary_service=kind,spread_pct=row['spread_pct'],mapping=mapping.to_dict()))
        print(row['id'],kind,round(reference,3),'->',round(training,3),'target',round(target,3),flush=True)
    profile=dict(schema='software-stage-services-1',hardware_sha256=digest(h.to_dict()),dtype='fp16',
                 rhs_storage='nk-contiguous-transposed',minimum_hbm_scale=h.hbm_bandwidth_Bps/2e12,
                 runtime='BW1100 gfx938; PyTorch 2.11.0 / HIP 6.3.26113; AIOB-style matmul(X,W.T)',
                 interpolation='linear',points=points,
                 scope='Effective software stage services, not measured physical hardware peaks. Independent calibration only; fixed hardware mapper plans.',
                 original_acceptance_excluded=True,unseen_holdout_excluded=True)
    # Select interpolation only by leaving out independent calibration shapes.
    loo=[]
    for method in ['linear','nearest']:
        errors=[]
        for i,row in enumerate(data):
            v={**profile,'interpolation':method,'points':points[:i]+points[i+1:]}
            svc=ServiceProfile(v)
            req=GemmRequest(*row['shape'],rhs_storage='nk-contiguous-transposed')
            result=svc.evaluate(req,Mapping(**row['best']['mapping']),h,p)
            error=100*abs(result['duration_us']/row['actual_us']-1)
            errors.append(error)
        loo.append(dict(method=method,n=len(errors),mape=st.mean(errors),passes10=sum(e<=10 for e in errors),
                        median=st.median(errors),errors=errors))
    chosen=min(loo,key=lambda r:(r['mape'],-r['passes10']))['method']
    profile['interpolation']=chosen
    profile['calibration_leave_one_out']=[{k:v for k,v in r.items() if k!='errors'} for r in loo]
    (OUT/'service_profile_v1.json').write_text(json.dumps(profile,indent=2)+'\n')
    (OUT/'service_fit_diagnostics.json').write_text(json.dumps(dict(rows=diagnostics,loo=loo),indent=2)+'\n')
    print('LOO',profile['calibration_leave_one_out'],'chosen',chosen,flush=True)

if __name__=='__main__':main()
