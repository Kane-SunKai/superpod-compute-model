"""V1 experiment: logical M tiles with conservative physical storage accounting.
No calibration, no changes to production model; original best remains a candidate.
"""
import dataclasses,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from superpod_compute.contracts import Hardware,Policy,GemmRequest,Mapping
from superpod_compute.backends import make_backend
from superpod_compute.operators import GemmModel,align

class LogicalSmallM(GemmModel):
    version='experiment-logical-small-M-physical-MMAC-resources'
    @staticmethod
    def capacity(p,h):
        # Retain padded storage and matrix operand resources. This experiment
        # changes task partitioning, not a claim of sparse physical registers.
        physical=dataclasses.replace(p,bm=align(p.bm,h.instruction_m),bn=align(p.bn,h.instruction_n))
        cap=GemmModel.capacity(physical,h)
        return {**cap,'logical_bm':p.bm,'physical_bm':physical.bm}

def main():
    work=ROOT.parents[2]
    plan=json.loads((work/'graph50-recheck-20261103/frozen_plan.json').read_text())
    predictions=json.loads((work/'llmcompass-method-validation-20261103/predictions.json').read_text())['cases']
    measured={r['id']:r for r in json.loads((work/'graph50-recheck-20261103/measurements.json').read_text())['cases']}
    h=Hardware(**plan['hardware']);p=Policy(**plan['policy']);model=LogicalSmallM();backend=make_backend('auto',h,p.matrix_efficiency)
    target=ROOT/'small_m_results.json';rows=[];start=time.perf_counter()
    for row in sorted(predictions,key=lambda r:r['id']):
        if row['group']!='decode':continue
        req=GemmRequest(*row['shape']);original=row['best'];mapping=Mapping(**original['mapping'])
        fixed=[]
        for bm in (1,2,4,8,16):
            trial=dataclasses.replace(mapping,bm=bm)
            result=model.evaluate(req,h,p,backend,trial)
            if result:fixed.append(dict(bm=bm,best=result))
        best=original;count=feasible=0
        sp=dataclasses.replace(p,bm_values=(1,2,4,8),add_shape_candidate=False)
        for trial in model.candidates(req,h,sp):
            count+=1
            if not model.capacity(trial,h)['valid']:continue
            feasible+=1;result=model.evaluate(req,h,p,backend,trial)
            if model.rank_key(result)<model.rank_key(best):best=result
        record=dict(id=row['id'],shape=row['shape'],baseline=original,best=best,
                    fixed_mapping=fixed,searched=count,feasible=feasible,measured_us=measured[row['id']]['measured_graph_us'])
        rows.append(record)
        target.write_text(json.dumps(dict(completed=False,cases=rows,wall_s=time.perf_counter()-start),indent=2)+'\n')
        print(row['id'],round(original['duration_us'],3),'->',round(best['duration_us'],3),best['mapping'],flush=True)
    target.write_text(json.dumps(dict(completed=True,cases=rows,wall_s=time.perf_counter()-start),indent=2)+'\n')
if __name__=='__main__':main()
