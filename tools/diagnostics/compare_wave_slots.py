"""Independent ablations of wave ownership and supplied Shared slot counts.

Only two execution plans have instruction evidence in this experiment. These
are structural inputs, not regressions of measured GEMM time. Other shapes
remain explicitly outside this diagnostic; this is not the final target scope.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from superpod_compute import Hardware,Policy,GemmRequest,OperatorRegistry,predict
from superpod_compute.kernel_plan import parse_tensile_kernel
from superpod_compute.wave_plan import WavePlan,WaveMmacBackend,WaveGemmModel,ScheduledWaveGemmModel


class PinnedWaveModel(WaveGemmModel):
    def __init__(self,backend,slots,register_slots):
        super().__init__(backend);self.slots=slots;self.register_slots=register_slots

    def candidates(self,request,hardware,policy):
        for mapping in super().candidates(request,hardware,policy):
            if mapping.shared_buffers==self.slots and mapping.register_input_buffers==self.register_slots and mapping.l2_buffers in (1,2):
                yield mapping


class PinnedScheduledWaveModel(PinnedWaveModel,ScheduledWaveGemmModel):
    version=ScheduledWaveGemmModel.version


class MeasuredReadWaveModel(PinnedScheduledWaveModel):
    """Sensitivity only: effective LDS instruction-mix service, not peak BW.

    The measured phase includes loop/checksum overhead. The barrier-inclusive
    variant is an alternative whole-phase sensitivity, not a second barrier
    latency added to the no-barrier measurement. Only full K16 panels qualify.
    """
    version='wave-lds-phase-sensitivity-1.0'

    def __init__(self,backend,slots,register_slots,panel_seconds):
        super().__init__(backend,slots,register_slots)
        self.panel_seconds=panel_seconds

    def run_operand_pipeline(self,segments,mapping,hardware,**kwargs):
        if mapping.bk!=16:raise ValueError('Measured phase only covers K16')
        reference=self.backend.estimate(mapping.bm,mapping.bn,16)
        original=reference.shared_read_bytes/(hardware.shared_read_bytes_per_cycle_per_core*hardware.clock_hz)
        updated=[]
        for count,h,r,s,c in segments:
            if abs(s-original)>max(1e-15,original*1e-12):
                raise ValueError('Measured phase cannot represent a partial K panel')
            updated.append((count,h,r,self.panel_seconds,c))
        result=super().run_operand_pipeline(updated,mapping,hardware,**kwargs)
        result['read_phase_seconds_per_panel']=self.panel_seconds
        result['read_phase_scope']='Independent instruction-mix microbenchmark including loop/checksum; sensitivity only'
        return result


def load_lds_phases(path):
    data=json.loads(path.read_text())
    if not data.get('completed') or len(data['rounds'])!=2:
        raise ValueError('Two completed independent measurement rounds required')
    if not all(c['correct'] for r in data['rounds'] for c in r['cases']):
        raise ValueError('LDS correctness check failed')
    phases={}
    for waves in (4,8):
        for barrier in (0,1):
            selected=[s['ns_per_panel'] for r in data['rounds'] for s in r['slopes']
                      if (s['waves'],s['blocks'],s['barrier'])==(waves,72,barrier)]
            if len(selected)!=2 or min(selected)<=0:raise ValueError('Missing LDS phase')
            phases[(waves,barrier)]=statistics.mean(selected)*1e-9
    return phases


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--lds-profile',type=Path,help='Run full-K-only independent read-phase sensitivity')
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Use a new report path')
    data=ROOT/'results/boundary_20261102/server_evidence'
    rows=json.loads((data/'measurements.json').read_text())['cases']
    metadata={r['id']:r for r in json.loads((data/'kernels.json').read_text())['cases']}
    hw=Hardware.load(ROOT/'superpod_compute/data/bw1100.json')
    base=Policy(**json.loads((ROOT/'examples/bw1100_graph_calibrated.json').read_text()))
    phases=load_lds_phases(args.lds_profile) if args.lds_profile else None
    configs={(192,128):(WavePlan(192,128,96,64,shared_panel_stride_bytes=10752,barrier_after_reads=False,read_waits_for_group_release=True),6,2),
             (384,256):(WavePlan(384,256,96,128,shared_panel_stride_bytes=20480),3,1)}
    report={'completed':False,'cases':[],
            'scope':'Supplied-wave and Shared-slot ablations for two inspected kernels; not a dispatch predictor. No change to launch calibration, throughput or bandwidth. Scalar read-latency proxy only in separately labeled sensitivity mode.',
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'superpod_compute').glob('*.py'))+[data/'measurements.json',data/'kernels.json']}}
    report['source_sha256'][str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if phases:
        report['lds_profile']={'path':str(args.lds_profile),'sha256':hashlib.sha256(args.lds_profile.read_bytes()).hexdigest(),
                              'phase_seconds':{f'waves{w}_barrier{b}':v for (w,b),v in phases.items()},
                              'scope':'Full K16 panels only; effective measured instruction phase, not isolated bandwidth. No GEMM-time fitting.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for row in rows:
        if row['layout']!='contiguous':continue
        if phases and row['shape'][2]%16:continue
        names=[n for n in metadata[row['id']]['kernels'] if '_MT' in n]
        if len(names)!=1:continue
        plan=parse_tensile_kernel(names[0],orientation='row-major-transposed-blas')
        if (plan.bm,plan.bn) not in configs or plan.depth_u!=16 or plan.global_split_u!=1:continue
        wave,slots,register_slots=configs[(plan.bm,plan.bn)]
        policy=replace(base,bm_values=(plan.bm,),bn_values=(plan.bn,),bk_values=(16,),add_shape_candidate=False)
        req=GemmRequest(*row['shape'])
        modes=('macro_baseline','wave_schedule','wave_schedule_lds_phase','wave_schedule_lds_barrier_phase') if phases else ('macro_baseline','wave_only','wave_slots','wave_slots_readiness','wave_schedule','wave_schedule_readiness')
        for mode in modes:
            backend='auto';registry=None;p=policy
            if mode!='macro_baseline':
                backend=WaveMmacBackend(hw,wave);registry=OperatorRegistry()
                if mode=='wave_only':model=WaveGemmModel(backend)
                else:
                    p=replace(p,buffer_counts=tuple(sorted({1,2,slots})))
                    model=(PinnedScheduledWaveModel if mode.startswith('wave_schedule') else PinnedWaveModel)(backend,slots,register_slots)
                    if mode.endswith('_phase'):
                        model=MeasuredReadWaveModel(backend,slots,register_slots,phases[(wave.waves,int(mode=='wave_schedule_lds_barrier_phase'))])
                    if mode.endswith('_readiness'):p=replace(p,input_read_latency_s=239.43020399505616e-9)
                registry.register(model)
            result=predict(req,hw,policy=p,backend=backend,registry=registry)
            measured=statistics.mean(row[f'graph_round{i}']['p50_us'] for i in (1,2))
            best=result['best']
            entry={'id':row['id'],'shape':row['shape'],'mode':mode,'model_us':best['duration_us'],'measured_us':measured,
                   'error_percent':100*(best['duration_us']/measured-1),'mapping':best['mapping'],'capacity':best['capacity'],
                   'waves':wave.waves,'wave_shape':[wave.wave_m,wave.wave_n],'wave_count':best['wave_count'],
                   'shared_read_bytes_per_panel_per_cu':best['wave_classes'][0]['full_panel_compute']['shared_read_bytes']}
            report['cases'].append(entry)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(row['shape'],mode,round(entry['model_us'],3),'error',round(entry['error_percent'],2),flush=True)
    report['completed']=True;report['summary']={}
    for mode in sorted({r['mode'] for r in report['cases']}):
        selected=[r for r in report['cases'] if r['mode']==mode]
        report['summary'][mode]={'count':len(selected),'passes':sum(abs(r['error_percent'])<=15 for r in selected),
                                 'mean_absolute_error_percent':statistics.mean(abs(r['error_percent']) for r in selected)}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
