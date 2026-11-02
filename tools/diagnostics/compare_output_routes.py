"""Frozen-mapping ablation of a structurally supplied C output route.

Does not choose mappings or fit parameters using GEMM measurements. Only the
two previously inspected kernels have evidence for the direct-register route.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics

from compare_wave_slots import ROOT,Hardware,Policy,WavePlan,WaveMmacBackend,PinnedScheduledWaveModel,MeasuredReadWaveModel,load_lds_phases
from superpod_compute import GemmRequest,Mapping
from superpod_compute.backends import make_backend
from superpod_compute.operators import GemmModel


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior report')
    folder=ROOT/'results/boundary_20261102'
    reports=[(folder/'wave_schedule_comparison.json',{'macro_baseline','wave_schedule'}),
             (folder/'wave_lds_phase_comparison.json',{'wave_schedule_lds_phase','wave_schedule_lds_barrier_phase'})]
    phase_path=folder/'server_evidence/lds_instruction_mix_measurements.json'
    phases=load_lds_phases(phase_path)
    hardware_path=ROOT/'superpod_compute/data/bw1100.json';policy_path=ROOT/'examples/bw1100_graph_calibrated.json'
    h=Hardware.load(hardware_path);policy=Policy(**json.loads(policy_path.read_text()))
    configs={(192,128):(WavePlan(192,128,96,64,shared_panel_stride_bytes=10752,barrier_after_reads=False,read_waits_for_group_release=True),6,2),
             (384,256):(WavePlan(384,256,96,128,shared_panel_stride_bytes=20480),3,1)}
    report={'completed':False,'cases':[],
            'scope':'Only output route changes; original mappings, compute, input transfer and independently measured LDS phase stay fixed. Direct output evidence is limited to two inspected kernels. Tail and output instruction issue costs remain unresolved.',
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__),hardware_path,policy_path,phase_path]+[p for p,_ in reports]+list((ROOT/'superpod_compute').glob('*.py'))}}
    for path,modes in reports:
        original=json.loads(path.read_text());assert original['completed']
        for row in original['cases']:
            if row['mode'] not in modes:continue
            mapping=Mapping(**row['mapping']);q=GemmRequest(*row['shape']);mode=row['mode']
            if mode=='macro_baseline':
                model=GemmModel();backend=make_backend('auto',h,policy.matrix_efficiency)
            else:
                wave,slots,regs=configs[mapping.bm,mapping.bn];backend=WaveMmacBackend(h,wave)
                if mode.endswith('_phase'):
                    model=MeasuredReadWaveModel(backend,slots,regs,phases[wave.waves,int('barrier' in mode)])
                else:model=PinnedScheduledWaveModel(backend,slots,regs)
            baseline=model.evaluate(q,h,policy,backend,mapping,trace_limit=0)
            assert math.isclose(baseline['duration_us'],row['model_us'],abs_tol=1e-8,rel_tol=1e-10),(row['id'],mode)
            direct=model.evaluate(q,h,policy,backend,replace(mapping,output_route='direct-register'),trace_limit=2)
            for key in ('hbm_read_bytes','l2_to_core_bytes','output_bytes','executed_tile_flops'):
                assert baseline[key]==direct[key]
            for a,b in zip(baseline['wave_classes'],direct['wave_classes']):
                assert a['pipeline']['duration_s']==b['pipeline']['duration_s']
            entry={'id':row['id'],'shape':row['shape'],'mode':mode,'mapping':mapping.to_dict(),
                   'measured_us':row['measured_us'],'via_shared_us':baseline['duration_us'],
                   'direct_register_us':direct['duration_us'],'saved_us':baseline['duration_us']-direct['duration_us'],
                   'via_shared_error_percent':100*(baseline['duration_us']/row['measured_us']-1),
                   'direct_register_error_percent':100*(direct['duration_us']/row['measured_us']-1)}
            report['cases'].append(entry)
            print(row['id'],mode,f"{entry['via_shared_error_percent']:.2f}% -> {entry['direct_register_error_percent']:.2f}%",flush=True)
    report['summary']={}
    for mode in sorted({r['mode'] for r in report['cases']}):
        rows=[r for r in report['cases'] if r['mode']==mode]
        report['summary'][mode]={route:{'count':len(rows),'passes':sum(abs(r[route+'_error_percent'])<=15 for r in rows),
                                              'mape_percent':statistics.mean(abs(r[route+'_error_percent']) for r in rows)}
                                for route in ('via_shared','direct_register')}
    report['completed']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
