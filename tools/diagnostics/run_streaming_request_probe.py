"""Paired full-K versus repeated-prefix delivery with identical GPU code."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/workspace/gemm_boundary_20261102')


def main():
    out=ROOT/'request_pattern_streaming';out.mkdir(exist_ok=False)
    source=ROOT/'request_pattern_probe.cpp';pattern_path=ROOT/'request_patterns.json'
    layout_path=ROOT/'group_source_shifts.json'
    cmd=['/opt/dtk/bin/hipcc','-O3','--offload-arch=gfx938','-std=c++17',str(source)]
    build=subprocess.run(cmd+['-o',str(out/'probe')],cwd=out,capture_output=True,text=True)
    (out/'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:print(build.stderr,flush=True);raise SystemExit(build.returncode)
    asm=subprocess.run(cmd+['--cuda-device-only','-S','-o',str(out/'probe.s')],cwd=out,capture_output=True,text=True)
    (out/'assembly_build.log').write_text(asm.stdout+asm.stderr)
    if asm.returncode:raise SystemExit(asm.returncode)
    patterns=json.loads(pattern_path.read_text())['cases']
    layout=json.loads(layout_path.read_text());assert layout['completed']
    layouts={c['id']:c for c in layout['cases']}
    report={'completed':False,'rounds':[],
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'pattern_sha256':hashlib.sha256(pattern_path.read_bytes()).hexdigest(),
            'layout_sha256':hashlib.sha256(layout_path.read_bytes()).hexdigest(),
            'scope':'Delivery only, same compiled kernel: repeatedly read prefix Q panels or stream cyclically over all 64 K16 panels. Actual spatial group sources, raw-offen instructions, wait/barrier after each Q panels, no computation/output. Timing includes address updates and control. Nine loop lengths validate final batch including K wrap; only 512/2048 form service slopes.'}
    for repeat in (1,2):
        rows=[];slopes=[]
        for case in patterns:
            threads=case['waves']*64
            text=out/(case['id']+'.txt')
            lines=[f"{threads} {case['panels']} {case['a_bytes']} {case['b_bytes']} {case['a_base_mod_256']} {case['b_base_mod_256']}"]
            lines += [f'{offset} {m0}' for stage in case['stages'] for offset,m0 in zip(stage['offsets_bytes'],stage['lds_m0'])]
            text.write_text('\n'.join(lines)+'\n')
            current=layouts[case['id']];groups=current['groups'];assert len(groups)==72 and current['k']==1024
            shifts=out/(case['id']+'_group_shifts.txt')
            shifts.write_text('\n'.join(f"{g['a_shift_bytes']} {g['b_shift_bytes']}" for g in groups)+'\n')
            modes=('repeat','stream') if repeat==1 else ('stream','repeat')
            for mode in modes:
                command=[str(out/'probe'),str(text),str(shifts),str(current['n']),mode]
                run=subprocess.run(command,cwd=out,capture_output=True,text=True)
                (out/f"round_{repeat}_{case['id']}_{mode}.log").write_text(run.stdout+run.stderr)
                if run.returncode:print(run.stdout[-2000:],run.stderr,flush=True);raise SystemExit(run.returncode)
                measured=[json.loads(s)|{'id':case['id'],'k_traversal':mode} for s in run.stdout.splitlines() if s.startswith('{')]
                assert len(measured)==(72 if threads==256 else 54)
                assert all(r['correct'] and r['local_bytes']==0 for r in measured)
                rows.extend(measured)
                for panels in sorted({r['queued_panels'] for r in measured}):
                    for blocks in (1,72):
                        a,b=sorted([r for r in measured if (r['queued_panels'],r['blocks'])==(panels,blocks) and r['iterations'] in (512,2048)],key=lambda r:r['iterations'])
                        slopes.append({'id':case['id'],'k_traversal':mode,'queued_panels':panels,'blocks':blocks,
                                       'ns_per_panel':1000*(b['median_us']-a['median_us'])/(b['iterations']-a['iterations'])/panels,
                                       'cycles_per_panel':(b['ticks']-a['ticks'])/(b['iterations']-a['iterations'])/panels})
                print('round',repeat,case['id'],mode,'passed',flush=True)
        report['rounds'].append({'cases':rows,'slopes':slopes})
        (out/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
    report['completed']=True;(out/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
