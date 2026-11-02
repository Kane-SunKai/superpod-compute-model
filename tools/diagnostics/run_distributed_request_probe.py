"""Paired shared/distributed source experiment, all files under /workspace."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/workspace/gemm_boundary_20261102')


def main():
    out=ROOT/'request_pattern_distributed';out.mkdir(exist_ok=False)
    source=ROOT/'request_pattern_probe.cpp';pattern_path=ROOT/'request_patterns.json'
    layout_path=ROOT/'group_source_shifts.json'
    cmd=['/opt/dtk/bin/hipcc','-O3','--offload-arch=gfx938','-std=c++17',str(source)]
    build=subprocess.run(cmd+['-o',str(out/'probe')],cwd=out,capture_output=True,text=True)
    (out/'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:print(build.stderr,flush=True);raise SystemExit(build.returncode)
    patterns=json.loads(pattern_path.read_text())['cases']
    layout=json.loads(layout_path.read_text());assert layout['completed']
    layouts={c['id']:c for c in layout['cases']}
    report={'completed':False,'rounds':[],
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'pattern_sha256':hashlib.sha256(pattern_path.read_bytes()).hexdigest(),
            'layout_sha256':hashlib.sha256(layout_path.read_bytes()).hexdigest(),
            'scope':'Delivery only, paired identical-group-0 versus actual spatial group sources. Repeated first Q panels, raw-offen instructions, no full K streaming or computation. Both modes use the same compiled kernel.'}
    for repeat in (1,2):
        rows=[];slopes=[]
        for case in patterns:
            threads=case['waves']*64
            text=out/(case['id']+'.txt')
            lines=[f"{threads} {case['panels']} {case['a_bytes']} {case['b_bytes']} {case['a_base_mod_256']} {case['b_base_mod_256']}"]
            lines += [f'{offset} {m0}' for stage in case['stages'] for offset,m0 in zip(stage['offsets_bytes'],stage['lds_m0'])]
            text.write_text('\n'.join(lines)+'\n')
            groups=layouts[case['id']]['groups'];assert len(groups)==72
            shifts=out/(case['id']+'_group_shifts.txt')
            shifts.write_text('\n'.join(f"{g['a_shift_bytes']} {g['b_shift_bytes']}" for g in groups)+'\n')
            modes=('shared','distributed') if repeat==1 else ('distributed','shared')
            for mode in modes:
                command=[str(out/'probe'),str(text)]+([str(shifts)] if mode=='distributed' else [])
                run=subprocess.run(command,cwd=out,capture_output=True,text=True)
                (out/f"round_{repeat}_{case['id']}_{mode}.log").write_text(run.stdout+run.stderr)
                if run.returncode:print(run.stdout[-2000:],run.stderr,flush=True);raise SystemExit(run.returncode)
                measured=[json.loads(s)|{'id':case['id'],'source_layout':mode} for s in run.stdout.splitlines() if s.startswith('{')]
                assert len(measured)==(16 if threads==256 else 12)
                assert all(r['correct'] and r['local_bytes']==0 for r in measured)
                rows.extend(measured)
                for panels in sorted({r['queued_panels'] for r in measured}):
                    for blocks in (1,72):
                        a,b=sorted([r for r in measured if (r['queued_panels'],r['blocks'])==(panels,blocks)],key=lambda r:r['iterations'])
                        slopes.append({'id':case['id'],'source_layout':mode,'queued_panels':panels,'blocks':blocks,
                                       'ns_per_panel':1000*(b['median_us']-a['median_us'])/(b['iterations']-a['iterations'])/panels,
                                       'cycles_per_panel':(b['ticks']-a['ticks'])/(b['iterations']-a['iterations'])/panels})
                print('round',repeat,case['id'],mode,'passed',flush=True)
        report['rounds'].append({'cases':rows,'slopes':slopes})
        (out/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
    report['completed']=True;(out/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
