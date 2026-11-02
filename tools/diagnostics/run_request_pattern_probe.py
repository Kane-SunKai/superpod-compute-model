"""Independent delivery measurements with private runtime address inputs."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/workspace/gemm_boundary_20261102')

def main():
    out=ROOT/'request_pattern_delivery';out.mkdir(exist_ok=False)
    source=ROOT/'request_pattern_probe.cpp';patterns=ROOT/'request_patterns.json'
    cmd=['/opt/dtk/bin/hipcc','-O3','--offload-arch=gfx938','-std=c++17',str(source)]
    build=subprocess.run(cmd+['-o',str(out/'probe')],cwd=out,capture_output=True,text=True)
    (out/'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:print(build.stderr,flush=True);raise SystemExit(build.returncode)
    asm=subprocess.run(cmd+['--cuda-device-only','-S','-o',str(out/'probe.s')],cwd=out,capture_output=True,text=True)
    (out/'assembly_build.log').write_text(asm.stdout+asm.stderr)
    if asm.returncode:raise SystemExit(asm.returncode)
    cases=json.loads(patterns.read_text())['cases']
    report={'completed':False,'rounds':[],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'pattern_sha256':hashlib.sha256(patterns.read_bytes()).hexdigest(),
            'scope':'Independent delivery-only test using supplied source addresses and LDS destination control. Raw offen source addressing, no original scalar/indexed address setup. All workgroups share the group-0 source addresses. Queued panels then wait/barrier; not full GEMM or steady pipeline overlap.'}
    for repeat in (1,2):
        measured=[];slopes=[]
        for case in cases:
            text=out/(case['id']+'.txt');threads=case['waves']*64
            lines=[f"{threads} {case['panels']} {case['a_bytes']} {case['b_bytes']} {case['a_base_mod_256']} {case['b_base_mod_256']}"]
            lines += [f'{offset} {m0}' for stage in case['stages'] for offset,m0 in zip(stage['offsets_bytes'],stage['lds_m0'])]
            text.write_text('\n'.join(lines)+'\n')
            run=subprocess.run([str(out/'probe'),str(text)],cwd=out,capture_output=True,text=True)
            (out/f"round_{repeat}_{case['id']}.log").write_text(run.stdout+run.stderr)
            if run.returncode:print(run.stdout[-2000:],run.stderr,flush=True);raise SystemExit(run.returncode)
            rows=[json.loads(s)|{'id':case['id']} for s in run.stdout.splitlines() if s.startswith('{')]
            if len(rows)!=(16 if threads==256 else 12) or not all(r['correct'] and r['local_bytes']==0 for r in rows):raise RuntimeError('Incorrect, incomplete or spilling delivery probe')
            measured.extend(rows)
            for panels in sorted({r['queued_panels'] for r in rows}):
                for blocks in (1,72):
                    a,b=sorted([r for r in rows if (r['queued_panels'],r['blocks'])==(panels,blocks)],key=lambda r:r['iterations'])
                    slopes.append({'id':case['id'],'queued_panels':panels,'blocks':blocks,
                                   'ns_per_panel':1000*(b['median_us']-a['median_us'])/(b['iterations']-a['iterations'])/panels,
                                   'cycles_per_panel':(b['ticks']-a['ticks'])/(b['iterations']-a['iterations'])/panels})
            print('round',repeat,case['id'],json.dumps([r for r in slopes if r['id']==case['id'] and r['blocks']==72]),flush=True)
        report['rounds'].append({'cases':measured,'slopes':slopes});(out/'measurements.json').write_text(json.dumps(report,indent=2))
    report['completed']=True;(out/'measurements.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
