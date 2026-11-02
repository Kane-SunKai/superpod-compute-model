"""Rolling queue test: validate readiness before timing the consumption-free loop."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/workspace/gemm_boundary_20261102')


def main():
    out=ROOT/'request_pattern_rolling';out.mkdir(exist_ok=False)
    source=ROOT/'rolling_request_probe.cpp'
    cmd=['/opt/dtk/bin/hipcc','-O3','--offload-arch=gfx938','-std=c++17',str(source)]
    build=subprocess.run(cmd+['-o',str(out/'probe')],cwd=out,capture_output=True,text=True)
    (out/'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:print(build.stderr,flush=True);raise SystemExit(build.returncode)
    asm=subprocess.run(cmd+['--cuda-device-only','-S','-o',str(out/'probe.s')],cwd=out,capture_output=True,text=True)
    (out/'assembly_build.log').write_text(asm.stdout+asm.stderr)
    if asm.returncode:raise SystemExit(asm.returncode)
    source_input=ROOT/'request_pattern_streaming'
    original=json.loads((source_input/'measurements.json').read_text());assert original['completed']
    cases=sorted({r['id'] for r in original['rounds'][0]['cases']})
    report={'completed':False,'rounds':[],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'input_measurements_sha256':hashlib.sha256((source_input/'measurements.json').read_bytes()).hexdigest(),
            'scope':'Rolling direct-to-LDS queue over K=1024, actual spatial source groups. Prefill Q panels, wait for oldest panel using VM count, workgroup barrier, reuse only that slot. Readiness validation sums every consumed panel across a full K wrap; that extra LDS consumption is excluded from timing. No matrix computation or output epilogue.'}
    for repeat in (1,2):
        rows=[];slopes=[]
        for case in cases:
            run=subprocess.run([str(out/'probe'),str(source_input/(case+'.txt')),str(source_input/(case+'_group_shifts.txt'))],cwd=out,capture_output=True,text=True)
            (out/f'round_{repeat}_{case}.log').write_text(run.stdout+run.stderr)
            if run.returncode:print(run.stdout[-3000:],run.stderr,flush=True);raise SystemExit(run.returncode)
            measured=[json.loads(s)|{'id':case} for s in run.stdout.splitlines() if s.startswith('{')]
            assert len(measured)==(20 if measured[0]['threads']==256 else 15)
            assert all(r['correct'] and r['local_bytes']==0 for r in measured)
            rows.extend(measured)
            for panels in sorted({r['queued_panels'] for r in measured}):
                for blocks in (1,72):
                    a,b=sorted([r for r in measured if (r['queued_panels'],r['blocks'])==(panels,blocks) and not r['validation_only']],key=lambda r:r['iterations'])
                    slopes.append({'id':case,'queued_panels':panels,'blocks':blocks,
                                   'ns_per_panel':1000*(b['median_us']-a['median_us'])/(b['iterations']-a['iterations'])/panels,
                                   'cycles_per_panel':(b['ticks']-a['ticks'])/(b['iterations']-a['iterations'])/panels})
            print('round',repeat,case,'readiness and timing passed',flush=True)
        report['rounds'].append({'cases':rows,'slopes':slopes});(out/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
    report['completed']=True;(out/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
