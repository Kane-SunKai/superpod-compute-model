"""Independent combined-phase diagnostic; never fits GEMM observations."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/workspace/gemm_boundary_20261102')

def main():
    out=ROOT/'lds_mmac_phase_v3';out.mkdir(exist_ok=False)
    source=ROOT/'lds_mmac_phase_probe.cpp'
    cmd=['/opt/dtk/bin/hipcc','-O3','--offload-arch=gfx938','-std=c++17',str(source)]
    build=subprocess.run(cmd+['-o',str(out/'probe')],cwd=out,capture_output=True,text=True)
    (out/'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:print(build.stderr,flush=True);raise SystemExit(build.returncode)
    assembly=subprocess.run(cmd+['--cuda-device-only','-S','-o',str(out/'probe.s')],cwd=out,capture_output=True,text=True)
    (out/'assembly_build.log').write_text(assembly.stdout+assembly.stderr)
    if assembly.returncode:raise SystemExit(assembly.returncode)
    report={'completed':False,'rounds':[],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':'Four-wave independent register-only vs serial-read vs prefetched-read matrix loops with per-panel workgroup barriers. Includes issue, control and synchronization; no global memory input within loop. Not vendor kernel timing.',
            'excluded':'Eight-wave prototype exceeded compiler register budget and failed correctness. Its times are invalid and excluded; prior failed logs preserved.'}
    for repeat in (1,2):
        rows=[]
        for waves in (4,):
            pattern=ROOT/f'lds_instruction_mix/pattern_{waves}.txt'
            result=subprocess.run([str(out/'probe'),str(pattern)],cwd=out,capture_output=True,text=True)
            (out/f'round_{repeat}_waves_{waves}.log').write_text(result.stdout+result.stderr)
            if result.returncode:print(result.stdout,result.stderr,flush=True);raise SystemExit(result.returncode)
            part=[json.loads(s) for s in result.stdout.splitlines() if s.startswith('{')]
            if len(part)!=12 or not all(r['correct'] and r['local_bytes']==0 for r in part):raise RuntimeError('Incorrect, spilling or incomplete measurement')
            rows.extend(part)
        slopes=[]
        for waves in (4,):
            for mode in range(3 if waves==4 else 2):
                for blocks in (1,72):
                    a,b=sorted([r for r in rows if (r['waves'],r['mode'],r['blocks'])==(waves,mode,blocks)],key=lambda r:r['iterations'])
                    slopes.append({'waves':waves,'mode':mode,'blocks':blocks,'ns_per_panel':1000*(b['median_us']-a['median_us'])/(b['iterations']-a['iterations']),
                                   'cycles_per_panel':(b['ticks']-a['ticks'])/(b['iterations']-a['iterations']),
                                   'local_bytes':max(a['local_bytes'],b['local_bytes'])})
        report['rounds'].append({'cases':rows,'slopes':slopes})
        (out/'measurements.json').write_text(json.dumps(report,indent=2))
        print('round',repeat,json.dumps(slopes),flush=True)
    report['completed']=True;(out/'measurements.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
