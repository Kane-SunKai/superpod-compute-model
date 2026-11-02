"""Independent read-service measurements; optional private address data input.

Only writes inside /root/sk's /workspace mount. No performance regression
against GEMM target times. Raw address data must not be uploaded to GitHub.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/workspace/gemm_boundary_20261102')


def main():
    out=ROOT/'lds_instruction_mix';out.mkdir(exist_ok=False)
    source=ROOT/'lds_instruction_mix_probe.cpp';patterns=ROOT/'lds_access_patterns.json'
    command=['/opt/dtk/bin/hipcc','-O3','--offload-arch=gfx938','-std=c++17',str(source)]
    build=subprocess.run(command+['-o',str(out/'probe')],cwd=out,capture_output=True,text=True)
    (out/'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:print(build.stderr,flush=True);raise SystemExit(build.returncode)
    assembly=subprocess.run(command+['--cuda-device-only','-S','-o',str(out/'probe.s')],cwd=out,capture_output=True,text=True)
    (out/'assembly_build.log').write_text(assembly.stdout+assembly.stderr)
    if assembly.returncode:raise SystemExit(assembly.returncode)
    report={'completed':False,'rounds':[],
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'pattern_sha256':hashlib.sha256(patterns.read_bytes()).hexdigest(),
            'scope':'LDS-only instruction mix plus checksum/control/barrier; slope removes one-time initialization/launch. Effective service for this independent microbenchmark, not a fitted GEMM penalty.'}
    rows=json.loads(patterns.read_text())['cases']
    for repeat in (1,2):
        results=[]
        for pattern in rows:
            path=out/f"pattern_{pattern['waves']}.txt"
            path.write_text(f"{pattern['waves']} {pattern['matrix_reads_per_wave']}\n"+'\n'.join(' '.join(map(str,r)) for r in pattern['addresses_bytes'])+'\n')
            result=subprocess.run([str(out/'probe'),str(path)],cwd=out,capture_output=True,text=True)
            (out/f"run_{repeat}_{pattern['waves']}.log").write_text(result.stdout+result.stderr)
            if result.returncode:print(result.stdout,result.stderr,flush=True);raise SystemExit(result.returncode)
            measured=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            if len(measured)!=8 or not all(r['correct'] for r in measured):raise RuntimeError('Incomplete or incorrect measurement')
            results.extend(measured)
            print('completed round',repeat,'waves',pattern['waves'],flush=True)
        slopes=[]
        for waves in (4,8):
            for blocks in (1,72):
                for barrier in (0,1):
                    a,b=sorted([r for r in results if (r['waves'],r['blocks'],r['barrier'])==(waves,blocks,barrier)],key=lambda r:r['iterations'])
                    ns=(b['median_us']-a['median_us'])*1000/(b['iterations']-a['iterations'])
                    cycles=(b['ticks']-a['ticks'])/(b['iterations']-a['iterations'])
                    slopes.append({'waves':waves,'blocks':blocks,'barrier':barrier,'ns_per_panel':ns,
                                   'cycles_per_panel':cycles,'effective_bytes_per_cycle_per_cu':a['bytes_per_panel_per_block']/cycles})
        report['rounds'].append({'cases':results,'slopes':slopes})
        (out/'measurements.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(slopes,indent=2),flush=True)
    report['completed']=True;(out/'measurements.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
