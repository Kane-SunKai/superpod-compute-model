"""Capture sanitized launch scalars for the two selected GEMMs only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/workspace/gemm_boundary_20261102')

def main():
    out=ROOT/'kernel_arguments';out.mkdir(exist_ok=False)
    src=ROOT/'kernel_argument_probe.cpp';library=out/'probe.so'
    build=subprocess.run(['/opt/dtk/bin/hipcc','-O2','-std=c++17','-shared','-fPIC',str(src),'-ldl','-o',str(library)],cwd=out,capture_output=True,text=True)
    (out/'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:print(build.stderr,flush=True);raise SystemExit(build.returncode)
    metadata={r['metadata']['.name']:r['metadata'] for r in json.loads((ROOT/'kernel_instruction_evidence/selected_abi.json').read_text())['kernels']}
    report={'completed':False,'cases':[],'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
            'scope':'Selected GEMM launch scalar arguments only. Device pointers replaced by synthetic bases retaining modulo 256 and aliases. No tensor data or binary copied.'}
    for m,n in ((1536,1152),(3072,2304)):
        for offset in (0,2):
            case=f'base_alignment_{m}x{n}x1024_a{offset}_b0_c0'
            env={**os.environ,'LD_PRELOAD':str(library)}
            run=subprocess.run(['python3','-B',str(ROOT/'gpu_pmc_driver.py'),'--plan','base-alignment','--case',case],cwd=ROOT,env=env,capture_output=True,text=True)
            (out/(case+'.log')).write_text(run.stdout+run.stderr)
            if run.returncode:print(run.stderr,flush=True);raise SystemExit(run.returncode)
            captures=[line.split('\t') for line in run.stderr.splitlines() if line.startswith('GEMM_ARGS\t')]
            if len(captures)!=1:raise RuntimeError(f'Expected one selected launch capture: {case}, {len(captures)}')
            _,name,grid,block,hex_data=captures[0];data=bytes.fromhex(hex_data)
            arguments={arg['.name']:int.from_bytes(data[arg['.offset']:arg['.offset']+arg['.size']],'little') for arg in metadata[name]['.args']}
            report['cases'].append({'id':case,'kernel':name,'global_work_items':list(map(int,grid.split(','))),
                                    'block':list(map(int,block.split(','))),'argument_bytes_hex':hex_data,'arguments':arguments})
            (out/'sanitized_arguments.json').write_text(json.dumps(report,indent=2))
            print(case,json.dumps(arguments),flush=True)
    report['completed']=True;(out/'sanitized_arguments.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
