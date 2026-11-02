"""Counter-only validation of fixed-shape pointer-offset interventions."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/workspace/gemm_boundary_20261102')

def main():
    status=ROOT/'base_alignment_pmc_batch.json'
    if status.exists():raise FileExistsError('Inspect existing acquisition; do not restart blindly')
    report={'completed':False,'cases':[]}
    cases=[(1536,1152,0),(1536,1152,2),(3072,2304,0),(3072,2304,2),(3072,2304,4)]
    for m,n,a in cases:
        case=f'base_alignment_{m}x{n}x1024_a{a}_b0_c0'
        result=subprocess.run([sys.executable,'-B',str(ROOT/'run_gpu_pmc.py'),'--case',case,'--plan','base-alignment','--read-focus'],cwd=ROOT)
        report['cases'].append({'id':case,'returncode':result.returncode});status.write_text(json.dumps(report,indent=2))
        if result.returncode:raise SystemExit(result.returncode)
    report['completed']=True;status.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
