"""Serialize counter acquisitions; never profile two GPU jobs concurrently."""
import json,subprocess,sys
from pathlib import Path

ROOT=Path('/workspace/gemm_boundary_20261102')
CASES=[
    'large_axis_K_1536x1152x1023_contiguous',
    'large_axis_K_1536x1152x1025_contiguous',
    'large_axis_M_1537x1152x1024_contiguous',
    'historical_irregular_1537x1153x1023_contiguous',
    'stride_intervention_1537x1153x1023_pad_a16',
]


def main():
    status=ROOT/'pmc/boundary_batch_status.json'
    if status.exists():raise FileExistsError('Preserve previous batch; inspect before retry')
    report={'completed':False,'cases':[],'scope':'Counter-only diagnosis; not timing validation'}
    for case in CASES:
        result=subprocess.run([sys.executable,'-B',str(ROOT/'run_gpu_pmc.py'),'--case',case],cwd=ROOT)
        report['cases'].append({'id':case,'returncode':result.returncode})
        status.write_text(json.dumps(report,indent=2))
        if result.returncode:raise SystemExit(result.returncode)
    report['completed']=True;status.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
