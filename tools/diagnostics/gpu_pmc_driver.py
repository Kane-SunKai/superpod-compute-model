"""Small counter-only target. Profiler durations are NOT benchmark latency.

One case per process makes counter rows unambiguous. First three GEMMs warm the
case; last three are the requested observation calls. Run with hipprof's explicit
output and temporary paths under /root/sk (container /workspace).
"""
import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from boundary_experiment import plan,tensors


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True)
    ap.add_argument('--plan',choices=('boundary','k-stride','base-alignment'),default='boundary');args=ap.parse_args()
    selected_plan=plan;selected_tensors=tensors
    if args.plan=='k-stride':
        from k_stride_isolation import plan as selected_plan
    elif args.plan=='base-alignment':
        from base_alignment_isolation import plan as selected_plan,tensors as selected_tensors
    rows=[r for r in selected_plan() if r['id']==args.case]
    if len(rows)!=1:raise ValueError('Unknown or ambiguous case')
    torch.cuda.set_device(0);torch.manual_seed(20261102)
    sys.path.insert(0,'/workspace')
    from benchmark_gemm import configure_precision
    configure_precision(torch)
    with torch.inference_mode():
        a,b,c=selected_tensors(torch,rows[0])
        for i in range(6):
            torch.mm(a,b,out=c);torch.cuda.synchronize()
            print('CALL',i,'warmup' if i<3 else 'observation',args.case,flush=True)


if __name__=='__main__':main()
