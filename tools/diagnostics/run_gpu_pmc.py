"""Counter acquisition only; all outputs and profiler scratch inside /workspace."""
import argparse,json,subprocess
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True)
    ap.add_argument('--plan',choices=('boundary','k-stride'),default='boundary')
    ap.add_argument('--read-focus',action='store_true');args=ap.parse_args()
    root=Path('/workspace/gemm_boundary_20261102')
    rows=json.loads((root/('k_stride_kernels.json' if args.plan=='k-stride' else 'kernels.json')).read_text())['cases']
    row=next(r for r in rows if r['id']==args.case)
    names=[name for name in row['kernels'] if '_MT' in name]
    if len(names)!=1:raise ValueError('Expected one matrix kernel name')
    if not args.case.replace('_','').replace('x','').isalnum():raise ValueError('Invalid case ID')
    out=root/('pmc_read' if args.read_focus else 'pmc')/args.case
    out.mkdir(parents=True,exist_ok=False)
    scratch=out/'scratch';scratch.mkdir()
    command=['/opt/dtk/bin/hipprof','--pmc-read' if args.read_focus else '--pmc','--pmc-type','3','--kernel-name',names[0],
             '-o',str(out/'report'),'-d',str(scratch),'python3','-B',str(root/'gpu_pmc_driver.py'),'--case',args.case,'--plan',args.plan]
    (out/'command.json').write_text(json.dumps(command,indent=2))
    with (out/'acquisition.log').open('w') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    (out/'status.json').write_text(json.dumps({'returncode':result.returncode,'case':args.case,
        'scope':'Hardware counters only. Profiling duration not used as benchmark latency.'},indent=2))
    print(args.case,'exit',result.returncode,'files',[p.name for p in out.iterdir()],flush=True)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
