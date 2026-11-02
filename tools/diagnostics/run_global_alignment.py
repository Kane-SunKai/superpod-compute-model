"""Read-alignment experiment; explicit outputs stay below /workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/workspace/gemm_boundary_20261102')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--route',choices=('vgpr','lds'),default='vgpr');args=ap.parse_args()
    out=ROOT/('global_alignment' if args.route=='vgpr' else 'global_to_lds_alignment');out.mkdir(exist_ok=False)
    source=ROOT/('global_alignment_probe.cpp' if args.route=='vgpr' else 'global_to_lds_alignment_probe.cpp')
    command=['/opt/dtk/bin/hipcc','-O3','--offload-arch=gfx938','-std=c++17',str(source)]
    build=subprocess.run(command+['-o',str(out/'probe')],cwd=out,capture_output=True,text=True)
    (out/'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:print(build.stderr,flush=True);raise SystemExit(build.returncode)
    assembly=subprocess.run(command+['--cuda-device-only','-S','-o',str(out/'probe.s')],cwd=out,capture_output=True,text=True)
    (out/'assembly_build.log').write_text(assembly.stdout+assembly.stderr)
    if assembly.returncode:raise SystemExit(assembly.returncode)
    report={'completed':False,'rounds':[],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':'Independent cache-resident byte-addressed buffer loads. Same input region across workgroups; four requests per wave then wait. Not HBM bandwidth or full GEMM timing.',
            'route':args.route,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for repeat in (1,2):
        result=subprocess.run([str(out/'probe')],cwd=out,capture_output=True,text=True)
        (out/f'round_{repeat}.log').write_text(result.stdout+result.stderr)
        if result.returncode:print(result.stdout[-1500:],result.stderr,flush=True);raise SystemExit(result.returncode)
        rows=[json.loads(s) for s in result.stdout.splitlines() if s.startswith('{')]
        if len(rows)!=72 or not all(r['correct'] and r['local_bytes']==0 for r in rows):raise RuntimeError('Invalid probe')
        groups=sorted({(r['load_bytes'],r['blocks'],r['stride_bytes'],r['base_bytes']) for r in rows})
        slopes=[]
        for key in groups:
            a,b=sorted([r for r in rows if (r['load_bytes'],r['blocks'],r['stride_bytes'],r['base_bytes'])==key],key=lambda r:r['iterations'])
            slopes.append(dict(zip(('load_bytes','blocks','stride_bytes','base_bytes'),key))|{
                'ns_per_iteration':1000*(b['median_us']-a['median_us'])/(b['iterations']-a['iterations']),
                'cycles_per_iteration':(b['ticks']-a['ticks'])/(b['iterations']-a['iterations'])})
        report['rounds'].append({'cases':rows,'slopes':slopes})
        (out/'measurements.json').write_text(json.dumps(report,indent=2))
        print('round',repeat,'correct cases',len(rows),flush=True)
        print(json.dumps([r for r in slopes if r['blocks']==72]),flush=True)
    report['completed']=True;(out/'measurements.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
