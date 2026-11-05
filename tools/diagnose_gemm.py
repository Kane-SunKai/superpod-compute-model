"""Optional tile-only diagnosis; regular predictions still use the own mapper."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from superpod_compute import GemmRequest, Hardware, Policy
from superpod_compute.diagnostics import ObservedTile, diagnose_tile


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for dim in ('m','n','k'):
        p.add_argument('--'+dim, type=int, required=True)
    p.add_argument('--observed-tile',required=True,help='bm,bn,bk; posthoc observed macro tile')
    p.add_argument('--dtype',choices=('fp16','bf16'),default='fp16')
    p.add_argument('--hardware',type=Path)
    p.add_argument('--policy',type=Path)
    p.add_argument('--trace-panels',type=int,default=8)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        tile=[int(v) for v in a.observed_tile.split(',')]
        if len(tile)!=3:raise ValueError('observed tile requires bm,bn,bk')
        if not 0<=a.trace_panels<=10000:raise ValueError('trace-panels must be 0..10000')
        h=Hardware.load(a.hardware)
        policy=Policy(**json.loads(a.policy.read_text(encoding='utf-8-sig'))) if a.policy else Policy()
        result=diagnose_tile(GemmRequest(a.m,a.n,a.k,dtype=a.dtype),ObservedTile(*tile),h,policy=policy,trace_limit=a.trace_panels)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        best=result['prediction']['best']
        print('Diagnostic only: matching tiles does not imply matching kernels.')
        if best:
            print(f"Tile-constrained duration: {best['duration_us']:.6f} us; mapping: {best['mapping']}")
            print('Stage service demand (not additive elapsed contributions):',
                  {k:round(v*1e6,6) for k,v in result['ledger']['service_demand_s'].items()})
        print('Saved:',a.output.resolve())
        return 0
    except (ValueError,TypeError,OSError) as exc:
        print('Error:',exc,file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
