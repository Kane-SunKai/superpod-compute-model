"""Write-bandwidth sensitivity for one shape, printed as it goes."""
import argparse, json, sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy, predict
from superpod_compute.batched import predict_batched_gemm

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_policy.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True, help='M,N,K')
    ap.add_argument('--batch', type=int, default=0)
    args = ap.parse_args()
    m, n, k = (int(x) for x in args.shape.split(','))
    hw = Hardware.load(HW)
    pol = Policy(**json.loads(POL.read_text(encoding='utf-8-sig')))
    out = []
    for w in (0.8e12, 1.05e12, 1.72e12):
        h = replace(hw, hbm_write_bandwidth_Bps=w)
        if args.batch:
            us = predict_batched_gemm(args.batch, m, n, k, h, pol)['best']['duration_us']
        else:
            us = predict(GemmRequest(m, n, k, 'fp16', layout='row-major-contiguous'), h,
                         policy=pol, backend='mmac', cache_path=None)['best']['duration_us']
        out.append(us)
        print('  W=%.2f -> %.1f us' % (w / 1e12, us), flush=True)
    print('SPREAD %+.1f%%' % (100 * (max(out) / min(out) - 1)), flush=True)


if __name__ == '__main__':
    main()
