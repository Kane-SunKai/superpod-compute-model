"""Predict the two attention matmuls as batched GEMMs.

GPT-7B, TP=1, seq 4096, 32 heads, head_dim 128:
  QK^T : batch=32, M=4096, N=4096, K=128   (scores, materialised)
  PV   : batch=32, M=4096, N=128,  K=4096  (context)
Also the 1024/2048 shapes we already measured on the machine, for comparison.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from superpod_compute import Hardware, Policy
from superpod_compute.batched import predict_batched_gemm

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
PO = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_policy.json'
CASES = [('qk  s=1024 (measured 7.19)', 32, 1024, 1024, 128),
         ('qk  s=2048 (measured 15.03)', 32, 2048, 2048, 128),
         ('QK^T s=4096', 32, 4096, 4096, 128),
         ('PV   s=4096', 32, 4096, 128, 4096)]


def main():
    hw = Hardware.load(HW)
    pol = Policy(**json.loads(PO.read_text(encoding='utf-8-sig')))
    print(f"{'case':<32}{'batch':>6}{'M':>6}{'N':>6}{'K':>6}{'model us':>10}{'tile':>14}{'waves':>7}{'cands':>7}")
    for label, b, m, n, k in CASES:
        try:
            r = predict_batched_gemm(b, m, n, k, hw, pol)
            mp = r['best']['mapping']
            print(f"{label:<32}{b:>6}{m:>6}{n:>6}{k:>6}{r['best']['duration_us']:>10.1f}"
                  f"{'%dx%dx%d' % (mp['bm'], mp['bn'], mp['bk']):>14}{r['best']['wave_count']:>7}"
                  f"{r['feasible_candidates']:>7}", flush=True)
        except Exception as exc:
            print(f"{label:<32} ERROR {type(exc).__name__}: {exc}", flush=True)


if __name__ == '__main__':
    main()
