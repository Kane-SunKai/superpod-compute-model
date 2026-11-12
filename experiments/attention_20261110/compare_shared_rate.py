"""How much does the Shared read rate move the GEMM predictions?

`shared_read_bytes_per_cycle_per_core` was 64 (the measured sustained rate) until
2026-11-05, when it became 128 (the 32-bank x 4-byte theoretical peak). The
policy's `effective_shared_read_bytes_per_cycle_per_core` is 0, so the hardware
value is what the model uses. This walks a few shapes from the regression plan at
both rates and reports the difference.
"""
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path('/workspace/vector_model_20261111')
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy, predict

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = ROOT / 'experiments/policy_no192.json'

SHAPES = [(4096, 4096, 4096), (4096, 16384, 4096), (4096, 4096, 16384),
          (1024, 4096, 11008), (2048, 5120, 13824), (4096, 50304, 4096),
          (512, 16384, 1024), (4096, 4096, 32), (8192, 512, 2048)]


def run(hw, pol, m, n, k):
    r = predict(GemmRequest(m, n, k, 'fp16', layout='row-major-contiguous'), hw,
                policy=pol, backend='mmac', cache_path=None)['best']
    mp = r['mapping']
    return r['duration_us'], '%dx%dx%d' % (mp['bm'], mp['bn'], mp['bk'])


def main():
    base = Hardware.load(HW)
    pol = Policy(**json.loads(POL.read_text(encoding='utf-8')))
    slow = dataclasses.replace(base, shared_read_bytes_per_cycle_per_core=64)
    fast = dataclasses.replace(base, shared_read_bytes_per_cycle_per_core=128)
    print('profile carries shared_read = %s; comparing 64 (measured) against 128 (bank peak)'
          % base.shared_read_bytes_per_cycle_per_core)
    print('%-22s %10s %10s %8s  %-16s' % ('shape', '64 B/cyc', '128 B/cyc', 'delta', 'tile'))
    deltas = []
    for m, n, k in SHAPES:
        a, ta = run(slow, pol, m, n, k)
        b, tb = run(fast, pol, m, n, k)
        d = 100 * (b / a - 1)
        deltas.append(d)
        print('%-22s %10.1f %10.1f %+7.1f%%  %s' % ('%dx%dx%d' % (m, n, k), a, b, d, ta))
    print('\nmean %+.1f%%  max |delta| %.1f%%' % (
        sum(deltas) / len(deltas), max(abs(d) for d in deltas)))


if __name__ == '__main__':
    main()
