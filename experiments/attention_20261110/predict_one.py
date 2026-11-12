"""Single-shape prediction: 1024 x 21760 x 4096 (GPT-7B gate/up projection)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy, predict

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
PO = ROOT / 'experiments/regression_100_v2_20261109/policy_no192.json'
SHAPE = (1024, 21760, 4096)
DIVISOR = 0.96

hw = Hardware.load(HW)
pol = Policy(**json.loads(PO.read_text(encoding='utf-8')))
r = predict(GemmRequest(*SHAPE, 'fp16', layout='row-major-contiguous'), hw, policy=pol,
            backend='mmac', cache_path=None)
best = r['best']
m = best['mapping']
aspect = max(SHAPE[0], SHAPE[1]) / min(SHAPE[0], SHAPE[1])
factor = 1 / DIVISOR if SHAPE[0] >= 64 and aspect >= 4 else 1.0

print(f"shape            {SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}")
print(f"tile             {m['bm']}x{m['bn']}x{m['bk']}  gm={m['gm']} gn={m['gn']}")
print(f"waves            {best['wave_count']}")
print(f"model raw        {best['duration_us']:.1f} us")
print(f"aspect           {aspect:.2f}  -> V2 factor {factor:.4f}")
print(f"model final      {best['duration_us'] * factor:.1f} us")
print(f"candidates       {r['searched_candidates']}")
