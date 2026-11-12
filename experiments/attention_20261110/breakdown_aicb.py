"""Stage breakdown for the two AICB GEMMs that the model gets wrong."""
import json, sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy, predict

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = ROOT / 'experiments/regression_100_v2_20261109/policy_no192.json'
CASES = [(4096, 50304, 4096, 7433.9, 'logit'),
         (4096, 4096, 16384, 2829.3, 'mlp_linear_2'),
         (4096, 16384, 4096, 2022.0, 'mlp_linear_1'),
         (4096, 12288, 4096, 1329.2, 'atten_qkv')]


def main():
    hw = Hardware.load(HW)
    pol = Policy(**json.loads(POL.read_text(encoding='utf-8')))
    print(f"{'case':<14}{'tile':<13}{'waves':>6}{'panels':>7}{'H':>8}{'R':>8}{'S':>8}{'C':>8}"
          f"{'out':>8}{'model':>9}{'meas':>9}{'TFLOPS':>8}")
    for m, n, k, meas, name in CASES:
        r = predict(GemmRequest(m, n, k, 'fp16', layout='row-major-contiguous'), hw,
                    policy=pol, backend='mmac', cache_path=None)['best']
        mp = r['mapping']
        cls = r['wave_classes'][0]
        run = cls['panel_byte_runs'][0]
        H = 1e6 * run['HBM_bytes_per_panel'] / hw.hbm_bandwidth_Bps
        R = 1e6 * run['R_service_s']
        S = 1e6 * run['Shared_read_service_s_per_core']
        C = 1e6 * cls['full_panel_compute']['duration_s']
        out = 1e6 * cls['output_stream']['duration_s']
        flops = 2 * m * n * k
        print(f"{name:<14}{'%dx%dx%d' % (mp['bm'], mp['bn'], mp['bk']):<13}{r['wave_count']:>6}"
              f"{cls['full_k_panels']:>7}{H:>8.3f}{R:>8.3f}{S:>8.3f}{C:>8.3f}{out:>8.1f}"
              f"{r['duration_us']:>9.1f}{meas:>9.1f}{flops / (meas * 1e-6) / 1e12:>8.0f}", flush=True)


if __name__ == '__main__':
    main()
