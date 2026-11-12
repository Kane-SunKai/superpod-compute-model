"""What the model does when the hardware changes.

The premise this model has to meet: fit on one machine, swap the hardware
parameters, predict the other machine. This walks the same two operators across
several hardware variants and prints what moves. Nothing is re-fitted - the
vector efficiency stays at its measured value, because a kernel's efficiency is
a property of the library, not of the chip.

Usage: python predict_on_hardware.py
"""
import dataclasses
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_root(start):
    for p in (start, *start.parents):
        if (p / 'superpod_compute').is_dir():
            return p
    raise SystemExit('superpod_compute not found above %s' % start)


ROOT = find_root(HERE)
sys.path.insert(0, str(ROOT))
from superpod_compute import Hardware
from superpod_compute.vector_ops import VectorRequest, predict

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
PROFILE = ROOT / 'superpod_compute/data/bw1100_vector_profile.json'
OUT = HERE / 'vector_hardware_variants.json'

SOFTMAX = VectorRequest('softmax', 32 * 4096, 4096, 'fp16')
GELU = VectorRequest('gelu', 1, 4096 * 16384, 'fp16')
LONG = VectorRequest('softmax', 8192, 65536, 'fp16')   # row 128 KiB, spills past L1D

VARIANTS = [
    ('BW1100 基线', {}),
    ('矢量单元加宽一倍', dict(vector_fp16_flops_per_cycle_per_core=7.866753472222222)),
    ('核数加倍 (144 CU)', dict(cores=144)),
    ('时钟 1.6 GHz', dict(clock_hz=1.6e9)),
    ('HBM 2.0 TB/s 连续', dict(hbm_bandwidth_Bps=2.0e12)),
    ('L1D 128 KiB', dict(l1d_bytes=131072)),
    ('以上全部', dict(cores=144, clock_hz=1.6e9, l1d_bytes=131072,
                 vector_fp16_flops_per_cycle_per_core=7.866753472222222,
                 hbm_bandwidth_Bps=2.0e12)),
]


def main():
    base = Hardware.load(HW)
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    print('矢量效率保持不变（库的性质，不随芯片变）：softmax gmem %.4f、gelu %.4f\n'
          % (profile['efficiency']['softmax']['fp16']['gmem'], profile['efficiency']['gelu']['fp16']['elem']))
    print('%-22s %12s %12s %12s   %-8s' % (
        '硬件', 'softmax µs', 'gelu µs', '长行 µs', 'softmax受限'))
    ref = None
    rows = []
    for name, changes in VARIANTS:
        hw = dataclasses.replace(base, **changes)
        s = predict(SOFTMAX, hw, profile)
        g = predict(GELU, hw, profile)
        l = predict(LONG, hw, profile)
        if ref is None:
            ref = (s['duration_us'], g['duration_us'], l['duration_us'])
        rows.append((name, s, g, l))
        print('%-22s %12.1f %12.1f %12.1f   %-8s  (%+.0f%% / %+.0f%% / %+.0f%%)' % (
            name, s['duration_us'], g['duration_us'], l['duration_us'], s['bound'],
            100 * (s['duration_us'] / ref[0] - 1), 100 * (g['duration_us'] / ref[1] - 1),
            100 * (l['duration_us'] / ref[2] - 1)))

    print('\n逐项分解（softmax 131072x4096 fp16）：')
    print('%-22s %10s %10s %10s %10s %8s' % ('硬件', 't_mem', 't_vector', 'spill', 'occupancy', '总计'))
    for name, s, _, _ in rows:
        print('%-22s %10.1f %10.1f %10.1f %10.4f %8.1f' % (
            name, s['t_mem_s'] * 1e6, s['t_vector_s'] * 1e6, s['spill_s'] * 1e6,
            s['occupancy'], s['duration_us']))

    print('\n逐项分解（gelu 67.1M fp16）：')
    print('%-22s %10s %10s %10s %8s' % ('硬件', 't_mem', 't_vector', 'occupancy', '总计'))
    for name, _, g, _ in rows:
        print('%-22s %10.1f %10.1f %10.4f %8.1f' % (
            name, g['t_mem_s'] * 1e6, g['t_vector_s'] * 1e6, g['occupancy'], g['duration_us']))

    print('\n逐项分解（长行 softmax 8192x65536 fp16，行 128 KiB > L1D）：')
    print('%-22s %10s %10s %10s %8s' % ('硬件', 't_mem', 't_vector', 'spill', '总计'))
    for name, _, _, l in rows:
        print('%-22s %10.1f %10.1f %10.1f %8.1f' % (
            name, l['t_mem_s'] * 1e6, l['t_vector_s'] * 1e6, l['spill_s'] * 1e6, l['duration_us']))

    OUT.write_text(json.dumps(dict(
        efficiency=dict(softmax=profile['efficiency']['softmax']['fp16']['gmem'],
                        gelu=profile['efficiency']['gelu']['fp16']['elem']),
        rows=[dict(name=n, softmax_us=s['duration_us'], gelu_us=g['duration_us'],
                   long_us=l['duration_us'], bound=s['bound'],
                   softmax_pct=round(100 * (s['duration_us'] / ref[0] - 1), 1),
                   gelu_pct=round(100 * (g['duration_us'] / ref[1] - 1), 1),
                   long_pct=round(100 * (l['duration_us'] / ref[2] - 1), 1))
              for n, s, g, l in rows]), indent=1) + '\n')
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
