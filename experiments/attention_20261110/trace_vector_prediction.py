"""Walk a shape through the vector model and print every term.

Nothing is computed here that `predict()` does not already do; this only makes
the arithmetic visible so each number can be traced back to a measurement or to
a hardware field.

Usage: python trace_vector_prediction.py
"""
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

CASES = [
    (VectorRequest('softmax', 32 * 4096, 4096, 'fp16'), 3376.2, 'AICB 的 attention softmax'),
    (VectorRequest('softmax', 32 * 1024, 2048, 'fp16'), None, 'warp 区间示例'),
    (VectorRequest('gelu', 1, 4096 * 16384, 'fp16'), 639.1, 'AICB 的 mlp gelu'),
]


def report(req, measured, why, hw, profile):
    p = predict(req, hw, profile)
    b = req.element_bytes
    row_bytes = req.length * b
    waves = -(-p['workgroups'] // hw.cores)
    reduction = req.kind in ('softmax', 'layernorm')

    print('=' * 78)
    print('%s %s   (%s)' % (req.kind, req.dtype, why))

    if reduction:
        print('  形状: rows=%d, length=%d  -> 归约是"行"方向，共有 %d 行' % (
            req.rows, req.length, req.rows))
        print('  1) 归约核分类: length=%d %s %d (row_capacity_elements) -> %s' % (
            req.length, '<=' if p['regime'] == 'warp' else '>',
            profile['row_capacity_elements'], p['regime']))
        print('      行字节 %d x %d B = %d B，L1D %d B -> 第二遍%s' % (
            req.length, b, row_bytes, hw.l1d_bytes,
            '命中缓存' if row_bytes <= hw.l1d_bytes else '会溢出到 L2'))
        print('  2) 读遍数 scans = %d  (%s)' % (
            p['scans'], 'warp 核把整行留在寄存器里' if p['regime'] == 'warp'
            else 'gmem 核把每行流式读两遍'))
    else:
        print('  形状: N=%d 个元素，逐元素，没有"行"的概念' % req.elements)
        print('  1) 归约核分类: 不适用 -> elem 路径')
        print('  2) 读遍数 scans = 1 (读一遍写一遍)')

    print('  3) 内存项  t_mem = (scans+1)·N·b / B_stream')
    print('            = %d · %d · %d / %.3g = %.1f us' % (
        p['scans'] + 1, req.elements, b,
        (hw.hbm_bandwidth_Bps * profile.get('stream_bandwidth_ratio', 1.0)), p['t_mem_s'] * 1e6))

    if reduction:
        print('  4) 每行成本 per_row = row_cycles + length·elem_cycles  (elem_cycles = %.0f ÷ (%.3f × %.4f) = %.6f)' % (p['ops_per_element'], hw.vector_flops_per_cycle, p['efficiency'], p['elem_cycles']))
        print('            = %.2f + %d x %.6f = %.1f CU-cycles' % (
            p['row_cycles'], req.length, p['elem_cycles'], p['per_row_cycles']))
        print('  5) 计算项  t_alu = rows·per_row / (cores·clock)')
        print('            = %d · %.1f / (%d · %.3g) = %.1f us' % (
            req.rows, p['per_row_cycles'], hw.cores, hw.clock_hz, p['t_vector_s'] * 1e6))
    else:
        print('  4) 每元素周期 elem_cycles = ops/元素 ÷ (矢量算力 × 效率) = %.0f ÷ (%.3f × %.4f) = %.6f' % (p['ops_per_element'], hw.vector_flops_per_cycle, p['efficiency'], p['elem_cycles']))
        print('  5) 计算项  t_alu = N·elem_cycles / (cores·clock)')
        print('            = %d · %.6f / (%d · %.3g) = %.1f us' % (
            req.elements, p['elem_cycles'], hw.cores, hw.clock_hz, p['t_vector_s'] * 1e6))

    print('  6) 占用率  occupancy = WGs/(waves·cores) = %d/(%d·%d) = %.4f' % (
        p['workgroups'], waves, hw.cores, p['occupancy']))
    print('  7) 物理项  t_phys = max(t_mem, t_alu)/occupancy = %.1f us  (受限: %s)' % (
        p['t_physical_s'] * 1e6, p['bound']))

    if reduction and p['regime'] == 'gmem':
        print('  8) L1D 溢出 spill = rows·max(0, row_bytes-L1D) / (L2 读带宽·clock)')
        print('            = %d · %d / (%.0f · %.3g) = %.2f us' % (
            req.rows, max(0, row_bytes - hw.l1d_bytes),
            hw.l2_read_bytes_per_cycle, hw.clock_hz, p['spill_s'] * 1e6))
    else:
        print('  8) L1D 溢出 spill = 0 (这一项只对 gmem 归约核起作用)')

    print('  9) 延迟    latency = (alpha-1)·per_row/clock, alpha=%s -> %.2f us' % (
        profile['latency_factor'].get(p['regime'], 1.0), p['latency_s'] * 1e6))
    print(' 10) 发射    launch = %.1f us' % (profile['launch_s'] * 1e6))
    print(' 11) 总计 = t_phys + spill + latency + launch = %.1f us' % p['duration_us'])
    if measured:
        print(' 12) 实测 %.1f us -> 误差 %+.1f%%' % (
            measured, 100 * (p['duration_us'] / measured - 1)))


def main():
    hw = Hardware.load(HW)
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    for req, measured, why in CASES:
        report(req, measured, why, hw, profile)


if __name__ == '__main__':
    main()
