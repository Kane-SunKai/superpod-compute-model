"""What the measured per-element cost means against the vector ceiling.

The predictive constant in vector_ops is `elem_cycles`: serial core-cycles per
element, measured. That is one fitted number per operator, dtype and regime. It
is not free-floating any more: the hardware table gives the vector throughput,
so the same number can be read as "effective flops per element at the peak",
which is checkable against what the operator actually needs.

Usage: python report_vector_utilization.py [profile.json]
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
PROFILE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'superpod_compute/data/bw1100_vector_profile.json'

# What each operator should cost if the kernel were purely the arithmetic the
# definition implies. These are estimates from the operation, not measurements,
# and they are only used to turn the fitted cost into a utilisation.
ISA_ESTIMATE = {
    ('gelu', 'fp16'): 30.0,      # erf on float, polynomial evaluated in fp32
    ('gelu', 'bf16'): 30.0,
    ('gelu', 'fp32'): 30.0,
    ('gelu-tanh', 'fp16'): 20.0,  # tanh approximation, one polynomial
    ('softmax', 'fp16'): 10.0,    # max, subtract, exp, accumulate, divide
    ('softmax', 'bf16'): 10.0,
    ('layernorm', 'fp16'): 12.0,  # mean and variance, then scale and shift
}


def main():
    hw = Hardware.load(HW)
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    peak = hw.vector_flops_per_cycle
    print('vector ceiling: %.3f flops/cycle/vector core x %d cores = %.2f flops/cycle/CU'
          % (hw.vector_fp16_flops_per_cycle_per_core, hw.vector_cores_per_core, peak))
    print('                at %.2f GHz that is %.1f TFLOPS for the card\n'
          % (hw.clock_hz / 1e9, peak * hw.cores * hw.clock_hz / 1e12))

    print('%-10s %-5s %-5s %10s %12s %14s %10s' % (
        'op', 'dtype', 'regime', 'elem_cyc', 'elem/CU/cyc', 'implied flops', 'est. util'))
    refs = {('softmax', 'warp'): (131072, 2048), ('softmax', 'gmem'): (131072, 4096),
            ('layernorm', 'warp'): (131072, 2048), ('layernorm', 'gmem'): (4096, 16384),
            ('gelu', 'elem'): (1, 67108864), ('gelu-tanh', 'elem'): (1, 67108864)}
    for kind, dtypes in profile['elem_cycles'].items():
        for dtype, regimes in dtypes.items():
            for key, e in regimes.items():
                rows, length = refs.get((kind, key), (1, 1))
                rc = profile.get('row_cycles', {}).get(kind, {}).get(dtype, {}).get(key, 0.0)
                t_alu = rows * (rc + length * e) / (hw.cores * hw.clock_hz)
                rate = rows * length / (t_alu * hw.cores * hw.clock_hz) if t_alu else 0.0
                implied = peak / rate if rate else 0.0
                isa = ISA_ESTIMATE.get((kind, dtype))
                util = ('%.1f%%' % (100 * isa / implied)) if isa else '—'
                print('%-10s %-5s %-5s %10.4f %12.3f %14.1f %10s' % (
                    kind, dtype, key, e, rate, implied, util))
    print('\nimplied flops = elem_cycles x vector ceiling: the arithmetic the kernel')
    print('would have to be doing to be vector-bound at the speed it actually runs.')
    print('est. util = an ISA-level op estimate divided by that, i.e. how much of the')
    print('vector unit the kernel is really using. The estimate is an assumption; the')
    print('measured column is not.')

    for req in (VectorRequest('softmax', 32 * 4096, 4096, 'fp16'),
                VectorRequest('gelu', 1, 4096 * 16384, 'fp16')):
        p = predict(req, hw, profile)
        print('\n%-8s %.1f us measured/model, vector-only ceiling %.1f us at 1 op/element, '
              'memory floor %.1f us'
              % (req.kind, p['duration_us'], p['vector_ceiling_s'] * 1e6,
                 p['t_mem_s'] * 1e6 / p['occupancy']))


if __name__ == '__main__':
    main()
