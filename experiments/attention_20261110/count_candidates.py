"""How many candidate mappings one shape actually enumerates, and from what."""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy
from superpod_compute.operators import default_registry

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = ROOT / 'experiments/regression_100_v2_20261109/policy_no192.json'
SHAPES = [(4096, 4096, 16384), (8192, 8192, 32768)]


def main():
    hw = Hardware.load(HW)
    pol = Policy(**json.loads(POL.read_text(encoding='utf-8')))
    model = default_registry().get('gemm')
    print(f'bm_values {pol.bm_values}  (+1 shape candidate)')
    print(f'bn_values {pol.bn_values}')
    print(f'bk_values {pol.bk_values}')
    print(f'buffer_counts {pol.buffer_counts}   output_chunk_values {pol.output_chunk_values}'
          f'   output_buffer_counts {pol.output_buffer_counts}'
          f'   register_buffer_counts {pol.register_buffer_counts}'
          f'   operand_layouts {pol.operand_layouts}')
    for m, n, k in SHAPES:
        req = GemmRequest(m, n, k, 'fp16', layout='row-major-contiguous')
        tiles, geos, buffers, total = Counter(), Counter(), Counter(), 0
        for mp in model.candidates(req, hw, pol):
            total += 1
            tiles[(mp.bm, mp.bn, mp.bk)] += 1
            geos[(mp.gm, mp.gn)] += 1
            buffers[(mp.l2_buffers, mp.shared_buffers, mp.output_chunk_bytes,
                     mp.output_buffers, mp.register_input_buffers)] += 1
        print('\n%d×%d×%d -> %d candidates' % (m, n, k, total))
        print('  distinct tiles (bm,bn,bk): %d' % len(tiles))
        print('  distinct (gm,gn):         %d' % len(geos))
        print('  distinct buffer combos:   %d' % len(buffers))
        print('  tiles × geos × buffers =  %d' % (len(tiles) * len(geos) * len(buffers)))


if __name__ == '__main__':
    main()
