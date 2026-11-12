"""Why does a one-panel GEMM prefer shallow buffers? Print the ranking."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy
from superpod_compute.backends import make_backend
from superpod_compute.operators import GemmModel

SMALL = Policy(bm_values=(16, 32), bn_values=(16, 32), bk_values=(16, 32),
               add_shape_candidate=False)


def main():
    h = Hardware()
    req = GemmRequest(128, 128, 16)
    model = GemmModel()
    backend = make_backend('array', h, SMALL.matrix_efficiency)
    rows = []
    for p in model.candidates(req, h, SMALL):
        r = model.evaluate(req, h, SMALL, backend, p)
        if r is None:
            continue
        rows.append(((p.l2_buffers, p.shared_buffers, p.output_chunk_bytes,
                      p.output_buffers, p.register_input_buffers),
                     '%dx%dx%d' % (p.bm, p.bn, p.bk), r['duration_us']))
    rows.sort(key=lambda x: x[2])
    print('K/bk = %d panel(s), %d candidates' % (16 // 16, len(rows)))
    for combo, tile, us in rows[:6]:
        print('   combo %-26s %-13s %8.2f us' % (combo, tile, us))
    worst = rows[-1]
    print('   worst %-26s %-13s %8.2f us' % (worst[0], worst[1], worst[2]))


if __name__ == '__main__':
    main()
