"""Model prediction for the operators of the AICB GPT-7B path.

Seven are GEMM-shaped and go through the matrix model. Two are vector-path
operators (softmax over the score matrix, gelu over the FFN activations) and go
through superpod_compute.vector_ops, which models the row streaming, the overlap
and the wave occupancy instead of treating them as pure bandwidth.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from superpod_compute import GemmRequest, Hardware, Policy, predict
from superpod_compute.batched import predict_batched_gemm
from superpod_compute.vector_ops import VectorRequest, predict as predict_vector

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
POL = ROOT / 'experiments/regression_100_v2_20261109/policy_no192.json'
OUT = HERE / 'aicb_ops_model.json'
SQ, H, FFN, VOCAB, NP, HD = 4096, 4096, 16384, 50304, 32, 128

GEMMS = [('atten_qkv', SQ, 3 * H, H), ('atten_linear', SQ, H, H),
         ('mlp_linear_1', SQ, FFN, H), ('mlp_linear_2', SQ, H, FFN),
         ('logit_time', SQ, VOCAB, H)]
BATCHED = [('atten_core_qk', NP, SQ, SQ, HD), ('atten_core_contex', NP, SQ, HD, SQ)]
VECTOR = [('atten_core_softmax', 'softmax', NP * SQ, SQ, 'fp16'),
          ('mlp_gelu', 'gelu', 1, SQ * FFN, 'fp16')]


def main():
    hw = Hardware.load(HW)
    pol = Policy(**json.loads(POL.read_text(encoding='utf-8')))
    rows = []
    for name, m, n, k in GEMMS:
        r = predict(GemmRequest(m, n, k, 'fp16', layout='row-major-contiguous'), hw,
                    policy=pol, backend='mmac', cache_path=None)['best']
        mp = r['mapping']
        rows.append(dict(op=name, shape=[m, n, k], us=r['duration_us'],
                         tile='%dx%dx%d' % (mp['bm'], mp['bn'], mp['bk'])))
        print(name, round(r['duration_us'], 1), flush=True)
    for name, b, m, n, k in BATCHED:
        r = predict_batched_gemm(b, m, n, k, hw, pol)['best']
        mp = r['mapping']
        rows.append(dict(op=name, shape=[b, m, n, k], us=r['duration_us'],
                         tile='%dx%dx%d' % (mp['bm'], mp['bn'], mp['bk'])))
        print(name, round(r['duration_us'], 1), flush=True)
    for name, kind, nrows, length, dtype in VECTOR:
        r = predict_vector(VectorRequest(kind, nrows, length, dtype), hw)
        rows.append(dict(op=name, shape=[nrows, length], us=r['duration_us'],
                         tile='%s occ %.2f' % (r['bound'], r['occupancy'])))
        print(name, r['duration_us'], flush=True)
    OUT.write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
