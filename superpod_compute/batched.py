"""Independent-head batched GEMM (attention QK^T / PV) on BW1100.

Two structural regimes, modelled separately because the vendor library lands
in different kernels for them (measured 2026-11-17, CUDA-graph protocol; all
constants live in `data/bw1100_bmm_profile.json`):

  PV-like  (n <= k)  output-narrow, K-streaming. The N dimension is small
                     (a head dim), so each member contributes few output
                     tiles and the A/B stream runs along K.
  QKT-like (n > k)   output-wide, short K. The N dimension is the sequence
                     length, so each member contributes many output tiles and
                     the cost is set by the B stream and the C write.

Within each, the library switches once more on the member count:
  batch == 1        the non-batched kernel (far cheaper per byte);
  m < 64            a vector/GEMV kernel whose per-member work is small
                    enough that a single block's stream time floors the few
                    member case;
  m >= 64           the tiled kernel: members are cut into output tiles and
                    the elapsed time is the larger of the A (PV) or C (QKT)
                    stream at its measured rate, floored by one tile's
                    stream time when only a handful of tiles are in flight.

The byte volumes are exact (A: batch*m*k*2, B: batch*n*k*2, C: batch*m*n*2);
only the per-regime launch offset and achievable stream rate are calibrated.
No operand is shared between batch members and each member's output is
written once.
"""
import json
from pathlib import Path

from .contracts import integer

PROFILE = Path(__file__).resolve().parent / 'data/bw1100_bmm_profile.json'


def load_profile(path=None):
    return json.loads(Path(path or PROFILE).read_text(encoding='utf-8'))


def _pv(n, k):
    """PV-like: output-narrow, the stream runs along K."""
    return n <= k


def predict_batched_gemm(batch, m, n, k, hardware, policy, *, transpose_b=False,
                         profile=None):
    for label, value in [('batch', batch), ('M', m), ('N', n), ('K', k)]:
        integer(value, label)
    prof = profile or load_profile()

    a_bytes = batch * m * k * 2
    b_bytes = batch * n * k * 2
    c_bytes = batch * m * n * 2
    footprint = a_bytes + b_bytes + c_bytes
    if footprint > hardware.hbm_capacity_bytes:
        raise ValueError('Batched inputs/output exceed HBM')

    gemv_threshold = prof.get('gemv_m_threshold', 64)
    regime = None

    if batch == 1 and m < gemv_threshold:
        single = prof['single']
        regime = 'single-%s' % ('pv' if _pv(n, k) else 'qk')
        launch = single['launch_pv_s'] if _pv(n, k) else single['launch_qk_s']
        duration = launch + footprint / single['stream_bw_Bps']

    elif m < gemv_threshold:
        if _pv(n, k):
            pv = prof['gemv_pv']
            if m == 1:
                # One block streams the member's whole B row set by itself.
                regime = 'gemv-pv-single-row'
                floor = n * k * 2 / pv['block_bw_Bps']
                stream = pv['launch_s'] + footprint / pv['stream_bw_Bps']
            else:
                regime = 'gemv-pv'
                floor = pv['multi_floor_s']
                stream = pv['multi_launch_s'] + footprint / pv['multi_stream_bw_Bps']
            duration = max(floor, stream)
        else:
            qk = prof['gemv_qk']
            regime = 'gemv-qk'
            duration = qk['launch_s'] + footprint / qk['stream_bw_Bps']

    else:
        tile = prof['tile']
        if _pv(n, k):
            pv = tile['pv']
            regime = 'tile-pv'
            # A is re-read for every member and dominates; the C write is
            # small next to it for these shapes. With a single member the
            # tile floor does not bind (one member's tiles already fill
            # enough of the card on their own).
            stream = pv['launch_s'] + a_bytes / pv['a_bw_Bps']
            duration = stream if batch == 1 else max(pv['floor_s'], stream)
        else:
            qk = tile['qk']
            regime = 'tile-qk'
            # The C write stream sets the time; its measured rate falls as
            # the reduction length grows because the B reads interleave.
            k_ref = qk.get('c_bw_k_ref', 0)
            c_bw = qk['c_bw_Bps']
            if k_ref and k > 0:
                exponent = qk.get('c_bw_k_exponent', 0.0)
                c_bw = c_bw * (k_ref / k) ** exponent
            duration = max(qk['floor_s'], qk['launch_s'] + c_bytes / c_bw)

    best = dict(
        duration_us=duration * 1e6,
        duration_s=duration,
        regime=regime,
        mapping=dict(bm=min(m, 64), bn=min(n, 64), bk=k, gm=1, gn=1,
                     l2_buffers=1, shared_buffers=1, output_chunk_bytes=16384,
                     output_buffers=1, register_input_buffers=1,
                     operand_layout='compact'),
        output_tasks=batch,
        wave_count=batch,
        wave_classes=[],
        hbm_read_bytes=a_bytes + b_bytes,
        l2_read_bytes=0,
        output_bytes=c_bytes,
    )
    return dict(
        operator='batched_gemm', batch=batch, shape=[m, n, k],
        transpose_b=transpose_b, best=best,
        searched_candidates=0, feasible_candidates=1,
        scope=('Independent heads; exact operand byte volumes (A, B, C); '
               'per-regime launch offset and stream rate calibrated on BW1100 '
               '(换机必须重测); no inter-member operand sharing; no fusion'),
        minimum_batched_tile=gemv_threshold,
    )
