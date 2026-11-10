"""BW1100 Decode, library-replay mode: predict the library's kernel, then time it.

Separate from the autonomous mapper and from decode.py. Kernel choice comes
from a measured shape -> kernel-class table (nearest neighbour in log2 M,N,K);
service time is a per-class wave model fitted on measurements that exclude
the held-out validation shapes:

    t = fixed + max(ceil(blocks / CUs) * bytes_per_block / r_block,
                    streamed_bytes / bandwidth_cap)

`r_block` is one block's sustained input stream rate for that kernel class. The
first term is the latency-bound wave regime (low prefetch, one resident block);
the second is the device bandwidth ceiling. The reported value is the fast
address mode; classes seen to change with B's placement carry an interval.
"""
import json, math, pathlib

DATA = pathlib.Path(__file__).parent / 'data/bw1100_decode_replay.json'


def class_key(tile, prefetch_global, workgroup):
    return '%dx%dx%d_pg%d_wg%dx%dx%d' % (*tile, prefetch_global, *workgroup)


def load(path=None):
    return json.loads(pathlib.Path(path or DATA).read_text(encoding='utf-8'))


def select(shape, profile):
    """Nearest measured shape in log2 space; returns (class_key, source_shape)."""
    lm = [math.log2(x) for x in shape]
    row = min(profile['table'], key=lambda r: (
        sum((a - math.log2(b)) ** 2 for a, b in zip(lm, r['shape'])), r['shape']))
    return row['class'], row['shape']


def service(shape, key, profile):
    m, n, k = shape
    cls = profile['classes'][key]
    bm, bn, _ = cls['tile']
    blocks = math.ceil(m / bm) * math.ceil(n / bn)
    rounds = math.ceil(blocks / profile['cus'])
    per_block = (bm + bn) * k * 2
    streamed = (m * k + k * n) * 2
    wave_s = rounds * per_block / cls['r_block_Bps']
    cap_s = streamed / profile['bandwidth_cap_Bps']
    return dict(blocks=blocks, rounds=rounds, wave_s=wave_s, cap_s=cap_s,
                bound='wave' if wave_s >= cap_s else 'bandwidth',
                duration_s=profile['fixed_s'] + max(wave_s, cap_s))


def predict(shape, profile=None):
    profile = profile or load()
    if len(shape) != 3 or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in shape):
        raise ValueError('M, N and K must be positive integers')
    m, n, k = shape
    lo, hi = profile['domain']['M']
    if not lo <= m <= hi:
        raise ValueError(f'Decode replay domain is {lo}<=M<={hi}')
    key, source = select(shape, profile)
    svc = service(shape, key, profile)
    cls = profile['classes'][key]
    result = dict(shape=list(shape), kernel_class=key, tile=cls['tile'],
                  prefetch_global=cls['prefetch_global'], selected_from=source,
                  duration_us=svc['duration_s'] * 1e6, regime=svc['bound'],
                  blocks=svc['blocks'], rounds=svc['rounds'],
                  model_version=profile['version'],
                  prediction_mode='BW1100 measured library-kernel replay (fast address mode)')
    ratio = cls.get('slow_over_fast', 1.0)
    if ratio > 1.1:
        result['address_sensitive'] = True
        result['interval_us'] = [result['duration_us'], result['duration_us'] * ratio]
    nmin, nmax = profile['domain']['N']; kmin, kmax = profile['domain']['K']
    if not (nmin <= n <= nmax and kmin <= k <= kmax):
        result['extrapolated'] = True
    return result
