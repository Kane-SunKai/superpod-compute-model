"""Command-line entry for the BW1100 Decode library-replay model."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from superpod_compute.decode_replay import predict


def main():
    p = argparse.ArgumentParser(description='Predict FP16 A[M,K] @ B[K,N] on BW1100 as the library runs it; 1<=M<=32.')
    for name in ('m', 'n', 'k'):
        p.add_argument('--' + name, type=int, required=True)
    p.add_argument('--output', type=pathlib.Path)
    a = p.parse_args()
    r = predict([a.m, a.n, a.k])
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(r, indent=2) + '\n', encoding='utf-8')
    extra = ''
    if r.get('address_sensitive'):
        extra += '; address-sensitive, slow mode up to %.3f us' % r['interval_us'][1]
    if r.get('extrapolated'):
        extra += '; outside measured N/K range'
    print('M/N/K=%d/%d/%d; kernel %s; %s-bound; predicted=%.3f us%s' % (
        a.m, a.n, a.k, r['kernel_class'], r['regime'], r['duration_us'], extra))


if __name__ == '__main__':
    main()
