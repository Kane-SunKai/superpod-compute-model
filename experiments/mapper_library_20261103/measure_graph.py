"""Fifty fixed-input Graph cases; use the historical timing protocol."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time


def stats(values):
    ordered = sorted(values)
    return dict(p50_us=statistics.median(ordered), p10_us=ordered[2],
                p90_us=ordered[22], samples_us=values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    target = Path(args.output).resolve()
    if not any(target.is_relative_to(Path(v)) for v in ('/workspace', '/root/sk')):
        raise ValueError('Output outside authorized server root')
    if target.exists():
        raise FileExistsError(target)
    plan = json.loads(Path(args.plan).read_text())
    import torch
    sys.path.insert(0, '/workspace')
    from benchmark_gemm import configure_precision, correctness_sample
    torch.cuda.set_device(0)
    torch.manual_seed(20261028)
    flags, warnings = configure_precision(torch)
    data = dict(completed=False, protocol=plan['protocol'], cases=[],
                plan_sha256=hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                environment=dict(torch=torch.__version__, hip=torch.version.hip,
                                 device=str(torch.cuda.get_device_properties(0)),
                                 precision_flags=flags), warnings=warnings)
    target.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        temp.replace(target)

    def timing(graph, calls):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        end.record()
        end.synchronize()
        values = []
        for _ in range(25):
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            values.append(begin.elapsed_time(end) * 1000 / calls)
        return stats(values)

    order = list(plan['cases'])
    random.Random(20261028).shuffle(order)
    save()
    start = time.monotonic()
    with torch.inference_mode():
        for row in order:
            m, n, k = row['shape']
            a = torch.randn(m, k, device='cuda', dtype=torch.float16) * .1
            b = torch.randn(k, n, device='cuda', dtype=torch.float16) * .1
            c = torch.empty(m, n, device='cuda', dtype=torch.float16)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(30):
                    torch.matmul(a, b, out=c)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            calls = 64 if 2*m*n*k > 1e9 else 256
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(calls):
                    torch.matmul(a, b, out=c)
            for _ in range(6):
                graph.replay()
            torch.cuda.synchronize()
            rounds = [timing(graph, calls) for _ in range(2)]
            check = correctness_sample(torch, a, b, c, 'fp16')
            record = dict(id=row['id'], group=row['group'], shape=row['shape'],
                          graph_calls=calls, rounds=rounds,
                          measured_graph_us=statistics.mean(r['p50_us'] for r in rounds),
                          correctness=check, a_stride=list(a.stride()), b_stride=list(b.stride()))
            record['round_drift_pct'] = 100 * abs(rounds[0]['p50_us'] - rounds[1]['p50_us']) / record['measured_graph_us']
            data['cases'].append(record)
            data['wall_seconds'] = time.monotonic() - start
            save()
            if not check['passed']:
                raise RuntimeError('Numerical failure: ' + row['id'])
            print(len(data['cases']), row['id'], round(record['measured_graph_us'], 3), 'us', flush=True)
            del graph, a, b, c
    data['completed'] = True
    save()


if __name__ == '__main__':
    main()
