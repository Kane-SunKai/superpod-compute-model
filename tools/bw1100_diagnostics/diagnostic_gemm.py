"""Controlled timing diagnostics; all outputs/caches stay in /workspace mount."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import torch
from benchmark_gemm import configure_precision, correctness_sample


def summary(values):
    s = sorted(values)
    return {'p50_us': statistics.median(s), 'min_us': s[0],
            'p90_us': s[math.ceil(.9 * len(s)) - 1], 'max_us': s[-1],
            'mean_us': statistics.mean(s), 'samples_us': values}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def timing(call, count, *, flush=None, divisor=1):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(); end.record(); end.synchronize()
    gpu, host = [], []
    for i in range(count):
        if flush is not None:
            flush.fill_(i % 251)
            torch.cuda.synchronize()
        start.record()
        t0 = time.perf_counter_ns()
        call()
        end.record()
        end.synchronize()
        host.append((time.perf_counter_ns() - t0) / 1000 / divisor)
        gpu.append(start.elapsed_time(end) * 1000 / divisor)
    return {'event': summary(gpu), 'host': summary(host), 'divisor': divisor}


def capture(a, b, c, count):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(20):
            torch.mm(a, b, out=c)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(count):
            torch.mm(a, b, out=c)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def profile_case(a, b, c, flush, name, outdir):
    info = {'supported_activities': [str(x) for x in torch.profiler.supported_activities()]}
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    for mode in ('warm', 'evict'):
        path = outdir / f'{name}_{mode}_trace.json'
        with torch.profiler.profile(activities=activities, record_shapes=False,
                                    with_stack=False, profile_memory=False) as prof:
            for i in range(10):
                if mode == 'evict':
                    flush.fill_(i)
                torch.cuda.synchronize()
                with torch.profiler.record_function(f'{name}/{mode}/mm/{i}'):
                    torch.mm(a, b, out=c)
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
        kernels = [{'name': x.get('name'), 'duration_us': x.get('dur'),
                    'ts': x.get('ts'), 'args': x.get('args', {})}
                   for x in trace.get('traceEvents', [])
                   if x.get('cat') == 'kernel' and x.get('ph') == 'X']
        info[mode] = {'trace': str(path), 'kernels': kernels,
                      'kernel_names': dict(Counter(x['name'] for x in kernels)),
                      'category_counts': dict(Counter(x.get('cat', '') for x in trace.get('traceEvents', [])))}
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='diagnostics_r1.json')
    p.add_argument('--samples', type=int, default=60)
    p.add_argument('--graph-repeats', type=int, default=64)
    p.add_argument('--skip-profiler', action='store_true')
    args = p.parse_args()
    output = Path(args.output).resolve()
    root = Path('/workspace/compute_diagnostics_20261028').resolve()
    if not output.is_relative_to(root):
        raise ValueError('Output must stay in authorized /workspace diagnostics directory')
    torch.cuda.set_device(0)
    torch.manual_seed(2026)
    flags, warnings = configure_precision(torch)
    record = {'utc': datetime.now(timezone.utc).isoformat(),
              'torch': torch.__version__, 'hip': torch.version.hip,
              'device': str(torch.cuda.get_device_properties(0)),
              'flags': flags, 'warnings': warnings,
              'samples': args.samples, 'graph_repeats': args.graph_repeats, 'cases': [],
              'scope': 'Graph batch amortizes submission overhead but changes scheduling/reuse. '
                       'Profiler has overhead; kernel names/grids are direct trace evidence. '
                       'Eager timings reproduce original single-call measurement scope.'}
    with torch.inference_mode():
        flush = torch.empty(256 * 1024**2, dtype=torch.uint8, device='cuda')
        for name, m, n, k in [('regular', 3072, 2304, 1024),
                              ('non_multiple', 130, 97, 270),
                              ('single_row', 1, 8192, 8192)]:
            print(f'CASE {name} {m}x{n}x{k}', flush=True)
            a = torch.randn(m, k, device='cuda', dtype=torch.float16)
            b = torch.randn(k, n, device='cuda', dtype=torch.float16)
            a.mul_(.1); b.mul_(.1)
            c = torch.empty(m, n, device='cuda', dtype=torch.float16)
            for _ in range(100):
                torch.mm(a, b, out=c)
            torch.cuda.synchronize()
            row = {'name': name, 'M': m, 'N': n, 'K': k,
                   'correctness': correctness_sample(torch, a, b, c, 'fp16'), 'timings': {}}
            if not row['correctness']['passed']:
                raise RuntimeError('Correctness failure')
            record['cases'].append(row)
            for mode in ('warm', 'evict'):
                for _ in range(30):
                    torch.mm(a, b, out=c)
                torch.cuda.synchronize()
                row['timings']['eager_' + mode] = timing(lambda: torch.mm(a, b, out=c),
                                                       args.samples, flush=flush if mode == 'evict' else None)
                print(mode, row['timings']['eager_' + mode]['event']['p50_us'], flush=True)
            try:
                graph = capture(a, b, c, args.graph_repeats)
                row['timings']['graph_batch_warm'] = timing(graph.replay, 30, divisor=args.graph_repeats)
                print('graph_batch', row['timings']['graph_batch_warm']['event']['p50_us'], flush=True)
                graph_one = capture(a, b, c, 1)
                row['timings']['graph_single_warm'] = timing(graph_one.replay, args.samples)
                row['timings']['graph_single_evict'] = timing(graph_one.replay, args.samples, flush=flush)
                row['graph_correctness'] = correctness_sample(torch, a, b, c, 'fp16')
                if not row['graph_correctness']['passed']:
                    raise RuntimeError('Graph correctness failure')
                del graph, graph_one
            except Exception:
                row['graph_error'] = traceback.format_exc()
                print(row['graph_error'], flush=True)
            save(output, record)
            if not args.skip_profiler:
                try:
                    row['profiler'] = profile_case(a, b, c, flush, name, root / 'traces')
                    print('profile kernel names', {mode: row['profiler'][mode]['kernel_names']
                                                    for mode in ('warm', 'evict')}, flush=True)
                except Exception:
                    row['profiler_error'] = traceback.format_exc()
                    print(row['profiler_error'], flush=True)
            save(output, record)
            del a, b, c
    record['completed'] = True
    save(output, record)
    print('SAVED', output, flush=True)


if __name__ == '__main__':
    main()
