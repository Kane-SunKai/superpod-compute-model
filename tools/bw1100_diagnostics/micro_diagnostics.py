"""Small GEMM service floor and graph amortization; no clock configuration writes."""
from pathlib import Path
import json
import threading
import time
import torch
from diagnostic_gemm import capture, timing, profile_case, save
from benchmark_gemm import configure_precision, correctness_sample

root = Path('/workspace/compute_diagnostics_20261028')
torch.cuda.set_device(0)
torch.manual_seed(2026)
configure_precision(torch)
stop = threading.Event()
samples = []


def monitor():
    files = [p for p in Path('/sys/class/drm').glob('card*/device/pp_dpm_sclk')]
    while not stop.is_set():
        row = {'monotonic_s': time.monotonic(), 'devices': {}}
        for clock in files:
            entry = {}
            for name in ('pp_dpm_sclk', 'pp_dpm_mclk', 'gpu_busy_percent'):
                try:
                    entry[name] = (clock.parent / name).read_text().strip()
                except OSError as exc:
                    entry[name] = str(exc)
            row['devices'][str(clock.parent)] = entry
        samples.append(row)
        stop.wait(.02)


thread = threading.Thread(target=monitor, daemon=True)
thread.start()
report = {'cases': [], 'telemetry': samples,
          'scope': 'Graph batch time per GEMM; contains device graph scheduling and kernel service. '
                   'Sysfs clock snapshots are reported states, not cycle-weighted frequency.'}
try:
    with torch.inference_mode():
        flush = torch.empty(256 * 1024**2, dtype=torch.uint8, device='cuda')
        for name, m, n, k in [('tiny16', 16, 16, 16), ('tiny64', 64, 64, 64),
                              ('non_multiple', 130, 97, 270),
                              ('regular', 3072, 2304, 1024), ('single_row', 1, 8192, 8192)]:
            a = torch.randn(m, k, device='cuda', dtype=torch.float16) * .1
            b = torch.randn(k, n, device='cuda', dtype=torch.float16) * .1
            c = torch.empty(m, n, device='cuda', dtype=torch.float16)
            row = {'name': name, 'shape': [m, n, k], 'graph_batch_sizes': {},
                   'start_monotonic_s': time.monotonic()}
            report['cases'].append(row)
            for batch in (1, 8, 64, 256):
                g = capture(a, b, c, batch)
                row['graph_batch_sizes'][str(batch)] = timing(g.replay, 30, divisor=batch)
                print(name, batch, row['graph_batch_sizes'][str(batch)]['event']['p50_us'], flush=True)
                del g
            row['correctness'] = correctness_sample(torch, a, b, c, 'fp16')
            if not row['correctness']['passed']:
                raise RuntimeError('Graph correctness failed')
            if name in ('tiny16', 'tiny64'):
                row['profiler'] = profile_case(a, b, c, flush, name, root / 'traces')
            row['end_monotonic_s'] = time.monotonic()
            save(root / 'micro_diagnostics.json', report)
            del a, b, c
    report['completed'] = True
finally:
    stop.set()
    thread.join()
    save(root / 'micro_diagnostics.json', report)
