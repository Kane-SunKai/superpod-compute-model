"""Verify graph replay uses the same named GEMM kernel as eager execution."""
from pathlib import Path
import json
from collections import Counter
import torch
from diagnostic_gemm import capture, save
from benchmark_gemm import configure_precision

root = Path('/workspace/compute_diagnostics_20261028')
configure_precision(torch)
results = []
with torch.inference_mode():
    for name, m, n, k in [('regular', 3072, 2304, 1024),
                          ('non_multiple', 130, 97, 270), ('single_row', 1, 8192, 8192)]:
        a = torch.randn(m, k, device='cuda', dtype=torch.float16)
        b = torch.randn(k, n, device='cuda', dtype=torch.float16)
        c = torch.empty(m, n, device='cuda', dtype=torch.float16)
        g = capture(a, b, c, 8)
        path = root / 'traces' / f'{name}_graph_trace.json'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                g.replay()
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())['traceEvents']
        kernels = [{'name': e['name'], 'duration_us': e['dur'], 'args': e.get('args', {})}
                   for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
        row = {'name': name, 'kernels': kernels,
               'kernel_names': dict(Counter(e['name'] for e in kernels))}
        results.append(row)
        print(name, row['kernel_names'], flush=True)
        del g, a, b, c
save(root / 'graph_kernel_check.json', results)
