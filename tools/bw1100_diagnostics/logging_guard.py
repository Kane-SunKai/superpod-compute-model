"""Verify the known runtime log sink stays untouched with process-local guards."""
import json
from pathlib import Path

def snapshot():
    p = Path('/var/log/hylog')
    return {f.name: [f.stat().st_size, f.stat().st_mtime_ns]
            for f in p.iterdir() if f.is_file()} if p.is_dir() else {}

before = snapshot()
import torch
with torch.inference_mode():
    a = torch.ones(16, 16, dtype=torch.float16, device='cuda')
    c = a @ a
    torch.cuda.synchronize()
    correct = bool(torch.all(c == 16).item())
after = snapshot()
result = {'before': before, 'after': after, 'unchanged': before == after,
          'correctness': correct,
          'scope': 'Known /var/log/hylog log sink; no claim of a full-filesystem audit'}
Path('/workspace/compute_diagnostics_20261028/logging_guard.json').write_text(json.dumps(result, indent=2))
print('runtime_log_directory_unchanged:', result['unchanged'])
print('small_gemm_correct:', correct)
if before != after or not correct:
    raise SystemExit(1)
