"""Read-focused counters for the paired K-tail/row-stride experiment."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/workspace/gemm_boundary_20261102')


def main():
    root = ROOT / 'pmc_read'
    root.mkdir(exist_ok=True)
    status = root / 'batch_status.json'
    if status.exists():
        raise FileExistsError('Inspect the existing acquisition before retrying')
    report = {'completed': False, 'cases': [], 'scope': 'Read counters only; not benchmark timing'}
    for k in (1024, 1023, 1025):
        for layout in ('contiguous', 'pad_a16'):
            case = f'k_stride_isolation_1536x1152x{k}_{layout}'
            command = [sys.executable, '-B', str(ROOT/'run_gpu_pmc.py'),
                       '--case', case, '--plan', 'k-stride', '--read-focus']
            result = subprocess.run(command, cwd=ROOT)
            report['cases'].append({'id': case, 'returncode': result.returncode})
            status.write_text(json.dumps(report, indent=2))
            if result.returncode:
                raise SystemExit(result.returncode)
    report['completed'] = True
    status.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
