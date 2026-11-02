"""Observe dependency paths only; all explicit outputs stay in /workspace."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/workspace/gemm_boundary_20261102')


def main():
    out = ROOT / 'module_paths_v3'
    out.mkdir(exist_ok=False)
    library = out / 'libmodule_path_probe.so'
    command = ['/opt/dtk/bin/hipcc', '-O2', '-std=c++17', '-shared', '-fPIC',
               str(ROOT/'module_path_probe.cpp'), '-ldl', '-o', str(library)]
    compiled = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (out/'build.log').write_text(compiled.stdout+compiled.stderr)
    if compiled.returncode:
        raise SystemExit(compiled.returncode)
    env = {**os.environ, 'LD_PRELOAD': str(library)}
    command = ['python3', '-B', str(ROOT/'gpu_pmc_driver.py'), '--case',
               'k_stride_isolation_1536x1152x1024_contiguous', '--plan', 'k-stride']
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    (out/'run.log').write_text(result.stdout+result.stderr)
    paths = [line.split('\t', 2)[1:] for line in result.stderr.splitlines() if line.startswith('MODULE_PATH\t')]
    handles = [line.split('\t') for line in result.stderr.splitlines() if line.startswith(('MODULE_HANDLE\t','MODULE_DATA\t','MODULE_DATA_EX\t','MODULE_FUNCTION\t','MODULE_HEADER\t'))]
    (out/'paths.json').write_text(json.dumps({'returncode': result.returncode, 'paths': paths, 'handle_events': handles,
        'scope': 'Runtime loader path metadata only; no dependency file copied or modified'}, indent=2))
    print('exit', result.returncode, 'observed loader paths', len(paths), flush=True)
    print(json.dumps(paths), flush=True)
    functions=[r for r in handles if r[0]=='MODULE_FUNCTION' and '_MT' in r[-1] and r[1]=='0']
    selected={r[2] for r in functions}
    print(json.dumps([r for r in handles if (r[0]=='MODULE_DATA' and r[2] in selected) or (r[0]=='MODULE_HEADER' and r[1] in selected)]),flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
