"""Read-only inspection of the dependency paths observed by the HIP loader.

Never copies or alters dependency code objects. Only selected textual metadata
and disassembly (when supported) are written in the approved experiment folder.
"""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/workspace/gemm_boundary_20261102')


def main():
    out = ROOT/'kernel_instruction_evidence'
    out.mkdir(exist_ok=False)
    loaded = json.loads((ROOT/'module_paths/paths.json').read_text())
    bases = [Path('/opt/dtk')/s for s in ('bin','llvm/bin','hip/bin','hcc/bin','compiler/bin')]
    names = ('llvm-objdump','llvm-readelf','llvm-readobj','objdump','readelf','hy-objdump')
    tools = sorted({str(p) for base in bases for name in names if (p:=base/name).is_file()}
                   | {path for name in names if (path:=shutil.which(name))})
    files = [{'path': p, 'bytes': Path(p).stat().st_size} for status,p in loaded['paths'] if status=='0']
    report = {'tools': tools, 'loaded_files': files, 'scope': 'Observed dependencies inspected read-only; no binary copy'}
    (out/'inspection.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    # This pass only checks tool/ELF metadata. Symbol selection follows after inspection.


if __name__=='__main__':
    main()
