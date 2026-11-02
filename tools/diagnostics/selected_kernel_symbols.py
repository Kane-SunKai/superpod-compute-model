"""Read only ELF64 symbol tables from observed HIP dependencies.

No complete binary is copied or exported. Reports contain the exact selected
symbol's location, size, and available compiler-support library filenames.
"""
import ctypes.util
import json
from pathlib import Path
import struct

ROOT = Path('/workspace/gemm_boundary_20261102')


def symbols(path, wanted):
    with path.open('rb') as stream:
        header = struct.unpack('<16sHHIQQQIHHHHHH', stream.read(64))
        if header[0][:6] != b'\x7fELF\x02\x01':
            raise ValueError('Only little-endian ELF64 supported')
        shoff, shsize, count = header[6], header[11], header[12]
        stream.seek(shoff)
        raw = stream.read(shsize*count)
        sections = [struct.unpack('<IIQQQQIIQQ', raw[i*shsize:i*shsize+64]) for i in range(count)]
        seen = set()
        for section in sections:
            if section[1] not in (2, 11):
                continue
            strings = sections[section[6]]
            stream.seek(strings[4]); names = stream.read(strings[5])
            stream.seek(section[4]); table = stream.read(section[5])
            entries = [struct.unpack('<IBBHQQ', table[i:i+24]) for i in range(0,len(table),section[9])]
            function_addresses = sorted({e[4] for e in entries if e[1]&15==2 and e[3]})
            for entry in entries:
                if not entry[3] or entry[3] >= len(sections):
                    continue
                end = names.find(b'\0', entry[0])
                name = names[entry[0]:end].decode('utf-8', errors='replace')
                if name not in wanted or name in seen:
                    continue
                seen.add(name)
                code = sections[entry[3]]
                offset = code[4]+entry[4]-code[3]
                if not (code[4] <= offset and offset+entry[5] <= code[4]+code[5]):
                    raise ValueError('Symbol extends outside its section')
                bound = min([code[3]+code[5]]+[v for v in function_addresses if v>entry[4]])-entry[4]
                yield {'name': name, 'symbol_type': entry[1]&15, 'file': str(path), 'file_offset': offset,
                       'virtual_address': entry[4], 'size': entry[5], 'next_function_bound_bytes':bound}


def main():
    rows = json.loads((ROOT/'kernels.json').read_text())['cases']
    wanted = {name for r in rows if r['shape'] in ([1536,1152,1024], [3072,2304,1024])
              for name in r['kernels'] if '_MT' in name}
    loaded = json.loads((ROOT/'module_paths/paths.json').read_text())['paths']
    result = []
    # Runtime handle tracing identified librocblas.so.4 as the selected loader.
    libraries = [p for p in Path('/opt/dtk/lib/rocblas').glob('library*') if p.is_dir()]
    candidates = sorted((p for library in libraries for p in library.iterdir() if p.is_file() and
                         'gfx938' in p.name and p.suffix in ('.co','.hsaco')),
                        key=lambda p: ('Ailk_Bljk' not in p.name, 'HH_HPA' not in p.name, p.name))
    print('rocBLAS gfx938 code objects',len(candidates),flush=True)
    scanned=[]
    for path in candidates:
        matches=list(symbols(path,wanted))
        result.extend(matches);scanned.append(str(path))
        if matches:print('MATCH',path.name,[x['size'] for x in matches],flush=True)
        if {r['name'] for r in result}==wanted:break
    support = sorted({str(p) for base in (Path('/opt/dtk/lib'), Path('/opt/dtk/llvm/lib'))
                      for p in base.glob('*comgr*.so*')})
    found = ctypes.util.find_library('amd_comgr')
    if found:
        support.append(found)
    report = {'symbols': result, 'requested': sorted(wanted), 'comgr_candidates': support,'scanned':scanned,
              'scope': 'Read-only symbol metadata; no binary exported'}
    (ROOT/'kernel_instruction_evidence/rocblas_selected_symbols.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
