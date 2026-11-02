"""Read selected-kernel ABI metadata; never copy a dependency binary."""
import json
from pathlib import Path
import struct

ROOT=Path('/workspace/gemm_boundary_20261102')

def notes(path):
    with path.open('rb') as stream:
        h=struct.unpack('<16sHHIQQQIHHHHHH',stream.read(64))
        if h[0][:6]!=b'\x7fELF\x02\x01':raise ValueError('Expected little-endian ELF64')
        stream.seek(h[6]);raw=stream.read(h[11]*h[12])
        sections=[struct.unpack('<IIQQQQIIQQ',raw[i*h[11]:i*h[11]+64]) for i in range(h[12])]
        for section in sections:
            if section[1]!=7:continue
            stream.seek(section[4]);data=stream.read(section[5]);offset=0
            while offset+12<=len(data):
                n,d,t=struct.unpack_from('<III',data,offset);offset+=12
                name=data[offset:offset+n].rstrip(b'\0');offset+=(n+3)//4*4
                desc=data[offset:offset+d];offset+=(d+3)//4*4
                if len(desc)!=d:raise ValueError('Truncated ELF note')
                yield name,t,desc

def main():
    import msgpack
    selected=json.loads((ROOT/'kernel_instruction_evidence/rocblas_selected_symbols.json').read_text())['symbols']
    wanted={r['name'] for r in selected};matches=[]
    for source in sorted({r['file'] for r in selected}):
        for name,kind,data in notes(Path(source)):
            if name!=b'AMDGPU' or kind!=32:continue
            metadata=msgpack.unpackb(data,raw=False,strict_map_key=False)
            for kernel in metadata.get('amdhsa.kernels',[]):
                if kernel.get('.name') in wanted:
                    matches.append({'source':source,'metadata':kernel})
    if {r['metadata']['.name'] for r in matches}!=wanted:raise RuntimeError('Incomplete selected ABI metadata')
    target=ROOT/'kernel_instruction_evidence/selected_abi.json'
    if target.exists():raise FileExistsError('Preserve prior metadata')
    target.write_text(json.dumps({'kernels':matches,'scope':'Selected argument/resource metadata only; no binary exported'},indent=2))
    for r in matches:
        k=r['metadata'];print(k['.name'].split('_MT')[1].split('_SE')[0],k.get('.kernarg_segment_size'),flush=True)
        print(json.dumps(k.get('.args',[])),flush=True)

if __name__=='__main__':main()
