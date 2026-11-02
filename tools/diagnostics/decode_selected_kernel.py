"""Decode selected runtime kernel symbols through the installed COMGR API.

API reference: ROCprofiler-SDK disassembly.hpp, AMD ROCm 6.3.3 docs.
This implementation reads bounded symbol bytes into memory, writes TEXT only,
and never exports a code-object binary or modifies an installed dependency.
"""
from collections import Counter
import ctypes as C
import hashlib
import json
from pathlib import Path

ROOT = Path('/workspace/gemm_boundary_20261102')


class Handle(C.Structure):
    _fields_ = [('handle', C.c_uint64)]


def main():
    evidence = ROOT/'kernel_instruction_evidence'
    symbols = json.loads((evidence/'rocblas_selected_symbols.json').read_text())['symbols']
    out = evidence/'decoded'
    out.mkdir(exist_ok=False)
    lib = C.CDLL('/opt/dtk/lib/libamd_comgr.so')
    lib.amd_comgr_get_isa_count.argtypes = [C.POINTER(C.c_size_t)]
    lib.amd_comgr_get_isa_name.argtypes = [C.c_size_t,C.POINTER(C.c_char_p)]
    count=C.c_size_t()
    if lib.amd_comgr_get_isa_count(C.byref(count)):raise RuntimeError('COMGR ISA query failed')
    names=[]
    for i in range(count.value):
        value=C.c_char_p()
        if not lib.amd_comgr_get_isa_name(i,C.byref(value)) and b'gfx938' in value.value:
            names.append(value.value.decode())
    print('ISA names',names,flush=True)
    if not names:raise RuntimeError('Installed COMGR cannot decode gfx938')
    isa=min(names,key=len)
    Read=C.CFUNCTYPE(C.c_uint64,C.c_uint64,C.c_void_p,C.c_uint64,C.c_void_p)
    Print=C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
    Address=C.CFUNCTYPE(None,C.c_uint64,C.c_void_p)
    create=lib.amd_comgr_create_disassembly_info
    create.argtypes=[C.c_char_p,Read,Print,Address,C.POINTER(Handle)]
    decode=lib.amd_comgr_disassemble_instruction
    decode.argtypes=[Handle,C.c_uint64,C.c_void_p,C.POINTER(C.c_uint64)]
    destroy=lib.amd_comgr_destroy_disassembly_info;destroy.argtypes=[Handle]
    summaries=[]
    for index,symbol in enumerate(symbols):
        length=symbol['size'] or symbol['next_function_bound_bytes']
        if length<=0 or length>262144:raise ValueError('Unexpected symbol extent; inspect before decoding')
        with Path(symbol['file']).open('rb') as stream:
            stream.seek(symbol['file_offset']);raw=stream.read(length)
        if len(raw)!=length:raise ValueError('Short symbol read')
        address=symbol['virtual_address'];parts=[]
        def read(pc,destination,size,user):
            offset=pc-address
            if offset<0 or offset>=len(raw):return 0
            size=min(size,len(raw)-offset)
            C.memmove(destination,raw[offset:offset+size],size)
            return size
        def emit(text,user):parts.append(text.decode() if text else '')
        def annotation(pc,user):parts.append(f' @0x{pc:x}')
        callbacks=(Read(read),Print(emit),Address(annotation));handle=Handle()
        status=create(isa.encode(),*callbacks,C.byref(handle))
        if status:raise RuntimeError(f'COMGR create failed: {status}')
        decoded=[];offset=0
        try:
            while offset<len(raw):
                parts.clear();size=C.c_uint64()
                status=decode(handle,address+offset,None,C.byref(size))
                if status or not size.value:break
                text=''.join(parts).strip()
                decoded.append({'offset':offset,'address':address+offset,'size':size.value,'text':text})
                offset+=size.value
        finally:destroy(handle)
        counts=Counter(r['text'].split()[0] for r in decoded if r['text'])
        summary={**symbol,'isa':isa,'bounded_bytes':length,'decoded_bytes':offset,'stop_status':status,
                 'instruction_count':len(decoded),'symbol_bytes_sha256':hashlib.sha256(raw).hexdigest(),
                 'opcode_counts':dict(counts),'scope':'Static bounded disassembly, not a dynamic instruction trace; zero symbol size uses next function as an upper bound'}
        (out/f'kernel_{index}.json').write_text(json.dumps({'summary':summary,'instructions':decoded},indent=2))
        (out/f'kernel_{index}.txt').write_text('\n'.join(f"{r['offset']:06x}: {r['text']}" for r in decoded)+'\n')
        summaries.append(summary)
        print('decoded',index,len(decoded),'instructions',offset,'/',length,'bytes','status',status,flush=True)
        print('memory/matrix', {k:v for k,v in counts.items() if any(t in k for t in ('load','store','read','write','mmac','mfma','branch','barrier'))},flush=True)
    (out/'summary.json').write_text(json.dumps({'kernels':summaries},indent=2))


if __name__=='__main__':main()
