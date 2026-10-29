"""Register -> LDS stores only; keep all generated files in the existing /workspace probe directory."""
import json,pathlib,subprocess,re,hashlib
root=pathlib.Path('/workspace/compute_mmac_20261028')
def logs():
    p=pathlib.Path('/var/log/hylog')
    return {str(f):[f.stat().st_size,f.stat().st_mtime_ns] for f in p.iterdir() if f.is_file()} if p.exists() else {}
before=logs()
common=['/opt/dtk/bin/hipcc','-O3','--offload-arch=gfx938','-std=c++17','lds_write_probe.cpp']
subprocess.run(common+['-o','lds_write_probe'],check=True)
subprocess.run(common+['--cuda-device-only','-S','-o','lds_write_device.s'],check=True,capture_output=True)
result=subprocess.run([str(root/'lds_write_probe')],check=True,capture_output=True,text=True)
rows=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
assert len(rows)==18 and all(x['correct'] for x in rows)
asm=(root/'lds_write_device.s').read_text()
slopes=[]
for b in (1,72,288):
    for w in (1,4,8):
        lo,hi=sorted([x for x in rows if x['blocks']==b and x['waves']==w],key=lambda x:x['iterations'])
        size=(hi['iterations']-lo['iterations'])*lo['bytes_per_iteration_per_block']
        slopes.append(dict(blocks=b,waves=w,bytes_per_block_tick=size/(hi['block_ticks']-lo['block_ticks']),
            gpu_event_TBps=size*b/(hi['median_us']-lo['median_us'])/1e6))
after=logs()
report={'cases':rows,'slopes':slopes,'correct':True,'log_sink_unchanged':before==after,
        'source_sha256':hashlib.sha256((root/'lds_write_probe.cpp').read_bytes()).hexdigest(),
        'assembly_shared_base':'src_shared_base' in asm,
        'vector_store_instruction_lines':len(re.findall(r'^\s*flat_store_dwordx4',asm,re.M)),
        'scope':'Volatile aligned 16-byte register-to-LDS stores. Per-block tick median across 9 launches; two iteration lengths remove fixed overhead. Output checks outside tick timing. Not a full GEMM epilogue measurement.'}
(root/'lds_write_final.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if before!=after:raise SystemExit('Known external log sink changed; stop further tests')
