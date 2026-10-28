import json,pathlib,subprocess,re
root=pathlib.Path('/workspace/compute_mmac_20261028')
common=['/opt/dtk/bin/hipcc','-O3','--offload-arch=gfx938','-std=c++17','lds_probe.cpp']
subprocess.run(common+['-o','lds_probe'],check=True)
subprocess.run(common+['--cuda-device-only','-S','-o','lds_device.s'],check=True,capture_output=True)
result=subprocess.run([str(root/'lds_probe')],check=True,capture_output=True,text=True)
rows=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
assert len(rows)==18 and all(x['correct'] for x in rows)
asm=(root/'lds_device.s').read_text()
instructions=[s.strip() for s in asm.splitlines() if re.search(r'(?i)^\s*(ds_|lds_|v_lds)',s)]
(root/'lds_final.json').write_text(json.dumps({'cases':rows,'instruction_examples':instructions[:12],
    'lds_instruction_lines':len(instructions),'scope':'Volatile aligned vector LDS reads plus vector accumulation; no GEMM fitting'},indent=2)+'\n')
print(json.dumps({'completed_cases':len(rows),'instruction_examples':instructions[:8]},indent=2))
