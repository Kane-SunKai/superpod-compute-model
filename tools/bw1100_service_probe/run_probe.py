import collections,json,pathlib,re,subprocess,threading,time
root=pathlib.Path('/workspace/compute_mmac_20261028')
counts=collections.Counter();stop=threading.Event()
def monitor():
    paths=list(pathlib.Path('/sys/class/drm').glob('card*/device/pp_dpm_sclk'))
    while not stop.is_set():
        for p in paths:
            for line in p.read_text().splitlines():
                if '*' in line:counts[str(p)+': '+line.strip()]+=1
        stop.wait(.02)
t=threading.Thread(target=monitor,daemon=True);t.start()
try:
    data=subprocess.run([str(root/'mmac_probe')],text=True,capture_output=True,check=True)
finally:
    stop.set();t.join()
rows=[json.loads(line) for line in data.stdout.splitlines() if line.startswith('{')]
assert len(rows)==54,(len(rows),data.stdout[-1000:],data.stderr)
assert all(x['correct'] for x in rows)
asm=(root/'mmac_device.s').read_text()
mmac=[s.strip() for s in asm.splitlines() if re.search(r'(?i)^\s*(?:v_|hcu_|s_)?mmac[\s._]',s)]
clock=[s.strip() for s in asm.splitlines() if re.search(r'(?i)memtime|clock|timestamp',s)]
result={'completed':True,'instruction':'vendor DU_MMA_F32_16x16x16F16 / gfx938 builtin',
        'cases':rows,'clock_state_sample_counts':dict(counts),
        'assembly_mmac_line_count':len(mmac),'assembly_mmac_examples':mmac[:8],
        'assembly_clock_examples':clock[:8],
        'assembly_instruction_candidates':[s.strip() for s in asm.splitlines() if re.search(r'(?i)mmac|f16|clock|memtime',s)][:20],
        'scope':'Register-input repeated MMAC, no GEMM-shape fitting. clock state sampling is not cycle-weighted frequency.'}
(root/'mmac_final.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
