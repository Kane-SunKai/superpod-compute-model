import json,pathlib,re
p=pathlib.Path('/workspace/compute_mmac_20261028')
s=(p/'lds_device.s').read_text();r=json.loads((p/'lds_final.json').read_text())
r['assembly_uses_shared_base']='src_shared_base' in s
r['assembly_group_segment_bytes']=int(re.search(r'\.group_segment_fixed_size:\s*(\d+)',s)[1])
r['assembly_vector_load_count']=len(re.findall(r'(?m)^\s*flat_load_dwordx4\s',s))
r['instruction_examples']=[line.strip() for line in s.splitlines() if 'src_shared_base' in line or 'flat_load_dwordx4' in line][:10]
assert r['assembly_uses_shared_base'] and r['assembly_group_segment_bytes']==16384 and r['assembly_vector_load_count']>=64
(p/'lds_final.json').write_text(json.dumps(r,indent=2)+'\n')
print('verified_shared_base, group_segment_bytes, vector_load_count:',r['assembly_uses_shared_base'],r['assembly_group_segment_bytes'],r['assembly_vector_load_count'])
