"""Compile-only resource evidence; no GPU execution or fitted durations."""
import json,pathlib,re,subprocess
p=pathlib.Path('/workspace/compute_mmac_20261028')
s=(p/'mmac_probe.cpp').read_text().split('int main()')[0]
s=s.replace('iterations/128','iterations/16').replace('u<128','u<16')
s+='\nvoid instantiate(float* out,unsigned long long* clocks,int iterations) {\n'
for chains in (32,48,64,80,96):
    s+=f'hipLaunchKernelGGL(probe<{chains}>,dim3(1),dim3(512),0,0,out,clocks,iterations);\n'
s+='}\n'
(p/'register_probe.cpp').write_text(s)
subprocess.run(['/opt/dtk/bin/hipcc','-O3','--offload-arch=gfx938','-std=c++17','--cuda-device-only','-S','register_probe.cpp','-o','register_probe.s'],check=True,capture_output=True)
asm=(p/'register_probe.s').read_text()
# Extract only the final metadata, not a full instruction trace.
ix=asm.find('.amdgpu_metadata')
meta=asm[ix:]
rows=[]
for item in re.split(r'(?m)^  - \.args:',meta)[1:]:
    name=re.search(r'(?m)^    \.name:\s+(\S+)',item)
    vgpr=re.search(r'(?m)^    \.vgpr_count:\s+(\d+)',item)
    scratch=re.search(r'(?m)^    \.private_segment_fixed_size:\s+(\d+)',item)
    if name and vgpr and scratch:rows.append({'name':name[1],'vgpr_count':int(vgpr[1]),'scratch_bytes_per_thread':int(scratch[1])})
result={'kernels':rows,'scope':'Same compiler and launch_bounds(512); register count/scratch size only; not proof of an immutable physical limit'}
(p/'register_probe_final.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
