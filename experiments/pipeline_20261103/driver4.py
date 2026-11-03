import pathlib,subprocess
r=pathlib.Path('/workspace/pipeline_20261103')
src=(r/'memory_probe.cpp').read_text()
src=src[:src.index('int main()')]+'''int main(){CK(hipSetDevice(0));for(int round:{1,2})for(int n:{6144,12288,24576})for(int blocks:{288,576,1152,2304})run(6144,n,0,blocks,false,round);}\n'''
(r/'memory_saturation.cpp').write_text(src)
for args,name in [(['/opt/dtk/bin/hipcc','-O3','--offload-arch=gfx938','-std=c++17','memory_saturation.cpp','-o','memory_saturation'],'memory_saturation_compile'),([str(r/'memory_saturation')],'memory_saturation'),(['python3','-B','cache_rotation.py'],'cache_rotation')]:
 print('START',name,flush=True)
 with (r/(name+'.log')).open('w') as f:subprocess.run(args,cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,flush=True)
