from pathlib import Path
import os,subprocess,json
r=Path("/workspace/gemm_boundary_20261102")
cmd=["/opt/dtk/bin/hipcc","-O2","-std=c++17","-shared","-fPIC",str(r/"kernel_resource_probe_v3.cpp"),"-ldl","-o",str(r/"libkernel_resource_probe_v3.so")]
p=subprocess.run(cmd,cwd=r,capture_output=True,text=True)
(r/"resource_build_v3.log").write_text(p.stdout+p.stderr)
if p.returncode:raise SystemExit(p.returncode)
env={**os.environ,"LD_PRELOAD":str(r/"libkernel_resource_probe_v3.so")}
p=subprocess.run(["python3","-B",str(r/"resource_driver.py")],env=env,cwd=r,capture_output=True,text=True)
(r/"resource_probe_v3.log").write_text(p.stdout+p.stderr)
rows=[]
for line in p.stderr.splitlines():
 if line.startswith("RESOURCE\t"):
  _,name,status,registers,shared,local,threads,occ_status,resident,actual_threads=line.split("\t")
  rows.append(dict(kernel=name,status=int(status),registers=int(registers),static_shared_bytes=int(shared),local_bytes=int(local),max_threads=int(threads),occupancy_status=int(occ_status),max_resident_workgroups_per_cu=int(resident),actual_threads=int(actual_threads)))
(r/"resources_v3.json").write_text(json.dumps({"returncode":p.returncode,"cases":rows,"scope":"runtime attributes queried by hipModuleGetFunction interception; not a timing run"},indent=2))
print("returncode",p.returncode,"resource records",len(rows),"successful queries",sum(x["status"]==0 for x in rows))
