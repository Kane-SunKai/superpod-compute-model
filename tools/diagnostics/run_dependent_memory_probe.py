from pathlib import Path
import subprocess,json,re
r=Path("/workspace/gemm_boundary_20261102")
common=["/opt/dtk/bin/hipcc","-O3","--offload-arch=gfx938","-std=c++17",str(r/"dependent_memory_probe.cpp")]
p=subprocess.run(common+["-o",str(r/"dependent_memory_probe")],cwd=r,capture_output=True,text=True)
(r/"memory_probe_build.log").write_text(p.stdout+p.stderr)
if p.returncode:raise SystemExit(p.returncode)
subprocess.run(common+["--cuda-device-only","-S","-o",str(r/"dependent_memory_probe.s")],cwd=r,check=True,capture_output=True)
report={"rounds":[],"scope":"Dependent volatile scalar global loads, one active lane per workgroup; 64-byte node stride; pointer-chain recurrence checked. Footprint suggests cache regime but does not prove hit rates. Do not substitute wholesale for bulk-copy latency."}
for repeat in range(2):
 p=subprocess.run([str(r/"dependent_memory_probe")],cwd=r,capture_output=True,text=True)
 (r/f"memory_probe_round{repeat+1}.log").write_text(p.stdout+p.stderr)
 if p.returncode:raise SystemExit(p.returncode)
 rows=[json.loads(line) for line in p.stdout.splitlines() if line.startswith("{")]
 assert len(rows)==16 and all(x["correct"] for x in rows)
 slopes=[]
 for footprint in sorted({x["footprint_bytes"] for x in rows}):
  for blocks in (1,72):
   a,b=sorted((x for x in rows if x["footprint_bytes"]==footprint and x["blocks"]==blocks),key=lambda x:x["iterations"])
   slopes.append({"footprint_bytes":footprint,"blocks":blocks,"event_slope_ns_per_load":(b["median_event_us"]-a["median_event_us"])*1000/(b["iterations"]-a["iterations"]),"clock_ticks_per_load":(b["median_block_clock_ticks"]-a["median_block_clock_ticks"])/(b["iterations"]-a["iterations"])})
 report["rounds"].append({"cases":rows,"slopes":slopes})
asm=(r/"dependent_memory_probe.s").read_text()
report["assembly_load_examples"]=[s.strip() for s in asm.splitlines() if re.search(r"(?i)global_load|buffer_load|flat_load|v_ld|ld_g|s_load",s)][:25]
(r/"dependent_memory_results.json").write_text(json.dumps(report,indent=2))
print(json.dumps({"slopes":[x["slopes"] for x in report["rounds"]],"assembly_load_examples":report["assembly_load_examples"]},indent=2))
