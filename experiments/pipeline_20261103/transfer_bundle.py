"""Only upload named study inputs under /root/sk/pipeline_20261103."""
import pathlib,subprocess
root=pathlib.Path(__file__).resolve().parent
work=root.parents[2]
ssh=[r'C:\Windows\System32\OpenSSH\ssh.exe','-F',r'C:\Users\sunkai\.ssh\codex_bw1100.conf','-o','BatchMode=yes','codex-bw1100']
files={n:root/n for n in ('calibration_plan.json','driver.py','run_server.sh')}
files['measure_graph.py']=work/'graph50-recheck-20261103/measure_graph.py'
for n in ('mmac_probe.cpp','lds_probe.cpp'):files[n]=root.parents[1]/'tools/bw1100_service_probe'/n
for name,path in files.items():
    target='/root/sk/pipeline_20261103/'+name
    data=path.read_bytes().replace(b'\r\n',b'\n')
    subprocess.run(ssh+[f'mkdir -p /root/sk/pipeline_20261103 && test ! -e {target} && cat > {target}'],input=data,check=True)
    print('uploaded',name,flush=True)
