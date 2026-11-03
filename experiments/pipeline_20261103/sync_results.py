import pathlib,subprocess,sys
root=pathlib.Path(__file__).resolve().parent
ssh=[r'C:\Windows\System32\OpenSSH\ssh.exe','-F',r'C:\Users\sunkai\.ssh\codex_bw1100.conf','-o','BatchMode=yes','codex-bw1100']
for name in sys.argv[1:]:
 if not all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in name):raise ValueError(name)
 data=subprocess.run(ssh+['cat /root/sk/pipeline_20261103/'+name],capture_output=True,check=True).stdout
 (root/name).write_bytes(data);print(name,len(data),flush=True)
