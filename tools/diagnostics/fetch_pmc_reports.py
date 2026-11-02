"""Copy only explicitly named counter CSV/metadata reports from the user's host."""
import argparse,concurrent.futures,hashlib,json,re,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ssh-config',required=True)
    ap.add_argument('--host',default='codex-bw1100');ap.add_argument('--cases',nargs='+',required=True)
    ap.add_argument('--category',choices=('pmc','pmc_read'),default='pmc')
    args=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.host):raise ValueError('Use a configured SSH alias')
    for case in args.cases:
        if not re.fullmatch(r'[A-Za-z0-9_]+',case):raise ValueError('Invalid case ID')
    def fetch(case):
        destination=ROOT/'results/boundary_20261102'/args.category/case
        destination.mkdir(parents=True,exist_ok=True)
        names=('report.csv','status.json','command.json')
        sources=[f'{args.host}:/root/sk/gemm_boundary_20261102/{args.category}/{case}/{name}' for name in names]
        p=subprocess.run(['scp','-F',args.ssh_config,'-o','BatchMode=yes',*sources,str(destination)],capture_output=True,text=True)
        if p.returncode:raise RuntimeError(case+': '+p.stderr)
        status=json.loads((destination/'status.json').read_text())
        if status['returncode']!=0:raise RuntimeError('Counter acquisition failed: '+case)
        hashes={name:hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in names}
        (destination/'transfer_manifest.json').write_text(json.dumps({'case':case,'files':hashes,'scope':'Explicit CSV/status/command allowlist; no binaries or log directories'},indent=2))
        return case
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for case in pool.map(fetch,args.cases):print('Fetched explicit reports:',case,flush=True)


if __name__=='__main__':main()
