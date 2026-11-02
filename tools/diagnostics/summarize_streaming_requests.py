"""Compare independent prefix, full-K batch, and rolling request measurements."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',type=Path,required=True)
    ap.add_argument('--rolling',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior report')
    batch=json.loads(args.batch.read_text());rolling=json.loads(args.rolling.read_text())
    for data in (batch,rolling):assert data['completed'] and len(data['rounds'])==2
    br=[r for run in batch['rounds'] for r in run['cases']]
    rr=[r for run in rolling['rounds'] for r in run['cases']]
    assert len(br)==1008 and len(rr)==140
    assert all(r['correct'] and r['local_bytes']==0 for r in br+rr)
    assert sum(r['validation_only'] for r in rr)==28
    groups={}
    for data in (batch,rolling):
        for run in data['rounds']:
            for r in run['slopes']:
                mode=r.get('k_traversal','rolling')
                groups.setdefault((r['id'],r['blocks'],r['queued_panels'],mode),[]).append(r['ns_per_panel'])
    rows=[]
    for (case,blocks,q,mode),values in sorted(groups.items()):
        assert len(values)==2 and min(values)>0
        reference=groups[case,blocks,q,'stream']
        aligned=groups[case.replace('_a2_','_a0_'),blocks,q,mode]
        rows.append({'id':case,'blocks':blocks,'queued_panels':q,'mode':mode,
                     'ns_per_panel':mean(values),'equivalent_64_panels_us':mean(values)*64/1000,
                     'ratio_to_batch_full_k':mean(values)/mean(reference),
                     'ratio_to_aligned':mean(values)/mean(aligned),
                     'round_spread_percent':100*(max(values)-min(values))/mean(values)})
    result={'completed':True,'summaries':rows,'batch_validated_records':len(br),
            'rolling_timing_records':sum(not r['validation_only'] for r in rr),'rolling_readiness_checks':28,
            'scope':'Independent component loops, not GEMM prediction or an additive/lower-bound memory cost. Rolling validation consumes data separately from the timed kernel; full-K batch and rolling use different implementations.',
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.batch,args.rolling)}}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        if r['blocks']==72 and r['mode']=='rolling':print(r['id'],'Q',r['queued_panels'],f"{r['ns_per_panel']:.3f} ns/panel",f"64 panels {r['equivalent_64_panels_us']:.3f} us",flush=True)


if __name__=='__main__':main()
