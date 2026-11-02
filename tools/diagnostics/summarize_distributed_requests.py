"""Summarize paired address-distribution measurements, no private input needed."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior report')
    data=json.loads(args.input.read_text());assert data['completed'] and len(data['rounds'])==2
    cases=[r for run in data['rounds'] for r in run['cases']]
    assert len(cases)==224 and all(r['correct'] and r['local_bytes']==0 for r in cases)
    groups={}
    for run in data['rounds']:
        for row in run['slopes']:
            groups.setdefault((row['id'],row['queued_panels'],row['blocks'],row['source_layout']),[]).append(row['ns_per_panel'])
    rows=[]
    for (case,panels,blocks,mode),values in sorted(groups.items()):
        assert len(values)==2 and min(values)>0
        base=groups[case,panels,blocks,'shared']
        aligned=groups[case.replace('_a2_','_a0_'),panels,blocks,mode]
        rows.append({'id':case,'queued_panels':panels,'blocks':blocks,'source_layout':mode,
                     'ns_per_panel':mean(values),'ratio_to_shared_sources':mean(values)/mean(base),
                     'ratio_to_aligned':mean(values)/mean(aligned),
                     'two_round_spread_percent':100*(max(values)-min(values))/mean(values)})
    result={'completed':True,'scope':data['scope'],'validated_measurements':len(cases),'summaries':rows,
            'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        if r['blocks']==72 and r['source_layout']=='distributed':
            print(r['id'],'Q',r['queued_panels'],f"{r['ns_per_panel']:.3f} ns",f"vs shared {r['ratio_to_shared_sources']:.4f}",f"vs aligned {r['ratio_to_aligned']:.4f}")


if __name__=='__main__':main()
