"""Summarize independent delivery measurements; never reads private patterns."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior summary')
    data=json.loads(args.input.read_text())
    assert data['completed'] and len(data['rounds'])==2
    rows=[r for run in data['rounds'] for r in run['cases']]
    assert len(rows)==112 and all(r['correct'] and r['local_bytes']==0 for r in rows)
    groups={}
    for run in data['rounds']:
        for row in run['slopes']:
            groups.setdefault((row['id'],row['queued_panels'],row['blocks']),[]).append(row)
    summary=[]
    for (case,panels,blocks),replicates in sorted(groups.items()):
        assert len(replicates)==2
        values=[r['ns_per_panel'] for r in replicates]
        assert all(v>0 for v in values)
        summary.append({'id':case,'queued_panels':panels,'blocks':blocks,
                        'ns_per_panel':mean(values),'round_spread_percent':100*(max(values)-min(values))/mean(values),
                        'cycles_per_panel':mean(r['cycles_per_panel'] for r in replicates)})
    lookup={(r['id'],r['queued_panels'],r['blocks']):r for r in summary}
    for row in summary:
        reference=row['id'].replace('_a2_','_a0_')
        row['relative_to_aligned']=row['ns_per_panel']/lookup[reference,row['queued_panels'],row['blocks']]['ns_per_panel']
    result={'completed':True,'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
            'validated_measurements':len(rows),'scope':data['scope'],'summaries':summary}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in summary:
        if r['blocks']==72:
            print(r['id'],'queue',r['queued_panels'],f"{r['ns_per_panel']:.3f} ns/panel",f"ratio {r['relative_to_aligned']:.4f}")


if __name__=='__main__':main()
