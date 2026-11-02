"""Private request-pattern input for an independent memory microbenchmark.

Input and generated data are local-only and must never be staged or pushed.
Validate complete-panel element coverage before making experiment inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior private evidence')
    trace=json.loads(args.input.read_text())
    if not trace['completed']:raise ValueError('Incomplete trace')
    report={'scope':'Private supplied address/Shared-destination patterns, not vendor kernel code. No GitHub upload.',
            'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'cases':[]}
    for case in trace['cases']:
        waves=len(case['waves']);panels=6 if waves==4 else 3;bm,bn=(192,128) if waves==4 else (384,256)
        named=case['arguments'];reads=[w['reads'][:3*panels] for w in case['waves']]
        validation={}
        for panel in range(panels):
            for operand in ('A','B'):
                chosen=[r for wave in reads for r in wave[panel*3:panel*3+3] if r['logical_operand']==operand]
                elements={offset//2+j for r in chosen for offset in r['tensor_offsets_bytes'] if offset is not None for j in range(r['load_bytes']//2)}
                if operand=='A':expected={r*named['strideB0']+k for r in range(bm) for k in range(panel*16,(panel+1)*16)}
                else:expected={k*named['strideA0']+c for k in range(panel*16,(panel+1)*16) for c in range(bn)}
                requested=sum(r['geometry']['requested_bytes'] for r in chosen)
                if elements!=expected or requested!=2*len(expected):raise ValueError('Missing, duplicated or unexpected panel input')
                validation[f'{operand}{panel}']={'bytes':requested,'unique_bytes':len(elements)*2}
        stages=[]
        for stage in range(panels*3):
            expected_operand='B' if stage%3==0 else 'A';expected_width=8 if stage%3==2 else 16
            offsets=[];destinations=[]
            for wave in reads:
                r=wave[stage]
                if r['logical_operand']!=expected_operand or r['load_bytes']!=expected_width or any(a is None for a in r['tensor_offsets_bytes']):raise ValueError('Unexpected opcode/operand/mask')
                offsets.extend(r['tensor_offsets_bytes']);destinations.extend([r['lds_base_bytes']]*64)
            stages.append({'operand':expected_operand,'load_bytes':expected_width,'offsets_bytes':offsets,'lds_m0':destinations})
        report['cases'].append({'id':case['id'],'waves':waves,'panels':panels,'block_m':bm,'block_n':bn,
                               'a_bytes':named['sizeB']*2,'b_bytes':named['sizeA']*2,
                               'a_base_mod_256':named['B']%256,'b_base_mod_256':named['A']%256,
                               'coverage':validation,'stages':stages})
        print(case['id'],'validated full panels',panels,flush=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
