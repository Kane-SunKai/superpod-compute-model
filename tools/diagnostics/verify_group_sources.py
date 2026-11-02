"""Privately verify every workgroup's sources against the decoded prologue.

Output is private derived address data; never add it to Git/GitHub.
"""
import argparse
import hashlib
import json
from pathlib import Path
from trace_address_prologue import PrologueMachine


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--private-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior evidence')
    captures=json.loads((args.private_dir/'sanitized_arguments.json').read_text())
    reference=json.loads((args.private_dir/'prologue_reads.json').read_text())
    assert captures['completed'] and reference['completed']
    references={c['id']:c for c in reference['cases']}
    codes={}
    for path in ('kernel_0.json','kernel_1.json'):
        data=json.loads((args.private_dir/path).read_text())
        codes[data['summary']['name']]=data['instructions']
    result={'completed':False,'scope':'Private derived group-source offsets. No GitHub upload.',
            'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':[]}
    for case in captures['cases']:
        named=case['arguments'];ref=references[case['id']]
        waves=len(ref['waves']);bm,bn=(192,128) if waves==4 else (384,256)
        count=18 if waves==4 else 9
        groups=[];origins=set()
        for y in range(named['NumWorkGroups1']):
            for x in range(named['NumWorkGroups0']):
                shifts={}
                for w in range(waves):
                    run=PrologueMachine(codes[case['kernel']],bytes.fromhex(case['argument_bytes_hex']),
                                        named,wave=w,group=(x,y,0)).run()
                    reads=run['reads'][:count];baseline=ref['waves'][w]['reads'][:count]
                    assert len(reads)==len(baseline)==count
                    for actual,original in zip(reads,baseline):
                        for key in ('logical_operand','load_bytes','lds_base_bytes'):
                            assert actual[key]==original[key],(case['id'],x,y,w,key)
                        offsets=actual['tensor_offsets_bytes'];prior=original['tensor_offsets_bytes']
                        assert all(a is not None and b is not None for a,b in zip(offsets,prior))
                        delta={a-b for a,b in zip(offsets,prior)}
                        assert len(delta)==1,(case['id'],x,y,w,'nonuniform source shift')
                        operand=actual['logical_operand'];value=delta.pop()
                        if operand in shifts:assert shifts[operand]==value
                        shifts[operand]=value
                assert shifts['A']%(2*named['strideB0'])==0 and shifts['B']%2==0
                row=shifts['A']//(2*named['strideB0']);col=shifts['B']//2
                assert row%bm==0 and col%bn==0
                assert 0<=row<named['SizesFree1'] and 0<=col<named['SizesFree0']
                assert (row,col) not in origins
                origins.add((row,col));groups.append({'x':x,'y':y,'a_shift_bytes':shifts['A'],
                                                      'b_shift_bytes':shifts['B'],'row':row,'column':col})
        expected={(r,c) for r in range(0,named['SizesFree1'],bm) for c in range(0,named['SizesFree0'],bn)}
        assert origins==expected
        result['cases'].append({'id':case['id'],'groups':groups,'all_wave_reads_verified':True,
                                'm':named['SizesFree1'],'n':named['SizesFree0'],'k':named['SizesSum0']})
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(case['id'],len(groups),'groups verified',flush=True)
    result['completed']=True;args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
