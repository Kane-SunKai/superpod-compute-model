"""Bounded private control trace through the compute loop; NOT cycle simulation.

Tensor values remain unknown. Their use in an address/branch fails closed.
Stops at the first output FP32->FP16 conversion; requires exact matrix-work
conservation for the supplied full-K execution before accepting the result.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.diagnostics.trace_address_prologue import PrologueMachine,regspec


class ControlMachine(PrologueMachine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.counts=Counter();self.timeline=[];self.matrix_count=0

    def poison(self,token):
        kind,lo,hi=regspec(token)
        if kind!='v':raise ValueError('Expected tensor-data vector destination')
        for index in range(lo,hi+1):self.v.pop(index,None)

    def step(self):
        ins=self.code[self.pc];text=ins['text'];op,_,rest=text.partition(' ')
        op=re.sub(r'_e(?:32|64)$','',op)
        if op=='v_cvt_f16_f32' and self.matrix_count:return False
        operands=[a.strip() for a in rest.split(',')]
        self.counts[op]+=1
        if op in ('s_barrier','s_waitcnt') or op.startswith(('buffer_load_','ds_read_')):
            self.timeline.append({'kind':op,'matrix_instructions_before':self.matrix_count,
                                  'global_reads_before':len(self.loads),
                                  'wait':rest if op=='s_waitcnt' else None})
        if op.startswith('v_mmac'):
            self.poison(operands[0]);self.matrix_count+=1
        elif op=='s_setprio':pass
        elif op.startswith('s_cmpk_'):
            relation,kind=op.removeprefix('s_cmpk_').split('_')
            self.special['scc']=int(self.compare(relation,kind,self.get(operands[0]),self.get(operands[1])))
        elif op in ('v_mul_f32','v_fma_mix_f32'):
            self.poison(operands[0])
        elif op=='v_cvt_f32_f16':
            source=self.lanes(operands[1])
            self.put(operands[0],[struct.unpack('<I',struct.pack('<f',struct.unpack('<e',struct.pack('<H',value&65535))[0]))[0] for value in source])
        else:
            result=super().step()
            # Logical scalar ops and shifts set SCC according to the masked
            # result. Their result cannot depend on unknown tensor data here.
            if op.startswith(('s_and_','s_or_','s_lshl_','s_lshr_')):
                self.special['scc']=int(self.get(operands[0])!=0)
            return result
        self.pc+=1;self.steps+=1;return True

    def run_control(self):
        for _ in range(100000):
            try:
                if not self.step():return {'matrix_instructions':self.matrix_count,'steps':self.steps,
                                           'opcode_counts':dict(self.counts),'events':self.timeline}
            except Exception as exc:
                raise ValueError(f'At instruction {self.pc}: {self.code[self.pc]["text"]}: {exc}') from exc
        raise RuntimeError('Control trace exceeded bounded instruction budget')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--private-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior evidence')
    captures=json.loads((args.private_dir/'sanitized_arguments.json').read_text());assert captures['completed']
    code={}
    for name in ('kernel_0.json','kernel_1.json'):
        data=json.loads((args.private_dir/name).read_text());code[data['summary']['name']]=data['instructions']
    report={'completed':False,'cases':[],'scope':'Private bounded scalar control-flow reconstruction for group0 wave0. No tensor values or instruction timing; stops at first output conversion. No GitHub upload.',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for row in captures['cases']:
        machine=ControlMachine(code[row['kernel']],bytes.fromhex(row['argument_bytes_hex']),row['arguments'])
        trace=machine.run_control();per_panel=24 if row['block'][0]==256 else 48
        assert row['arguments']['SizesSum0']%16==0
        expected=per_panel*(row['arguments']['SizesSum0']//16)
        assert trace['matrix_instructions']==expected,(row['id'],trace['matrix_instructions'],expected)
        barriers=[e['matrix_instructions_before']/per_panel for e in trace['events'] if e['kind']=='s_barrier']
        report['cases'].append({'id':row['id'],'matrix_instructions_per_panel':per_panel,'trace':trace,
                                'barriers_after_panel_equivalents':barriers})
        print(row['id'],'MMAC',expected,'barriers',len(barriers),'first positions',barriers[:20],flush=True)
    report['completed']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
