"""Interpret address-only prologues from explicitly supplied instruction text.

This is an independent, bounded integer interpreter, NOT a GPU timing model.
Unsupported instructions fail closed. It stops at the first matrix operation;
global read addresses are recorded without reading any tensor data. Input text
and generated private access patterns must remain outside Git/GitHub.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from superpod_compute.memory_access import describe_read

MASK32=(1<<32)-1
MASK64=(1<<64)-1

def signed(value):
    value&=MASK32
    return value-(1<<32) if value&(1<<31) else value

def regspec(token):
    m=re.fullmatch(r'([sv])(?:\[(\d+):(\d+)\]|(\d+))',token)
    if not m:return None
    lo=int(m[2] or m[4]);hi=int(m[3] or m[4])
    return m[1],lo,hi

class PrologueMachine:
    def __init__(self,instructions,argument_bytes,arguments,*,wave=0,group=(0,0,0)):
        self.code=instructions;self.locations={x['address']:i for i,x in enumerate(instructions)}
        self.args=argument_bytes;self.named=arguments
        self.s={0:0,1:0,2:group[0],3:group[1],4:group[2]}
        self.v={0:[wave*64+i for i in range(64)]}
        self.special={'m0':0,'vcc':0,'exec':MASK64,'scc':0}
        self.wave=wave;self.loads=[];self.events=[];self.pc=0;self.steps=0

    def get(self,token):
        token=token.strip()
        if token in self.special:return self.special[token]
        spec=regspec(token)
        if not spec:
            # ISA inline floating constants denote their FP32 bit pattern,
            # even when an integer/bitwise instruction consumes the operand.
            if re.fullmatch(r'-?\d+\.\d+',token):return struct.unpack('<I',struct.pack('<f',float(token)))[0]
            return int(token,0)
        kind,lo,hi=spec;registers=self.s if kind=='s' else self.v
        if any(i not in registers for i in range(lo,hi+1)):raise ValueError('Uninitialized '+token)
        if kind=='s':return sum(registers[i]<<(32*(i-lo)) for i in range(lo,hi+1))
        return [sum(registers[i][lane]<<(32*(i-lo)) for i in range(lo,hi+1)) for lane in range(64)]

    def put(self,token,value):
        if token in self.special:
            self.special[token]=int(value)&(MASK64 if token in ('vcc','exec') else MASK32);return
        spec=regspec(token)
        if not spec:raise ValueError('Invalid destination '+token)
        kind,lo,hi=spec
        if kind=='s':
            for i in range(lo,hi+1):self.s[i]=(value>>(32*(i-lo)))&MASK32
        else:
            if not isinstance(value,list):value=[value]*64
            for i in range(lo,hi+1):self.v[i]=[(x>>(32*(i-lo)))&MASK32 for x in value]

    def lanes(self,token):
        value=self.get(token)
        return value if isinstance(value,list) else [value]*64

    def compare(self,relation,kind,a,b):
        if kind=='i32':a,b=signed(a),signed(b)
        elif kind=='f32':a,b=(struct.unpack('<f',struct.pack('<I',x&MASK32))[0] for x in (a,b))
        else:a,b=a&MASK32,b&MASK32
        return {'eq':a==b,'lt':a<b,'le':a<=b,'gt':a>b,'ge':a>=b,'ne':a!=b}[relation]

    def record_load(self,op,rest,instruction):
        match=re.fullmatch(r'(v(?:\[\d+:\d+\]|\d+)), (s\[\d+:\d+\]), (\d+)(.*)',rest)
        if not match:raise ValueError('Unsupported buffer read syntax: '+rest)
        vector,resource,scalar,flags=match.groups()
        if 'lds' not in flags:raise ValueError('Prologue tracer currently requires direct-to-LDS reads')
        descriptor=self.get(resource);base=descriptor&((1<<48)-1);stride=(descriptor>>48)&0x3fff
        records=(descriptor>>64)&MASK32
        widths={'buffer_load_dword':4,'buffer_load_dwordx2':8,'buffer_load_dwordx4':16}
        width=widths[op];index_enabled='idxen' in flags;offset_enabled='offen' in flags
        values=self.lanes(vector);addresses=[]
        # A resource base may be biased slightly before the tensor while its
        # lane offsets compensate. Bounds apply to final lane addresses.
        source=next((name for name in ('A','B') if self.named[name]-65536<=base<self.named[name]+2*self.named['size'+name]),None)
        if source is None:raise ValueError(f'Unknown input resource base {base}')
        lo=self.named[source];hi=lo+2*self.named['size'+source]
        for lane,value in enumerate(values):
            index=(value&MASK32) if index_enabled else 0
            offset=((value>>32)&MASK32) if index_enabled and offset_enabled else ((value&MASK32) if offset_enabled else 0)
            address=base+index*stride+offset+int(scalar)
            addresses.append(address if lo<=address and address+width<=hi and self.special['exec']>>lane&1 else None)
        self.loads.append({'instruction_address':instruction['address'],'wave':self.wave,'abi_operand':source,
                           'logical_operand':'B' if source=='A' else 'A','load_bytes':width,'route':'lds',
                           'resource_stride_bytes':stride,'resource_records':records,'lds_base_bytes':self.get('m0'),
                           'addresses_bytes':addresses,'tensor_offsets_bytes':[a-lo if a is not None else None for a in addresses],
                           'geometry':describe_read(addresses,width)})

    def step(self):
        ins=self.code[self.pc];text=ins['text'];op,_,rest=text.partition(' ')
        op=re.sub(r'_e(?:32|64)$','',op)
        if op.startswith('v_mmac'):return False
        args=[x.strip() for x in rest.split(',')];next_pc=self.pc+1
        if op.startswith('s_load_dword'):
            count=int(op.split('x')[1]) if 'x' in op else 1
            offset=self.get(args[1])+self.get(args[2])
            data=self.args[offset:offset+count*4]
            if len(data)!=count*4:raise ValueError('Argument load outside supplied ABI bytes')
            self.put(args[0],int.from_bytes(data,'little'))
        elif op in ('s_waitcnt','s_nop','s_barrier'):
            if op=='s_barrier':self.events.append({'kind':'barrier','after_loads':len(self.loads)})
        elif op.startswith('buffer_load_'):self.record_load(op,rest,ins)
        elif op.startswith('ds_read_'):
            # Data values only feed the following MMAC. The address prologue
            # must not consume these unknown values for further addressing.
            spec=regspec(args[0]);kind,lo,hi=spec
            for i in range(lo,hi+1):self.v.pop(i,None)
        elif op=='s_getpc_b64':self.put(args[0],ins['address']+ins['size'])
        elif op=='s_setpc_b64':next_pc=self.locations[self.get(args[0])]
        elif op=='s_branch' or op.startswith('s_cbranch_'):
            if op not in ('s_branch','s_cbranch_scc1','s_cbranch_scc0','s_cbranch_vccz','s_cbranch_vccnz'):
                raise ValueError('Unsupported branch '+text)
            take=op=='s_branch'
            if op=='s_cbranch_scc1':take=bool(self.special['scc'])
            if op=='s_cbranch_scc0':take=not self.special['scc']
            if op=='s_cbranch_vccz':take=not self.special['vcc']
            if op=='s_cbranch_vccnz':take=bool(self.special['vcc'])
            if take:
                target=re.search(r'@0x([0-9a-f]+)',rest)
                if not target:raise ValueError('Branch requires decoded target')
                next_pc=self.locations[int(target[1],16)]
        elif op.startswith('s_cmp_'):
            _,_,relation,kind=op.split('_');self.special['scc']=int(self.compare(relation,kind,self.get(args[0]),self.get(args[1])))
        elif op.startswith('v_cmp_'):
            _,_,relation,kind=op.split('_')
            mask=sum(int(self.compare(relation,kind,a,b))<<lane for lane,(a,b) in enumerate(zip(self.lanes(args[1]),self.lanes(args[2]))))
            self.put(args[0],mask)
        elif op=='v_readfirstlane_b32':self.put(args[0],self.lanes(args[1])[next(i for i in range(64) if self.special['exec']>>i&1)])
        elif op=='v_cndmask_b32':
            mask=self.get(args[3]) if len(args)>3 else self.get('vcc')
            self.put(args[0],[b if mask>>i&1 else a for i,(a,b) in enumerate(zip(self.lanes(args[1]),self.lanes(args[2])))])
        elif op.startswith(('s_mov_','v_mov_')):self.put(args[0],self.get(args[1]))
        elif op=='s_cmov_b32':
            if self.special['scc']:self.put(args[0],self.get(args[1]))
        elif op=='s_cselect_b32':self.put(args[0],self.get(args[1] if self.special['scc'] else args[2]))
        elif op.startswith('s_'):
            a,b=self.get(args[1]),self.get(args[2]);carry=self.special['scc'];bits=64 if op.endswith('b64') else 32
            if op in ('s_add_u32','s_addc_u32'):
                value=a+b+(carry if op=='s_addc_u32' else 0);self.special['scc']=int(value>MASK32)
            elif op in ('s_sub_u32','s_subb_u32'):
                value=a-b-(1-carry if op=='s_subb_u32' else 0);self.special['scc']=int(value>=0)
            elif op in ('s_mul_i32','s_mul_hi_u32'):value=a*b if op=='s_mul_i32' else (a*b)>>32
            elif op.startswith('s_lshl_'):value=a<<(b&(bits-1))
            elif op.startswith('s_lshr_'):value=a>>(b&(bits-1))
            elif op.startswith('s_and_'):value=a&b
            elif op.startswith('s_or_'):value=a|b
            else:raise ValueError('Unsupported scalar instruction '+text)
            self.put(args[0],value&((1<<bits)-1))
        elif op.startswith('v_'):
            carry_op=op in ('v_add_co_u32','v_sub_co_u32')
            first=2 if carry_op else 1;a,b=self.lanes(args[first]),self.lanes(args[first+1]);values=[];mask=0
            for lane,(x,y) in enumerate(zip(a,b)):
                if op in ('v_add_co_u32','v_add_u32'):value=x+y
                elif op in ('v_sub_co_u32','v_sub_i32','v_sub_u32'):value=x-y
                elif op=='v_add_lshl_u32':value=(x+y)<<(self.get(args[3])&31)
                elif op=='v_lshlrev_b32':value=y<<(x&31)
                elif op=='v_lshrrev_b32':value=y>>(x&31)
                elif op=='v_and_b32':value=x&y
                elif op=='v_mul_lo_u32':value=x*y
                elif op=='v_mul_u32_u24':value=(x&0xffffff)*(y&0xffffff)
                else:raise ValueError('Unsupported vector instruction '+text)
                if carry_op:mask|=int(value>MASK32 if op=='v_add_co_u32' else value<0)<<lane
                values.append(value&MASK32)
            self.put(args[0],values)
            if carry_op:self.put(args[1],mask)
        else:raise ValueError('Unsupported instruction '+text)
        self.pc=next_pc;self.steps+=1
        return True

    def run(self):
        for _ in range(10000):
            try:
                if not self.step():return {'wave':self.wave,'steps':self.steps,'reads':self.loads,'events':self.events}
            except Exception as exc:raise ValueError(f'At instruction {self.pc}: {self.code[self.pc]["text"]}: {exc}') from exc
        raise RuntimeError('Address prologue exceeded instruction budget')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--private-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior private evidence')
    capture=json.loads((args.private_dir/'sanitized_arguments.json').read_text())
    if not capture['completed']:raise ValueError('Incomplete argument capture')
    code={}
    for name in ('kernel_0.json','kernel_1.json'):
        data=json.loads((args.private_dir/name).read_text());code[data['summary']['name']]=data['instructions']
    report={'completed':False,'cases':[],'scope':'Private address-only reconstruction before first MMAC; no timing model or tensor values. No GitHub upload.',
            'interpreter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for row in capture['cases']:
        waves=row['block'][0]//64;output={'id':row['id'],'arguments':row['arguments'],'waves':[]}
        for wave in range(waves):
            output['waves'].append(PrologueMachine(code[row['kernel']],bytes.fromhex(row['argument_bytes_hex']),row['arguments'],wave=wave).run())
        report['cases'].append(output)
        print(row['id'],'waves',waves,'loads per wave',[len(w['reads']) for w in output['waves']],flush=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    report['completed']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
