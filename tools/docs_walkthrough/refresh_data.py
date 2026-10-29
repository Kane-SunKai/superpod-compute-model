"""Rebuild documentation numbers from current code; no hand-entered timings."""
import json,sys
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from superpod_compute import Hardware,Policy,GemmRequest,Mapping,evaluate_mapping
here=Path(__file__).parent
h=Hardware.load(ROOT/'superpod_compute/data/bw1100.json')
p=Policy(**json.loads((ROOT/'examples/bw1100_graph_calibrated.json').read_text()))
mp=Mapping(256,384,16,12,6,1,2,4096,2,2)
q=GemmRequest(3072,2304,1024)
main=evaluate_mapping(q,mp,h,policy=p,trace_limit=100)
reuse=evaluate_mapping(replace(q,N=4608),mp,h,policy=p,trace_limit=100)
no=evaluate_mapping(replace(q,N=4608),mp,h,policy=replace(p,input_reuse=False,initial_cache_state='cold'),trace_limit=100)
hm=Hardware.load(ROOT/'superpod_compute/data/bw1100_mac256.json')
pm=Policy(**json.loads((ROOT/'examples/bw1100_mac256_comparison.json').read_text()))
mac=evaluate_mapping(q,mp,hm,policy=pm,trace_limit=100)
data={'main':main,'reuse':reuse,'no_reuse':no,'mac256':mac,'hardware':h.to_dict(),'policy':p.to_dict()}
report=ROOT/'results/validation_20261029_v32/comparison_modes.json'
if report.exists():
    r=json.loads(report.read_text())
    if r.get('completed'):data['comparison']=r['summaries']
data['write_evidence']=json.loads((ROOT/'tools/bw1100_service_probe/lds_write_summary.json').read_text())
data['chunk_sweep']=[]
for size in (4096,16384,32768):
    rr=evaluate_mapping(q,replace(mp,output_chunk_bytes=size),h,policy=p)
    data['chunk_sweep'].append({'chunk_bytes':size,'duration_us':rr['duration_us'],'output_us':rr['wave_classes'][0]['output_stream']['duration_s']*1e6})
(here/'example_data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'duration_us':main['duration_us'],'pipeline_us':main['wave_classes'][0]['pipeline']['duration_s']*1e6,
 'output_us':main['wave_classes'][0]['output_stream']['duration_s']*1e6,'reuse_us':reuse['duration_us'],'no_reuse_us':no['duration_us'],'mac256_us':mac['duration_us']}))
