"""Fixed-shape alignment evidence and exact storage-address geometry."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from superpod_compute.memory_access import RowMajorStorage,describe_read
from summarize_pmc import summarize


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Use a new evidence report')
    root=ROOT/'results/boundary_20261102';data=root/'server_evidence'
    source=data/'base_alignment_measurements.json';profiles=data/'base_alignment_kernels.json'
    raw=json.loads(source.read_text());metadata=json.loads(profiles.read_text())
    if not raw['completed'] or not metadata['completed'] or len(raw['cases'])!=14:raise ValueError('Incomplete GEMM measurements')
    kernels={r['id']:[n for n in r['kernels'] if '_MT' in n] for r in metadata['cases']}
    references={tuple(r['shape']):r for r in raw['cases'] if r['base_offsets_bytes']==[0,0,0]}
    counters=('TA_TA_BUSY','TCC_EA_RDREQ','TCC_EA_RDREQ_32B','TCC_EA1_RDREQ','TCC_EA1_RDREQ_32B','TCP_TCP_TA_DATA_STALL_CYCLES')
    counter_data={}
    for path in (root/'pmc_read').glob('base_alignment_*/report.csv'):
        row=next(r for r in raw['cases'] if r['id']==path.parent.name)
        if json.loads((path.parent/'status.json').read_text())['returncode']:raise ValueError('Failed counter collection')
        counter_data[row['id']]=summarize(path,row,counters)
    if len(counter_data)!=5:raise ValueError('Expected five completed counter conditions')
    report={'scope':'Graph times and counters collected separately. Address histograms are exact; no inferred cache hit, transaction count, or fitted latency correction.',
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,profiles,ROOT/'superpod_compute/memory_access.py',Path(__file__))},
            'cases':[],'counter_cases':list(counter_data.values()),'counter_pairs':[]}
    def time(row):return statistics.mean(row[f'graph_round{i}']['p50_us'] for i in (1,2))
    for r in sorted(raw['cases'],key=lambda r:(r['shape'],r['base_offsets_bytes'])):
        if not r['correctness']['passed']:raise ValueError('GEMM correctness failed')
        ref=references[tuple(r['shape'])];m,n,k=r['shape'];storages={}
        for i,(name,rows,cols) in enumerate((('A',m,k),('B',k,n),('C',m,n))):
            storage=RowMajorStorage(rows,cols,r['strides'][i][0],base_mod_256_bytes=r['pointer_modulo_bytes']['256'][i])
            storages[name]={'storage':asdict(storage),'row_start_residues':{str(q):storage.row_start_residues(q) for q in (4,16,64)}}
        entry={'id':r['id'],'shape':r['shape'],'offsets_bytes':r['base_offsets_bytes'],'graph_us':time(r),
               'reference_us':time(ref),'time_ratio':time(r)/time(ref),'same_kernel':kernels[r['id']]==kernels[ref['id']],
               'same_strides':r['strides']==ref['strides'],'storage_geometry':storages}
        report['cases'].append(entry)
        print(r['shape'],r['base_offsets_bytes'],round(time(r),3),'ratio',round(entry['time_ratio'],4),'same kernel',entry['same_kernel'])
        if r['id']!=ref['id'] and r['id'] in counter_data:
            a,b=counter_data[ref['id']],counter_data[r['id']]
            pair={'reference':ref['id'],'variant':r['id'],'same_geometry':a['geometry']==b['geometry'],
                  'counter_ratios':{key:b['counters'][key]['median']/v['median'] if v['median'] else None for key,v in a['counters'].items()}}
            report['counter_pairs'].append(pair)
            print(' counter ratios',json.dumps(pair['counter_ratios']))
    report['primitive_cases']=[]
    for route,name in (('vgpr','global_alignment_measurements.json'),('lds','global_to_lds_alignment_measurements.json')):
        primitive_path=data/name;primitive=json.loads(primitive_path.read_text())
        if not primitive['completed'] or not all(c['correct'] and not c['local_bytes'] for r in primitive['rounds'] for c in r['cases']):raise ValueError('Invalid primitive measurement')
        report['source_sha256'][str(primitive_path.relative_to(ROOT))]=hashlib.sha256(primitive_path.read_bytes()).hexdigest()
        primitive_groups={}
        for rnd in primitive['rounds']:
            for row in rnd['slopes']:
                key=(row['blocks'],row['load_bytes'],row['stride_bytes'],row['base_bytes']);primitive_groups.setdefault(key,[]).append(row['ns_per_iteration'])
        for (blocks,width,stride,offset),times in sorted(primitive_groups.items()):
            record={'route':route,'blocks':blocks,'load_bytes':width,'stride_bytes':stride,'base_bytes':offset,'ns_per_iteration':statistics.mean(times),
                    'ratio_to_same_width_aligned_stride':statistics.mean(times)/statistics.mean(primitive_groups[(blocks,width,2048,0)]),
                    'first_wave_read_geometry':[describe_read([offset+lane*stride+j*16 for lane in range(64)],width) for j in range(4)]}
            report['primitive_cases'].append(record)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
