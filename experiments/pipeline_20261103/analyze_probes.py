import json,pathlib,statistics,collections
r=pathlib.Path(__file__).resolve().parent
def lines(name):return [json.loads(s) for s in (r/name).read_text().splitlines() if s.startswith('{')]
summary={}
mm=[]
for rd in (1,2):
 rows=lines(f'mmac_probe_round{rd}.log');assert len(rows)==54 and all(x['correct'] for x in rows)
 for w in (1,4,8):
  for c in (1,4,8):
   a,b=[next(x for x in rows if (x['blocks'],x['waves_per_block'],x['chains'],x['iterations'])==(72,w,c,i)) for i in (4096,16384)]
   denom=(16384-4096)*c*max(1,w/4)
   mm.append(dict(round=rd,waves=w,chains=c,event_slope_cycles=(b['median_us']-a['median_us'])*1300/denom,
                  clock_slope_ticks=(b['median_block_clock64_ticks']-a['median_block_clock64_ticks'])/denom))
summary['mmac_slopes']=mm
print('MMAC slopes',json.dumps(mm,indent=1))
if (r/'mixed_probe.log').exists():
 mix=lines('mixed_probe.log');assert len(mix)==96 and all(x['correct'] for x in mix)
 slopes=[]
 for rd in (1,2):
  for w in (4,8):
   for c in (1,4,8):
    row=dict(round=rd,waves=w,chains=c)
    for mode in range(4):
     a,b=[next(x for x in mix if (x['round'],x['waves'],x['chains'],x['mode'],x['iterations'])==(rd,w,c,mode,i)) for i in (4096,16384)]
     row[str(mode)]=(b['median_us']-a['median_us'])/(16384-4096)
    row['independent_over_max']=row['2']/max(row['0'],row['1'])
    row['dependent_over_max']=row['3']/max(row['0'],row['1'])
    slopes.append(row)
 summary['mixed_slopes_us_per_iteration']=slopes
 print('MIXED',json.dumps(slopes,indent=1))
if (r/'memory_probe.log').exists():
 rows=lines('memory_probe.log');assert len(rows)==84 and all(x['correct'] for x in rows)
 summary['memory']=rows
 for n in (6144,12288,24576):
  rows2=[x for x in rows if x['round']==2 and x['N']==n]
  print('MEMORY N',n,[(x['column_half_elements'],x['kparts_or_blocks'],x['copy'],round(x['median_us'],2),round(x['read_bytes']/x['median_us']/1e6,3)) for x in rows2])
if (r/'execution_metadata.json').exists():
 meta=json.loads((r/'execution_metadata.json').read_text());summary['execution']=meta
 for x in meta:
  print('KERNEL',x['shape'])
  for k in x['kernels']:print(k['name'],{a:b for a,b in k['args'].items() if a in ('grid','block','registers per thread','shared memory')})
if (r/'cache_diagnostic.json').exists():
 cache=json.loads((r/'cache_diagnostic.json').read_text());summary['cache']=cache
 for x in cache:print('CACHE',x['shape'],[(a['hot']['p50_us'],a['evict']['p50_us']) for a in x['rounds']])
(r/'probe_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
