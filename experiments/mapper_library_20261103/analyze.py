import collections,dataclasses,hashlib,json,pathlib,statistics,time
from mapper import Hardware,Policy,search,predict
r=pathlib.Path(__file__).resolve().parent
load=lambda n:json.loads((r/n).read_text())
p=load('frozen_plan.json');pred=load('predictions.json');meas=load('validation_measure.json');seal=load('prediction_seal.json')
assert pred['completed'] and meas['completed']
assert hashlib.sha256((r/'predictions.json').read_bytes()).hexdigest()==seal['predictions_sha256']
assert pred['plan_sha256']==meas['plan_sha256']==seal['plan_sha256']
assert seal['finished_unix'] < load('validation_start.json')['unix'], 'Predictions must be frozen before validation starts'
byid={x['id']:x for x in meas['cases']};post={x['id']:x for x in load('posthoc_metadata.json')}
assert len(byid)==len(pred['cases'])==24
rows=[];h=Hardware(**p['hardware']);po=dataclasses.replace(Policy(**p['policy']),matrix_efficiency=p['frozen_cost']['matrix_efficiency'],calibration_hardware_sha256='')
for x in pred['cases']:
    y=byid[x['id']];assert y['correctness']['passed'];assert y['shape']==x['shape']
    methods={}
    for key,v in x['methods'].items():
        best=v['best'];err=best['duration_us']/y['measured_graph_us']-1
        methods[key]=dict(predicted_us=best['duration_us'],signed_error=err,absolute_error=abs(err),pass10=abs(err)<=.1,
            mapping=best['mapping'],wave_count=best['wave_count'],hbm_read_bytes=best['hbm_read_bytes'],
            compute_flops=best['executed_tile_flops'],search_seconds=v['search_seconds'])
    meta=post[x['id']];diag=None
    if meta['supported']:
        v=search(x['shape'],h,po,p['frozen_cost']['input_hbm_bandwidth_Bps'],[meta['tile']])
        if v['best']:diag=dict(predicted_us=v['best']['duration_us'],absolute_error=abs(v['best']['duration_us']/y['measured_graph_us']-1))
    tile=lambda method:[methods[method]['mapping'][f] for f in ('bm','bn','bk')]
    rows.append(dict(id=x['id'],group=x['group'],shape=x['shape'],measured_us=y['measured_graph_us'],rounds=y['rounds'],
        round_drift_pct=y['round_drift_pct'],methods=methods,posthoc=meta,observed_tile_diagnostic=diag,
        nearest_source_id=x['entries'][0]['source_id'],nearest_source_shape=x['entries'][0]['source_shape'],
        nearest_tile=tile('nearest'),automatic_tile=tile('automatic'),library_min_tile=tile('library_min'),
        nearest_hits_actual=meta['supported'] and tile('nearest')==meta['tile'],
        top5_contains_actual=meta['supported'] and meta['tile'] in [v['tile'] for v in x['entries']]))
summary={}
for k in rows[0]['methods']:
    es=[x['methods'][k]['absolute_error'] for x in rows]
    summary[k]=dict(count=24,pass10=sum(x<=.1 for x in es),pass15=sum(x<=.15 for x in es),mape=statistics.mean(es),median=statistics.median(es),max=max(es),
        signed_mean=statistics.mean(x['methods'][k]['signed_error'] for x in rows))
groups={g:{key:dict(count=len(ss),pass10=sum(x['methods'][key]['pass10'] for x in ss),mape=statistics.mean(x['methods'][key]['absolute_error'] for x in ss)) for key in summary}
    for g in dict.fromkeys(x['group'] for x in rows) for ss in [[x for x in rows if x['group']==g]]}
cm=load('controlled_measure.json');cp=load('controlled_predictions.json');assert cm['completed'] and len(cm['cases'])==72
assert all(x['status']=='ok' and x['correctness']['passed'] for x in cm['cases'])
cd={(x['id'],tuple(x['tile'])):x for x in cp};control=[]
for case in load('controlled_plan.json')['cases']:
    samples=[x for x in cm['cases'] if x['id']==case['id'] and x['tile']]
    best=min(samples,key=lambda x:x['measured_us']);worst=max(samples,key=lambda x:x['measured_us'])
    chosen=min(samples,key=lambda x:cd[x['id'],tuple(x['tile'])]['best']['duration_us'])
    baseline=next(x for x in cm['cases'] if x['id']==case['id'] and x['tile'] is None)
    control.append(dict(id=case['id'],shape=case['shape'],fastest_tile=best['tile'],selected_tile=chosen['tile'],
        fastest_us=best['measured_us'],slowest_us=worst['measured_us'],selected_us=chosen['measured_us'],library_us=baseline['measured_us'],
        slowdown=worst['measured_us']/best['measured_us'],regret=chosen['measured_us']/best['measured_us']-1,
        top1=best['tile']==chosen['tile']))
data=dict(summary=summary,groups=groups,cases=rows,controlled=control,
    controlled_top1=sum(x['top1'] for x in control),controlled_regret=statistics.mean(x['regret'] for x in control),
    round_drift_max=max(x['round_drift_pct'] for x in rows),round_drift_median=statistics.median(x['round_drift_pct'] for x in rows),
    actual_supported=sum(x['posthoc']['supported'] for x in rows),nearest_hits=sum(x['nearest_hits_actual'] for x in rows),top5_hits=sum(x['top5_contains_actual'] for x in rows),
    identity=seal,checks=load('cache_checks.json'),audit=load('disjointness_audit.json'))
(r/'analysis.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({k:data[k] for k in ('summary','groups','controlled_top1','controlled_regret','round_drift_max','nearest_hits','top5_hits','actual_supported')},indent=2))
