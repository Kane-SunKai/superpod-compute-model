import hashlib,json,pathlib
r=pathlib.Path(__file__).resolve().parent;p=json.loads((r/'frozen_plan.json').read_text())
ids=['cal_decode_1x6144x6144','cal_decode_8x12288x6144','hold_decode_4x24576x8192',
     'cal_prefill_384x6144x6144','cal_prefill_768x6144x6144','14_prefill_1024x8192x8192',
     '23_square_256x256x256','26_square_768x768x768','cal_square_1280x1280x1280',
     '36_long_k_512x768x4096','hold_long_k_384x1024x2048','43_aspect_1536x1152x1024']
lookup={x['id']:x for x in p['development_cases']}
out=r/'controlled_plan.json';assert not out.exists()
obj=dict(cases=[lookup[x] for x in ids],tiles=[[16,128,32],[64,128,32],[128,128,32],[128,256,32],[128,128,64]],
         protocol=dict(warmup_calls=10,graph_calls=16,graph_warmup_replays=3,event_samples_per_round=10,rounds=2,
         num_warps=4,num_stages=1,dtype='fp16',measurement='GPU event mean of two medians; same A/B/C per case',
         scope='Triton controlled output tiles/K steps, not a replication of vendor GEMM kernels; development only'),
         no_fitting='These results do not adjust the frozen validation predictions or retrieval rules.')
out.write_text(json.dumps(obj,indent=2)+'\n');print(hashlib.sha256(out.read_bytes()).hexdigest())
