"""Reproducible local verification, reports only; no GPU or network access."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from superpod_compute import GemmRequest, Mapping, predict, evaluate_mapping

def main():
    log=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'unittest.log').write_text(log.getvalue(),encoding='utf-8')
    if not tests.wasSuccessful():
        print(log.getvalue());return 1
    cases=[('baseline',GemmRequest(3072,2304,1024)),('tail',GemmRequest(130,97,270)),('decode_shape',GemmRequest(1,8192,8192))]
    summaries=[];cache=ROOT/'.cache/validation.sqlite3'
    for name,request in cases:
        started=time.perf_counter();r=predict(request,cache_path=cache,force=True);cold=time.perf_counter()-started
        started=time.perf_counter();hit=predict(request,cache_path=cache);warm=time.perf_counter()-started
        if not hit['cache_hit'] or r['best']!=hit['best']:raise RuntimeError('Cache verification failed')
        (out/(name+'.json')).write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
        b=r['best'];summaries.append({'case':name,'shape':[request.M,request.N,request.K],'mapping':b['mapping'],'model_duration_us':b['duration_us'],'search_wall_seconds':cold,'cached_wall_seconds':warm,'searched':r['searched_candidates'],'feasible':r['feasible_candidates'],'source_counts':b['array_sources_wave_panels']})
    peak=predict(cases[0][1],backend='peak');fixed=evaluate_mapping(cases[0][1],Mapping(128,128,128,12,9,2,2))
    report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'tests':tests.testsRun,'tests_passed':True,'reference_lut_entries_checked':1049,'pipeline_random_comparisons':60*2*2*2,'cases':summaries,'baseline_peak_us':peak['best']['duration_us'],'baseline_fixed_teaching_mapping_us':fixed['duration_us'],'scope':'model verification only; output conversion/layout remains zero and uncalibrated by default'}
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# 首版验证记录','',f"执行时间（UTC）：{report['checked_at_utc']}",'',f"- {tests.testsRun}项单元/集成测试全部通过。",'- 1049个不同来源LUT键逐条验证OS周期公式。','- 480组随机流水线配置与独立DAG调度对照，含四种缓冲组合和共享端口。','- 完成非整除尺寸、低并行度、数据覆盖、容量拒绝、长K快进与缓存失效检查。','','| 算例 M×N×K | 最佳 bm×bn×bk | gm×gn / L,S | 模型µs | 搜索墙钟s | 查表墙钟s | 候选/可行 |','|---|---|---|---:|---:|---:|---:|']
    for r in summaries:
        m=r['mapping'];lines.append(f"| {'×'.join(map(str,r['shape']))} | {m['bm']}×{m['bn']}×{m['bk']} | {m['gm']}×{m['gn']} / {m['l2_buffers']},{m['shared_buffers']} | {r['model_duration_us']:.6f} | {r['search_wall_seconds']:.4f} | {r['cached_wall_seconds']:.4f} | {r['searched']}/{r['feasible']} |")
    lines+=['',f"教学固定分块128×128×128、12×9 Core、双/双缓冲：{fixed['duration_us']:.6f} µs。",f"峰值后端自动搜索：{peak['best']['duration_us']:.6f} µs。",'', '这些数值包含4个输出波（基准算例）和21 µs启动，不能与旧HTML中单波、L2预就绪的12.60 µs主循环直接比较。','', '搜索墙钟与模型µs是不同指标；墙钟计时在同一Python进程内，不含解释器/PowerShell启动。','', '**限制：** 这是软件逻辑和参考阵列数据验证，不是真机A100准确率验证。输出转换/布局整理默认0、跨波冷读及资源简化均保留在每个JSON的scope_notes中。','', '复现：从仓库根目录运行 `python tools/validate.py`。详细测试日志与JSON在results/，Git仅提交本摘要。']
    (ROOT/'docs/validation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
