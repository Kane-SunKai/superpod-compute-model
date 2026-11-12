"""Build 100组回归测试V3_20261112.html.

Columns: NT default (what Megatron actually gets), fastest found by the kernel
sweep (only for the shapes that did not converge), the model, and the error
against both references.
"""
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V2 = ROOT / 'experiments/regression_100_v2_20261109'
PROBE = ROOT.parent / 'bw_probe'
OUT = ROOT.parent / '100组回归测试V3_20261112.html'

GROUPS = [('decode_normal', 'Decode · 常规 N'), ('decode_largeN', 'Decode · 大 N'),
          ('prefill', 'Prefill'), ('gpt7b', 'GPT-7B'), ('gpt13b', 'GPT-13B')]

OP_SHAPE = {
    'Emb': 'ids (1,4096) → 取表 (50304,4096) int64 + 位置编码 + transpose',
    'layernorm': '(4096,1,4096) fp16 layer_norm',
    'atten_qkv': '(4096×4096) × (4096×12288)',
    'atten_core_qk': '32 × baddbmm (4096×128)×(128×4096)',
    'atten_core_softmax': '32 × softmax 4096×4096',
    'atten_core_contex': '32 × bmm (4096×4096)×(4096×128)',
    'atten_linear': '(4096×4096) × (4096×4096)',
    'mlp_linear_1': '(4096×4096) × (4096×16384)',
    'mlp_gelu': '(4096,1,16384) fp16 gelu',
    'mlp_linear_2': '(4096×16384) × (16384×4096)',
    'logit_time': '(4096×4096) × (4096×50304)',
    'param_time': '7.68 B float32 逐元素除',
}
ORDER = list(OP_SHAPE)


def cls(e):
    if e is None:
        return 'muted'
    a = abs(e)
    return 'ok' if a <= 10 else ('warn' if a <= 20 else 'bad')


def pct(v):
    return f'{v:+.1f}%' if v is not None else '—'


def main():
    nt = {c['id']: c for c in
          json.loads((PROBE / 'nt100_20261111.json').read_text(encoding='utf-8'))['cases']}
    model = {r['id']: r['predicted_us'] for r in
             json.loads((V2 / 'predictions_bw2.json').read_text(encoding='utf-8'))}
    plan = {p['id']: p for p in json.loads((V2 / 'plan.json').read_text(encoding='utf-8'))}
    sweep = {}
    sf = PROBE / 'sweep_v3_20261112.json'
    if sf.exists():
        for c in json.loads(sf.read_text(encoding='utf-8')).get('cases', []):
            if c.get('fastest_us'):
                sweep[c['id']] = c
    ops = json.loads((PROBE / 'aicb_ops_20261111.json').read_text(encoding='utf-8'))['cases']
    opmodel = {r['op']: r['us'] for r in
               json.loads((HERE / 'aicb_ops_model.json').read_text(encoding='utf-8'))}

    rows = []
    for rid, p in plan.items():
        if rid not in nt:
            continue
        sw = sweep.get(rid)
        best = sw['fastest_us'] if sw else None
        rows.append(dict(id=rid, cat=p['category'], shape=p['shape'],
                         dflt=nt[rid]['default_us'], best=best,
                         model=model[rid], kernel=nt[rid]['kernel'],
                         fast_kernel=(sw['fastest_kernel'] if sw else ''),
                         e_def=100 * (model[rid] / nt[rid]['default_us'] - 1),
                         e_best=(100 * (model[rid] / best - 1)) if best else None,
                         gain=(nt[rid]['default_us'] / best) if best else None))

    def agg(rs, key):
        v = [abs(r[key]) for r in rs if r[key] is not None]
        return sum(x <= 10 for x in v), len(v), (st.mean(v) if v else 0)

    a_def = agg(rows, 'e_def')
    a_best = agg(rows, 'e_best')
    swept = [r for r in rows if r['best']]

    def group_block(cat, label):
        rs = [r for r in rows if r['cat'] == cat]
        d = agg(rs, 'e_def')
        b = agg(rs, 'e_best')
        head = (f"<tr class='sect'><td colspan='9'>{label} · {len(rs)} 个 · 对默认 {d[0]}/{d[1]}"
                f"（MAPE {d[2]:.1f}%）· 对最快 {b[0]}/{b[1]}（MAPE {b[2]:.1f}%）</td></tr>")
        out = []
        for r in sorted(rs, key=lambda r: -abs(r['e_def'])):
            btxt = f"{r['best']:.1f}" if r['best'] else '<span class="muted">未扫描</span>'
            gtxt = f"{r['gain']:.2f}×" if r['gain'] else '—'
            out.append(
                f"<tr><td class='id'>{r['id']}</td>"
                f"<td class='num'>{'×'.join(map(str, r['shape']))}</td>"
                f"<td class='num'>{r['dflt']:.1f}</td>"
                f"<td class='num'>{btxt}</td>"
                f"<td class='num {'ok' if (r['gain'] or 1) > 1.5 else 'muted'}'>{gtxt}</td>"
                f"<td class='num'>{r['model']:.1f}</td>"
                f"<td class='num {cls(r['e_def'])}'>{pct(r['e_def'])}</td>"
                f"<td class='num {cls(r['e_best'])}'>{pct(r['e_best'])}</td>"
                f"<td class='tile muted'>{r['kernel']}</td></tr>")
        return head + ''.join(out)

    op_rows = []
    for name in ORDER:
        c = next(x for x in ops if x['op'] == name)
        mo = opmodel.get(name)
        e = 100 * (mo / c['us'] - 1) if mo else None
        op_rows.append(
            f"<tr><td class='id'>{name}</td><td class='sh'>{OP_SHAPE[name]}</td>"
            f"<td class='num'><b>{c['us']:.1f}</b></td>"
            f"<td class='num muted'>{c['gbps']:.0f}</td>"
            f"<td class='num'>{f'{mo:.1f}' if mo else '—'}</td>"
            f"<td class='num {cls(e)}'>{pct(e) if e is not None else '无模型'}</td>"
            f"<td class='tile muted'>{c['kernels'][0][:44] if c['kernels'] else ''}</td></tr>")

    sweep_rows = ''.join(
        f"<tr><td class='id'>{r['id']}</td><td class='num'>{'×'.join(map(str, r['shape']))}</td>"
        f"<td class='num'>{r['dflt']:.1f}</td><td class='tile muted'>{r['kernel']}</td>"
        f"<td class='num'>{r['best']:.1f}</td><td class='tile muted'>{r['fast_kernel']}</td>"
        f"<td class='num ok'>{r['gain']:.2f}×</td>"
        f"<td class='num {cls(r['e_best'])}'>{pct(r['e_best'])}</td></tr>"
        for r in sorted(swept, key=lambda r: -(r['gain'] or 0)))

    opmap = {c['op']: c['us'] for c in ops}
    per_layer = sum(opmap[o] for o in ('layernorm', 'atten_qkv', 'atten_core_qk',
                                       'atten_core_softmax', 'atten_core_contex',
                                       'atten_linear', 'layernorm', 'mlp_linear_1',
                                       'mlp_gelu', 'mlp_linear_2'))
    total_op = (opmap['Emb'] + 36 * per_layer + opmap['layernorm']
                + opmap['logit_time'] + opmap['param_time'])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>100 组回归测试 V3</title>
<style>
:root {{ --ok:#1b7f3b; --okbg:#e8f5ec; --warn:#9a6b00; --warnbg:#fff6e0; --bad:#b3261e; --badbg:#fdeceb; }}
body {{ font:13px/1.6 "Segoe UI","Microsoft YaHei",sans-serif; margin:24px; color:#1b1b1b; background:#fafafa; }}
h1 {{ font-size:21px; margin:0 0 4px; }} h2 {{ font-size:15px; margin:26px 0 8px; }}
.sub {{ color:#666; margin-bottom:14px; }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 8px; }}
.card {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:12px 16px; min-width:205px; }}
.card h3 {{ font-size:12px; color:#666; margin:0 0 6px; font-weight:600; }}
.card .v {{ font-size:21px; font-weight:600; }} .card .d {{ font-size:12px; color:#777; }}
table {{ border-collapse:collapse; background:#fff; width:100%; margin-bottom:6px; }}
th,td {{ border:1px solid #e0e0e0; padding:5px 8px; }}
th {{ background:#f2f4f7; font-weight:600; font-size:12px; text-align:left; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
td.id {{ white-space:nowrap; font-family:Consolas,monospace; font-size:12px; }}
td.sh {{ font-size:12px; }}
td.tile {{ font-family:Consolas,monospace; font-size:10px; color:#888; max-width:290px; overflow:hidden; text-overflow:ellipsis; }}
tr.sect td {{ background:#e9edf3; font-weight:600; }}
.muted {{ color:#888; }} td.ok {{ background:var(--okbg); color:var(--ok); }}
td.warn {{ background:var(--warnbg); color:var(--warn); }} td.bad {{ background:var(--badbg); color:var(--bad); }}
.note {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:12px 16px; margin-top:10px; }}
.note li {{ margin:4px 0; }}
code {{ background:#f0f2f5; padding:1px 5px; border-radius:3px; font-size:12px; }}
</style></head><body>
<h1>100 组回归测试 · V3</h1>
<div class="sub">BW1100 单卡 · FP16 · 实机 nmz-sh-1 · 布局按 Megatron 实际用法（NT：<code>matmul(x, w.t())</code>）·
协议：3 次独立分配、预热 Graph、每次回放 ≥8 次调用、2×25 中位数、取最小</div>

<div class="cards">
  <div class="card"><h3>模型 vs 库默认</h3><div class="v">{a_def[0]}/{a_def[1]}</div>
    <div class="d">±10% 以内 · MAPE {a_def[2]:.1f}%</div></div>
  <div class="card"><h3>模型 vs NN 形式上界</h3><div class="v">{a_best[0]}/{a_best[1]}</div>
    <div class="d">仅 {len(swept)} 个未收敛形状 · MAPE {a_best[2]:.1f}% · 该上界在 NT 下不可达</div></div>
  <div class="card"><h3>默认 / 上界（最大）</h3><div class="v">{max([r['gain'] for r in swept] or [1]):.1f}×</div>
    <div class="d">中位 {st.median([r['gain'] for r in swept]) if swept else 1:.2f}× · 即库默认离硬件极限有多远</div></div>
  <div class="card"><h3>GPT-7B 一次前向</h3><div class="v">{total_op / 1000:.0f} ms</div>
    <div class="d">12 个算子按 36 层展开实测（seq 4096、mb 1）</div></div>
</div>

<h2>1. 100 组逐条</h2>
<div class="note" style="margin-bottom:8px">
<b>三列口径不同，别混着看：</b>
「NT 默认」= 库在这块卡上实际选的 kernel 在 NT 布局下的耗时，也就是 Megatron 真正拿到的；
「NN 上界」= 对未收敛形状做全量枚举后在<b>转置 NN 形式</b>下测到的最小值，代表这块卡上这个数学问题能达到的上限，
<b>不是 NT 调用能拿到的值</b>——在 NT 形式下强制指定 solution 会触发 GPU 非法访存，实测不可行；
「模型」= 我们自己的分块模型，与布局无关。
</div>
<table><thead><tr><th>编号</th><th>M×N×K</th><th>NT 默认 µs</th><th>NN 上界 µs</th><th>默认/上界</th>
<th>模型 µs</th><th>误差 vs 默认</th><th>误差 vs 上界</th><th>默认 kernel</th></tr></thead>
<tbody>{''.join(group_block(c, l) for c, l in GROUPS)}</tbody></table>

<h2>2. 未收敛形状的 kernel 扫描（上界参考）</h2>
<div class="note" style="margin-bottom:8px">
这一节是「模型 vs NT 默认」误差超过 20% 的 24 个形状。扫描在转置 NN 形式下做全量枚举，
列出的最小值是<b>上界</b>：它说明这些形状离硬件极限有多远，不代表 Megatron 能拿到。
</div>
<table><thead><tr><th>编号</th><th>形状</th><th>NT 默认 µs</th><th>默认 kernel</th>
<th>NN 上界 µs</th><th>上界 kernel</th><th>默认/上界</th><th>模型 vs 上界</th></tr></thead>
<tbody>{sweep_rows}</tbody></table>

<h2>3. GPT-7B 的 12 个算子</h2>
<div class="note" style="margin-bottom:8px">
配置：36 层、hidden 4096、32 头（head_dim 128）、seq 4096、micro_batch 1、词表 50304、FFN 16384、未开 flash attention。
按 AICB <code>AiobMegatron.py</code> 的调用方式逐个复刻。非 GEMM 的五个算子没有 kernel 可选，只给实测值。
</div>
<table><thead><tr><th>算子</th><th>形状 / 调用方式</th><th>实测 µs</th><th>有效 GB/s</th>
<th>模型 µs</th><th>误差</th><th>实际 kernel</th></tr></thead>
<tbody>{''.join(op_rows)}</tbody></table>

<h2>4. 说明</h2>
<div class="note">
<ul>
<li><b>为什么用 NT。</b>Megatron 与 AICB 都是 <code>matmul(x, weight.t())</code>，权重存成 (N,K)。同一形状下 NT 与 NN 的耗时中位差 6%，范围 0.19~5.97 倍——库在两套布局下选的是不同族的 kernel。</li>
<li><b>为什么不给模型加布局系数。</b>实测乘 NT/NN 中位比值 1.056 之后，带内形状从 59/78 掉到 57/78、MAPE 从 8.5% 升到 9.2%。布局差小于模型自身的散布，加系数只会把原本偏高的推得更远。</li>
<li><b>上界为什么不是 NT 可达值。</b>rocBLAS 的 <code>gemm_ex_get_solutions</code> 在 NT 形式下确实能列出候选（4096³ 有 1064 个），
但把索引喂回 <code>gemm_ex</code> 会触发 GPU 非法访存（VMFault、进程 core dump），连第一个索引都崩。
同样的写法在转置 NN 形式下跑了几千次都没事，所以这是该 DTK 版本 NT 路径上的缺陷。结论：NT 下除了库默认，没有安全的选核方式。</li>
<li><b>模型的定位。</b>模型预测的是理想分块下的硬件极限，对标的就是「NN 上界」这一列；换硬件后更换硬件参数即可重新对标，而库默认的行为不在模型职责内。</li>
</ul>
</div>
</body></html>
"""
    OUT.write_text(html, encoding='utf-8')
    print(f'{OUT}  ({len(rows)} shapes, {len(swept)} swept, {len(ops)} ops)')


if __name__ == '__main__':
    main()
