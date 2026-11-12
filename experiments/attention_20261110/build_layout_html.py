"""Build 布局与选核排查_20261111.html from this round's measurement data."""
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V2 = ROOT / 'experiments/regression_100_v2_20261109'
PROBE = ROOT.parent / 'bw_probe'
OUT = ROOT.parent / '布局与选核排查_20261111.html'

GROUPS = [('decode_normal', 'Decode · 常规 N'), ('decode_largeN', 'Decode · 大 N'),
          ('prefill', 'Prefill'), ('gpt7b', 'GPT-7B'), ('gpt13b', 'GPT-13B')]

# K scan, M=N=4096, NT layout (console output of measure_kscan.py)
KSCAN = [(1024, 155.3, 221.2, 57.7, 146.9), (2048, 268.5, 255.9, 66.8, 255.7),
         (4096, 491.1, 279.9, 73.0, 473.1), (8192, 1137.7, 241.6, 63.0, 908.0),
         (16384, 2830.8, 194.2, 50.7, 1777.7), (32768, 6382.4, 172.3, 44.9, 3517.3)]
# same total work, different shape of run (measure_longk.py / measure_splitk.py)
LONGK = [('4 × K=4096（独立）', 1959.3, 280.6), ('1 × K=16384', 2832.9, 194.1),
         ('4 × K=4096（累加）', 3441.5, 159.7)]


def cls(e):
    if e is None:
        return 'muted'
    a = abs(e)
    return 'ok' if a <= 10 else ('warn' if a <= 20 else 'bad')


def pct(v):
    return f'{v:+.1f}%' if v is not None else '—'


def main():
    nnmap = {c['id']: c['default_us'] for c in
             json.loads((V2 / 'r100v2.json').read_text(encoding='utf-8'))['cases']}
    ntmap = {c['id']: c['default_us'] for c in
             json.loads((PROBE / 'nt100_20261111.json').read_text(encoding='utf-8'))['cases']}
    kern = {c['id']: c['kernel'] for c in
            json.loads((PROBE / 'nt100_20261111.json').read_text(encoding='utf-8'))['cases']}
    model = {r['id']: r['predicted_us'] for r in
             json.loads((V2 / 'predictions_bw2.json').read_text(encoding='utf-8'))}
    plan = {p['id']: p for p in json.loads((V2 / 'plan.json').read_text(encoding='utf-8'))}
    ab = json.loads((PROBE / 'layout_ab_20261111.json').read_text(encoding='utf-8'))['cases']
    sweep = json.loads((PROBE / 'kernel_check_20261111.json').read_text(encoding='utf-8'))['cases']

    rows = []
    for rid, p in plan.items():
        if rid not in ntmap:
            continue
        rows.append(dict(id=rid, cat=p['category'], shape=p['shape'], nn=nnmap[rid], nt=ntmap[rid],
                         ratio=ntmap[rid] / nnmap[rid], model=model[rid],
                         e_nn=100 * (model[rid] / nnmap[rid] - 1),
                         e_nt=100 * (model[rid] / ntmap[rid] - 1), kernel=kern.get(rid, '')))

    def agg(rs, key):
        v = [abs(r[key]) for r in rs]
        return sum(x <= 10 for x in v), len(v), st.mean(v)

    band = [r for r in rows if 0.8 <= r['ratio'] <= 1.5]
    outb = [r for r in rows if not (0.8 <= r['ratio'] <= 1.5)]
    a_nn, a_nt = agg(rows, 'e_nn'), agg(rows, 'e_nt')
    b_nt = agg(band, 'e_nt')
    b_nn = agg(band, 'e_nn')

    def group_block(cat, label):
        rs = [r for r in rows if r['cat'] == cat]
        n_ok, _n, n_m = agg(rs, 'e_nn')
        t_ok, _t, t_m = agg(rs, 'e_nt')
        head = (f"<tr class='sect'><td colspan='9'>{label} · {len(rs)} 个 · "
                f"±10% 以内 对 NN {n_ok}/{len(rs)}（MAPE {n_m:.1f}%） · "
                f"对 NT {t_ok}/{len(rs)}（MAPE {t_m:.1f}%） · "
                f"NT/NN 中位 {st.median([r['ratio'] for r in rs]):.2f}</td></tr>")
        body = []
        for r in sorted(rs, key=lambda r: -abs(r['e_nt'])):
            body.append(
                f"<tr><td class='id'>{r['id']}</td>"
                f"<td class='num'>{'×'.join(map(str, r['shape']))}</td>"
                f"<td class='num'>{r['nn']:.1f}</td><td class='num'><b>{r['nt']:.1f}</b></td>"
                f"<td class='num {'bad' if (r['ratio'] > 1.5 or r['ratio'] < 0.8) else 'muted'}'>"
                f"{r['ratio']:.2f}</td>"
                f"<td class='num'>{r['model']:.1f}</td>"
                f"<td class='num {cls(r['e_nn'])}'>{pct(r['e_nn'])}</td>"
                f"<td class='num {cls(r['e_nt'])}'>{pct(r['e_nt'])}</td>"
                f"<td class='tile muted'>{r['kernel']}</td></tr>")
        return head + ''.join(body)

    ab_rows = ''.join(
        f"<tr><td class='id'>{'×'.join(map(str, c['shape']))}</td>"
        f"<td class='num'>{c['nn_us']:.1f}</td><td class='num'><b>{c['nt_us']:.1f}</b></td>"
        f"<td class='num {'bad' if (c['nt_over_nn'] > 1.5 or c['nt_over_nn'] < 0.8) else 'warn'}'>"
        f"{c['nt_over_nn']:.2f}</td>"
        f"<td class='tile muted'>{c['nn_kernel']}</td><td class='tile muted'>{c['nt_kernel']}</td></tr>"
        for c in ab)

    sweep_rows = ''.join(
        f"<tr><td class='id'>{c['id']}</td><td class='num'>{'×'.join(map(str, c['shape']))}</td>"
        f"<td class='num'>{c['default_us']:.1f}</td><td class='tile muted'>{c['default_kernel']}</td>"
        f"<td class='num'>{c['fastest_us']:.1f}</td><td class='tile muted'>{c['fastest_kernel']}</td>"
        f"<td class='num {'bad' if c['speedup'] >= 1.5 else 'ok'}'>{c['speedup']:.2f}×</td></tr>"
        for c in sweep)

    kscan_rows = ''.join(
        f"<tr><td class='num'>{k}</td><td class='num'>{us:.1f}</td><td class='num'>{tf:.0f}</td>"
        f"<td class='num'>{pk:.1f}%</td><td class='num'>{mo:.1f}</td>"
        f"<td class='num {cls(100 * (mo / us - 1))}'>{pct(100 * (mo / us - 1))}</td></tr>"
        for k, us, tf, pk, mo in KSCAN)

    longk_rows = ''.join(
        f"<tr><td class='id'>{n}</td><td class='num'>{us:.1f}</td><td class='num'>{tf:.1f}</td></tr>"
        for n, us, tf in LONGK)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>NT/NN 布局与选核排查</title>
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
td.tile {{ font-family:Consolas,monospace; font-size:10px; color:#888; max-width:300px; overflow:hidden; text-overflow:ellipsis; }}
tr.sect td {{ background:#e9edf3; font-weight:600; }}
.muted {{ color:#888; }} td.ok {{ background:var(--okbg); color:var(--ok); }}
td.warn {{ background:var(--warnbg); color:var(--warn); }} td.bad {{ background:var(--badbg); color:var(--bad); }}
.note {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:12px 16px; margin-top:10px; }}
.note li {{ margin:4px 0; }}
code {{ background:#f0f2f5; padding:1px 5px; border-radius:3px; font-size:12px; }}
</style></head><body>
<h1>NT / NN 布局与选核排查</h1>
<div class="sub">BW1100 单卡 · 实机 nmz-sh-1 · 100 个形状全部按两种布局各测一遍 ·
NN = <code>matmul(a, b)</code>（B 存 K×N），NT = <code>matmul(a, w.t())</code>（B 存 N×K，Megatron/AICB 的实际写法）·
协议：3 次独立分配、预热 Graph、每次回放 ≥8 次调用、2×25 中位数、取最小</div>

<div class="cards">
  <div class="card"><h3>NT / NN 中位比值</h3><div class="v">{st.median([r['ratio'] for r in rows]):.2f}×</div>
    <div class="d">范围 {min(r['ratio'] for r in rows):.2f} ~ {max(r['ratio'] for r in rows):.2f}</div></div>
  <div class="card"><h3>模型 vs NT</h3><div class="v">{a_nt[0]}/{a_nt[1]}</div>
    <div class="d">±10% 以内 · MAPE {a_nt[2]:.1f}%（对 NN 是 {a_nn[0]}/{a_nn[1]}、{a_nn[2]:.1f}%）</div></div>
  <div class="card"><h3>带内 78 个形状</h3><div class="v">{b_nt[0]}/{b_nt[1]}</div>
    <div class="d">NT/NN 在 0.8~1.5 · 模型对 NT 的 MAPE {b_nt[2]:.1f}%</div></div>
  <div class="card"><h3>带外 22 个</h3><div class="v">{len(outb)}</div>
    <div class="d">绝大多数是 decode 大 N，NT 比 NN 慢 1.8~6 倍</div></div>
</div>

<h2>1. 100 组逐条：NN / NT / 模型</h2>
<table><thead><tr><th>编号</th><th>M×N×K</th><th>NN µs</th><th>NT µs</th><th>NT/NN</th>
<th>模型 µs</th><th>误差 vs NN</th><th>误差 vs NT</th><th>NT 实际 kernel</th></tr></thead>
<tbody>{''.join(group_block(c, l) for c, l in GROUPS)}</tbody></table>

<h2>2. 四个形状的布局对拍（含 kernel 名）</h2>
<table><thead><tr><th>形状</th><th>NN µs</th><th>NT µs</th><th>NT/NN</th>
<th>NN kernel</th><th>NT kernel</th></tr></thead><tbody>{ab_rows}</tbody></table>
<div class="note">
<li>NT 用的是<b>另一族 kernel</b>：名字前缀从 <code>Cijk_Ailk</code> 变成 <code>Cijk_Alik</code>，并且多了 <code>DTL1</code>（Direct-To-LDS）和 <code>SU</code>（SplitU）开关，分块也常从 bk=16 换成 bk=32。</li>
<li>所以差距不是"同一个内核换个方向"，而是库按布局选了不同的实现，比值因此不固定。</li>
</div>

<h2>3. 选核枚举：默认是不是最快</h2>
<table><thead><tr><th>用例</th><th>形状</th><th>默认 µs</th><th>默认 kernel</th>
<th>最快 µs</th><th>最快 kernel</th><th>倍数</th></tr></thead><tbody>{sweep_rows}</tbody></table>
<div class="note">
<li><code>logit</code> 那一组是 NN 布局：默认选中 64x128x16，比最快的 256x256x16 慢 <b>8.27 倍</b>。而 NT 路径下库选的是 256x256x32，属于正常水平——所以"lm_head 选核灾难"是 NN 布局特有的。</li>
<li><code>mlp2</code>（K=16384）默认就是最快（1.00×），说明大 K 的偏差不是选核问题，是内核本身。</li>
</div>

<h2>4. 大 K：吞吐随 K 衰减</h2>
<table><thead><tr><th>K</th><th>实测 µs</th><th>TFLOPS</th><th>峰值占比</th><th>模型 µs</th><th>误差</th></tr></thead>
<tbody>{kscan_rows}</tbody></table>
<div class="note">
固定 M=N=4096、NT 布局。吞吐在 K=4096 达到 73%，之后一路掉到 K=32768 的 45%。模型按面板线性计，所以偏差随 K 单调放大。
</div>

<h2>5. 大 K 的三个判别实验</h2>
<table><thead><tr><th>跑法</th><th>耗时 µs</th><th>TFLOPS</th></tr></thead><tbody>{longk_rows}</tbody></table>
<div class="note">
<li>同样 550 GFLOP、同样字节量，拆成 4 个独立的 K=4096 是 1959 µs（280 TFLOPS），一个 K=16384 是 2832.9 µs（194 TFLOPS）→ <b>不是降频，是长 K 内核本身慢</b>。</li>
<li>真实拆 K 必须累加（<code>addmm</code> beta=1），实测 3441.5 µs，比不拆还慢 → <b>拆 K 这条优化路走不通</b>。</li>
<li>M 不是因素：2048×4096×16384 是 52.5%，4096×4096×16384 是 50.5%。</li>
</div>

<h2>6. 结论</h2>
<div class="note">
<ul>
<li><b>不加布局系数。</b>实测给模型乘 NT/NN 中位比值 1.056 之后，带内形状从 59/78 掉到 57/78、MAPE 从 8.5% 升到 9.2%——因为布局差（5.6%）小于模型自身的散布（8%），乘系数只会把原本偏高的推得更远。</li>
<li><b>两种布局各有各的坏区，而且不是同一批形状。</b>NN 在大 N（lm_head）上会选灾难 kernel（慢 8.3 倍）；NT 在大 N 的 decode 上会选差 kernel（最差 6 倍）。没有哪个布局普遍更好。</li>
<li><b>Megatron 用 NT，就以 NT 为基准。</b>模型在 NT 上的误差是带内 78 个形状 MAPE 8.5%，可以作为仿真里外推形状的精度预期。</li>
<li><b>带外 22 个是 kernel 选择问题</b>，不是带宽、不是布局系数，需要按形状查"库选了哪族 kernel"才能修。</li>
</ul>
</div>
</body></html>
"""
    OUT.write_text(html, encoding='utf-8')
    print(f'{OUT}  ({len(rows)} shapes, {len(sweep)} sweep, {len(ab)} layout pairs)')


if __name__ == '__main__':
    main()
