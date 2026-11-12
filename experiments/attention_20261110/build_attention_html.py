"""Build 带宽与注意力建模_20261110.html (regenerated 2026-11-11 with the A/B read split)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # compute_model
V2 = ROOT / 'experiments/regression_100_v2_20261109'
PROBE = ROOT.parent / 'bw_probe'
OUT = ROOT.parent / '带宽与注意力建模_20261110.html'

GROUPS = [('decode_normal', 'Decode · 常规 N'), ('decode_largeN', 'Decode · 大 N'),
          ('prefill', 'Prefill'), ('gpt7b', 'GPT-7B'), ('gpt13b', 'GPT-13B')]

# name, measured, original model, after write-rate split, after A/B read split
BMM = [
    ('QK^T  32×1024×1024×128', 110.1, 77.4, 102.1, 102.5),
    ('QK^T  32×4096×4096×128', 1384.4, 1095.5, 1493.9, 1495.4),
    ('PV    32×4096×128×4096', 989.0, 673.8, 686.3, 1038.1),
    ('PV    32×2048×128×2048', 258.5, None, 193.4, 275.9),
]


def cls(e):
    if e is None:
        return 'muted'
    a = abs(e)
    return 'ok' if a <= 10 else ('warn' if a <= 20 else 'bad')


def pct(v):
    return f'{v:+.1f}%' if v is not None else '—'


def main():
    curve = json.loads((PROBE / 'stream_curve_20261110.json').read_text(encoding='utf-8'))['cases']
    l2 = json.loads((PROBE / 'pv_l2_20261111.json').read_text(encoding='utf-8'))['cases']
    readp = json.loads((PROBE / 'read_path_20261111.json').read_text(encoding='utf-8'))['cases']
    opnd = json.loads((PROBE / 'read_operand_20261111.json').read_text(encoding='utf-8'))['cases']

    meas = {tuple(c['shape']): c for c in
            json.loads((V2 / 'r100v2.json').read_text(encoding='utf-8'))['cases']}
    plan = {p['id']: p for p in json.loads((V2 / 'plan.json').read_text(encoding='utf-8'))}
    before = {r['id']: r for r in json.loads((V2 / 'predictions.json').read_text(encoding='utf-8'))}
    after = {r['id']: r for r in json.loads((V2 / 'predictions_bw2.json').read_text(encoding='utf-8'))}

    rows = []
    for rid, p in plan.items():
        m = meas.get(tuple(p['shape']), {})
        d = m.get('default_us')
        rows.append(dict(id=rid, cat=p['category'], shape=p['shape'], meas=d,
                         b=before[rid]['predicted_us'], a=after[rid]['predicted_us'],
                         eb=100 * (before[rid]['predicted_us'] / d - 1) if d else None,
                         ea=100 * (after[rid]['predicted_us'] / d - 1) if d else None))

    def stats(key):
        e = [abs(r[key]) for r in rows if r[key] is not None]
        return (sum(x <= 10 for x in e), len(e), sum(e) / len(e) if e else 0, max(e) if e else 0)

    sb, sa = stats('eb'), stats('ea')
    keep = [r for r in rows if not (r['meas'] and r['meas'] > 3000)]
    kb = [abs(r['eb']) for r in keep]; ka = [abs(r['ea']) for r in keep]

    curve_rows = ''.join(
        f"<tr><td class='id'>{c['kind']}</td><td class='num'>{c['size_mb']}</td>"
        f"<td class='num'>{c['us']:.1f}</td><td class='num'><b>{c['gbps']:.0f}</b></td>"
        f"<td class='num muted'>{c['spread_pct']:.1f}%</td></tr>" for c in curve)

    def num_cells(v, ref):
        if v is None:
            return "<td class='num muted'>—</td><td class='num muted'>—</td>"
        e = 100 * (v / ref - 1)
        return f"<td class='num'>{v:.1f}</td><td class='num {cls(e)}'>{pct(e)}</td>"

    bmm_rows = ''.join(
        f"<tr><td class='id'>{n}</td><td class='num'>{m:.1f}</td>"
        f"{num_cells(o, m)}{num_cells(w, m)}{num_cells(f, m)}</tr>"
        for n, m, o, w, f in BMM)

    l2_rows = ''.join(
        f"<tr><td class='id'>{c['label']}</td><td class='num'>{c['unique_MB']:.0f}</td>"
        f"<td class='num'>{c['us']:.1f}</td><td class='num'><b>{c['gbps_unique']:.0f}</b></td>"
        f"<td class='num'>{c['reread_MB']:.0f}</td><td class='num muted'>{c['gbps_reread']:.0f}</td></tr>"
        for c in l2)

    readp_rows = ''.join(
        f"<tr><td class='num'>N={c['n']}</td><td class='num'>{c['read_MB']:.0f}</td>"
        f"<td class='num'>{c['us']:.1f}</td><td class='num'><b>{c['gbps_read']:.0f}</b></td>"
        f"<td class='num muted'>{c['tflops']:.1f}</td></tr>" for c in readp)

    opnd_rows = ''.join(
        f"<tr><td class='id'>{c['label']}</td><td class='num'>{c['read_MB']:.0f}</td>"
        f"<td class='num'>{c['us']:.1f}</td><td class='num'><b>{c['gbps']:.0f}</b></td>"
        f"<td class='num muted'>{c['tflops']:.1f}</td></tr>" for c in opnd)

    def group_block(cat, label):
        rs = [r for r in rows if r['cat'] == cat]
        b = [abs(r['eb']) for r in rs if r['eb'] is not None]
        a = [abs(r['ea']) for r in rs if r['ea'] is not None]
        head = (f"<tr class='sect'><td colspan='7'>{label} · {len(rs)} 个 · "
                f"±10% 以内 {sum(x <= 10 for x in b)}→{sum(x <= 10 for x in a)} · "
                f"MAPE {sum(b) / len(b):.1f}%→{sum(a) / len(a):.1f}%</td></tr>")
        body = []
        for r in sorted(rs, key=lambda r: -abs(r['ea'] or 0)):
            body.append(
                f"<tr><td class='id'>{r['id']}</td>"
                f"<td class='num'>{'×'.join(map(str, r['shape']))}</td>"
                f"<td class='num'>{r['meas']:.1f}</td>"
                f"<td class='num'>{r['b']:.1f}</td><td class='num {cls(r['eb'])}'>{pct(r['eb'])}</td>"
                f"<td class='num'><b>{r['a']:.1f}</b></td><td class='num {cls(r['ea'])}'>{pct(r['ea'])}</td></tr>")
        return head + ''.join(body)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>带宽实测与注意力建模</title>
<style>
:root {{ --ok:#1b7f3b; --okbg:#e8f5ec; --warn:#9a6b00; --warnbg:#fff6e0; --bad:#b3261e; --badbg:#fdeceb; }}
body {{ font:13px/1.6 "Segoe UI","Microsoft YaHei",sans-serif; margin:24px; color:#1b1b1b; background:#fafafa; }}
h1 {{ font-size:21px; margin:0 0 4px; }} h2 {{ font-size:15px; margin:26px 0 8px; }}
h3 {{ font-size:13px; margin:16px 0 6px; }}
.sub {{ color:#666; margin-bottom:14px; }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 8px; }}
.card {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:12px 16px; min-width:215px; }}
.card h3 {{ font-size:12px; color:#666; margin:0 0 6px; font-weight:600; }}
.card .v {{ font-size:21px; font-weight:600; }} .card .d {{ font-size:12px; color:#777; }}
table {{ border-collapse:collapse; background:#fff; width:100%; margin-bottom:6px; }}
th,td {{ border:1px solid #e0e0e0; padding:5px 8px; }}
th {{ background:#f2f4f7; font-weight:600; font-size:12px; text-align:left; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
td.id {{ white-space:nowrap; font-family:Consolas,monospace; font-size:12px; }}
tr.sect td {{ background:#e9edf3; font-weight:600; }}
.muted {{ color:#888; }} td.ok {{ background:var(--okbg); color:var(--ok); }}
td.warn {{ background:var(--warnbg); color:var(--warn); }} td.bad {{ background:var(--badbg); color:var(--bad); }}
.note {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:12px 16px; margin-top:10px; }}
.note li {{ margin:4px 0; }}
code {{ background:#f0f2f5; padding:1px 5px; border-radius:3px; font-size:12px; }}
</style></head><body>
<h1>HBM 带宽实测与注意力算子建模</h1>
<div class="sub">BW1100 单卡 · FP16 · 实机 nmz-sh-1 · 协议：3 次独立分配、预热 Graph、2×25 样本取中位数、取最小</div>

<div class="cards">
  <div class="card"><h3>写带宽上限</h3><div class="v">1.05 TB/s</div>
    <div class="d">模型原来按 1.72 TB/s，偏乐观 64%</div></div>
  <div class="card"><h3>读 A（M×K）上限</h3><div class="v">1.10 TB/s</div>
    <div class="d">流式读 A 只有这个速度，与输出无关</div></div>
  <div class="card"><h3>读 B（K×N）上限</h3><div class="v">1.72 TB/s</div>
    <div class="d">decode 实测 1.60~1.83，与模型一致</div></div>
  <div class="card"><h3>100 组回归</h3><div class="v">{sa[0]}/{sa[1]}</div>
    <div class="d">±10% 以内（改前 {sb[0]}/{sb[1]}）· MAPE {sa[2]:.1f}%（改前 {sb[2]:.1f}%）</div></div>
</div>

<h2>1. 流式带宽实测（fill / sum / copy）</h2>
<table><thead><tr><th>类型</th><th>MiB</th><th>µs</th><th>GB/s</th><th>两次差</th></tr></thead>
<tbody>{curve_rows}</tbody></table>
<div class="note">
<li><code>fill</code> 纯写：64 MiB 起稳定在 <b>1.03~1.05 TB/s</b>。</li>
<li><code>sum</code> 纯读：256 MiB 起 <b>1.63~1.84 TB/s</b>。读写不对称，差 60~75%。</li>
<li>模型原来两个方向都用 1.72 TB/s，写方向错得最多。</li>
</div>

<h2>2. 模型改动</h2>
<div class="note">
<ul>
<li><b>第一次</b>：硬件档案新增 <code>hbm_write_bandwidth_Bps = 1.05e12</code>，输出排空改用写速率。修好了 QK^T，但 PV 没动（它是读主导）。</li>
<li><b>第二次</b>：新增 <code>hbm_read_a_Bps = 1.10e12</code>，输入填充拆成两段——A 视图 <code>m·k·2 / 1.10</code>，B 视图 <code>n·k·2 / 1.72</code>，两段相加。</li>
<li>两个字段留空时都回落到 <code>hbm_bandwidth_Bps</code>，A100 等旧档案行为不变；108 个测试全过。</li>
</ul>
</div>

<h2>3. 批量注意力：实测 vs 三个模型版本</h2>
<table><thead><tr><th>用例</th><th>实测 µs</th><th>原始</th><th>误差</th><th>写带宽修正</th><th>误差</th><th>A 读拆分</th><th>误差</th></tr></thead>
<tbody>{bmm_rows}</tbody></table>
<div class="note">
<li><b>批量确实有效，但没有理想的那么好。</b>s=1024 时 32 头一次 bmm 是 110 µs，32 次单头循环 241 µs（快 2.06 倍）；s=4096 时只快 1.30 倍，单头已经铺满机器。</li>
<li><b>缩放不是免费的。</b><code>baddbmm(alpha=1/√d)</code> 比纯 bmm 慢 3.2%（s=1024）到 9.6%（s=4096），和输出元素数成正比，属 epilogue 的逐元素开销。</li>
</div>

<h2>4. 诊断：PV 为什么慢</h2>
<h3>4.1 先否掉"B 的副本重复读"</h3>
<p>PV 每个块都要重读本头的 B（1 MiB），32 头合计 32 MiB，超过 8 MiB 的 L2，所以一开始怀疑这些重读要回 HBM。把 B 换成 32 个头共用一份（广播，stride 0）来消除重读：</p>
<table><thead><tr><th>用例</th><th>唯一字节 MB</th><th>µs</th><th>唯一带宽 GB/s</th><th>含重读 MB</th><th>含重读 GB/s</th></tr></thead>
<tbody>{l2_rows}</tbody></table>
<div class="note">
<b>结论：否证。</b>共用 B 反而更慢（1062.7 → 1098.6 µs）。B 的重读本来就被 L2 吃掉，PV 的缺口不在重复流量上。
</div>

<h3>4.2 再定位到"读 A 的上限"</h3>
<p>固定 A（32×4096×4096 = 1.07 GB），把输出宽度 N 从 8 加到 128。N=8 时算力只占峰值 2%，几乎纯读：</p>
<table><thead><tr><th>N</th><th>读 MB</th><th>µs</th><th>读带宽 GB/s</th><th>TFLOPS</th></tr></thead>
<tbody>{readp_rows}</tbody></table>
<div class="note">
读速率对 N 几乎不敏感（963 → 1010 GB/s），说明<b>这是 GEMM 读 A 的上限，和输出、和算力都无关</b>。
</div>

<h3>4.3 A 和 B 的读取速率不一样</h3>
<table><thead><tr><th>用例</th><th>读 MB</th><th>µs</th><th>GB/s</th><th>TFLOPS</th></tr></thead>
<tbody>{opnd_rows}</tbody></table>
<div class="note">
<li><b>A 主导</b>（65536×8×4096，流 M×K）只有 1174 GB/s；<b>B 主导</b>（8×65536×4096，流 K×N）能到 1414 GB/s。</li>
<li>拿 100 组里的 decode（权重主导）对照：实测折算 <b>1.60~1.83 TB/s</b>，和模型的 1.72 吻合——所以 B 的速率不能降。</li>
<li>于是输入填充按算子拆开：A 用 1.10，B 用 1.72。PV 因此从 686 → 1038 µs。</li>
</ul>
</div>

<h3>4.4 硬件计数器（hipprof PMC）：A 的每个 L2 行被取两次</h3>
<table><thead><tr><th>用例</th><th>kernel 分块</th><th>cycles</th><th>TCC_MISS</th>
<th>唯一数据的 128 B 行数</th><th>每行 miss 次数</th></tr></thead><tbody>
<tr><td class="id">A 主导 65536×8×4096</td><td class="id">MT32x64x128</td><td class="num">562 K</td><td class="num">8.43 M</td><td class="num">4.19 M</td><td class="num bad">2.01</td></tr>
<tr><td class="id">B 主导 8×65536×4096</td><td class="id">MT256x64x16</td><td class="num">441 K</td><td class="num">4.21 M</td><td class="num">4.19 M</td><td class="num ok">1.00</td></tr>
<tr><td class="id">PV 32×4096×128×4096</td><td class="id">MT128x128x32</td><td class="num">1.40 M</td><td class="num">17.7 M</td><td class="num">8.39 M</td><td class="num bad">2.10</td></tr>
</tbody></table>
<div class="note">
<li>同样 537 MB 唯一数据，A 主导的 kernel 产生 <b>8.43M 次 L2 miss</b>，B 主导只有 <b>4.21M</b> —— 而 537 MB / 128 B = 4.19M 行。所以 <b>A 的每个 128 B 行被取了两次</b>，B 只取一次。</li>
<li>真实的 PV 形状（A = 1074 MB = 8.39M 行）复现了同样的 2.10 倍。</li>
<li>但耗时只差 20~28%（457 vs 380 µs），说明重复的请求有一部分被合并/吸收，没有全部打到 DRAM。所以<b>有效读取率降到约 1.1 TB/s，而不是直接减半</b>——这和按实测值标定出来的 1.10 TB/s 一致。</li>
</div>

<h4 style="margin:14px 0 6px">第二次访问去哪了：它真的又取了一遍</h4>
<table><thead><tr><th>A 的大小</th><th>形状</th><th>kernel 分块</th><th>TCC_MISS</th>
<th>A 的 128 B 行数</th><th>每行 miss</th></tr></thead><tbody>
<tr><td>4 MiB（放得进 L2）</td><td class="id">2048×8×1024</td><td class="id">MT64x32x64</td><td class="num">34.0 K</td><td class="num">32.8 K</td><td class="num ok">1.04</td></tr>
<tr><td>32 MiB</td><td class="id">16384×8×1024</td><td class="id">MT64x64x32</td><td class="num">529 K</td><td class="num">262 K</td><td class="num bad">2.02</td></tr>
<tr><td>32 MiB（同样字节，块数少 4 倍）</td><td class="id">4096×8×4096</td><td class="id">MT64x64x64</td><td class="num">527 K</td><td class="num">262 K</td><td class="num bad">2.01</td></tr>
<tr><td>537 MiB，K=1024</td><td class="id">262144×8×1024</td><td class="id">MT32x64x128</td><td class="num">8.42 M</td><td class="num">4.19 M</td><td class="num bad">2.01</td></tr>
<tr><td>537 MiB，K=32768</td><td class="id">8192×8×32768</td><td class="id">MT32x16x128</td><td class="num">8.47 M</td><td class="num">4.19 M</td><td class="num bad">2.02</td></tr>
</tbody></table>
<div class="note">
<li><b>A 装得进 L2 时是 1.04 次/行</b>——第二次访问命中缓存；一旦超过 8 MiB 就变成 2.02。说明第二遍是真的又去取了，只是小的时候能被缓存接住。</li>
<li><b>与 K 无关</b>（1024 / 4096 / 32768 都是 2.0），<b>与并发块数无关</b>（64 块和 256 块同为 2.01）。所以不是"面板宽度切到半条线"，也不是"并发压力挤出"。</li>
<li>剩下唯一没定位的是：kernel 为什么要对 A 读两遍。要回答这个需要反汇编厂商 kernel，按仓库约定不进 Git，只能到此为止。</li>
</div>

<h2>5. 100 组回归：原始 vs 最终</h2>
<div class="cards">
  <div class="card"><h3>全部 100 个 · ±10% 以内</h3><div class="v">{sb[0]} → {sa[0]}</div>
    <div class="d">MAPE {sb[2]:.1f}% → {sa[2]:.1f}% · 最大 {sb[3]:.1f}% → {sa[3]:.1f}%</div></div>
  <div class="card"><h3>剔除 lm_head 三行后的 97 个</h3>
    <div class="v">{sum(1 for x in kb if x <= 10)} → {sum(1 for x in ka if x <= 10)}</div>
    <div class="d">MAPE {sum(kb) / len(kb):.1f}% → {sum(ka) / len(ka):.1f}%</div></div>
</div>
<table><thead><tr><th>编号</th><th>M×N×K</th><th>实测 µs</th><th>原始 µs</th><th>误差</th><th>最终 µs</th><th>误差</th></tr></thead>
<tbody>{''.join(group_block(c, l) for c, l in GROUPS)}</tbody></table>

<h2>6. 结论与剩余问题</h2>
<div class="note">
<ul>
<li><b>QK^T / PV 不需要矢量单元。</b>它们是矩阵乘，走 MMAC；实测也证明瓶颈在读取和写出，不在算力（N=8 时算力只占 2%，时间由读决定）。</li>
<li><b>真正需要的矢量单元场景</b>是 softmax / layernorm / gelu / embedding 四个算子，届时要先测 SIMD 的逐元素与规约吞吐（现在只知道 Shared 读 128 B/cycle、写 59.35 B/cycle）。</li>
<li><b>lm_head 三行（gpt7b_13/14、gpt13b_14）不是模型的错</b>：库默认选了慢 3.5~8.5 倍的 kernel。对最快 kernel 时误差是 +8.3% / −0.5% / −3.4%。</li>
<li><b>流 A 只有 1.10 TB/s、流 B 有 1.72 的机理</b>：PMC 显示 A 的每个 128 B 行被取两次（decoding 侧只取一次），重复请求部分被合并，净损失约 36%。具体是哪一段代码造成二次读取还没定位。</li>
<li><b>结构性问题</b>：长 K 形状里输出排空应当和下一波计算重叠，现在模型是整段串行加到波次时间上，所以低 K 形状修好了、中 K 形状会略微过冲。</li>
</ul>
</div>
</body></html>
"""
    OUT.write_text(html, encoding='utf-8')
    print(f'{OUT}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
