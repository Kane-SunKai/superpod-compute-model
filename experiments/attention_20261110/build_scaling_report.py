"""Join the nine scaling measurements with the model and report both a table and an HTML."""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBE = ROOT.parent / 'bw_probe'
OUT = ROOT.parent / '缩放对照_MNK_20261112.html'
CORES, CLK, L2 = 72, 1.3e9, 8 * 1024 * 1024
SHAPES = [
    ('half_all', 2048, 2048, 8192, '全部 ÷2'),
    ('half_M', 2048, 4096, 16384, '只 M ÷2'),
    ('half_N', 4096, 2048, 16384, '只 N ÷2'),
    ('half_K', 4096, 4096, 8192, '只 K ÷2'),
    ('base', 4096, 4096, 16384, '基准'),
    ('dbl_M', 8192, 4096, 16384, '只 M ×2'),
    ('dbl_N', 4096, 8192, 16384, '只 N ×2'),
    ('dbl_K', 4096, 4096, 32768, '只 K ×2'),
    ('dbl_all', 8192, 8192, 32768, '全部 ×2'),
]


def main():
    meas_f = PROBE / 'scaling_20261112.json'
    if not meas_f.exists():
        sys.exit('missing %s' % meas_f)
    meas = {c['label']: c for c in json.loads(meas_f.read_text(encoding='utf-8'))['cases']}
    mf = HERE / 'scaling_model.json'
    model = {r['label']: r for r in json.loads(mf.read_text(encoding='utf-8'))} if mf.exists() else {}

    rows = []
    for label, m, n, k, note in SHAPES:
        c = meas.get(label)
        if not c:
            continue
        us = c['min_us']
        bm, bn, bk = (int(x) for x in c['tile'].split('x')) if c['tile'] else (0, 0, 0)
        macs = m * n * k
        ideal = macs / (CORES * 2048 * CLK) * 1e6
        wgs = math.ceil(m / bm) * math.ceil(n / bn) if bm else 0
        waves = math.ceil(wgs / CORES) if wgs else 0
        util = wgs / (waves * CORES) if waves else 0
        panel = bm * k * 2
        steps = k // bk if bk else 0
        per_step = (us * 1e-6 * CLK / waves / steps) if steps else 0
        step_ideal = bm * bn * bk / 2048 if bk else 0
        mu = model.get(label, {}).get('us')
        rows.append(dict(label=label, note=note, m=m, n=n, k=k, us=us, model=mu,
                         err=(100 * (mu / us - 1) if mu else None),
                         tile=c['tile'], su=c['su'], wgs=wgs, waves=waves, util=util,
                         panel=panel, ratio=panel / L2, ideal=ideal,
                         meas_over_ideal=us / ideal,
                         model_step=(mu / ideal * util) if mu else None,
                         per_step_ratio=(per_step / step_ideal) if step_ideal else 0))

    print('%-9s %-20s %-6s %-13s %-5s %8s %8s %8s %7s %7s %6s %6s' % (
        '组', 'M×N×K', '变化', '分块/SplitU', 'WGs', '切片MB', '实测µs',
        '模型µs', '误差', '实测/理想', '每步', '模型每步'))
    for r in rows:
        print('%-9s %-20s %-6s %-13s %-5d %8.2f %8.1f %8s %7s %7.2f %6.2f %6s' % (
            r['label'], '%d×%d×%d' % (r['m'], r['n'], r['k']), r['note'],
            '%s %s' % (r['tile'], r['su']), r['wgs'], r['panel'] / 2**20, r['us'],
            ('%.1f' % r['model']) if r['model'] else '—',
            ('%+.1f%%' % r['err']) if r['err'] is not None else '—',
            r['meas_over_ideal'], r['per_step_ratio'],
            ('%.2f' % r['model_step']) if r['model_step'] else '—'))

    def cls(e):
        if e is None:
            return 'muted'
        a = abs(e)
        return 'ok' if a <= 10 else ('warn' if a <= 20 else 'bad')

    body = ''.join(
        f"<tr><td class='id'>{r['label']}</td>"
        f"<td class='num'>{r['m']}×{r['n']}×{r['k']}</td><td>{r['note']}</td>"
        f"<td class='tile'>{r['tile']} {r['su']}</td>"
        f"<td class='num'>{r['wgs']}</td><td class='num'>{r['util'] * 100:.0f}%</td>"
        f"<td class='num'>{r['panel'] / 2**20:.2f}</td><td class='num'>{r['ratio']:.2f}</td>"
        f"<td class='num'><b>{r['us']:.1f}</b></td>"
        f"<td class='num'>{('%.1f' % r['model']) if r['model'] else '—'}</td>"
        f"<td class='num {cls(r['err'])}'>"
        f"{('%+.1f%%' % r['err']) if r['err'] is not None else '—'}</td>"
        f"<td class='num'>{r['meas_over_ideal']:.2f}</td>"
        f"<td class='num'>{r['per_step_ratio']:.2f}</td>"
        f"<td class='num muted'>{('%.2f' % r['model_step']) if r['model_step'] else '—'}</td>"
        f"</tr>"
        for r in rows)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>缩放对照 · MNK</title>
<style>
:root {{ --ok:#1b7f3b; --okbg:#e8f5ec; --warn:#9a6b00; --warnbg:#fff6e0; --bad:#b3261e; --badbg:#fdeceb; }}
body {{ font:13px/1.6 "Segoe UI","Microsoft YaHei",sans-serif; margin:24px; color:#1b1b1b; background:#fafafa; }}
h1 {{ font-size:21px; margin:0 0 4px; }} h2 {{ font-size:15px; margin:24px 0 8px; }}
.sub {{ color:#666; margin-bottom:14px; }}
table {{ border-collapse:collapse; background:#fff; width:100%; margin-bottom:6px; }}
th,td {{ border:1px solid #e0e0e0; padding:5px 8px; }}
th {{ background:#f2f4f7; font-weight:600; font-size:12px; text-align:left; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
td.id {{ font-family:Consolas,monospace; font-size:12px; }}
td.tile {{ font-family:Consolas,monospace; font-size:11px; color:#555; }}
.muted {{ color:#888; }} td.ok {{ background:var(--okbg); color:var(--ok); }}
td.warn {{ background:var(--warnbg); color:var(--warn); }} td.bad {{ background:var(--badbg); color:var(--bad); }}
.note {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:12px 16px; margin-top:10px; }}
.note li {{ margin:4px 0; }}
code {{ background:#f0f2f5; padding:1px 5px; border-radius:3px; font-size:12px; }}
</style></head><body>
<h1>mlp_linear_2 缩放对照 · 九个点</h1>
<div class="sub">基准 4096×4096×16384（GPT-7B mlp_linear_2）· 单卡 BW1100 · FP16 · NT 调用（<code>matmul(x, w.t())</code>）·
协议：3 次独立分配、预热 Graph、每次回放 ≥8 次调用、2×25 中位数、取最小</div>

<table><thead><tr><th>组</th><th>M×N×K</th><th>变化</th><th>库选分块</th><th>WG</th><th>核占用</th>
<th>切片 MB</th><th>切片/L2</th><th>实测 µs</th><th>模型 µs</th><th>误差</th>
<th>实测/理想</th><th>实测每步</th><th>模型每步</th></tr></thead>
<tbody>{body}</tbody></table>

<div class="note">
<ul>
<li><b>误差</b> = 模型/实测 − 1，负数表示模型比实测快。误差越大，模型越乐观。</li>
<li><b>切片</b> = 一个工作组在整个 K 循环里要读过的 A 那 bm 行（bm×K×2 字节），和 L2 的 8 MiB 比。
滑块越大，这个工作组越装不进 L2。</li>
<li><b>实测每步 / 模型每步</b> = 一个 K 步的实际周期 ÷ 理想周期，已扣掉核占用（WG 数除不尽 72 核）的影响。
两者之差就是模型缺的那部分。</li>
<li><b>实测/理想</b> = 实测时间 ÷ 纯矩阵阵列峰值所需时间。它同时含核占用损失和取数损失，
乘除一下就能从"每步低效"还原出来。</li>
</ul>
</div>
</body></html>
"""
    OUT.write_text(html, encoding='utf-8')
    print('\n%s' % OUT)


if __name__ == '__main__':
    main()
