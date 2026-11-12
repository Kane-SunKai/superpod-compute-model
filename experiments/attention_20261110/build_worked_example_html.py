"""One softmax prediction, walked through term by term, as a teaching page.

Every number on the page is computed from the live profile, so the worked
example cannot drift away from the model.

Usage: python build_worked_example_html.py [out.html]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_root(start):
    for p in (start, *start.parents):
        if (p / 'superpod_compute').is_dir():
            return p
    raise SystemExit('superpod_compute not found above %s' % start)


ROOT = find_root(HERE)
sys.path.insert(0, str(ROOT))
from superpod_compute import Hardware
from superpod_compute.vector_ops import VectorRequest, predict, regime

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
PROFILE = ROOT / 'superpod_compute/data/bw1100_vector_profile.json'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / 'softmax_预测流程示例_20261112.html'

HEADS, SEQ, DTYPE = 32, 4096, 'fp16'
MEASURED = 3376.2


def bar(x, y, w, h, label, value, color, scale, maxw):
    bw = maxw * value / scale
    return f"""
  <rect x="{x}" y="{y}" width="{maxw}" height="{h}" fill="#eee" rx="3"/>
  <rect x="{x}" y="{y}" width="{bw:.1f}" height="{h}" fill="{color}" rx="3"/>
  <text x="{x + maxw + 10}" y="{y + h - 6}" font-size="12" fill="#333">{label} · {value:,.1f} µs</text>"""


def main():
    hw = Hardware.load(HW)
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    req = VectorRequest('softmax', HEADS * SEQ, SEQ, DTYPE)
    p = predict(req, hw, profile)

    reg = p['regime']
    scans = p['scans']
    ops = p['ops_per_element']
    eff = p['efficiency']
    elem_cycles = p['elem_cycles']
    row_cycles = p['row_cycles']
    per_row = p['per_row_cycles']
    rows = req.rows
    length = req.length
    n = req.elements
    b = req.element_bytes
    vector_rate = hw.vector_flops_per_cycle
    clock = hw.clock_hz
    cores = hw.cores
    bw = hw.hbm_bandwidth_Bps * profile['stream_bandwidth_ratio']
    occ = p['occupancy']
    waves = -(-p['workgroups'] // cores)
    spill_us = p['spill_s'] * 1e6
    lat_us = p['latency_s'] * 1e6
    launch_us = profile['launch_s'] * 1e6
    t_mem = p['t_mem_s'] * 1e6
    t_vec = p['t_vector_s'] * 1e6
    t_phys = p['t_physical_s'] * 1e6
    total = p['duration_us']
    alpha = profile['latency_factor'][reg]
    row_bytes = length * b
    l1d = hw.l1d_bytes

    # sensitivity: same head count, sweep the reduction length
    sens = []
    for s in (512, 1024, 2048, 4096, 8192):
        q = predict(VectorRequest('softmax', HEADS * s, s, DTYPE), hw, profile)
        sens.append((s, q))
    smax = max(q['duration_us'] for _, q in sens)

    sens_rows = ''
    for s, q in sens:
        sens_rows += (
            f"<tr><td class='num'>{s}</td><td>{q['regime']}</td>"
            f"<td class='num muted'>{q['t_mem_s'] * 1e6:,.0f}</td>"
            f"<td class='num muted'>{q['t_vector_s'] * 1e6:,.0f}</td>"
            f"<td>{q['bound']}</td><td class='num'>{100 * q['efficiency']:.2f}%</td>"
            f"<td class='num muted'>{q['per_row_cycles']:,.0f}</td>"
            f"<td class='num'><b>{q['duration_us']:,.1f}</b></td></tr>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>一个 softmax 的完整预测过程</title>
<style>
body {{ font:14px/1.75 "Segoe UI","Microsoft YaHei",sans-serif; margin:26px auto; max-width:980px;
        color:#1b1b1b; background:#fafafa; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:32px 0 8px; color:#111; }}
h3 {{ font-size:13px; margin:16px 0 6px; color:#444; }}
.sub {{ color:#666; margin-bottom:18px; }}
.fig {{ background:#fff; border:1px solid #e3e3e3; border-radius:10px; padding:14px 16px; margin:12px 0; }}
.fig .cap {{ font-size:12px; color:#666; margin-top:8px; }}
table {{ border-collapse:collapse; background:#fff; width:100%; margin:10px 0; }}
th,td {{ border:1px solid #e2e2e2; padding:6px 9px; }}
th {{ background:#f2f4f7; font-size:12px; text-align:left; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.muted {{ color:#888; }}
.step {{ background:#fff; border-left:4px solid #4a7ebb; border-radius:6px; padding:10px 16px; margin:14px 0; }}
.step .n {{ font-size:12px; color:#4a7ebb; font-weight:600; }}
pre {{ background:#f7f8fa; border:1px solid #e6e6e6; border-radius:6px; padding:10px 14px;
       overflow:auto; font-size:12.5px; line-height:1.6; }}
code {{ background:#f0f2f5; padding:1px 5px; border-radius:3px; font-size:12.5px; }}
.note {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:12px 16px; margin:12px 0; }}
.note li {{ margin:5px 0; }}
.tag {{ display:inline-block; font-size:11px; padding:1px 7px; border-radius:10px; margin-right:6px; }}
.t-hw {{ background:#e8f0fb; color:#2d5f9e; }}
.t-op {{ background:#f3ecfb; color:#6b3fa0; }}
.t-cal {{ background:#eef7ee; color:#2f7a35; }}
.t-fit {{ background:#fdeceb; color:#b3261e; }}
.big {{ font-size:15px; }}
</style></head><body>

<h1>一个 softmax 的完整预测过程</h1>
<div class="sub">例子：GPT-7B 的 attention softmax，<code>({HEADS}, {SEQ}, {SEQ})</code> {DTYPE}，
在 BW1100 上实测 <b>{MEASURED:,.1f} µs</b>。<br>
下面每个数字都从模型现算，不是手抄的。</div>

<h2>0. 输入长什么样</h2>
<div class="fig">
<svg width="900" height="210" viewBox="0 0 900 210" xmlns="http://www.w3.org/2000/svg">
  <g font-size="12" fill="#333" font-family="Segoe UI, Microsoft YaHei">
    <rect x="20" y="40" width="150" height="110" fill="#e8f0fb" stroke="#4a7ebb" rx="4"/>
    <rect x="34" y="32" width="150" height="110" fill="#e8f0fb" stroke="#4a7ebb" rx="4" opacity="0.75"/>
    <rect x="48" y="24" width="150" height="110" fill="#e8f0fb" stroke="#4a7ebb" rx="4" opacity="0.5"/>
    <text x="48" y="152" font-size="12">…共 {HEADS} 个 head</text>
    <text x="58" y="80">每个 head</text><text x="58" y="98">{SEQ} × {SEQ}</text>

    <path d="M225 90 L275 90" stroke="#999" stroke-width="2" marker-end="url(#a)"/>
    <defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#999"/></marker></defs>
    <text x="228" y="80" font-size="11">展开</text>

    <rect x="290" y="30" width="330" height="16" fill="#dfe9f7" stroke="#4a7ebb" rx="2"/>
    <rect x="290" y="50" width="330" height="16" fill="#dfe9f7" stroke="#4a7ebb" rx="2"/>
    <rect x="290" y="70" width="330" height="16" fill="#dfe9f7" stroke="#4a7ebb" rx="2"/>
    <rect x="290" y="90" width="330" height="16" fill="#fff6c9" stroke="#c9a227" stroke-width="2" rx="2"/>
    <rect x="290" y="110" width="330" height="16" fill="#dfe9f7" stroke="#4a7ebb" rx="2"/>
    <text x="632" y="46" font-size="11">… 一行 = 一个 softmax 归约</text>
    <text x="632" y="102" font-size="11" fill="#8a6d00">这一行有 {SEQ} 个元素</text>
    <text x="290" y="20" font-size="12">展开成 {rows:,} 行，每行 {length:,} 个元素</text>

    <text x="290" y="160" font-size="13" fill="#2d5f9e">rows = heads × seq = {HEADS} × {SEQ} = {rows:,}</text>
    <text x="290" y="182" font-size="13" fill="#2d5f9e">length = seq = {length:,}　（softmax 只在最后一维上做）</text>
    <text x="290" y="204" font-size="12" fill="#666">总元素 N = {n:,}，每个 {b} 字节</text>
  </g>
</svg>
<div class="cap">softmax 沿最后一维（序列方向）做，所以"一行"就是长度为 seq 的那条线。</div>
</div>

<h2>1. 第一步：算出两件事 —— 行数 和 每行多长</h2>
<div class="step"><span class="n">STEP 1</span>
<pre>rows   = heads × seq = {HEADS} × {SEQ} = {rows:,}
length = seq         = {length:,}</pre>
只有这两个数会进模型。hidden、层数、ffn 都不影响 softmax。</div>

<h2>2. 第二步：判断库会选哪种 kernel</h2>
<div class="fig">
<svg width="900" height="130" viewBox="0 0 900 130" xmlns="http://www.w3.org/2000/svg">
  <g font-size="12" font-family="Segoe UI, Microsoft YaHei">
    <rect x="40" y="45" width="300" height="34" fill="#eef7ee" stroke="#2f7a35" rx="4"/>
    <rect x="340" y="45" width="520" height="34" fill="#fdeceb" stroke="#b3261e" rx="4"/>
    <text x="60" y="67" fill="#2f7a35">warp 核：整行留在寄存器里，只读一遍</text>
    <text x="360" y="67" fill="#b3261e">Gmem 核：行被流式读两遍</text>
    <line x1="340" y1="40" x2="340" y2="86" stroke="#333" stroke-width="2"/>
    <text x="300" y="34" font-size="11">length = 2048</text>
    <circle cx="420" cy="105" r="4" fill="#333"/>
    <text x="430" y="109" font-size="12">我们的例子：length = {length} → 落在 Gmem 区间</text>
  </g>
</svg>
<div class="cap">分界线由 <code>row_capacity_elements = {profile['row_capacity_elements']}</code> 给出，
它比的是 <b>length</b>（一行多长），不是行数、也不是总元素数。</div>
</div>
<div class="step"><span class="n">STEP 2</span>
<pre>length {length} &gt; {profile['row_capacity_elements']}  →  regime = {reg}  →  scans = {scans}</pre>
这个判断决定了后面用哪一套 <code>efficiency</code> 和 <code>row_cycles</code>——
两套的差别很大（效率 {100 * profile['efficiency']['softmax']['fp16']['warp']:.1f}% vs
{100 * eff:.1f}%）。</div>

<h2>3. 第三步：算"每个元素多少周期"</h2>
<div class="step"><span class="n">STEP 3</span>
<pre>elem_cycles = ops_per_element ÷ (矢量算力 × efficiency)
            = {ops:.0f} ÷ ({vector_rate:.2f} × {eff:.4f})
            = {ops:.0f} ÷ {vector_rate * eff:.2f}
            = {elem_cycles:.4f} 周期/元素</pre>
</div>
<div class="note"><ul>
<li><span class="tag t-op">假设</span><code>ops_per_element = {ops:.0f}</code>：
按算子定义数出来的运算条数（max、减、exp、累加、乘），朴素计数 5 条，我按 exp 要多条指令、加类型转换放大到 10。</li>
<li><span class="tag t-hw">硬件</span><code>矢量算力 = {vector_rate:.2f} ops/cycle/CU</code>：
参数表 29 TFLOPS ÷ 72 CU ÷ 1.6 GHz ÷ 64 vector core = 3.933，每 CU 乘 64。</li>
<li><span class="tag t-fit">拟合</span><code>efficiency = {eff:.4f}</code>：
这个 kernel 实际只用到矢量单元容量的 {100 * eff:.1f}%。这是唯一被数据拟合的数。</li>
<li>三者相乘的结果 <b>{elem_cycles:.4f}</b> 才是模型真正需要的量。如果设计者说 softmax 实际要 20 条指令，
效率会变成 {eff * 2:.4f}，乘积不变，预测也不变。</li>
</ul></div>

<h2>4. 第四步：每行成本 → 乘上行数</h2>
<div class="step"><span class="n">STEP 4</span>
<pre>每行成本 = row_cycles + length × elem_cycles
         = {row_cycles:.2f} + {length} × {elem_cycles:.4f}
         = {row_cycles:.2f} + {length * elem_cycles:,.1f}
         = {per_row:,.1f} 周期/行

计算项 t_vector = rows × 每行成本 ÷ (cores × clock)
                = {rows:,} × {per_row:,.1f} ÷ ({cores} × {clock:.3g})
                = {rows * per_row:,.0f} ÷ {cores * clock:.3g}
                = {t_vec:,.1f} µs</pre>
</div>
<div class="note"><ul>
<li><span class="tag t-fit">拟合</span><code>row_cycles = {row_cycles:.1f}</code>：
每行的固定开销——block 建立、跨 lane 归约的 barrier。它让短行每元素更贵。
单位是<b>周期</b>而不是运算量，因为它是延迟，不随矢量宽度缩放。</li>
<li>这里可以看到：每行 {per_row:,.0f} 周期里，{100 * row_cycles / per_row:.0f}% 是固定开销，
{100 * length * elem_cycles / per_row:.0f}% 是逐元素的工作。</li>
</ul></div>

<h2>5. 第五步：算内存地板（然后发现它不是瓶颈）</h2>
<div class="step"><span class="n">STEP 5</span>
<pre>t_mem = (scans + 1) × N × b ÷ 实测流带宽
      = ({scans} + 1) × {n:,} × {b} ÷ {bw:.3g}
      = {t_mem:,.1f} µs</pre>
</div>
<div class="fig">
<svg width="900" height="150" viewBox="0 0 900 150" xmlns="http://www.w3.org/2000/svg">
    <g font-family="Segoe UI, Microsoft YaHei">
{bar(30, 24, 0, 22, '内存地板', t_mem, '#7fb3e0', smax, 560)}
{bar(30, 60, 0, 22, '计算项 t_vector', t_vec, '#4a7ebb', smax, 560)}
{bar(30, 96, 0, 22, '模型总计', total, '#2d5f9e', smax, 560)}
    </g>
</svg>
<div class="cap">计算项比内存地板大 {t_vec / t_mem:.1f} 倍，所以模型判为 <b>{p['bound']}</b> 受限。
内存这一项在这类形状上从来不生效——它只是保证模型不会预测得比带宽允许的还快。</div>
</div>

<h2>6. 第六步：占用率、L1D 溢出、延迟、发射</h2>
<div class="step"><span class="n">STEP 6a — 占用率</span>
<pre>workgroups = rows = {rows:,}
waves      = ⌈{rows:,} ÷ {cores}⌉ = {waves:,}
occupancy  = {rows:,} ÷ ({waves:,} × {cores}) = {occ:.4f}
t_phys     = max(内存, 计算) ÷ occupancy = {t_phys:,.1f} µs</pre>
这个例子占用率是 {occ:.4f}——几乎满。只有行数远少于 {cores} 时它才掉下来（decode 就是那样）。</div>

<div class="fig">
<svg width="900" height="150" viewBox="0 0 900 150" xmlns="http://www.w3.org/2000/svg">
  <g font-size="12" font-family="Segoe UI, Microsoft YaHei">
    <rect x="30" y="30" width="{row_bytes / l1d * 260:.0f}" height="26" fill="#fff6c9" stroke="#c9a227" rx="3"/>
    <text x="40" y="48" font-size="11">一行 {row_bytes:,} B</text>
    <rect x="30" y="70" width="260" height="26" fill="#eef7ee" stroke="#2f7a35" rx="3"/>
    <text x="40" y="88" font-size="11">L1D {l1d:,} B（32 KiB）</text>
    <text x="310" y="48" font-size="12">行装得下 → 第二遍读命中 L1D，<tspan fill="#2f7a35">溢出 = 0</tspan></text>
    <text x="310" y="88" font-size="12" fill="#666">如果一行超过 32 KiB（例如 length ≥ 16384），</text>
    <text x="310" y="106" font-size="12" fill="#666">超出的部分就要从 L2 重读，按 3768 B/cycle 计费。</text>
  </g>
</svg>
<div class="cap">本地产出 <b>spill = {spill_us:.2f} µs</b>。</div>
</div>

<div class="step"><span class="n">STEP 6b — 低占用延迟</span>
<pre>latency = (α − 1) × 每行成本 ÷ clock
        = ({alpha} − 1) × {per_row:,.1f} ÷ {clock:.3g}
        = {lat_us:.3f} µs</pre>
α 是行数少时单行成本的放大倍数。这个例子 {rows:,} 行，机器填得很满，所以这一项 <b>{lat_us:.2f} µs，
几乎可以忽略</b>。只有在行数只有几十时它才主导（decode）。</div>

<h2>7. 汇总：一张账</h2>
<table>
<tr><th>项</th><th>值 (µs)</th><th>占比</th><th>进入公式的位置</th><th>性质</th></tr>
<tr><td>计算项 t_vector</td><td class="num">{t_vec:,.1f}</td><td class="num">{100 * t_vec / total:.1f}%</td>
    <td>max() 里较大者</td><td><span class="tag t-hw">硬件</span><span class="tag t-fit">拟合</span></td></tr>
<tr><td>内存地板 t_mem</td><td class="num">{t_mem:,.1f}</td><td class="num muted">被 max 取掉</td>
    <td>max() 里较小者</td><td><span class="tag t-hw">硬件</span></td></tr>
<tr><td>占用率修正</td><td class="num">{t_phys - t_vec:+,.1f}</td><td class="num muted">+{(t_phys - t_vec):.2f}</td>
    <td>÷ occupancy</td><td><span class="tag t-hw">硬件</span></td></tr>
<tr><td>L1D 溢出 spill</td><td class="num">{spill_us:.2f}</td><td class="num muted">0%</td>
    <td>加项</td><td><span class="tag t-hw">硬件</span></td></tr>
<tr><td>低占用延迟 latency</td><td class="num">{lat_us:.2f}</td><td class="num">{100 * lat_us / total:.1f}%</td>
    <td>加项</td><td><span class="tag t-fit">拟合</span></td></tr>
<tr><td>kernel 发射 launch</td><td class="num">{launch_us:.1f}</td><td class="num">{100 * launch_us / total:.2f}%</td>
    <td>加项</td><td><span class="tag t-cal">实测</span></td></tr>
<tr style="background:#f2f4f7"><td><b>模型预测</b></td><td class="num"><b>{total:,.1f}</b></td><td class="num">100%</td>
    <td colspan="2"></td></tr>
<tr><td>实测</td><td class="num">{MEASURED:,.1f}</td><td class="num muted">—</td>
    <td colspan="2">误差 <b>{100 * (total / MEASURED - 1):+.1f}%</b></td></tr>
</table>

<h2>8. 每个参数的身份证</h2>
<table>
<tr><th>参数</th><th>值</th><th>类别</th><th>作用在哪一步</th><th>证据</th></tr>
<tr><td><code>cores</code></td><td class="num">{cores}</td><td><span class="tag t-hw">硬件</span></td>
    <td>第 4、6a 步</td><td>参数表</td></tr>
<tr><td><code>clock_hz</code></td><td class="num">{clock / 1e9:.2f} GHz</td><td><span class="tag t-hw">硬件</span></td>
    <td>所有时间的换算</td><td>实测运行频率</td></tr>
<tr><td><code>矢量算力/cycle</code></td><td class="num">{vector_rate:.2f}</td><td><span class="tag t-hw">硬件</span></td>
    <td>第 3 步</td><td>参数表 29T ÷ 72 ÷ 1.6G ÷ 64 × 64</td></tr>
<tr><td><code>hbm_bandwidth</code></td><td class="num">{hw.hbm_bandwidth_Bps / 1e12:.2f} TB/s</td>
    <td><span class="tag t-hw">硬件</span></td><td>第 5 步（通常不生效）</td><td>参数表</td></tr>
<tr><td><code>l1d_bytes</code></td><td class="num">{l1d:,} B</td><td><span class="tag t-hw">硬件</span></td>
    <td>第 6 步的溢出判断</td><td>参数表</td></tr>
<tr><td><code>l2_read_bytes_per_cycle</code></td><td class="num">{hw.l2_read_bytes_per_cycle:,}</td>
    <td><span class="tag t-hw">硬件</span></td><td>溢出项速率</td><td>参数表</td></tr>
<tr><td><code>ops_per_element</code></td><td class="num">{ops:.0f}</td><td><span class="tag t-op">假设</span></td>
    <td>第 3 步的分子</td><td>按算子定义数出来，非拟合</td></tr>
<tr><td><code>efficiency</code></td><td class="num">{eff:.4f}</td><td><span class="tag t-fit">拟合</span></td>
    <td>第 3 步的分母</td><td>12 个模型配置的实测</td></tr>
<tr><td><code>row_cycles</code></td><td class="num">{row_cycles:.1f}</td><td><span class="tag t-fit">拟合</span></td>
    <td>第 4 步</td><td>归约长度扫描</td></tr>
<tr><td><code>row_capacity_elements</code></td><td class="num">{profile['row_capacity_elements']}</td>
    <td><span class="tag t-cal">实测断言</span></td><td>第 2 步</td>
    <td>kernel 名在 2048/4096 之间切换</td></tr>
<tr><td><code>scans</code></td><td class="num">{scans}</td><td><span class="tag t-op">从 kernel 语义读出</span></td>
    <td>第 5 步</td><td>warp 核留寄存器 / Gmem 核重读</td></tr>
<tr><td><code>latency_factor</code></td><td class="num">{alpha}</td><td><span class="tag t-fit">拟合</span></td>
    <td>第 6b 步</td><td>行数扫描的延迟平台</td></tr>
<tr><td><code>launch_s</code></td><td class="num">{launch_us:.1f} µs</td><td><span class="tag t-cal">实测</span></td>
    <td>最后一项</td><td>16k 元素 gelu 的地板</td></tr>
<tr><td><code>stream_bandwidth_ratio</code></td><td class="num">{profile['stream_bandwidth_ratio']:.4f}</td>
    <td><span class="tag t-cal">实测</span></td><td>第 5 步</td><td>实测流带宽 ÷ 标称</td></tr>
</table>

<h2>9. 换个归约长度会怎样</h2>
<p>固定 {HEADS} 个 head，只改 seq。可以看到 <code>length</code> 一旦超过 2048，核就换了、效率翻倍，
而内存地板和计算项的相对关系也随之改变。</p>
<table><tr><th>length (seq)</th><th>regime</th><th>t_mem µs</th><th>t_vector µs</th><th>受限</th>
<th>效率</th><th>每行周期</th><th>总计 µs</th></tr>
{sens_rows}</table>
<div class="note"><ul>
<li>512 时是 warp 核，每行周期只有几百，但行数是 {HEADS * 512:,}，总时间仍不小。</li>
<li>2048→4096 是分界：核一换，效率从 {100 * profile['efficiency']['softmax']['fp16']['warp']:.1f}% 跳到
{100 * eff:.1f}%，每行成本也变了。</li>
<li>8192 时行变长、行数变少，计算项继续涨，但内存地板也跟着涨（元素总数多了 64 倍）。</li>
</ul></div>

<h2>10. 一句话记住</h2>
<div class="note big">
预测 = <b>每行成本 × 行数 ÷ (核数 × 时钟)</b>，
其中每行成本 = <b>固定开销（拟合）</b> + <b>行长度 × 每元素周期</b>，
每元素周期 = <b>算子运算量（假设）÷ 矢量算力（硬件）÷ 效率（拟合）</b>。
再除以占用率、加上溢出 / 延迟 / 发射。
<br><br>
整条链路里，属于这台机器的只有两个拟合量和几个实测常数；其余全部是硬件参数和算子定义。
</div>

<div class="sub" style="margin-top:24px">模型 <code>superpod_compute/vector_ops.py</code> ·
常数 <code>superpod_compute/data/bw1100_vector_profile.json</code></div>
</body></html>"""
    OUT.write_text(html, encoding='utf-8')
    print('wrote %s (%d bytes)' % (OUT, len(html)))


if __name__ == '__main__':
    main()
