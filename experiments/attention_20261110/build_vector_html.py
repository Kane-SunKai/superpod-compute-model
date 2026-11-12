"""Build the softmax / gelu model report.

Reads whatever is on disk and skips the sections whose data is not there yet, so
the report can be rebuilt as measurements land.

Usage: python build_vector_html.py [probe_dir] [model_dir] [out.html]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_root(start):
    for p in (start, *start.parents):
        if (p / 'superpod_compute').is_dir():
            return p
    raise SystemExit('superpod_compute not found above %s' % start)


ROOT = find_root(HERE)
PROBE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / 'bw_probe'
MODEL = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT.parent / '矢量算子模型_softmax_gelu_20261111.html'

HW = ROOT / 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
PROFILE = ROOT / 'superpod_compute/data/bw1100_vector_profile.json'


def jl(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default


def cls(e):
    if e is None:
        return 'muted'
    a = abs(e)
    return 'ok' if a <= 10 else ('warn' if a <= 20 else 'bad')


def main():
    anchors = jl(PROBE / 'vector_anchors_20261111.json', [])
    gaps = jl(PROBE / 'vector_gaps_20261111.json', [])
    fit = jl(MODEL / 'vector_fit_results.json', {})
    valid = jl(PROBE / 'vector_validation_20261111.json', {})
    profile = jl(PROFILE, {})
    aicb = jl(PROBE / 'aicb_ops_20261111.json', {})

    meas = {c['op']: c for c in aicb.get('cases', [])}
    soft = meas.get('atten_core_softmax', {})
    gelu = meas.get('mlp_gelu', {})
    total_us = 531800.0

    def card(title, value, detail):
        return (f'<div class="card"><h3>{title}</h3><div class="v">{value}</div>'
                f'<div class="d">{detail}</div></div>')

    cards = []
    if soft and gelu:
        cards.append(card('softmax 每层', f"{soft['us']:.0f} µs",
                          f"占一次前向 {100 * 36 * soft['us'] / total_us:.0f}%，此前未建模"))
        cards.append(card('gelu 每层', f"{gelu['us']:.0f} µs", '此前未建模'))
    if fit.get('rows'):
        cards.append(card('扫描形状平均误差', f"{fit['mean_abs_err_pct']:.1f}%",
                          f"{fit.get('n', 0)} 个形状，最大 {fit['max_abs_err_pct']:.1f}%"))
    vcases = valid.get('cases', [])
    if vcases:
        e = [abs(c['err_pct']) for c in vcases]
        cards.append(card('LLM 尺寸验证', f"{sum(e) / len(e):.1f}%",
                          f"{len(vcases)} 个未参与拟合的形状"))
    cards.append(card('算子覆盖率', '9 / 12', 'GEMM 7 个 + softmax、gelu'))

    # ---- anchors ----
    anchor_tbl = ''
    if anchors:
        rows = ''
        for a in anchors:
            rows += (f"<tr><td class='id'>{a['footprint']}</td><td>{a['dtype']}</td>"
                     f"<td>{a['probe']}</td><td class='num'>{a['us']:.1f}</td>"
                     f"<td class='num muted'>{a['bytes_MB']:.0f}</td>"
                     f"<td class='num'><b>{a['gbps']:.0f}</b></td></tr>")
        anchor_tbl = f"""
<h2>2. 流锚点：这块卡搬一个字节要多久</h2>
<p>要在没测过的形状上做预测，先得把搬运上限钉死。<b>在算子自己的 footprint 上重测</b>，不套用 GEMM 的旧值。</p>
<table><tr><th>footprint</th><th>dtype</th><th>探针</th><th>耗时 µs</th><th>字节 MB</th><th>GB/s</th></tr>
{rows}</table>
<div class="note"><ul>
<li><b>纯读 1842 GB/s，一读一写 1787 GB/s</b>：两者几乎相同，说明读写共用总线、总字节数决定耗时，
所以物理层用"总字节 / 流带宽"一笔即可。</li>
<li><code>copy_</code> 只有 908 GB/s —— 它走 DtoD 拷贝引擎而不是 kernel，不能当锚点。</li>
<li>流带宽 1.8 TB/s 远高于 GEMM 标定的 1.10 TB/s：<b>矩阵访问模式的达成率低，不等于卡的搬运能力低</b>。
这两个数必须分开，混用就是早前那些 −30% 误差的来源。</li>
</ul></div>"""

    # ---- model ----
    model_txt = """
<h2>3. 模型：物理层与库层分开</h2>
<p>物理层沿用矩阵模型的结构——分块、搬运算重叠、写回、占用率；库层只放"厂商库自己的选择"，
两者不混。这样换一版 DTK 只需重测库层，物理层不动。</p>

<h3>物理层</h3>
<pre><code>scans  = 1   行被归约核留在片上（warp 核，行驻留寄存器）
       = 2   行被分块流过片上缓冲，归一化前要重读（gmem 核）

elem_cycles = ops_per_element / (矢量算力/周期 · efficiency)      ← 硬件在这，换芯片就变
per_row     = row_cycles + L · elem_cycles                       ← row_cycles 是延迟，不随矢量宽度变
t_mem       = (scans + 1) · N · b / (HBM 带宽 · 流效率比)         读 scans 遍 + 写 1 遍
t_vector    = rows · per_row / (cores · clock)
spill       = rows · max(0, L·b − L1D) / (L2 读字节/周期 · clock)   第二遍超出 L1D 的部分
t           = max(t_mem, t_vector) / occupancy + spill + launch + latency

occupancy = WGs / (waves · cores)                           与矩阵模型同一项
launch    = 单次 kernel 的固定开销，测出来 3.3 µs
latency   = (α − 1) · per_row / clock   α_warp = 8，α_gmem = 1.15</code></pre>
<ul>
<li><code>row_cycles</code>：每行的固定成本——块建立与拆除、跨 lane 归约的 barrier。
它让<b>短行每元素更贵、长行更便宜</b>，这正是扫描里 R=512 比 R=16384 慢一倍的原因。</li>
<li><code>spill</code>：gmem 核要把每行读两遍。第二遍在行不超过 <b>L1D 32 KiB</b> 时命中缓存，
超出部分落到 L2，按硬件给的 3768 字节/周期计费。<b>16384 个 fp16 元素正好是 32 KiB</b>——
扫描里每元素成本的最低点就在 R=16384，越过它开始回升，这一项就是那个回升，不是拟合出来的曲率。
等价地，R=32768 与 R=65536 两个点<b>没有参与拟合</b>，是这一项的预测。</li>
<li><code>elem_cycles</code> <b>不是常数，是从硬件算出来的</b>：
<code>ops_per_element ÷（矢量算力 × efficiency）</code>。换一颗矢量宽一倍的芯片，它自动减半。
整条链路里唯一的标定量是 <code>efficiency</code>（实测 4%~10%）。
另外它<b>不按元素宽度缩放</b>：fp32 的 gelu 实测 55.2 周期/元素、fp16 是 57.0，几乎相同，
说明瓶颈是元素流水而不是字节，所以模型里没有"fp32 慢一倍"这种规则。</li>
<li><code>latency</code>：<b>低占用下的延迟放大</b>。行数少到机器填不满时，一个 block 无法躲在别的
block 后面把延迟盖住，于是单行的实际耗时是饱和时的 α 倍。实测依据是行数扫描：
固定归约长度、把行数从 16 加到 128，耗时<b>完全不变</b>（15.2 / 15.2 / 15.0 / 15.3 µs），
这正是延迟平台；越过约两千行后转为线性。α 由该平台与饱和斜率之比定出。
它对 R 缩放也是对的：32 行时 R=1024 要 8.8 µs，R=2048 要 15.5 µs。</li>
</ul>

<h3>库层：一个真实的核切换</h3>
<p>扫描时把完整 kernel 名抓了出来，切换点很干净：</p>
<table><tr><th>归约长度</th><th>kernel</th><th>行在哪里</th></tr>
<tr><td class='id'>R ≤ 2048</td><td class='id'>softmax_warp_forward&lt;…, log2(R), …&gt;</td><td>寄存器</td></tr>
<tr><td class='id'>R ≥ 4096</td><td class='id'>cunn_SoftMaxForwardGmem&lt;…&gt;</td><td>全局内存，分块重读</td></tr></table>
<p>两侧的每行、每元素成本是不同的常数，<b>没有一对参数能同时拟合两侧</b>。
所以 <code>row_capacity_elements = 2048</code> 是一个独立的库层参数，不是物理常数。</p>

<h3>达成率</h3>
<pre><code>t = t_phys / achievement[op][dtype]</code></pre>
<p>只在验证集上决定，而且<b>只有平坦残差才允许变成达成率</b>；随形状漂移的残差算结构错误，只报告、不吸收。</p>
<!--LATENCY-->"""

    lat = [c for c in gaps if c.get('group', '').startswith('softmax')]
    latency_tbl = ''
    if lat:
        by_rows = defaultdict(dict)
        for c in lat:
            by_rows[c['rows']][c['length']] = c['us']
        body = ''
        for r in sorted(by_rows):
            v = by_rows[r]
            body += (f"<tr><td class='num'>{r}</td>"
                     f"<td class='num'>{v.get(2048, float('nan')):.2f}</td>"
                     f"<td class='num'>{v.get(4096, float('nan')):.2f}</td></tr>")
        latency_tbl = f"""
<h3>低占用延迟的实测依据</h3>
<p>固定归约长度、只移动行数。两个区间在十几行到上百行之间<b>耗时完全不变</b>，
这是延迟平台，不是吞吐；越过平台后转为线性。gmem 核的平台远低于 warp 核，
这就是同一个 32 行张量在 4096 上只要 5.4 µs、在 2048 上要 15.5 µs 的原因。</p>
<table><tr><th>行数</th><th>R=2048（warp）µs</th><th>R=4096（gmem）µs</th></tr>
{body}</table>"""
    model_txt = model_txt.replace('<!--LATENCY-->', latency_tbl)

    # ---- fit table ----
    fit_tbl = ''
    fit_rows = fit.get('rows', [])
    if fit_rows:
        body = ''
        for r in sorted(fit_rows, key=lambda x: (x['kind'], x['dtype'], x['length'])):
            body += (f"<tr><td class='id'>{r['tag']}</td><td>{r['regime']}</td>"
                     f"<td class='num muted'>{r['rows']}</td><td class='num muted'>{r['length']}</td>"
                     f"<td class='num muted'>{r['t_mem_us']:.0f}</td>"
                     f"<td class='num muted'>{r['t_vector_us']:.0f}</td>"
                     f"<td class='num'><b>{r['measured_us']:.1f}</b></td>"
                     f"<td class='num'>{r['predicted_us']:.1f}</td>"
                     f"<td class='num {cls(r['err_pct'])}'>{r['err_pct']:+.1f}%</td>"
                     f"<td class='num muted'>{r['occupancy']:.3f}</td>"
                     f"<td class='sh muted'>{'拟合' if r['fitted'] else ''}"
                     f"{'分配坏点' if r.get('outlier') else ''}</td></tr>")
        fit_tbl = f"""
<h2>4. 拟合结果：全部实测形状</h2>
<p>平均误差 {fit.get('mean_abs_err_pct', 0):.1f}%，最大 {fit.get('max_abs_err_pct', 0):.1f}%（{fit.get('n_shapes', 0)} 个形状）。
其中 R=32768 与 R=65536 四个大归约长度的点<b>没有参与拟合</b>，由 L1D 溢出项直接预测。
另有两个大尺寸 gelu 点是分配位置造成的坏点，已用 67/80/100/110/120/150 Mi 的阶梯确认（见第 8 节）。</p>
<table><tr><th>case</th><th>regime</th><th>rows</th><th>length</th><th>t_mem µs</th><th>t_vector µs</th>
<th>实测</th><th>预测</th><th>误差</th><th>占用</th><th></th></tr>
{body}</table>"""

    # ---- validation ----
    valid_tbl = ''
    if vcases:
        body = ''
        for c in vcases:
            after = c.get('err_after_pct')
            body += (f"<tr><td class='id'>{c['label']}</td><td>{c['regime']}</td>"
                     f"<td class='num muted'>{c['rows']}</td><td class='num muted'>{c['length']}</td>"
                     f"<td class='num muted'>{c['dtype']}</td>"
                     f"<td class='num'><b>{c['measured_us']:.1f}</b></td>"
                     f"<td class='num'>{c['predicted_us']:.1f}</td>"
                     f"<td class='num {cls(c['err_pct'])}'>{c['err_pct']:+.1f}%</td>"
                     f"<td class='num {cls(after)}'>{('%+.1f%%' % after) if after is not None else '—'}</td>"
                     f"<td class='sh muted'>{c['why']}</td></tr>")
        e = [abs(c['err_pct']) for c in vcases]
        valid_tbl = f"""
<h2>6. 跨模型尺寸的精度验证</h2>
<p>这些形状<b>没有参与拟合</b>，覆盖了不同序列长度、头数、FFN 宽度和 bf16。
这是模型能不能拿出去用的判据。</p>
<table><tr><th>case</th><th>regime</th><th>rows</th><th>length</th><th>dtype</th>
<th>实测</th><th>预测</th><th>误差</th><th>加达成率后</th><th>对应的模型形状</th></tr>
{body}</table>
<p class="sub">mean |err| {sum(e) / len(e):.1f}% · max {max(e):.1f}%</p>"""

    # ---- residual structure ----
    structure = ''
    if fit_rows:
        groups = defaultdict(list)
        for r in fit_rows:
            if r.get('outlier') or r['fitted']:
                continue
            groups[(r['kind'], r['dtype'], r['regime'])].append(r['err_pct'])
        if groups:
            body = ''
            for key, xs in sorted(groups.items()):
                med = sorted(xs)[len(xs) // 2]
                spread = max(xs) - min(xs)
                verdict = ('点太少' if len(xs) < 3 else
                           '趋势性残差 → 记为结构误差' if spread >= 5 else '平坦且小')
                body += (f"<tr><td class='id'>{key[0]}</td><td>{key[1]}</td><td>{key[2]}</td>"
                         f"<td class='num'>{len(xs)}</td><td class='num'>{med:+.1f}%</td>"
                         f"<td class='num'>{spread:.1f}</td><td>{verdict}</td></tr>")
            structure = f"""
<h2>7. 残差结构</h2>
<p>拟合之后按算子、dtype、regime 分组看残差。这一栏决定了哪些东西可以进达成率、哪些必须留在模型外面。</p>
<table><tr><th>算子</th><th>dtype</th><th>regime</th><th>点数</th><th>中位</th><th>跨度</th><th>判读</th></tr>
{body}</table>
<div class="note">大尺寸 gelu 的分配坏点：在 134 Mi 和 180 Mi 各测到过一次 1.32 周期/元素，
而 150 / 200 / 220 / 235 / 250 / 268 Mi 全部在 0.89~1.00 之间。同缓冲区反复分配会因页对齐不同
换到不同访问路径，这类点作废，既不参与拟合也不进入误差统计。</div>"""

    # ---- vector ceiling / utilisation ----
    hardware = jl(HW, {})
    util_tbl = ''
    if profile and hardware:
        peak = (hardware.get('vector_fp16_flops_per_cycle_per_core', 0)
                * hardware.get('vector_cores_per_core', 1))
        isa = {('gelu', 'fp16'): 30, ('gelu', 'bf16'): 30, ('gelu', 'fp32'): 30,
               ('gelu-tanh', 'fp16'): 20, ('softmax', 'fp16'): 10,
               ('softmax', 'bf16'): 10, ('layernorm', 'fp16'): 12}
        body = ''
        refs = {('softmax', 'warp'): (131072, 2048), ('softmax', 'gmem'): (131072, 4096),
                ('layernorm', 'warp'): (131072, 2048), ('layernorm', 'gmem'): (4096, 16384),
                ('gelu', 'elem'): (1, 67108864), ('gelu-tanh', 'elem'): (1, 67108864)}
        cores = hardware.get('cores', 72)
        clock = hardware.get('clock_hz', 1.3e9)
        ops_map = profile.get('ops_per_element', {})
        for kind, dtypes in profile.get('efficiency', {}).items():
            for dtype, regimes in dtypes.items():
                for key, eff in regimes.items():
                    rows_, length_ = refs.get((kind, key), (1, 1))
                    rc = profile.get('row_cycles', {}).get(kind, {}).get(dtype, {}).get(key, 0.0)
                    ops = ops_map.get(kind, {}).get(dtype, 0.0)
                    elem_cycles = ops / (peak * eff) if (peak and eff) else 0.0
                    t_vec = rows_ * (rc + length_ * elem_cycles) / (cores * clock)
                    rate = rows_ * length_ / (t_vec * cores * clock) if t_vec else 0.0
                    body += (f"<tr><td class='id'>{kind}</td><td>{dtype}</td><td>{key}</td>"
                             f"<td class='num'>{ops:.0f}</td><td class='num'>{rc:.1f}</td>"
                             f"<td class='num'>{elem_cycles:.4f}</td>"
                             f"<td class='num'><b>{100 * eff:.1f}%</b></td>"
                             f"<td class='num'>{rate:.2f}</td></tr>")

        variants = jl(MODEL / 'vector_hardware_variants.json', {})
        variant_rows = ''
        for v in variants.get('rows', []):
            variant_rows += (f"<tr><td>{v['name']}</td>"
                             f"<td class='num'>{v['softmax_us']:.1f}</td>"
                             f"<td class='num'>{v['softmax_pct']:+.0f}%</td>"
                             f"<td class='num'>{v['gelu_us']:.1f}</td>"
                             f"<td class='num'>{v['gelu_pct']:+.0f}%</td>"
                             f"<td class='num'>{v['long_us']:.1f}</td>"
                             f"<td class='num'>{v['long_pct']:+.0f}%</td></tr>")
        variant_tbl = ''
        if variant_rows:
            variant_tbl = f"""
<h3>换硬件时预测怎么动</h3>
<p>同一个模型、同一个效率值，只改硬件参数。<b>没有重新拟合任何东西</b>——效率是库的性质，不随芯片变。
这正是这套模型的设计目的：在一台真机上标定，换参数预测另一台。</p>
<table><tr><th>硬件</th><th>softmax µs</th><th>相对</th><th>gelu µs</th><th>相对</th>
<th>长行 softmax µs</th><th>相对</th></tr>
{variant_rows}</table>
<div class="note"><ul>
<li><b>矢量单元加宽一倍</b> → gelu 正好减半（−50%），因为它是纯矢量受限；softmax 只 −37%，
因为核数加倍后它开始碰内存地板。</li>
<li><b>核数加倍</b> → softmax 只 −44% 而不是 −50%：<code>t_mem</code> 1790 µs 变成了新的下界，
模型标成 <code>mem</code> 受限。</li>
<li><b>时钟 1.6 GHz</b> → 两个都 −19%，正好是 1.3/1.6，因为时钟同时进内存项和计算项。</li>
<li><b>HBM 2.0 TB/s</b> 在基线上<b>没有影响</b>：这两个形状都是矢量受限，加带宽帮不上忙。
只有在"以上全部"那一行它才起作用——那时模型已经翻到内存受限。</li>
<li><b>L1D 加宽</b>只影响长行（−6%），因为 prefill 的 softmax 行只有 8 KiB，本来就装得下。</li>
</ul></div>"""

        util_tbl = f"""
<h2>5. 矢量效率：唯一被拟合的量</h2>
<p>模型里除了 <code>row_cycles</code>（每行的 block 建立与跨 lane 归约，单位是周期）之外，
唯一被拟合的就是<b>矢量效率</b>——这个 kernel 实际用到了矢量单元的百分之几。
它不是自由参数：算子需要多少运算量由定义给出（<code>ops_per_element</code>），
硬件每周期能做多少由参数表给出，两者相除再乘上实测，就是效率。</p>
<p class='sub'>矢量峰值：{hardware.get('vector_fp16_flops_per_cycle_per_core', 0):.3f} flops/cycle/vector core × {hardware.get('vector_cores_per_core', 1)} cores = <b>{peak:.2f} flops/cycle/CU</b>；
在实测运行频率 {hardware.get('clock_hz', 0) / 1e9:.2f} GHz 下整卡 {peak * hardware.get('cores', 72) * hardware.get('clock_hz', 0) / 1e12:.1f} TFLOPS。
以下的达成速率参考 softmax 131072×4096、gelu 67.1M 元素。</p>
<table><tr><th>算子</th><th>dtype</th><th>regime</th><th>ops/元素</th><th>每行周期</th>
<th>每元素周期</th><th>矢量效率</th><th>元素/CU/cycle</th></tr>
{body}</table>
<div class="note"><ul>
<li>两个 streamed kernel（gmem softmax 9.5%、gelu 9.8%）效率接近，说明它们受同一种机制限制；
warp 核只有 4.2~4.7%，因为它每个 block 只处理一行、填不满机器。</li>
<li>"ops/元素"是<b>算子定义</b>上的量，不是拟合量；"矢量效率"是拟合量。两者乘积才是实测成本，
所以单看效率那一列的数字不能脱离 ops/元素 单独解读。</li>
<li>对照 roofline：softmax 实测 3335 µs，内存地板 1790 µs（1.9 倍），矢量理想值 228 µs（14.6 倍）；
gelu 实测 640 µs，内存地板 149 µs（4.3 倍），矢量理想值 63 µs（10.2 倍）。</li>
<li><b>矢量单元对这两个算子从不构成约束</b>，所以模型里它不出现在 <code>max()</code> 的主导项里——
但它决定了"换个更好的 kernel 能快多少"。</li>
</ul></div>
{variant_tbl}"""

    limits = """
<h2>8. 已知限制</h2>
<div class="note"><ul>
<li><b>过渡形状偏慢 16~20%。</b>行数刚过核数（128~512 行）时，<code>occupancy</code> 项过于悲观：
它按整波次量化，而实际一个 CU 能同时驻留多个 block。不影响 prefill（十万行量级），只影响小 batch。</li>
<li><b>20 µs 以下的形状被 launch 常数主导。</b>3.3 µs 的固定项有 ±0.3 µs 不确定度，在 5 µs 的形状上就是 ±6%。</li>
<li><b>大缓冲区偶发分配坏点。</b>gelu 在 134 Mi 和 180 Mi 各测到过一次高出 50% 的结果，
而 150 / 200 / 220 / 235 / 250 / 268 Mi 全部正常。同缓冲区反复分配会因页对齐不同换到不同访问路径，
这类点已作废，不参与拟合。预测大尺寸时需要留意同一形状重测的离散度。</li>
<li><b>achievement 是 DTK 版本的函数。</b>换库要重测库层；物理层不受影响，这正是分层的原因。</li>
<li><b>只覆盖 AICB 的裸 torch 调用</b>（<code>softmax(dim=-1)</code>、<code>F.gelu</code> erf 版）。
融合 attention 或厂商算子不适用。</li>
<li><b>L1D 溢出项解释了大部分、但不是全部的上翘。</b>R=32768 与 R=65536 没有参与拟合，是用
"超出 32 KiB 的字节 ÷ L2 读带宽"直接算的：bf16 的 R=32768 落在 <b>+0.1%</b>，
fp16 的 R=32768 和两档 R=65536 分别偏低 7.7% / 13.4% / 7.4%。
剩余部分没有再加拟合项，作为残差报告。</li>
</ul></div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>矢量算子模型 · softmax + gelu</title>
<style>
:root {{ --ok:#1b7f3b; --okbg:#e8f5ec; --warn:#9a6b00; --warnbg:#fff6e0; --bad:#b3261e; --badbg:#fdeceb; }}
body {{ font:13px/1.6 "Segoe UI","Microsoft YaHei",sans-serif; margin:24px; color:#1b1b1b; background:#fafafa; }}
h1 {{ font-size:21px; margin:0 0 4px; }} h2 {{ font-size:15px; margin:26px 0 8px; }}
.sub {{ color:#666; margin-bottom:14px; }}
h3 {{ font-size:13px; margin:16px 0 6px; }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 8px; }}
.card {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:12px 16px; min-width:180px; }}
.card h3 {{ font-size:12px; color:#666; margin:0 0 6px; font-weight:600; }}
.card .v {{ font-size:21px; font-weight:600; }} .card .d {{ font-size:12px; color:#777; }}
table {{ border-collapse:collapse; background:#fff; width:100%; margin-bottom:6px; }}
th,td {{ border:1px solid #e0e0e0; padding:5px 8px; }}
th {{ background:#f2f4f7; font-weight:600; font-size:12px; text-align:left; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
td.id {{ white-space:nowrap; font-family:Consolas,monospace; font-size:12px; }}
td.sh {{ font-size:12px; color:#333; }}
.muted {{ color:#888; }} td.ok {{ background:var(--okbg); color:var(--ok); }}
td.warn {{ background:var(--warnbg); color:var(--warn); }} td.bad {{ background:var(--badbg); color:var(--bad); }}
.note {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:12px 16px; margin-top:10px; }}
.note li {{ margin:4px 0; }}
pre {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:10px 14px; overflow:auto; }}
code {{ background:#f0f2f5; padding:1px 5px; border-radius:3px; font-size:12px; }}
pre code {{ background:none; padding:0; font-size:12px; line-height:1.55; }}
</style></head><body>
<h1>矢量算子模型 · softmax + gelu</h1>
<div class="sub">BW1100 单卡 · nmz-sh-1 · 物理层（分块 / 重叠 / 写回 / 占用率）与库层（核切换 / 达成率）分离 ·
协议：3 次独立分配、预热 Graph、回放 ≥8 次、2×25 样本取中位、取最小</div>

<div class="cards">{''.join(cards)}</div>

<h2>1. 为什么这两个算子必须单独建模</h2>
<p>AICB 的 GPT-7B 一次前向 12 个算子，其中 softmax 一项就占
{100 * 36 * soft.get('us', 3376) / total_us:.0f}%，是最大的单项；gelu 占
{100 * 36 * gelu.get('us', 639) / total_us:.0f}%。</p>
<p>沿用 roofline（LLMCompass 的做法）会严重低估：softmax 的 roofline 是 1201 µs，实测
{soft.get('us', 3376):.0f} µs；gelu 的 roofline 是 150.6 µs，实测 {gelu.get('us', 639):.0f} µs。
<b>两者都不是带宽受限</b>，所以要建的是矢量发射率与占用率，而不是"字节 / 带宽"。</p>
{anchor_tbl}
{model_txt}
{fit_tbl}
{util_tbl}
{valid_tbl}
{structure}
{limits}

<div class="sub" style="margin-top:22px">模型 <code>superpod_compute/vector_ops.py</code> ·
库层常数 <code>superpod_compute/data/bw1100_vector_profile.json</code> ·
拟合 <code>experiments/attention_20261110/fit_vector_model.py</code> ·
实测 <code>bw_probe/measure_vector_sweep.py</code> / <code>measure_vector_validation.py</code></div>
</body></html>"""
    OUT.write_text(html, encoding='utf-8')
    print('wrote %s (%d bytes)' % (OUT, len(html)))


if __name__ == '__main__':
    main()
