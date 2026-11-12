"""Build AICB算子实测_20261111.html from the on-machine measurement and the model."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBE = ROOT.parent / 'bw_probe'
OUT = ROOT.parent / 'AICB算子实测_GPT7B_20261111.html'

ORDER = ['Emb', 'layernorm', 'atten_qkv', 'atten_core_qk', 'atten_core_softmax',
         'atten_core_contex', 'atten_linear', 'mlp_linear_1', 'mlp_gelu',
         'mlp_linear_2', 'logit_time', 'param_time']
SHAPE = {
    'Emb': 'ids (1,4096) → 取表 (50304,4096) int64 + 位置编码 + transpose/contiguous',
    'layernorm': '(4096,1,4096) fp16 → layer_norm',
    'atten_qkv': '(4096×4096) × (4096×12288)',
    'atten_core_qk': '32 × baddbmm (4096×128)×(128×4096)，alpha=1/√128',
    'atten_core_softmax': '32 × softmax over 4096×4096',
    'atten_core_contex': '32 × bmm (4096×4096)×(4096×128)',
    'atten_linear': '(4096×4096) × (4096×4096)',
    'mlp_linear_1': '(4096×4096) × (4096×16384)',
    'mlp_gelu': '(4096,1,16384) fp16 → gelu',
    'mlp_linear_2': '(4096×16384) × (16384×4096)',
    'logit_time': '(4096×4096) × (4096×50304)',
    'param_time': '7.68 B float32 逐元素除（AICB 的梯度规约占位）',
}
GROUP = {'Emb': 'other', 'layernorm': 'ln', 'atten_qkv': 'attn', 'atten_core_qk': 'attn',
         'atten_core_softmax': 'attn', 'atten_core_contex': 'attn', 'atten_linear': 'attn',
         'mlp_linear_1': 'mlp', 'mlp_gelu': 'mlp', 'mlp_linear_2': 'mlp',
         'logit_time': 'other', 'param_time': 'other'}


def cls(e):
    if e is None:
        return 'muted'
    a = abs(e)
    return 'ok' if a <= 10 else ('warn' if a <= 20 else 'bad')


def main():
    d = json.loads((PROBE / 'aicb_ops_20261111.json').read_text(encoding='utf-8'))
    cfg = d['config']
    meas = {c['op']: c for c in d['cases']}
    model = {}
    mf = HERE / 'aicb_ops_model.json'
    if mf.exists():
        model = {r['op']: r for r in json.loads(mf.read_text(encoding='utf-8'))}

    rows = []
    for op in ORDER:
        c = meas[op]
        m = model.get(op)
        e = 100 * (m['us'] / c['us'] - 1) if m else None
        rows.append(dict(op=op, us=c['us'], mb=c['bytes_MB'], gbps=c['gbps'],
                         kernel=(c['kernels'][0][:52] if c['kernels'] else ''),
                         model=(m['us'] if m else None), err=e,
                         tile=(m['tile'] if m else ''), group=GROUP[op]))

    per_layer = sum(meas[o]['us'] for o in
                    ['layernorm', 'atten_qkv', 'atten_core_qk', 'atten_core_softmax',
                     'atten_core_contex', 'atten_linear', 'layernorm', 'mlp_linear_1',
                     'mlp_gelu', 'mlp_linear_2'])
    layers = cfg['layers']
    total = (meas['Emb']['us'] + layers * per_layer + meas['layernorm']['us']
             + meas['logit_time']['us'] + meas['param_time']['us'])
    attn = sum(meas[o]['us'] for o in ['atten_qkv', 'atten_core_qk', 'atten_core_softmax',
                                       'atten_core_contex', 'atten_linear', 'layernorm'])
    mlp = sum(meas[o]['us'] for o in ['mlp_linear_1', 'mlp_gelu', 'mlp_linear_2', 'layernorm'])
    covered = [r for r in rows if r['model']]
    body = []
    for r in rows:
        mc = f"{r['model']:.1f}" if r['model'] else '—'
        ec = f"{r['err']:+.1f}%" if r['err'] is not None else '未建模'
        body.append(
            f"<tr><td class='id'>{r['op']}</td><td class='sh'>{SHAPE[r['op']]}</td>"
            f"<td class='num'><b>{r['us']:.1f}</b></td>"
            f"<td class='num muted'>{r['mb']:.0f}</td><td class='num muted'>{r['gbps']:.0f}</td>"
            f"<td class='num'>{mc}</td><td class='num {cls(r['err'])}'>{ec}</td>"
            f"<td class='tile muted'>{r['tile']}</td>"
            f"<td class='tile muted'>{r['kernel']}</td></tr>")

    roll = [('Emb', 1, meas['Emb']['us']),
            ('Transformer 层 × %d' % layers, layers, per_layer),
            ('layernorm_post', 1, meas['layernorm']['us']),
            ('logit_time', 1, meas['logit_time']['us']),
            ('param_time', 1, meas['param_time']['us'])]
    roll_rows = ''.join(
        f"<tr><td class='id'>{n}</td><td class='num'>{k}</td><td class='num'>{u:.1f}</td>"
        f"<td class='num'>{k * u:.0f}</td><td class='num'>{100 * k * u / total:.1f}%</td></tr>"
        for n, k, u in roll)

    # --- section 5: the two GEMMs the model gets wrong, chased down ---
    var = {}
    for f in ('two_ops_layout_20261112', 'logit_variants_20261112',
              'mlp2_verify_20261112', 'half_mlp2_20261112'):
        p = PROBE / f'{f}.json'
        if p.exists():
            var[f] = json.loads(p.read_text(encoding='utf-8'))

    def lg(variant):
        for c in var.get('logit_variants_20261112', {}).get('cases', []):
            if c['variant'] == variant:
                return c['us']
        return None

    def m2(kind):
        for c in var.get('mlp2_verify_20261112', {}).get('cases', []):
            if c['kind'] == kind:
                return c['min_us']
        return None

    def lay(label):
        for c in var.get('two_ops_layout_20261112', {}).get('cases', []):
            if c['label'] == label:
                return c
        return {}

    half = var.get('half_mlp2_20261112', {})
    hmodel = HERE / 'half_mlp2_model.json'
    half_model = None
    if hmodel.exists():
        for r in json.loads(hmodel.read_text(encoding='utf-8')):
            if r['shape'] == [2048, 2048, 8192]:
                half_model = r['us']

    def err(us, base):
        """Model over measured, minus one -- the convention of section 2."""
        return 100 * (base / us - 1) if (us and base) else None

    def fmt(us):
        return f'{us:.1f}' if us else '—'

    def pct(e):
        return f'{e:+.1f}%' if e is not None else '—'

    def row(label, us, base, note, strong=False):
        e = err(us, base)
        val = f"{us:.1f}" if us else '—'
        ec = f"{e:+.1f}%" if e is not None else '—'
        cls_ = cls(e)
        return (f"<tr><td class='id'>{'<b>' + label + '</b>' if strong else label}</td>"
                f"<td class='num'>{val}</td><td class='num {cls_}'>{ec}</td>"
                f"<td class='sh'>{note}</td></tr>")

    ml2 = model.get('mlp_linear_2', {}).get('us')
    lgt = model.get('logit_time', {}).get('us')
    logit_rows = ''.join([
        row('NT 默认（现状）', lg('NT default 4096x50304'), lgt,
            'AICB/Megatron 的实际写法，SU8', True),
        row('NN（权重存成 K×N）', lay('logit_time').get('NN_us'), lgt,
            '库给它选了 128×64×16 的小分块', False),
        row('NT 转置 50304×4096', lg('NT transposed 50304x4096'), lgt,
            '同一数学问题，M/N 对调，SU2', False),
        row('NT N 切 2 段', lg('NT N split x2'), lgt, '每段 25152 列，99 个面板，SU1', False),
        row('NT N 切 4 段', lg('NT N split x4'), lgt, '每段 12576 列', False),
    ])
    ml2_rows = ''.join([
        row('NT 默认（现状）', m2('NT'), ml2, 'AICB/Megatron 的实际写法', True),
        row('NN（权重存成 K×N）', m2('NN'), ml2, '转置权重，只快 6.6%', False),
        row('NN + K 切 2 段', m2('NN-K2'), ml2, '需要额外两次 C 累加', False),
        row('NT + K 切 2 段', m2('NT-K2'), ml2, '更慢', False),
    ])
    half_note = '半尺寸，SU1，面板 4.19 MB（L2 的一半）'
    half_rows = ''
    if half.get('min_us'):
        half_rows = row('2048×2048×8192（NT）', half['min_us'], half_model, half_note, True)
    half_txt = '模型值待补'
    if half.get('min_us') and half_model:
        half_txt = '实测 %.1f µs、模型 %.1f µs（误差 %+.1f%%）' % (
            half['min_us'], half_model, 100 * (half_model / half['min_us'] - 1))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>AICB 算子实测 · GPT-7B</title>
<style>
:root {{ --ok:#1b7f3b; --okbg:#e8f5ec; --warn:#9a6b00; --warnbg:#fff6e0; --bad:#b3261e; --badbg:#fdeceb; }}
body {{ font:13px/1.6 "Segoe UI","Microsoft YaHei",sans-serif; margin:24px; color:#1b1b1b; background:#fafafa; }}
h1 {{ font-size:21px; margin:0 0 4px; }} h2 {{ font-size:15px; margin:26px 0 8px; }}
.sub {{ color:#666; margin-bottom:14px; }}
h3 {{ font-size:13px; margin:16px 0 6px; }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 8px; }}
.card {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:12px 16px; min-width:200px; }}
.card h3 {{ font-size:12px; color:#666; margin:0 0 6px; font-weight:600; }}
.card .v {{ font-size:21px; font-weight:600; }} .card .d {{ font-size:12px; color:#777; }}
table {{ border-collapse:collapse; background:#fff; width:100%; margin-bottom:6px; }}
th,td {{ border:1px solid #e0e0e0; padding:5px 8px; }}
th {{ background:#f2f4f7; font-weight:600; font-size:12px; text-align:left; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
td.id {{ white-space:nowrap; font-family:Consolas,monospace; font-size:12px; }}
td.sh {{ font-size:12px; color:#333; }}
td.tile {{ font-family:Consolas,monospace; font-size:11px; color:#888; max-width:330px; overflow:hidden; text-overflow:ellipsis; }}
.muted {{ color:#888; }} td.ok {{ background:var(--okbg); color:var(--ok); }}
td.warn {{ background:var(--warnbg); color:var(--warn); }} td.bad {{ background:var(--badbg); color:var(--bad); }}
.note {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:12px 16px; margin-top:10px; }}
.note li {{ margin:4px 0; }}
code {{ background:#f0f2f5; padding:1px 5px; border-radius:3px; font-size:12px; }}
</style></head><body>
<h1>AICB · Megatron GPT-7B 算子级实测</h1>
<div class="sub">BW1100 单卡 · 实机 nmz-sh-1 · 按 AICB <code>AiobMegatron.py</code> 的调用方式逐个复刻 ·
协议：3 次独立分配、预热 Graph、每次回放 ≥8 次调用、2×25 样本取中位数、取最小</div>

<div class="cards">
  <div class="card"><h3>一次前向合计</h3><div class="v">{total / 1000:.0f} ms</div>
    <div class="d">seq {cfg['seq']} · micro_batch {cfg['mb']} · TP 1 · {layers} 层 · 无 flash attention</div></div>
  <div class="card"><h3>每层</h3><div class="v">{per_layer / 1000:.2f} ms</div>
    <div class="d">其中 attention 部分 {attn / 1000:.2f} ms、MLP {mlp / 1000:.2f} ms</div></div>
  <div class="card"><h3>softmax 单独</h3><div class="v">{100 * layers * meas['atten_core_softmax']['us'] / total:.0f}%</div>
    <div class="d">每层 {meas['atten_core_softmax']['us']:.0f} µs，是最大的单项</div></div>
  <div class="card"><h3>已有模型的算子</h3><div class="v">{len(covered)} / {len(rows)}</div>
    <div class="d">其余靠这次实测数据补位</div></div>
</div>

<h2>1. 配置</h2>
<div class="note">
隐藏层 {cfg['H']}，{cfg['heads']} 个注意力头（head_dim {cfg['hd']}），{layers} 层，
序列长度 {cfg['seq']}，micro_batch {cfg['mb']}，词表补齐到 {cfg['vocab']}（原始 50257，按 128 对齐），
FFN {cfg['ffn']}（未开 swiglu 时的 4×hidden），未开 flash attention。
和 <code>megatron_workload_with_aiob.sh -m 7 -t 1</code> 的默认配置一致。
</div>

<h2>2. 逐算子实测</h2>
<table><thead><tr><th>算子</th><th>形状 / 调用方式</th><th>实测 µs</th><th>搬运 MB</th>
<th>有效 GB/s</th><th>模型 µs</th><th>误差</th><th>模型分块</th><th>实际 kernel</th></tr></thead>
<tbody>{''.join(body)}</tbody></table>

<h2>3. 一次前向的分解</h2>
<table><thead><tr><th>组成</th><th>次数</th><th>单次 µs</th><th>小计 µs</th><th>占比</th></tr></thead>
<tbody>{roll_rows}
<tr><td class="id"><b>合计</b></td><td class="num"></td><td class="num"></td>
<td class="num"><b>{total:.0f}</b></td><td class="num"><b>100%</b></td></tr></tbody></table>

<h2>4. 读数说明</h2>
<div class="note">
<ul>
<li><b>softmax 是最大的单项</b>：每层 {meas['atten_core_softmax']['us']:.0f} µs，{layers} 层合计
{layers * meas['atten_core_softmax']['us'] / 1000:.0f} ms，占整步的
{100 * layers * meas['atten_core_softmax']['us'] / total:.0f}%。它要读+写 32 个 4096×4096 的分数矩阵（约 2.1 GB），
有效带宽 {meas['atten_core_softmax']['gbps']:.0f} GB/s——这是**矢量单元的活**，MMAC 不参与，所以现在没有模型。</li>
<li><b>attention 的两次 bmm 都不是计算受限</b>：QK^T 1499.7 µs 对应 137 GFLOP 只有约 92 TFLOPS（峰值的 24%），
主要成本是写出 1.07 GB 分数矩阵；PV 1077.7 µs 是读回 1.07 GB。</li>
<li><b>GEMM 类算子已有模型</b>（atten_qkv / atten_linear / mlp_linear_1 / mlp_linear_2 / logit_time /
QK^T / PV 共 7 个），其余 5 个是这次实测补上的：Emb、layernorm、softmax、gelu、param_time。</li>
<li><b>两个 GEMM 偏快较多</b>：mlp_linear_2（−37.2%）和 logit_time（−29.9%）。这两格追查下来性质完全不同——
一个是库选到了差的 kernel 变体，一个是真实的模型缺口，见第 5 节。</li>
<li><b>param_time 是 AICB 的合成算子</b>：对 7.68 B 个 float32 做一次逐元素除，读+写 61 GB，39 ms。
它占了整步的 {100 * meas['param_time']['us'] / total:.0f}%，但真实框架里这一步通常和通信重叠。</li>
<li><b>micro_batch = 1</b>：所以这版数据对应的是 batch=1、seq=4096 的一次前向，
不是训练吞吐口径。换 seq 或 micro_batch 需要重测。</li>
</ul>
</div>

<h2>5. 两个偏快的 GEMM：追查结果</h2>
<div class="note" style="margin-bottom:8px">
两格的实测都用 <b>NT</b>（<code>matmul(x, w.t())</code>，与 AICB 一致，kernel 名 <code>Cijk_Alik_*</code> 吻合）。
想直接枚举最优 kernel 走不通：把枚举出的 solution index 喂回 <code>gemm_ex</code>，在 NT 和 NN 两种形式下都会
触发 VMFault + core dump。所以改用「换调用形式」逼近上界——同一数学问题换方向、换切分。
表里的误差 = 模型 / 实测 − 1，与第 2 节同口径（负数表示模型比实测快）。
</div>

<h3>5.1 logit_time（4096×50304×4096）：库的选核退化，不是模型误差</h3>
<table><thead><tr><th>写法</th><th>实测 µs</th><th>模型误差（模型/实测−1）</th><th>说明</th></tr></thead>
<tbody>{logit_rows}</tbody></table>
<div class="note">
<ul>
<li>N ≤ 32768 时每列成本稳定在 0.112 µs，到 50304 变成 0.148 µs（默认写法）。50304/256 =
<b>197 个 N 面板，197 是质数</b>，WGM 的二维分组在这个面板数上退化，L2 局部性垮掉。</li>
<li>三种写法选到的<b>宏分块完全一样</b>（256×256×32，3152 个工作组），差别只在 SplitU：
默认 <code>SU8</code>、转置 <code>SU2</code>、N 切两段 <code>SU1</code>。</li>
<li>SplitU 把输出块拆成 U 趟算，每趟都要把另一个操作数从 Shared 重读一遍。PMC 实测：
SU8 下 TA 每条 LDS 指令忙 11.9 周期、每个 K 步是理想值的 1.65 倍；SU1/SU2 下是 7.6 周期和 1.21 倍。</li>
<li>所以这一格不该记在模型账上。N 切两段（输出布局不变，可直接落地）后模型误差
{pct(err(lg('NT N split x2'), lgt))}；转置是 {pct(err(lg('NT transposed 50304x4096'), lgt))}，
但输出是 Cᵀ，下游要按列归约才免费。</li>
</ul>
</div>

<h3>5.2 mlp_linear_2（4096×4096×16384）：真实的模型缺口</h3>
<table><thead><tr><th>写法</th><th>实测 µs</th><th>模型误差（模型/实测−1）</th><th>说明</th></tr></thead>
<tbody>{ml2_rows}{half_rows}</tbody></table>
<div class="note">
<ul>
<li>换 NN、切 K 都试过，五个独立分配取最小，最好仍是 {fmt(m2('NN'))} µs。没有一个调用形式能救它。</li>
<li>它是所有测过的形状里唯一让「每工作组操作数切片」正好压在 8 MiB L2 上的：
256×16384×2 = 8.39 MB。同一个模型在别的形状上都能落在 ±7% 内。</li>
<li>把 M/N/K 同时减半后切片降到 4.19 MB，{half_txt}——误差随这个比值一起消失。</li>
<li>结论：这一格是模型缺口，但它缺的是一个「面板贴近缓存容量」的代价项。
要补应该补在「库达成率」层（类似 <code>bw1100_tile_correction.json</code>），不要写成物理定律——
目前只有一台机器、一个库的数据。</li>
</ul>
</div>
</body></html>
"""
    OUT.write_text(html, encoding='utf-8')
    print(f'{OUT}  ({len(rows)} ops, total {total / 1000:.1f} ms)')


if __name__ == '__main__':
    main()
