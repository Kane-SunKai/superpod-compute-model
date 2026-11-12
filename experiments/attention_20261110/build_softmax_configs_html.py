"""Build the softmax / vector-operator validation report.

Primary content is the Megatron model table taken from SimAI's own
`megatron_workload_with_aiob.sh`, measured on this machine. Secondary sections
keep the earlier configuration runs (GPT-7B at TP=1/2/4, a seq ladder) and the
LLM-sized shapes used when the model was first validated, so all the evidence
sits on one page.

Usage: python build_softmax_configs_html.py [probe_dir] [model_dir] [out.html]
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
PROBE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / 'bw_probe'
MODEL = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT.parent / 'softmax_模型验证_20261112.html'

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
    simai = jl(PROBE / 'simai_models_20261112.json', {})
    cfg = jl(PROBE / 'softmax_configs_20261112.json', {}).get('cases', [])
    valid = jl(PROBE / 'vector_validation_20261111.json', {}).get('cases', [])
    ops = jl(PROBE / 'model_ops_20261112.json', {}).get('cases', [])
    profile = jl(PROFILE, {})

    scases = simai.get('cases', [])
    serr = [abs(c['err_pct']) for c in scases]
    cerr = [abs(c['err_pct']) for c in cfg]

    def grp(g):
        return [c for c in ops if c['group'] == g]

    dec, gel, ln = grp('decode_softmax'), grp('gelu'), grp('layernorm')

    def meanerr(xs):
        return sum(abs(c['err_pct']) for c in xs) / len(xs) if xs else 0.0

    def maxerr(xs):
        return max(abs(c['err_pct']) for c in xs) if xs else 0.0
    drift = [abs(c['trace_vs_measured_pct']) for c in cfg if c.get('trace_vs_measured_pct') is not None]

    def card(t, v, d):
        return (f'<div class="card"><h3>{t}</h3><div class="v">{v}</div><div class="d">{d}</div></div>')

    cards = []
    if serr:
        models = sorted({c['model'] for c in scases})
        cards.append(card('softmax prefill', f"{meanerr(scases):.1f}%",
                          f"{len(models)} 个模型 × TP=1/8，最大 {maxerr(scases):.1f}%"))
    if dec:
        cards.append(card('softmax decode', f"{meanerr(dec):.1f}%",
                          f"{len(dec)} 个形状，最大 {maxerr(dec):.1f}%"))
    if gel:
        cards.append(card('gelu 多模型', f"{meanerr(gel):.1f}%",
                          f"{len(gel)} 个形状，最大 {maxerr(gel):.1f}%"))
    if ln:
        cards.append(card('layernorm 多模型', f"{meanerr(ln):.1f}%",
                          f"{len(ln)} 个形状（本轮拟合，样本内）"))
    eff = profile.get('efficiency', {}).get('softmax', {}).get('bf16', {}).get('gmem', 0)
    cards.append(card('矢量效率', f"{100 * eff:.1f}%",
                      'softmax bf16 streamed kernel，唯一标定量'))

    # --- SimAI model table ---
    sbody = ''
    for c in sorted(scases, key=lambda x: (x['model'], x['tp'])):
        sbody += (f"<tr><td class='id'>{c['model']}</td>"
                  f"<td class='num muted'>{c['layers']}</td>"
                  f"<td class='num muted'>{c['hidden']}</td>"
                  f"<td class='num muted'>{c['heads']}</td>"
                  f"<td class='num muted'>{c['tp']}</td>"
                  f"<td class='num muted'>{c['local_heads']}</td>"
                  f"<td class='num muted'>{c['rows']}</td>"
                  f"<td>{c['regime']}</td>"
                  f"<td class='num'><b>{c['measured_us']:.1f}</b></td>"
                  f"<td class='num'>{c['model_us']:.1f}</td>"
                  f"<td class='num {cls(c['err_pct'])}'>{c['err_pct']:+.1f}%</td>"
                  f"<td class='num muted'>{c['layers_total_ms']:.0f}</td></tr>")

    cbody = ''
    for c in sorted(cfg, key=lambda x: (x['dtype'], x['seq'])):
        tr = f"{c['trace_us']:.1f}" if c.get('trace_us') else '—'
        d = f"{c['trace_vs_measured_pct']:+.1f}%" if c.get('trace_vs_measured_pct') is not None else '—'
        cbody += (f"<tr><td class='id'>{c['label']}</td><td class='num muted'>{c['heads']}</td>"
                  f"<td class='num muted'>{c['seq']}</td><td class='num muted'>{c['dtype']}</td>"
                  f"<td>{c['regime']}</td>"
                  f"<td class='num'><b>{c['measured_us']:.1f}</b></td>"
                  f"<td class='num'>{c['model_us']:.1f}</td>"
                  f"<td class='num {cls(c['err_pct'])}'>{c['err_pct']:+.1f}%</td>"
                  f"<td class='num muted'>{tr}</td>"
                  f"<td class='num {cls(c['trace_vs_measured_pct'])}'>{d}</td></tr>")

    vbody = ''
    for c in valid:
        vbody += (f"<tr><td class='id'>{c['label']}</td><td>{c['kind']}</td>"
                  f"<td class='num muted'>{c['rows']}</td><td class='num muted'>{c['length']}</td>"
                  f"<td class='num muted'>{c['dtype']}</td>"
                  f"<td class='num'>{c['measured_us']:.1f}</td>"
                  f"<td class='num'>{c['predicted_us']:.1f}</td>"
                  f"<td class='num {cls(c['err_pct'])}'>{c['err_pct']:+.1f}%</td>"
                  f"<td class='sh muted'>{c['why']}</td></tr>")

    def op_table(rows_):
        out = ''
        for c in rows_:
            out += (f"<tr><td class='id'>{c['label']}</td>"
                    f"<td class='num muted'>{c['rows']}</td>"
                    f"<td class='num muted'>{c['length']}</td>"
                    f"<td class='num muted'>{c['elements']:,}</td>"
                    f"<td>{c['regime']}</td>"
                    f"<td class='num'><b>{c['measured_us']:.1f}</b></td>"
                    f"<td class='num'>{c['model_us']:.1f}</td>"
                    f"<td class='num {cls(c['err_pct'])}'>{c['err_pct']:+.1f}%</td>"
                    f"<td class='num muted'>{100 * c['efficiency']:.1f}%</td></tr>")
        return out

    dbody, gbody, lbody = op_table(dec), op_table(gel), op_table(ln)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>矢量算子模型 · SimAI 多模型验证</title>
<style>
:root {{ --ok:#1b7f3b; --okbg:#e8f5ec; --warn:#9a6b00; --warnbg:#fff6e0; --bad:#b3261e; --badbg:#fdeceb; }}
body {{ font:13px/1.6 "Segoe UI","Microsoft YaHei",sans-serif; margin:24px; color:#1b1b1b; background:#fafafa; }}
h1 {{ font-size:21px; margin:0 0 4px; }} h2 {{ font-size:15px; margin:26px 0 8px; }}
.sub {{ color:#666; margin-bottom:14px; }} h3 {{ font-size:13px; margin:16px 0 6px; }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 8px; }}
.card {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:12px 16px; min-width:190px; }}
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
code {{ background:#f0f2f5; padding:1px 5px; border-radius:3px; font-size:12px; }}
pre {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:10px 14px; }}
</style></head><body>
<h1>矢量算子模型 · SimAI 多模型验证</h1>
<div class="sub">BW1100 · nmz-sh-1 · 模型表取自 SimAI 的 <code>megatron_workload_with_aiob.sh</code>，
覆盖 softmax prefill / softmax decode / gelu / layernorm 四组算子，在本机重测（PyTorch 2.11）·
协议：3 次独立分配、预热 Graph、回放 ≥8 次、2×25 样本取中位、取最小</div>

<div class="cards">{''.join(cards)}</div>

<h2>1. 模型表来自 SimAI 源码</h2>
<div class="note">
<pre><code>model_size  layers  hidden  heads   ffn
     7        36      4096     32     -
    13        40      5120     40     -
    22        48      6144     64     -
   175        96     12288     96     -
   405       128     16384    128   53248
   moe        32      4096     32   14336</code></pre>
<ul>
<li>softmax 算子只看到 <code>(heads/TP, seq, seq)</code>——hidden、layers、ffn 都不影响它。
所以 6 个模型在 TP=1 和 TP=8 下只产生 <b>10 个不同的形状</b>（本地头数 4/5/8/12/16/32/40/64/96/128）。</li>
<li>DeepSeek 系列（deepseek671/236/16）<b>没有纳入</b>：它们用 MLA，没有独立的 softmax 算子。</li>
<li>seq 统一 4096，dtype 统一 bf16，与 SimAI 脚本的默认值一致。</li>
</ul></div>

<h2>2. 结果</h2>
<table><tr><th>模型</th><th>layers</th><th>hidden</th><th>heads</th><th>TP</th><th>本地头</th>
<th>行数</th><th>regime</th><th>实测 µs</th><th>模型 µs</th><th>误差</th><th>整层累计 ms</th></tr>
{sbody}</table>
<p class="sub">"整层累计" = 该模型的 softmax 在全部 layers 上的总时间（每层一次 softmax）。</p>

<h2>3. decode 形状的 softmax</h2>
<p>推理时序列长度为 1、KV 缓存长度不断增长，softmax 变成 <code>(local_heads, 1, kv)</code>——
<b>行数极少、归约极长</b>。这是与 prefill 完全不同的区间：机器填不满，耗时由单块延迟决定。</p>
<table><tr><th>配置</th><th>rows</th><th>kv 长度</th><th>元素数</th><th>regime</th>
<th>实测 µs</th><th>模型 µs</th><th>误差</th><th>效率</th></tr>
{dbody}</table>

<h2>4. gelu 多模型</h2>
<p>gelu 的输入是 <code>seq × ffn / TP</code> 个元素。ffn 取 Megatron 的默认值 4×hidden
（405B 和 moe 在脚本里显式指定），所以元素数从 730 万到 2.18 亿，跨 30 倍。</p>
<table><tr><th>配置</th><th>rows</th><th>length</th><th>元素数</th><th>regime</th>
<th>实测 µs</th><th>模型 µs</th><th>误差</th><th>效率</th></tr>
{gbody}</table>

<h2>5. layernorm 多模型</h2>
<p>Megatron 不对 layernorm 切 hidden 维，所以 TP 不改变这个形状，只随 hidden 变化：
<code>(seq, hidden)</code>。</p>
<table><tr><th>配置</th><th>rows</th><th>hidden</th><th>元素数</th><th>regime</th>
<th>实测 µs</th><th>模型 µs</th><th>误差</th><th>效率</th></tr>
{lbody}</table>
<div class="note"><b>这一组暴露了一个真问题。</b>最初 layernorm 直接沿用 softmax 的矢量效率（当迁移测试），
结果六个尺寸<b>一致偏慢 1.78 倍</b>——误差 −43.4% ~ −43.9%，跨度只有 0.5%。
平坦的系统性偏差说明不是噪声，而是 layernorm 的 kernel 确实比 softmax 用得更差，尤其是 bf16。
按纪律，平坦残差允许变成独立参数，所以现在 layernorm 有自己的矢量效率：
<b>bf16 约 5.5%、fp16 约 12%</b>（softmax 是 9%）。它和 softmax 仍共用同一套"每行 + 每元素"结构，
只是效率值不同——这比强行共用一套参数更诚实。</div>

<h2>6. GPT-7B 的 TP 切分与 seq 扩展组</h2>
<p>同一模型在 TP=1/2/4 下的三种切分，加上固定 32 头、只改 seq 的一组，
用来检验模型对行长和行数的响应，尤其是 <code>warp</code>/<code>gmem</code> 两个核边界。</p>
<table><tr><th>配置</th><th>heads</th><th>seq</th><th>dtype</th><th>regime</th>
<th>本机实测 µs</th><th>模型 µs</th><th>误差</th><th>旧 trace µs</th><th>漂移</th></tr>
{cbody}</table>

<h2>7. 建模时用过的 12 个 LLM 尺寸</h2>
<p>这批形状最初用来验证模型，混合了 softmax 与 gelu；gelu 用的是同一套矢量效率。</p>
<table><tr><th>case</th><th>kind</th><th>rows</th><th>length</th><th>dtype</th>
<th>实测 µs</th><th>模型 µs</th><th>误差</th><th>形状</th></tr>
{vbody}</table>

<h2>8. 怎么读这些数</h2>
<div class="note"><ul>
<li><b>模型只吃 <code>rows</code> 和 <code>length</code></b>，没有 per-model、per-TP 的参数。
6 个模型 12 个组合共用同一套常数，说明它没有对某个配置过拟合。
<b>softmax 和 gelu 的常数全部来自更早的拟合，这一批是纯留出验证；只有 layernorm 是本轮拟合的。</b></li>
<li><b>误差随 regime 不同</b>：<code>gmem</code>（seq ≥ 4096）与 <code>warp</code>（seq ≤ 2048）
用两套效率值，这是库换核的直接后果，不是额外的拟合自由度。</li>
<li><b>旧 trace 不能直接当基准</b>：它们是 PyTorch 2.10 纪录的，与本机差最多 50%。
模型贴的是当前这套库，这正是它的定位。</li>
<li><b>矢量效率 {100 * eff:.1f}%</b>（bf16 streamed kernel）：整条链路里唯一标定量，
其余全部由硬件参数（矢量算力、core 数、时钟、L1D/L2 带宽、HBM）和算子定义推出。</li>
</ul></div>

<h2>9. 限制</h2>
<div class="note"><ul>
<li><b>已知缺口：softmax 在 256~1024 行、length=2048 时高估约 22%。</b>
第 7 节的 <code>dec_256x2048_bf16</code>（+23.6%）就是它。补测后 fp16 误差 +21.8%、
bf16 +23.5%，两种 dtype 完全一致，已排除 dtype、测量和 launch 的影响。
根因是"每行成本随并发 block 数变化"这条曲线还没建，详情见 <code>docs/待解决问题.md</code>。
这是目前唯一超出 10% 的尺寸带。</li>
<li><b>layernorm 是拟合的，不是预测的。</b>6 个 bf16 点拟 2 个参数，0.3% 的误差是样本内数字；
这批数据里已经没有独立的 layernorm 留出点。它证明了结构（每行 + 每元素）可用，
但没有证明效率值可以外推。</li>
<li><b>fp16 layernorm 只有 2 个点</b>，来自建模时的扫描，参数很弱。</li>
<li><b>warp 区间仍有行数依赖没建进去</b>：seq=2048、6.5 万行时偏低 12.7%。
decode 那组虽然只有 1.4% 误差，但行数只有 4~32，起作用的是延迟放大项，不是这条规律被修好了。</li>
<li><b>没有覆盖 masked softmax。</b>TP4 上实测 908.96 µs、普通 softmax 909.60 µs，差 0.07%，
但这是单点观测，不能推广到别的长度。</li>
<li><b>DeepSeek / Qwen 系列未纳入。</b>MLA 和 MoE 结构不同，没有对应的裸 softmax/gelu 算子；
它们的算子需要单独建模。</li>
</ul></div>

<div class="sub" style="margin-top:22px">模型 <code>superpod_compute/vector_ops.py</code> ·
常数 <code>superpod_compute/data/bw1100_vector_profile.json</code> ·
实测 <code>bw_probe/measure_simai_models.py</code> / <code>measure_softmax_configs.py</code></div>
</body></html>"""
    OUT.write_text(html, encoding='utf-8')
    print('wrote %s (%d bytes)' % (OUT, len(html)))


if __name__ == '__main__':
    main()
