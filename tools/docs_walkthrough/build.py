from pathlib import Path
import json,html
ROOT=Path(__file__).parent
D=json.loads((ROOT/'example_data.json').read_text(encoding='utf-8'))
project=Path('D:/03_Project_Work/02_WORK_Super node/compute_model')
def esc(s):return html.escape(str(s))
def txt(x,y,s,size=16,fill='#18364d',weight=400,anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{esc(s)}</text>'
def rect(x,y,w,h,fill='#edf4f8',stroke='#c4d4df',radius=10):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'
def svgstart(w,h,label):return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}" xmlns="http://www.w3.org/2000/svg"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8Z" fill="#7892a6"/></marker></defs>'
def arrow(x,y,x2,y2):return f'<path d="M{x},{y} L{x2},{y2}" fill="none" stroke="#7892a6" stroke-width="2" marker-end="url(#arrow)"/>'

gpu=svgstart(1060,590,'一张BW1100卡的模型结构：HBM、全卡共享L2、72个CU及各自Shared与寄存器')
gpu+=rect(15,12,1030,560,'#f8fbfd')+txt(38,45,'BW1100 · 单卡建模边界',23,weight=700)
gpu+=rect(45,68,970,66,'#e4efff')+txt(530,96,'HBM：A、B 和最终 C',20,anchor='middle',weight=700)+txt(530,120,'有效读写服务 1.72 TB/s · 本地显存，不包含卡间网络',15,anchor='middle')
gpu+=arrow(530,136,530,162)
gpu+=rect(45,165,970,67,'#dcefe9')+txt(530,193,'全卡共用 L2：8 MiB',20,anchor='middle',weight=700)+txt(530,216,'输入窗口 + 可保留的 A/B 数据 + 输出小片窗口；统一服务带宽 3.887 TB/s',15,anchor='middle')
for i,x in enumerate((45,375,705)):
    gpu+=arrow(x+150,234,x+150,264)+rect(x,269,310,259,'#fff')
    gpu+=txt(x+16,294,['CU 0','CU 1','CU 71'][i],19,weight=700)
    gpu+=rect(x+15,307,280,40,'#e0f0eb')+txt(x+155,333,'Shared / LDS：64 KiB',16,anchor='middle',weight=700)
    gpu+=arrow(x+155,349,x+155,368)
    gpu+=rect(x+15,373,280,49,'#ece4f7')+txt(x+155,394,'寄存器：A/B 输入槽 + FP32 C',15,anchor='middle',weight=700)+txt(x+155,413,'聚合预算 768 KiB / CU',14,anchor='middle')
    gpu+=arrow(x+155,423,x+155,442)
    for j in range(4):
        gpu+=rect(x+15+j*70,447,64,39,'#fff0d7')+txt(x+47+j*70,472,'MMAC',12,anchor='middle',weight=700)
    gpu+=txt(x+155,511,'4 个矩阵服务单元 / CU',15,anchor='middle')
gpu+=txt(530,551,'图中仅示意 3 个 CU；实际有 72 个。Shared 与寄存器各 CU 独立，L2 全卡共享。',15,anchor='middle')+'</svg>'

flow=svgstart(1180,250,'模型接口与执行流程')
labels=[('1 输入','本卡 M/N/K + 精度','Hardware + Policy'),('2 枚举','块尺寸 / CU 分组','各层单缓冲或双缓冲'),('3 容量检查','Shared / 寄存器 / L2','不满足预算则跳过'),('4 事件调度','H → R → S → C','再做输出分片流水'),('5 选择与输出','候选中 duration 最小','分块、流量、时间轴')]
for i,(a,b,c) in enumerate(labels):
    x=12+i*235;flow+=rect(x,37,218,139,'#fff')+txt(x+17,68,a,19,weight=700)+txt(x+17,104,b,15)+txt(x+17,133,c,14)
    if i<4:flow+=arrow(x+219,107,x+232,107)
flow+=txt(20,211,'精确缓存身份：算子形状 + 硬件参数 + 策略 + 模型/搜索版本。命中后复核已存分块；首次执行搜索。',16)+'</svg>'

partition=svgstart(1170,350,'整个矩阵、K面板和单CU子片的对应关系')
partition+=txt(25,31,'从完整 GEMM 到面板，再到每个 CU 的子片',22,weight=700)
partition+=rect(25,62,300,224,'#e4efff')+txt(45,91,'A：3072 × 1024',19,weight=700)
for i in range(8):partition+=rect(44+i*32,113,28,110,'#93b8e8' if i==0 else '#cfdef1',radius=3)
partition+=txt(45,246,'沿 K 取 16 列 → Aᵢ：3072 × 16',16)+txt(45,272,'i = 0…63；图中仅示意若干面板',14)
partition+=txt(349,177,'×',32)
partition+=rect(389,62,300,224,'#e0f0eb')+txt(409,91,'B：1024 × 2304',19,weight=700)
for i in range(5):partition+=rect(409,111+i*23,257,19,'#86c4b1' if i==0 else '#c8e3da',radius=3)
partition+=txt(409,246,'沿 K 取 16 行 → Bᵢ：16 × 2304',16)+txt(409,272,'同一 i 的 Aᵢ、Bᵢ 相乘并累加',14)
partition+=txt(710,177,'→',30)
partition+=rect(757,62,387,224,'#fff0d7')+txt(777,91,'C：3072 × 2304',19,weight=700)
for r in range(12):
    for c in range(6):partition+=rect(779+c*22,110+r*10,20,8,'#e3ae52' if r==c==0 else '#f3d8aa',radius=1)
partition+=txt(931,145,'12 × 6 = 72 块',17,weight=700)+txt(931,174,'每个 CU 负责',15)+txt(931,201,'256 × 384 输出',16,weight=700)
partition+=txt(777,252,'CU(r,c)：aᵢ 256×16，bᵢ 16×384',16)+txt(777,276,'C 子块驻留寄存器，跨 64 段 K 累加',15)
partition+=txt(25,325,'定义：Aᵢ = A[:, 16i:16(i+1)]；Bᵢ = B[16i:16(i+1), :]。小写 aᵢ / bᵢ 表示分给一个 CU 的子片。',16)+'</svg>'

reusemap=svgstart(1170,335,'沿N处理两批C时，A条带在L2保留并复用')
reusemap+=txt(22,32,'把 N 扩为 4608：同一 A，依次计算左右两批 C',22,weight=700)
reusemap+=rect(25,64,260,175,'#e4efff')+txt(45,96,'A：3072 × 1024',20,weight=700)+txt(45,132,'完整 A = 6 MiB',18)+txt(45,169,'第一批逐 K 面板读入',16)+txt(45,203,'在 L2 保留至第二批消费',16)
reusemap+=arrow(290,121,340,121)
for j,x in enumerate((355,756)):
    reusemap+=rect(x,64,375,175,'#e0f0eb' if j==0 else '#fff0d7')
    reusemap+=txt(x+18,95,f'输出批 {j}：C{j}',20,weight=700)
    reusemap+=txt(x+18,132,f'A × B[:, {j*2304}:{(j+1)*2304}]',17)
    reusemap+=txt(x+18,167,'HBM：A + 左半 B' if j==0 else 'HBM：只读右半 B；A 从 L2 取',16)
    reusemap+=txt(x+18,201,'输入 10.5 MiB' if j==0 else '输入 4.5 MiB，少搬 6 MiB',17,weight=700)
reusemap+=arrow(730,152,750,152)
reusemap+=txt(24,272,'同波复用：一份 Aᵢ 向 6 列 CU 分发，一份 Bᵢ 向 12 行 CU 分发；减少 HBM 读取，但 L2→CU 副本仍计费。',16)
reusemap+=txt(24,305,'跨波复用：第二批继续使用同一 A 条带。保留空间不足时，只保留可容纳的 K 前缀；未保留的部分继续从 HBM 读。',16)+'</svg>'

guide=r'''# 单卡计算模型 · 简明 User Guide

当前实现：V3.1。代码目录：`D:\03_Project_Work\02_WORK_Super node\compute_model`。
模型在CPU上预测本卡GEMM时间，不需要GPU、PyTorch或SSH。Python 3.10+，核心仅使用标准库。

## 1. 打开项目，选择Python

Windows PowerShell：

```powershell
cd 'D:\03_Project_Work\02_WORK_Super node\compute_model'
$py = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
```

上面是本机已有Python。换一台机器可把`$py`设成自己的Python可执行文件路径。Linux进入项目目录后，把下面`& $py`换成`python3`。

## 2. 先跑固定分块，复现HTML算例（很快，不搜索）

```powershell
& $py -B -m superpod_compute evaluate `
  --m 3072 --n 2304 --k 1024 --dtype fp16 `
  --hardware superpod_compute/data/bw1100.json `
  --policy examples/bw1100_graph_calibrated.json `
  --mapping 256,384,16,12,6,1,2,4096,2,2 `
  --trace-panels 8 --output results/tutorial_fixed.json
```

预期打印`duration_us=50.208132`，1批输出；其中固定设备服务约3.121245 µs、64个K面板输入/计算流水38.780914 µs、分片输出8.305974 µs。此数值是模型估计。相同形状已有Graph实测64.709 µs，误差约−22.41%，尚未达到15%。

映射参数依次为：

| 值 | 含义 |
|---|---|
| `256,384,16` | 每CU输出块bm×bn、K面板厚度bk |
| `12,6` | 一批输出在M/N方向使用的CU数，总共72 |
| `1,2` | L2输入窗口1份、Shared输入槽2份 |
| `4096,2` | 每CU输出片4 KiB、输出缓冲2份 |
| 最后`2` | A/B操作数寄存器槽2份；与C累加寄存器分别预算 |

`--trace-panels 8`仅保存代表性输出批前8个面板的时间轴，不影响完整64面板的duration计算。JSON中`wave_classes[].pipeline.trace`包含H/R/S/C的开始和结束时间，单位秒。

## 3. 自动搜索，再次运行精确查表

```powershell
& $py -B -m superpod_compute gemm `
  --m 3072 --n 2304 --k 1024 --dtype fp16 `
  --hardware superpod_compute/data/bw1100.json `
  --policy examples/bw1100_graph_calibrated.json `
  --cache .cache/bw1100_v31.sqlite3 `
  --output results/tutorial_search.json
```

首次枚举合法候选，耗时高于固定分块；重复相同指令可见`cache_hit=True`。缓存按算子、硬件、策略及版本精确匹配；不同尺寸不会直接复用一个近似答案。`--no-cache`不读写缓存，`--force`强制重搜。结果的“最优”只指当前候选和模型内的最小值。

## 4. 换算子尺寸、硬件或模型策略

- 改GEMM：修改`--m --n --k`。它们表示`A[M,K] × B[K,N]`，应由应用层提供TP等切分后的本卡尺寸。
- 改精度：`--dtype fp16`或`bf16`；目前按相同16位服务处理，BW1100独立指令证据来自FP16。
- 改硬件：复制`superpod_compute/data/bw1100.json`再编辑，传新文件给`--hardware`。关注CU数、频率、矩阵指令形状/周期、LDS读服务、L2/HBM带宽和各层容量。A100原array后端仍保留；显式S阶段目前由BW1100的mmac配置启用。
- 改分块候选/缓冲/初始状态：复制`examples/bw1100_graph_calibrated.json`修改`bm_values`、`bn_values`、`bk_values`、`buffer_counts`、`register_buffer_counts`等，再用`--policy`指定。
- `initial_cache_state=cold`表示开始时不假设A/B已在L2；`warm-if-fit`仅在A+B与工作窗口同时放得下时允许初始命中。跨输出批复用由`input_reuse`独立控制。
- 换硬件后原校准指纹会拒绝运行。做未校准的设计探索，可以不传`--policy`，此时设备固定服务为0、默认cold；需要完整设备时间时，应重新校准，不能只清空指纹后沿用旧时延。

当前只实现GEMM，Softmax/LayerNorm仅预留扩展接口。

## 5. 跑复用示例

把固定分块命令中的`--n 2304`改为`--n 4608`，会计算两批C。当前策略下A+B不能全放入L2，所以初始仍为冷输入；模型会选择保留A的跨批复用，预期`duration_us=97.237866`、HBM输入15 MiB。

如要做严格无复用对照：复制策略文件，在新文件里设`initial_cache_state`为`cold`、`input_reuse`为`false`。用相同映射重新evaluate，预期97.295020 µs、HBM输入21 MiB。搬运减少6 MiB，但本例计算与搬运高度重叠，总时间只减少约0.057153 µs。

## 6. 看结果、跑验证

| JSON字段 | 用途 |
|---|---|
| `best.duration_us` | 搜索输出的总预测时间；固定evaluate输出直接是`duration_us` |
| `best.mapping` | 选择的分块与缓冲 |
| `capacity` | 各层容量是否可容纳；寄存器含C、A/B输入和预留 |
| `wave_classes` | 输出批、K面板服务、流水时间轴与输出分片 |
| `hbm_read_bytes` | HBM读取总量，已扣除模型允许的复用 |
| `l2_to_core_bytes` | L2向各CU提供的副本总量 |
| `device_service_s` | 显式加入的单次设备固定服务 |

```powershell
& $py -B -m unittest discover -s tests
& $py -B tools/validation_20261028/verify_installed_results.py
```

55项实现测试通过。既有60组真机回归中，规则组41/50误差≤15%，非规则组0/10；规则组平均绝对误差7.52%。非规则组当前单独记录，不作为本阶段主要验收范围；仍有9组规则案例需要改进。

`tools/validation_20261028/compare_suite.py --version v31 --workers 4`可重新搜索既有72组（12校准+60回归），耗时较长；它读取已保存真机数据，不重新启动GPU测试。上述固定服务基于Graph设备时间校准，不能直接用早期eager调用的时间判断同一口径精度。
'''
(ROOT/'计算模型_User_Guide.md').write_text(guide.replace(chr(92)*2,chr(92)),encoding='utf-8')

body=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>单卡计算建模｜总体方案与完整流水线</title><style>
:root{--ink:#17354b;--muted:#566e7d;--line:#d9e3e9;--blue:#447bbd;--teal:#228775;--purple:#8252a1;--amber:#b77720;--bg:#eff3f6}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:85px}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.8 "Segoe UI","Microsoft YaHei",sans-serif}a{color:#265f99}header{background:#132e42;color:#fff;padding:64px max(28px,calc((100vw - 1250px)/2)) 42px}header p{max-width:960px;color:#c5d7e4;margin-bottom:0}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#8db2cb;font-weight:700}h1{font-size:40px;line-height:1.3;margin:15px 0 22px;letter-spacing:-1px}h2{font-size:28px;line-height:1.4;margin:8px 0 20px}h3{font-size:19px;margin:24px 0 10px}p{margin:10px 0 16px}.nav{position:sticky;top:0;z-index:10;background:#fffffff5;border-bottom:1px solid var(--line);display:flex;justify-content:center;gap:25px;padding:13px 22px;flex-wrap:wrap;backdrop-filter:blur(8px)}.nav a{text-decoration:none;font-size:14px;font-weight:600}main{max-width:1320px;margin:auto;padding:30px 26px 70px}section{background:#fff;border:1px solid var(--line);border-radius:16px;padding:32px;margin:0 0 24px;box-shadow:0 3px 14px #14354c04}.section-no{font-size:12px;letter-spacing:1.6px;color:#4c7997;font-weight:700}.lead{font-size:18px;color:#486475}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{border:1px solid var(--line);border-radius:12px;padding:20px;background:#f8fafc}.card b{display:block;font-size:18px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}.metric{background:#edf4f8;padding:17px;border-radius:10px}.metric strong{font-size:24px;display:block;line-height:1.4}.metric span{font-size:13px;color:var(--muted)}table{width:100%;border-collapse:collapse;font-size:14px}th{text-align:left;background:#edf3f7}th,td{padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top}td:first-child{font-weight:600}code{font-family:Consolas,monospace;font-size:.9em;background:#eef3f6;border-radius:4px;padding:2px 5px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#142f43;color:#dcecf7;border-radius:10px;padding:20px;font:14px/1.8 Consolas,monospace}.formula{background:#edf3f8;border-left:4px solid #4f809e;padding:16px 20px;border-radius:0 9px 9px 0;font-size:16px;margin:16px 0}.note{border-left:4px solid #bf8932;background:#fff7e9;padding:14px 18px;margin:18px 0}.small{font-size:13px;color:var(--muted)}.diagram{overflow:auto;border:1px solid #d9e4ed;border-radius:12px;background:#fbfdff;margin:18px 0}.diagram svg{display:block;width:100%;min-width:950px;height:auto}.toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:14px 0}button,select{font:inherit;padding:8px 13px;border:1px solid #bfd0db;background:#fff;border-radius:7px;color:var(--ink);cursor:pointer}button.active{background:#1e536f;color:#fff;border-color:#1e536f}button:focus-visible,select:focus-visible{outline:3px solid #83b7e5;outline-offset:2px}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:14px}.dot{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px}.detail{padding:14px 18px;background:#f3f7fa;border-radius:9px;min-height:68px}.steps{counter-reset:steps;display:grid;gap:12px}.step{padding:15px 18px 15px 65px;background:#f6f9fb;border-radius:9px;position:relative}.step:before{counter-increment:steps;content:counter(steps);position:absolute;left:18px;top:17px;background:#234f6c;color:white;border-radius:50%;width:28px;height:28px;line-height:28px;text-align:center}.pill{display:inline-block;border:1px solid #496b83;border-radius:40px;padding:4px 12px;font-size:12px;color:#c2d7e6;margin-right:6px}.table-scroll{overflow:auto}footer{font-size:13px;color:#607789;text-align:center;padding:10px}.code-head{display:flex;justify-content:space-between;align-items:center;margin-top:14px}.source{font-family:Consolas,monospace;font-size:13px;color:#5c7385}details{margin-top:15px}summary{cursor:pointer;font-weight:600}@media(max-width:760px){header{padding:35px 22px}h1{font-size:29px}main{padding:18px 12px}section{padding:22px 16px}.grid{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}.nav{gap:8px 16px;position:relative}.nav a{font-size:12px}h2{font-size:24px}.diagram svg{min-width:1000px}th,td{padding:9px}.table-scroll table{min-width:650px}}@media print{.nav,.toolbar,.copy{display:none}header{padding:25px;background:#fff;color:#17354b}header p{color:#17354b}section{break-inside:avoid;box-shadow:none}.diagram svg{min-width:0}.diagram{overflow:visible}body{background:#fff}}
</style></head><body>
<header><div class="eyebrow">Superpod Compute · Design & Walkthrough · V3.1</div><h1>一张卡的 GEMM，<br>如何从分块走到完整时间轴</h1><div><span class="pill">BW1100 · 72 CU</span><span class="pill">FP16 / FP32 累加</span><span class="pill">代码驱动的流水图</span></div><p>面向 LLM 的单卡计算建模。输入本卡矩阵尺寸与硬件参数，枚举资源可容纳的执行计划；把数据搬运、矩阵计算和缓冲释放组织成时间轴，输出预测时延。</p></header>
<nav class="nav"><a href="#architecture">总体方案</a><a href="#hardware">硬件结构</a><a href="#mapper">分块搜索</a><a href="#example">完整算例</a><a href="#pipeline">四阶段流水</a><a href="#reuse">HBM/L2复用</a><a href="#run">怎么运行</a></nav>
<main>
<section id="architecture"><div class="section-no">01 / OVERALL DESIGN</div><h2>先明确模块边界，再组织一条完整计算链</h2><p class="lead">应用层负责给出“本卡要算什么”。计算模型负责给出“这张卡怎么分块、怎么搬、怎么重叠，以及何时完成”。</p>
<div class="grid"><div class="card"><b>输入：三个独立对象</b><p><code>GemmRequest</code>：M、N、K，输入/输出精度；连续行主序 C=AB。<br><code>Hardware</code>：CU、矩阵服务、各级容量、带宽和时钟。<br><code>Policy</code>：候选分块、缓冲数、复用策略、初始数据状态、固定服务。</p></div><div class="card"><b>输出：可解释的预测</b><p>总 duration；选中分块与缓冲；各层容量与数据流量；每批输出的 H/R/S/C 时间轴和写回过程。跨卡 AllReduce 等通信由外部调度/网络模块处理。</p></div></div>
<div class="diagram">@@FLOW@@</div><p class="source">contracts.py → api.py:predict → operators.py:GemmModel → backends.py:MmacBackend + scheduler.py:operand_pipeline</p>
<div class="note">这里描述的是当前 V3.1 实现。BW1100 已启用四阶段输入/计算流水；A100仍使用原 array 后端。两种硬件路径的细节并不完全相同。</div></section>

<section id="hardware"><div class="section-no">02 / GPU STRUCTURE</div><h2>HBM → 全卡共享 L2 → 各 CU 的 Shared → 寄存器 → 矩阵单元</h2><div class="diagram">@@GPU@@</div>
<div class="table-scroll"><table><thead><tr><th>资源</th><th>当前配置</th><th>进入模型的方式</th></tr></thead><tbody><tr><td>矩阵计算</td><td>72 CU，每CU 4单元；16×16×16 MMAC；8 cycles</td><td>每条8192 FLOP；按指令分组和最慢单元服务计时。1.30 GHz下整卡383.3856 TFLOP/s。</td></tr><tr><td>Shared / LDS</td><td>64 KiB/CU；读取64 B/CU/cycle</td><td>A/B暂存与双缓冲；读取带宽83.2 GB/s/CU，显式加入S阶段。</td></tr><tr><td>寄存器</td><td>768 KiB/CU聚合预算；预留32 KiB</td><td>C的FP32累加区 + 1/2份A/B操作数槽 + 预留；必须放得下。</td></tr><tr><td>L2</td><td>8 MiB，全卡共用；2990 B/cycle</td><td>输入窗口、输出窗口与保留数据共用容量；统一服务带宽3.887 TB/s。</td></tr><tr><td>HBM</td><td>约144 GiB；有效带宽1.72 TB/s</td><td>输入未命中/未保留的数据，以及最终FP16输出，按实际字节数计费。</td></tr></tbody></table></div><p class="small">矩阵服务和LDS读取来自独立微基准；HBM/L2参数沿用现有资料。图中的寄存器与矩阵单元连接是性能模型抽象；每线程资源分配、LDS bank冲突和完整端口竞争尚未逐指令刻画。</p></section>

<section id="mapper"><div class="section-no">03 / TILING & CAPACITY</div><h2>任意尺寸怎么切？先枚举，再检查容量，最后比较时间</h2><div class="steps"><div class="step"><b>M/N 划分独立的 C 子块。</b>每个CU负责 bm×bn 的输出；一批安排 gm×gn 个CU。不同CU的C不重叠，所以无需跨CU合并C。</div><div class="step"><b>K 分段输入，C一直累加。</b>每个面板厚度bk；K未整除时处理尾段并按矩阵指令粒度补齐计算。C保留在寄存器直到全部K完成。</div><div class="step"><b>联合枚举缓冲和输出片。</b>L2、Shared、A/B寄存器各选1或2份；输出片可选4/16 KiB，输出缓冲1或2份。多缓冲增加重叠，也占用更多容量。</div><div class="step"><b>容量通过后，模拟整个时间轴。</b>对每个合法方案计算H/R/S/C及写回流水；再比较A复用、B复用或不复用的遍历顺序，选模型时间最小者。</div></div>
<div class="formula">Shared输入 = n<sub>S</sub> × 2(bm+bn)bk 字节<br>Shared输出 = min(2bm·bn, 输出片字节数 × n<sub>O</sub>)<br>Shared峰值 = max(输入, 输出)，因为本实现将输入/计算与最终输出分阶段执行。<br>寄存器 = 4bm·bn + n<sub>V</sub> × A/B操作数寄存器字节数 + 预留<br>L2工作窗口 = n<sub>L</sub> × 2(gm·bm+gn·bn)bk + gm·gn × Shared输出</div>
<p>Shared存放“每CU一份输入子片”；不同矩阵引擎使用的A/B副本进入寄存器后需要另外预算。因此Shared输入大小与寄存器操作数大小可能不同。L2工作窗口以外的剩余容量，才允许用来保留A或B。</p>
<p class="small">默认bm/bn候选16、32、64、128、192、256、384；bk候选16、32、64、128、256、512，并加入有限的形状相关候选。搜索的是这些合法方案，不是所有可能GPU程序。缓存命中使用完整输入身份，不能把一个尺寸的最优块无条件套到另一个尺寸。</p></section>

<section id="example"><div class="section-no">04 / ONE COMPLETE GEMM</div><h2>算例：A[3072,1024] × B[1024,2304]</h2><p>沿用之前的规则矩阵。采用已归档的分块 <code>bm=256, bn=384, bk=16, gm=12, gn=6</code>；72个CU一次覆盖完整C，每个CU依次处理64个K面板。</p><div class="diagram">@@PARTITION@@</div>
<div class="metrics"><div class="metric"><strong>64</strong><span>K面板：1024÷16</span></div><div class="metric"><strong>72</strong><span>并行CU：12×6</span></div><div class="metric"><strong>496 KiB</strong><span>每CU寄存器聚合预算使用</span></div><div class="metric"><strong>50.208 µs</strong><span>此映射的完整预测时间</span></div></div>
<h3>① 每个 CU 内，如何分给4个矩阵单元</h3><p>把256×384的C再分成2×2个128×192区域，每个矩阵服务单元负责一块。每个K面板对应96条16×16×16指令/单元，因此最慢单元用 <b>96×8=768 cycles</b>，即0.590769 µs。指令完成乘加，新的乘积直接加到已有C上。</p>
<h3>② 这一组数据放在哪里，放得下吗</h3><div class="table-scroll"><table><thead><tr><th>存储位置</th><th>算式</th><th>占用 / 预算</th></tr></thead><tbody><tr><td>Shared输入槽</td><td>每份2×(256+384)×16 =20 KiB；双缓冲</td><td>40 KiB / 64 KiB</td></tr><tr><td>FP32 C寄存器</td><td>256×384×4</td><td>384 KiB，跨全部K保留</td></tr><tr><td>A/B寄存器槽</td><td>4单元 × 2×(128+192)×16 =40 KiB；双缓冲</td><td>80 KiB</td></tr><tr><td>寄存器合计</td><td>384 + 80 + 32预留</td><td>496 KiB / 768 KiB</td></tr><tr><td>L2输入/输出窗口</td><td>168 KiB输入 + 72×8 KiB输出窗口</td><td>744 KiB / 8192 KiB</td></tr></tbody></table></div>
<h3>③ 每个 K 面板的四种服务，逐项算时间</h3><div class="table-scroll"><table><thead><tr><th>阶段</th><th>本阶段数据或工作量</th><th>服务公式</th><th>时长</th></tr></thead><tbody><tr><td>H：HBM→L2</td><td>整波Aᵢ 96 KiB + Bᵢ 72 KiB =168 KiB</td><td>172032 B ÷ 1.72×10¹² B/s</td><td>0.100019 µs</td></tr><tr><td>R：L2→Shared</td><td>72 CU ×20 KiB =1440 KiB，包含分发副本</td><td>1474560 B ÷ 3.887×10¹² B/s</td><td>0.379357 µs</td></tr><tr><td>S：Shared→A/B寄存器</td><td>40 KiB/CU；72个CU各用自己的读服务</td><td>40960 B ÷ 83.2×10⁹ B/s</td><td>0.492308 µs</td></tr><tr><td>C：MMAC计算</td><td>每CU 384条指令，分给4个单元</td><td>96×8 cycles ÷ 1.30×10⁹ cycles/s</td><td>0.590769 µs</td></tr></tbody></table></div><p class="small">R按全卡L2带宽计费，S按每CU独立LDS读取服务计费；不能把两者的流量口径混用。A/B→寄存器的复制、缓冲和等待通过S阶段体现。</p></section>

<section id="pipeline"><div class="section-no">05 / ACTUAL MODEL TIMELINE</div><h2>同一面板串行，连续面板流水重叠</h2><p class="lead">A₀/B₀必须走完 H→R→S 才能计算；但计算 a₀×b₀ 时，后续面板可以分别在搬入L2、搬入Shared或搬入寄存器。</p>
<div class="legend"><span><i class="dot" style="background:#b7d4fa"></i>H：HBM读取</span><span><i class="dot" style="background:#b2ded1"></i>R：L2分发</span><span><i class="dot" style="background:#d5bde9"></i>S：Shared读入寄存器</span><span><i class="dot" style="background:#f3d49e"></i>C：矩阵乘加</span></div>
<div class="toolbar"><label for="panel">跟踪面板</label><select id="panel"></select><button id="allPanels" type="button">显示全部</button><span class="small">横轴是真实模型时间，块宽按时长绘制；窄块悬停查看完整信息。</span></div>
<div class="diagram" id="mainTimeline"></div><div class="detail" id="panelDetail" aria-live="polite"></div>
<h3>④ 缓冲什么时候能覆盖</h3><div class="grid"><div class="card"><b>L2 / Shared 输入槽</b><p>本例L2输入窗口只有1份，必须等Rᵢ结束才能读Hᵢ₊₁；Shared有2槽，偶数/奇数面板交替。Sᵢ读完后，Shared槽可用于Rᵢ₊₂。</p></div><div class="card"><b>A/B寄存器槽 / C累加区</b><p>A/B有2槽，必须等Cᵢ计算完成后，Sᵢ₊₂才能覆盖同一槽。C累加区独立保留：C←C+aᵢbᵢ，i从0到63，期间不把部分C写回L2/HBM。</p></div></div>
<div class="formula">每个活动的开始时间 = max(输入数据就绪时间，本资源空闲时间，目标缓冲槽释放时间)<br>结束时间 = 开始时间 + 本阶段服务时长。<br>重叠由这些依赖自然产生，不能把“重叠时间”再作为一个正开销相加。</div>
<h3>⑤ 全部K结束后，把C分片写出</h3><p>每CU的FP32 C为384 KiB，最终FP16输出为192 KiB。按4 KiB/片分成48片；每轮72个CU合计288 KiB。输出有两套暂存窗口，允许“这一轮写HBM”与“下一轮写L2”重叠。</p>
<div class="diagram" id="outputTimeline"></div><p>每轮写L2为0.075871 µs，写HBM为0.171460 µs。48轮输出流水合计 <b>8.305974 µs</b>。FP32→FP16转换/重排的执行服务目前没有单独校准，图中只展示已计费的L2/HBM写出；Shared输出峰值为每CU8 KiB。</p>
<h3>⑥ 把完整算子的三段拼起来</h3><div class="diagram" id="overview"></div><div class="formula">总时间 = 固定设备服务 3.121245 + 64个K面板流水 38.780914 + 输出流水 8.305974<br><b>= 50.208132 µs</b>。单个首面板H+R+S+C=1.562452 µs，后续面板重叠，因此不是64×1.562452。</div>
<div class="note">此形状的既有Graph设备实测为64.709 µs，模型低估约22.41%。保留该例是为了延续可核对的完整推导，不把它包装成已达到15%的精度示例。</div></section>

<section id="reuse"><div class="section-no">06 / HBM TRAFFIC REUSE</div><h2>L2里保留A，下一批只从HBM搬新的B</h2><p>将N从2304扩为4608，其余分块保持不变。模型按N方向依次处理两批C。两批使用同一个A；第一批读入的A面板在L2留下，第二批继续复用。</p><div class="diagram">@@REUSEMAP@@</div>
<div class="formula">L2可保留空间 = 8192 − 744 = 7448 KiB<br>完整A = 3072×1024×2 B = 6144 KiB，可以保留。<br>第二批每个面板：HBM只读新的Bᵢ =72 KiB；L2仍向CU提供aᵢ和bᵢ，所以R/S/C不消失。</div>
<div class="toolbar"><button type="button" id="reuseOn" class="active" aria-pressed="true">第二批：开启A复用</button><button type="button" id="reuseOff" aria-pressed="false">第二批：不保留A</button></div>
<div class="diagram" id="reuseTimeline"></div><div class="detail" id="reuseDetail" aria-live="polite"></div>
<h3>跨两批的完整流水与A的存活区间</h3><div class="diagram" id="reuseOverview"></div>
<div class="metrics"><div class="metric"><strong>21 → 15 MiB</strong><span>两批HBM输入总量</span></div><div class="metric"><strong>28.57%</strong><span>HBM输入字节减少比例</span></div><div class="metric"><strong>97.295 → 97.238</strong><span>两批总预测时间，µs</span></div><div class="metric"><strong>0.057 µs</strong><span>本例真正节省的时间</span></div></div>
<p><b>复用减少了搬运，并不一定同比减少总时间。</b>本例S/C阶段比H更长，大部分HBM服务原本就被计算隐藏了。复用主要缩短第二批的流水启动部分；通过切换上图可以看到H块明显变窄，但C流水节奏基本不变。</p>
<details><summary>如果L2放不下完整A，mapper怎么做？</summary><p>代码按剩余容量计算可保留K长度：<code>keepK = floor(剩余L2字节 / (2×A条带行数))</code>；未覆盖完整K时向下对齐bk。后续批在已保留前缀只搬B，超出前缀继续搬A+B，因此同一批的H块宽度会变化。对称地也可保留B、沿M遍历；模型比较各方案时间。它是容量受限的计划驻留策略，并未模拟真实L2替换算法。</p></details>
<p class="small">为忠实于当前代码，两批输出在图中顺序执行：第一批输出写回结束后才开始第二批输入。当前实现没有把不同输出批的计算/写回进一步交错。</p></section>

<section id="run"><div class="section-no">07 / QUICK START</div><h2>先复现固定分块，再启动mapper搜索</h2><p>模型在本机CPU上运行，无需连接GPU。完整说明见旁边的 <a href="计算模型_User_Guide.md">计算模型_User_Guide.md</a>。</p>
<div class="code-head"><b>1. 固定映射，快速复现本页</b><button class="copy" data-copy="cmd1">复制指令</button></div><pre id="cmd1">cd 'D:\03_Project_Work\02_WORK_Super node\compute_model'
$py = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
&amp; $py -B -m superpod_compute evaluate `
  --m 3072 --n 2304 --k 1024 --dtype fp16 `
  --hardware superpod_compute/data/bw1100.json `
  --policy examples/bw1100_graph_calibrated.json `
  --mapping 256,384,16,12,6,1,2,4096,2,2 `
  --trace-panels 8 --output results/tutorial_fixed.json</pre>
<div class="code-head"><b>2. 自动搜索；后续相同输入查表</b><button class="copy" data-copy="cmd2">复制指令</button></div><pre id="cmd2">&amp; $py -B -m superpod_compute gemm `
  --m 3072 --n 2304 --k 1024 --dtype fp16 `
  --hardware superpod_compute/data/bw1100.json `
  --policy examples/bw1100_graph_calibrated.json `
  --cache .cache/bw1100_v31.sqlite3 `
  --output results/tutorial_search.json</pre>
<p>改算子尺寸：<code>--m --n --k</code>；改硬件：<code>--hardware</code>；改候选和缓冲：<code>--policy</code>。换硬件需重新校准设备固定服务，原策略中的硬件指纹会阻止误用旧校准值。</p>
<h3>当前验证范围</h3><div class="table-scroll"><table><thead><tr><th>分组</th><th>≤15%</th><th>平均绝对误差</th><th>本阶段定位</th></tr></thead><tbody><tr><td>规则形状</td><td>41 / 50</td><td>7.52%</td><td>面向LLM的主要验证范围；仍有9组需要改进</td></tr><tr><td>非规则形状</td><td>0 / 10</td><td>61.73%</td><td>按当前约定单独记录，不作为本阶段主要验收范围</td></tr></tbody></table></div><p class="small">这里按已有实验的irregular组分类，不表示LLM实际运行绝不会遇到尾块。实现逻辑有55项测试；精度结果来自原60组回归，未宣称全新盲测。Softmax和LayerNorm尚未实现。</p>
<details><summary>代码对应表与模型边界</summary><div class="table-scroll"><table><thead><tr><th>代码</th><th>职责</th></tr></thead><tbody><tr><td>contracts.py / Hardware, Policy, Mapping</td><td>硬件、策略与映射参数及单位</td></tr><tr><td>operators.py / candidates, capacity, evaluate</td><td>枚举、容量、全卡流量、输入复用、输出分片</td></tr><tr><td>backends.py / mmac_operand_plan, MmacBackend</td><td>引擎分组、指令计数、Shared到寄存器操作数流量</td></tr><tr><td>scheduler.py / operand_pipeline</td><td>H/R/S/C资源与缓冲依赖，推出每项活动开始/结束</td></tr><tr><td>api.py / predict；cache.py</td><td>比较合法候选，精确缓存与复核</td></tr></tbody></table></div><p>本例假设不同搬运服务可按配置重叠，LDS读端口与输入写服务独立；寄存器采用聚合容量限制，未模拟每线程分配、实际rocBLAS选择的kernel和所有边界路径。输出转换/重排的额外服务仍为0。完整范围随结果JSON的scope_notes输出。</p></details></section>
</main><footer>单卡计算建模总体方案 · V3.1 · 2026-10-28<br>图中时间轴由安装后的模型 evaluate_mapping 生成；宽度按模型时间绘制。旧版HTML保留，此页为独立整理版。</footer>
<script>const DATA=@@DATA@@;
const colors={H:'#b7d4fa',R:'#b2ded1',S:'#d5bde9',C:'#f3d49e'};
const labels={H:'HBM → L2',R:'L2 → Shared',S:'Shared → A/B寄存器',C:'MMAC：C += aᵢbᵢ'};
const us=v=>(v*1e6).toFixed(6);const X=250,W=940;
function text(x,y,s,size=14,anchor='start'){return `<text x="${x}" y="${y}" font-size="${size}" fill="#18364d" text-anchor="${anchor}">${s}</text>`}
function box(x,y,w,h,fill,extra=''){return `<rect x="${x}" y="${y}" width="${Math.max(0,w)}" height="${h}" rx="5" fill="${fill}" stroke="#a5bac9" ${extra}/>`}
function canvas(label,h){return `<svg viewBox="0 0 1250 ${h}" role="img" aria-label="${label}" xmlns="http://www.w3.org/2000/svg">`}
function timeline(trace,opts={}){
 const top=58,rowH=76,end=opts.end||Math.max(...trace.map(r=>r.C_end_s)),scale=W/end;
 let s=canvas('包含Shared到寄存器的四阶段K面板流水',opts.storage?675:438);
 s+=text(24,28,opts.title||'前8个K面板 · 横轴相对本输出批输入开始（µs）',19);
 ['H','R','S','C'].forEach((key,j)=>{const y=top+j*rowH;s+=box(12,y-4,1220,65,'#f3f7fa')+text(25,y+27,labels[key],16);
 trace.forEach(r=>{const a=r[key+'_start_s'],b=r[key+'_end_s'],x=X+a*scale,w=(b-a)*scale,selected=opts.selected==null||opts.selected===r.panel,opacity=selected?1:.25;
 const detail=`${key}${r.panel} · ${opts.reused&&key==='H'?'只搬新B':'A/B面板'} · ${us(a)}–${us(b)} µs`;
 s+=`<g opacity="${opacity}">`+box(x,y,w,47,colors[key],`data-panel="${r.panel}"`)+`<title>${detail}</title>`;
 let label=key==='C'?`a${r.panel}×b${r.panel}`:key==='H'?(opts.reused?`B${r.panel}`:`${r.panel}`):`a${r.panel}/b${r.panel}`;
 s+=text(x+Math.min(w/2,8),y+29,label,w<25?10:13,w<25?'middle':'start')+'</g>';
 });});
 for(let i=0;i<=5;i++){let x=X+W*i/5;s+=`<path d="M${x},45 V355" stroke="#cad6df" stroke-dasharray="3 4"/>`+text(x,382,(end*1e6*i/5).toFixed(2),13,'middle')}
 s+=text(25,417,opts.reused?'H只搬新B；复用的A仍需经过L2→Shared→寄存器，再参与MMAC。':'H行中的编号i表示整波Aᵢ/Bᵢ；R/S/C行对应本CU的aᵢ/bᵢ。',14);
 if(opts.storage){
 const last=trace.length-1; const rows=[['Shared 槽0',0,'R','S'],['A/B寄存器 槽0',0,'S','C']];
 rows.forEach(([label,mod,start,endKey],j)=>{const y=448+j*67;s+=text(25,y+24,label,15);trace.filter(r=>r.panel%2===mod).forEach(r=>{const x=X+r[start+'_start_s']*scale,w=(r[endKey+'_end_s']-r[start+'_start_s'])*scale;s+=box(x,y,w,39,'#ece4f7')+text(x+6,y+24,`a${r.panel}/b${r.panel}`,12)})});
 s+=text(25,600,'C 累加寄存器',15)+box(X,576,W,43,'#fff0d7')+text(X+16,602,'同一份 C：累加 K[0:16] → [0:32] → … → [0:128]；之后继续到 K=1024',14);
 s+=text(25,652,'槽1承担奇数面板（未展开）。彩色槽区间含填入与消费；C累加区与A/B输入槽分别占用容量。',14);
 }return s+'</svg>';
}
const wave=DATA.main.wave_classes[0], trace=wave.pipeline.trace;
const selector=document.querySelector('#panel');selector.innerHTML=trace.map(r=>`<option value="${r.panel}">i = ${r.panel} · K[${r.panel*16}:${(r.panel+1)*16}]</option>`).join('');
function showPanel(i=null){document.querySelector('#mainTimeline').innerHTML=timeline(trace,{storage:true,selected:i});const r=trace[i??0];document.querySelector('#panelDetail').innerHTML=`<b>面板 ${r.panel} 的依赖：</b>H ${us(r.H_start_s)}–${us(r.H_end_s)} → R ${us(r.R_start_s)}–${us(r.R_end_s)} → <b>S ${us(r.S_start_s)}–${us(r.S_end_s)}</b> → C ${us(r.C_start_s)}–${us(r.C_end_s)} µs。<br>其余面板可在不同资源上同时推进；后续面板等待缓冲时，等待已体现在开始时刻。`;}
selector.addEventListener('change',()=>showPanel(Number(selector.value)));document.querySelector('#allPanels').onclick=()=>showPanel();showPanel();
function overview(){let d=DATA.main,parts=[['固定设备服务',d.device_service_s,'#dce5ec'],['64个K面板：H / R / S / C',wave.pipeline.duration_s,'#d5bde9'],['48轮C分片输出',wave.output_stream.duration_s,'#b2ded1']],s=canvas('完整单次GEMM时间分解',170),scale=1120/d.duration_s,x=45;
 s+=text(25,28,'完整算子 · 从单次设备服务到最后一个C输出片写入HBM',18);parts.forEach(([name,t,c],i)=>{let w=t*scale;s+=box(x,62,w,50,c)+text(x+w/2,138,`${name} ${us(t)} µs`,i===1?15:12,'middle');s+=text(x+w/2,92,i===0?'固定':i===1?'输入与计算重叠':'输出',14,'middle');x+=w});return s+'</svg>'}
document.querySelector('#overview').innerHTML=overview();
function outputs(){let t=wave.output_stream.pipeline.trace,end=t[t.length-1].C_end_s,scale=W/end,s=canvas('C分片输出的L2和HBM重叠',245);s+=text(24,28,'输出流水放大：前8轮，每轮72个CU各写4 KiB（共288 KiB）',19);
 [['R','输出片 → L2','#b2ded1'],['C','L2 → HBM','#b7d4fa']].forEach(([k,l,c],j)=>{let y=60+j*72;s+=text(25,y+27,l,16);t.forEach(r=>{let x=X+r[k+'_start_s']*scale,w=(r[k+'_end_s']-r[k+'_start_s'])*scale;s+=box(x,y,w,45,c)+text(x+5,y+28,`片${r.panel}`,12)+`<title>第${r.panel}轮 ${us(r[k+'_start_s'])}–${us(r[k+'_end_s'])} µs</title>`})});
 for(let i=0;i<=5;i++)s+=text(X+W*i/5,215,(end*1e6*i/5).toFixed(3),13,'middle');return s+'</svg>'}
document.querySelector('#outputTimeline').innerHTML=outputs();
function reuseOverview(on){let d=on?DATA.reuse:DATA.no_reuse,first=DATA.reuse.wave_classes[0],second=on?DATA.reuse.wave_classes[1]:DATA.no_reuse.wave_classes[0],scale=1020/DATA.no_reuse.duration_s,x=155,s=canvas('两批C顺序执行以及A的L2保留区间',310);
 s+=text(25,28,'全局时间轴（含固定服务） · 先算C左半，再算C右半',19);
 let parts=[['固定',d.device_service_s,'#dce5ec'],['批0：输入与计算',first.pipeline.duration_s,'#d5bde9'],['写C₀',first.output_stream.duration_s,'#b2ded1'],['批1：输入与计算',second.pipeline.duration_s,'#d5bde9'],['写C₁',second.output_stream.duration_s,'#b2ded1']];
 parts.forEach(([n,t,c])=>{let w=t*scale;s+=box(x,65,w,49,c)+text(x+w/2,94,n,w<90?11:15,'middle');x+=w});
 s+=text(25,96,'执行阶段',16);
 let start=d.device_service_s+first.pipeline.trace[0].H_end_s,end=d.device_service_s+first.duration_s+second.pipeline.R_end_s;
 s+=text(25,163,'A在L2保留',16);
 if(on)s+=box(155+start*scale,136,(end-start)*scale,48,'#b7d4fa')+text(180+start*scale,165,'逐面板累积 → 完整A保留6 MiB → 第二批逐面板取用',16);
 else s+=text(180,165,'关闭跨批保留：第二批重新从HBM读取A。',16);
 s+=text(25,222,`两批HBM输入 ${(d.hbm_read_bytes/2**20).toFixed(1)} MiB；总时间 ${d.duration_us.toFixed(6)} µs。`,17);
 s+=text(25,257,'条带表示A从开始保留到最后消费的范围；占用并非从起点就瞬间达到6 MiB。',14);
 s+=text(25,286,'每批内部都有H/R/S/C流水；当前代码不把第二批输入与第一批C写回重叠。',14);return s+'</svg>';
}
function setReuse(on){const w=on?DATA.reuse.wave_classes[1]:DATA.no_reuse.wave_classes[0],t=w.pipeline.trace,end=trace[trace.length-1].C_end_s;
 document.querySelector('#reuseTimeline').innerHTML=timeline(t,{reused:on,end,title:'第二批输出：前8个K面板 · 两种模式保持同一时间刻度'});
 document.querySelector('#reuseOverview').innerHTML=reuseOverview(on);
 document.querySelector('#reuseDetail').innerHTML=on?'<b>开启复用：</b>每面板H只读新B：72 KiB，0.042865 µs；A从L2取。R仍为1440 KiB/波，S仍为40 KiB/CU。第二批完整耗时47.029734 µs。':'<b>关闭复用：</b>每面板H重读A+B：168 KiB，0.100019 µs。R/S/C工作量相同。第二批完整耗时47.086888 µs。';
 ['reuseOn','reuseOff'].forEach((id,i)=>{const active=on===(i===0),b=document.getElementById(id);b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active))});}
document.querySelector('#reuseOn').onclick=()=>setReuse(true);document.querySelector('#reuseOff').onclick=()=>setReuse(false);setReuse(true);
document.querySelectorAll('[data-copy]').forEach(b=>b.onclick=async()=>{try{await navigator.clipboard.writeText(document.getElementById(b.dataset.copy).textContent);b.textContent='已复制'}catch{b.textContent='请选中下方指令复制'}setTimeout(()=>b.textContent='复制指令',2000)});
</script></body></html>'''
for key,val in [('FLOW',flow),('GPU',gpu),('PARTITION',partition),('REUSEMAP',reusemap),('DATA',json.dumps(D,ensure_ascii=False))]:body=body.replace('@@'+key+'@@',val)
(ROOT/'单卡计算建模_总体方案与流水线.html').write_text(body.replace(chr(92)*2,chr(92)),encoding='utf-8')
print('Wrote clean HTML and user guide')
