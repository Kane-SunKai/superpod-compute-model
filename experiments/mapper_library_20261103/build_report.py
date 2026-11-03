"""Local XLSX fallback: artifact-tool native renderer is unavailable on this host."""
import collections,json,pathlib,statistics
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name
r=pathlib.Path(__file__).resolve().parent;out=r/'outputs/mapper_library_20261103';out.mkdir(parents=True,exist_ok=True)
load=lambda x:json.loads((r/x).read_text())
a=load('analysis.json');p=load('frozen_plan.json');hw=load('hardware_checks.json');raw=load('validation_measure.json');control=load('controlled_measure.json');cp=load('controlled_predictions.json')
methods={'original':'原mapper','automatic':'校准后mapper','nearest':'最近模板','library_min':'5模板模型择优','union':'mapper+模板合并'}
groups={'decode':'Decode','prefill':'Prefill','square':'方阵','long_k':'长K','aspect':'高瘦/宽扁'}
tile=lambda t:'×'.join(map(str,t))
counts={name:sum(x['best'] is not None for x in hw['cases'] if x['variant']==name) for name in dict.fromkeys(x['variant'] for x in hw['cases'])}
baseline=a['summary']['original'];near=a['summary']['nearest'];automatic=a['summary']['automatic']
table='\n'.join(f"|{methods[k]}|{s['pass10']}/24|{s['pass15']}/24|{s['mape']:.2%}|{s['max']:.2%}|" for k,s in a['summary'].items())
group_table='\n'.join(f"|{groups[g]}|{v['original']['count']}|{v['original']['pass10']}|{v['automatic']['pass10']}|{v['nearest']['pass10']}|{v['library_min']['pass10']}|" for g,v in a['groups'].items())
bad=[x for x in a['cases'] if not x['methods']['nearest']['pass10']]
bad_table='\n'.join(f"|{tile(x['shape'])}|{x['measured_us']:.3f}|{x['methods']['nearest']['predicted_us']:.3f}|{x['methods']['nearest']['signed_error']:+.2%}|{tile(x['nearest_tile'])}|{tile(x['posthoc']['tile']) if x['posthoc'].get('tile') else '未识别/多kernel'}|" for x in bad)
ct='\n'.join(f"|{tile(x['shape'])}|{tile(x['fastest_tile'])}|{tile(x['selected_tile'])}|{x['slowdown']:.2f}×|{x['regret']:.2%}|" for x in a['controlled'])
checks=a['checks'];diag=[x['observed_tile_diagnostic'] for x in a['cases'] if x['observed_tile_diagnostic']]
diag_pass=sum(x['absolute_error']<=.1 for x in diag)
md=f'''# BW1100 分块方案库与Mapper独立验证报告

日期：2026-11-03。代码基线：20d3cc3（10.29核心模型及上一轮独立校准实验）。全部尺寸顺序为 **M、N、K**：A[M,K]×B[K,N]→C[M,N]。本轮保留MMAC 8 cycle后端，没有使用256 MAC/cycle。

## 1. 本次回答什么问题

测试“把已有软件分块记录成模板，再交给mapper检查和估时”能否改善新尺寸的预测，以及模板迁移到不同硬件配置时是否仍可行。

**记录方案可以作为候选来源，但既不能保证它是厂商库对新尺寸的选择，也不能保证它是新硬件上的最优分块。** 本轮没有替换默认模型核心、没有按验证样例添加系数，也没有修改MMAC的8周期参数。

## 2. 数据隔离与冻结

|数据|数量|本轮用途|
|---|---:|---|
|历史开发池|78个不同形状|从旧44+22+12组中提取分块，旧验收数据在本轮明确降为开发数据|
|可用历史分块记录|77条、17种不同模板|只记录M/N/K与bm/bn/bk；1条Split-K方案不导入当前单kernel模板库|
|受控开发对照|12形状×5显式分块+12库对照=72条件|检查分块敏感性和模型排序；不用于更新本轮校准参数|
|全新独立验证|24形状×2轮|6 Decode、8 Prefill、4方阵、3长K、3高瘦/宽扁；全部保留|
|硬件约束对照|5配置×17模板|纯模型容量检查，不冒充新硬件精度验证|

进一步扫描{a['audit']['files_scanned']}份历史JSON，共发现{a['audit']['known_unique_shapes']}种已知形状。初选中的1152³、1792³、2560³曾出现在更早实验中，因此**在验证测量前**替换为1216³、1344³、1408³。最终验证集与已扫描历史形状的交集为0。原始草案、替换记录和SHA256均保留。

冻结顺序：开发库与检索规则 → 全部24组预测及哈希封存 → 启动实测 → 读取时延 → 事后记录真实分块。验证样例的真实分块不输入五种正式预测，也没有根据验证结果修改距离权重、模板数量、带宽或效率。

预测文件SHA256：`{a['identity']['predictions_sha256']}`。

## 3. 方案具体怎么工作

1. **输入**：M/N/K、FP16、连续row-major布局、硬件配置和已冻结的策略。用户不需要手工填bm/bn/bk。
2. **查候选**：对开发样例计算 `d=|log₂(M/M₀)|+|log₂(N/N₀)|+|log₂(K/K₀)|`，按距离和固定ID排序，取最近的5种不同分块。分块是离散模板，不插值、不平均bm/bn/bk，也不查表返回实测时延。
3. **检查与估时**：每个模板重新枚举CU分组、Shared/L2双缓冲、输出分段和操作数缓冲；检查Shared、寄存器、L2和CU数，再用现有流水模型算duration。
4. **保留两种用途**：最近模板用于尝试模拟现有软件选择；mapper和合并候选搜索用于模型内的架构探索。它们都不保证找到真实库的最优内核。
5. **结果缓存**：精确尺寸、硬件全参数、策略、模板库、读带宽、源码摘要等共同组成key。命中后读取预测与映射；更改硬件或检索规则会失效。首次/命中功能测试分别为{checks['first_seconds']:.4f}s/{checks['cache_hit_seconds']:.4f}s（小规模功能检查，不是完整搜索速度承诺）。

|正式方法|候选/选择方式|成本模型|
|---|---|---|
|原mapper|原候选空间，模型最小耗时|效率1.0，HBM 1.72 TB/s|
|校准后mapper|同一候选空间，模型最小耗时|沿用前轮独立校准：效率0.87，HBM输入读1.923451 TB/s|
|最近模板|取最近可行历史模板；其余参数由模型选|同上|
|5模板模型择优|五种模板中取预测最快者|同上|
|mapper+模板合并|原候选空间与模板并集，取预测最快者|同上|

以上所有方法HBM输出写带宽仍为1.72 TB/s。0.87是此前开发数据标定的有效持续计算效率，不是说一条MMAC物理指令改为某个周期。本轮没有再次拟合。

## 4. 真实机器怎么测

服务器BW1100，容器sunkai_compute，PyTorch 2.11.0 / HIP 6.3.26113。使用 `torch.matmul(A,B,out=C)`、连续FP16输入、FP32累加相关精度设置沿用原脚本，预先分配A/B/C。

主验证沿用用户要求的Graph稳态口径：预热30次；大算例Graph内64次GEMM、小算例256次；预热Graph回放6次；每轮测25次Graph回放，共两轮。GPU Event包住整个Graph，除以调用次数得到每次GEMM时间，最终取两轮p50的平均值。计时不包含张量初始化、CPU提交等待和编译；同地址输入反复使用。它验证这一稳态场景，不代表所有冷缓存/轮换工作集/端到端LLM场景。

24组均通过数值检查。两轮p50相对差的中位数{a['round_drift_median']:.2f}%，最大{a['round_drift_max']:.2f}%。没有删除偏差大的性能样例。GPU验证顺序按固定种子打乱，预测搜索只在本机CPU执行。全部服务器新代码、结果、临时文件、日志和缓存均在 `/root/sk/mapper_library_20261103`，未修改驱动、时钟或其他目录。

受控Triton对照为10次预热、Graph内16次、回放预热3次、2轮×10个Event样本；固定4 warps、num_stages=1，改变bm/bn/bk。这只是开发诊断，不能与主验证口径或厂商内核实现混为一谈。

## 5. 独立验证结果

误差=|预测−实测|/实测。下表分母始终为24。

|方法|≤10%|≤15%|平均绝对相对误差|最大绝对相对误差|
|---|---:|---:|---:|---:|
{table}

|类别|数量|原mapper≤10%|校准mapper≤10%|最近模板≤10%|5模板择优≤10%|
|---|---:|---:|---:|---:|---:|
{group_table}

**本轮没有达到24组全部误差≤10%的目标，不能将模板库作为已验证成功的默认替代方案。** 原mapper、校准mapper与模板选择分开列出，避免把参数变化的收益算成模板库的收益。合并候选只保证模型最小值不增，并不保证更接近真实库；如果原成本模型乐观，新增候选反而可能扩大偏快问题。

本次最值得关注的是：最近模板将通过数从8提高到14，**Prefill为8/8通过、平均误差3.69%**，但整体平均误差从16.29%变成16.30%，几乎没有改善，最大误差却由33.92%升到60.62%。这是“更多样例接近、少数样例更差”，不是整体精度已经收敛。五模板择优将平均误差降到13.49%，但只有11/24通过；合并候选在24组上全部回到校准mapper原来的选择，未获得额外收益。

最近模板仍超出10%的完整列表：

|M×N×K|实测μs|预测μs|有符号误差|最近模板bm×bn×bk|事后实际模板|
|---|---:|---:|---:|---|---|
{bad_table}

## 6. 分块查找和成本估算分别出了什么问题

事后粗粒度执行记录中，{a['actual_supported']}/24组可识别为当前支持的单kernel、Split-K=1路径。最近模板命中实际bm/bn/bk的有{a['nearest_hits']}组，5候选覆盖实际分块的有{a['top5_hits']}组。**相近尺寸不一定选同一分块；候选中有真实分块，也不代表按模型耗时择优就会选中它。**

另外做了一个事后诊断：固定每个验证算例观测到的真实bm/bn/bk，仅重算其余映射。结果为{diag_pass}/{len(diag)}组误差≤10%。这使用了验证样例执行信息，**不是正式预测成绩**；只是帮助区分“分块选择不匹配”与“同粗粒度分块下的成本/调度仍不匹配”。Excel的“分块选择”页单列该诊断，未加入五种正式方法排名。

本模型仍按总寄存器预算检查，而非复现编译器逐线程分配；CU任务波次、操作数复用和缓存保留是模型规则。真实库的调度与这些规则可能不同。因此不能把全部残差归因于HBM带宽或MMAC周期，也不能通过查一份分块表就宣布解决全部差异。

三个具体例子可以把问题分开：

- **候选择优不等于模拟库选择**：1024×5120×20480的实测为1138.950 μs。最近模板256×256×32与实际分块一致，预测1171.600 μs，误差2.87%；五模板择优却选择384×256×16，预测879.578 μs，偏快22.77%。同一成本模型、同一输入，主要变化就是选了哪个模板。
- **近邻规则未必能迁移到长K**：640×1024×12288的最近模板为96×64×64，实际为96×128×32；最近模板预测194.059 μs、实测120.821 μs，偏慢60.62%。而真实模板不在该样例的五个近邻候选中。这说明应改进候选覆盖/选择规则，不能直接认为MMAC的8 cycle又错了。
- **分块匹配后仍有残差**：1×10240×5120的查表分块和实际均为16×128×32，但预测76.218 μs、实测66.786 μs，仍偏慢14.12%。只记录粗粒度输出块和K步长，尚未完全描述真实的数据服务与任务调度。

验证两轮差异最大仅0.172%，比这些14%～61%的误差小得多；本次结果不能简单归因于两轮测量抖动。采集的173个时钟状态样本包括1300 MHz以及空闲/间隙的较低档位，并未锁定时钟，因此也不宣称每个kernel全程处于完全相同的频率。

## 7. 受控分块实验及硬件迁移

五个显式候选为16×128×32、64×128×32、128×128×32、128×256×32、128×128×64。12形状的60个自写内核条件、12个厂商库对照全部通过数值检查。

|M×N×K|实测最快分块|模型选择分块|最慢/最快实测比|选中方案比实测最快慢多少|
|---|---|---|---:|---:|
{ct}

模型选中实测最快候选{a['controlled_top1']}/12组，平均选择损失{a['controlled_regret']:.2%}。自写Triton实现普遍慢于厂商库，因此这里说明分块确实影响性能，也暴露模型排序限制；不能把其时延作为厂商库GEMM的标定值，更没有用它重新拟合参数。

硬件约束测试固定M/N/K=1024/8192/8192，仅修改硬件约束，重新枚举模板内映射：原配置17/17可行；Shared降到16 KiB后{counts['shared_16KiB']}/17可行；寄存器降到128 KiB后{counts['registers_128KiB']}/17可行。CU降到36、L2降到512 KiB仍有17种模板可通过调整分组/缓冲找到可行映射，但分组、时延可能改变。这里是模型约束敏感性检查，并无这些新硬件的实测精度证据。

所以，新硬件可以复用“候选模板和搜索代码”，不能原封不动继承“可行性、最优性、BW1100标定效率和固定开销”。实验入口发现硬件不匹配时停用BW1100专属校准，使用新硬件带宽、效率1与无校准device service重新计算。

## 8. 本轮交付、复现和下一步

核心分块/双缓冲/搬运流水代码未改。本轮新增实验入口 `experiments/mapper_library_20261103/mapper.py`，输出最近模板、五模板择优、纯mapper和合并方案，缓存与资源检查已验证；已有62项单元测试通过。

本机执行（使用安装本项目依赖的Python）：

```powershell
cd 'D:\\01codex_work\\03_super node\\compute_model'
python experiments/mapper_library_20261103/mapper.py --M 1 --N 10240 --K 5120 --output results/mapper_demo.json
# 第二次同样调用会命中缓存。不同硬件可加 --hardware 路径/硬件.json
```

服务器复测应保留本轮原始结果，复制脚本/冻结清单到 `/root/sk` 下的新实验目录，再使用同样的运行环境重定向。`measure_graph.py --plan frozen_plan.json --output 新结果.json` 会拒绝覆盖已有文件。

下一步建议：保留模板库做可选候选源；针对已暴露的选择失配检查少量有机制含义的尺寸/布局条件；对粗分块已匹配而仍超差的样例检查波次与搬运调度。若据本轮结果再改规则，这24组必须转为开发数据，再另设新的验证集。不能反复改到这24组全过，再称之为独立验证。

本轮所有详细时延、误差、模板来源、原始Event样本与资源对照见同名Excel；原始JSON、预测封存与运行脚本保存在本机实验目录。没有上传厂商二进制或反汇编，也没有自动推送GitHub。
'''
(out/'BW1100_分块方案库与Mapper独立验证_20261103.md').write_text(md,encoding='utf-8')

book=xlsxwriter.Workbook(out/'BW1100_分块方案库与Mapper独立验证_20261103.xlsx')
base={'font_name':'Microsoft YaHei','font_size':10,'valign':'vcenter'}
fmt={k:book.add_format({**base,**v}) for k,v in {
 'text':{},'num':{'num_format':'0.000'},'int':{'num_format':'0'},'pct':{'num_format':'0.00%'},
 'header':{'bold':True,'font_color':'white','bg_color':'#193449','text_wrap':True},
 'title':{'font_size':16,'bold':True,'font_color':'#193449'},'note':{'font_color':'#526779','text_wrap':True},
 'bad':{'bg_color':'#FCE5E4','font_color':'#9C2222'},'amber':{'bg_color':'#FFF2CC'},
 'section':{'bold':True,'bg_color':'#E9F1F6','font_color':'#193449'},'wrap':{'text_wrap':True},
 'na':{'align':'right','font_color':'#526779'},'center':{'align':'center'}}.items()}
def sheet(name,title,note,headers,widths):
 s=book.add_worksheet(name);s.hide_gridlines(2);s.set_zoom(85);s.set_default_row(23);s.set_row(0,30);s.set_row(1,36);s.set_row(3,34)
 s.write(0,0,title,fmt['title']);s.merge_range(1,0,1,len(headers)-1,note,fmt['note'])
 for j,(h,w) in enumerate(zip(headers,widths)):s.set_column(j,j,w,fmt['text']);s.write(3,j,h,fmt['header'])
 s.freeze_panes(4,1);s.set_landscape();s.set_paper(8);s.fit_to_pages(1,0);s.repeat_rows(3);s.set_margins(.25,.25,.35,.35)
 return s
def finish(s,end,col):s.autofilter(3,0,end,col);s.print_area(0,0,end,col)
def errfmt(s,col,end):
 cell=f'{xl_col_to_name(col)}5'
 s.conditional_format(4,col,end,col,{'type':'formula','criteria':f'=AND(ISNUMBER({cell}),{cell}>0.15)','format':fmt['bad']})
 s.conditional_format(4,col,end,col,{'type':'formula','criteria':f'=AND(ISNUMBER({cell}),{cell}>0.1,{cell}<=0.15)','format':fmt['amber']})

summary=sheet('结果总览','分块方案库与Mapper · 独立验证','78组历史开发池 → 17类模板；24组全新验证。没有使用验证时延或真实分块选预测方案。',
 ['预测方案','≤10%组数','总数','≤10%比例','平均绝对误差','最大绝对误差','≤15%组数'],[25,14,12,16,18,18,16])
val=sheet('24组独立验证','24组验证 · 时延与绝对相对误差','单位：μs。误差=ABS(预测−实测)/实测；≤10%为目标。红色>15%，浅黄10%～15%。',
 ['类别','M','N','K','实测μs','原mapperμs','误差','校准mapperμs','误差','最近模板μs','误差','5模板择优μs','误差','合并μs','误差'],[14,10,10,10,13,15,12,16,12,15,12,16,12,15,12])
raws=sheet('GPU原始计时','24组验证 · 原始GPU Event样本','原始每次Graph平均到单次GEMM的时间，保留两轮25样本；未剔除异常点。来源：validation_measure.json。',
 ['ID','轮次','样本序号','单次GEMM μs','Graph内调用数'],[48,10,12,20,20])
rr=4;raw_row_by_id={}
for x in raw['cases']:
 starts=[]
 for ri,rd in enumerate(x['rounds'],1):
    start=rr+1
    for si,t in enumerate(rd['samples_us'],1):
        raws.write_row(rr,0,[x['id'],ri,si],fmt['text']);raws.write_number(rr,3,t,fmt['num']);raws.write_number(rr,4,x['graph_calls'],fmt['int']);rr+=1
    starts.append((start,rr))
 raw_row_by_id[x['id']]=starts
finish(raws,rr-1,4)
for i,x in enumerate(a['cases'],4):
 excel=i+1;val.write_row(i,0,[groups[x['group']],*x['shape']],fmt['text'])
 intervals=raw_row_by_id[x['id']];formula='=AVERAGE('+','.join(f"MEDIAN('GPU原始计时'!D{s}:D{e})" for s,e in intervals)+')'
 val.write_formula(i,4,formula,fmt['num'],x['measured_us'])
 for j,key in enumerate(methods):
    col=5+j*2;v=x['methods'][key];val.write_number(i,col,v['predicted_us'],fmt['num'])
    val.write_formula(i,col+1,f'=ABS({xl_col_to_name(col)}{excel}/$E{excel}-1)',fmt['pct'],v['absolute_error'])
finish(val,27,14)
for j in range(6,15,2):errfmt(val,j,27)
for i,(key,name) in enumerate(methods.items(),4):
 col=xl_col_to_name(6+2*(i-4));rng=f"'24组独立验证'!{col}5:{col}28";s=a['summary'][key]
 summary.write(i,0,name);summary.write_formula(i,1,f'=COUNTIFS({rng},"<=0.1")',fmt['int'],s['pass10']);summary.write_number(i,2,24,fmt['int'])
 summary.write_formula(i,3,f'=B{i+1}/C{i+1}',fmt['pct'],s['pass10']/24)
 summary.write_formula(i,4,f'=AVERAGE({rng})',fmt['pct'],s['mape']);summary.write_formula(i,5,f'=MAX({rng})',fmt['pct'],s['max'])
 summary.write_formula(i,6,f'=COUNTIFS({rng},"<=0.15")',fmt['int'],s['pass15'])
summary.merge_range(11,0,12,6,'没有达到所有样例≤10%的目标。查历史分块、模型择优和真实库选择是三个不同问题，不能仅凭候选增多就推断精度提高。',fmt['wrap'])
summary.write_row(15,0,['类别','样例数','原mapper≤10%','校准mapper≤10%','最近模板≤10%','5模板≤10%','合并≤10%'],fmt['header']);summary.set_row(15,34)
for i,(g,stats) in enumerate(a['groups'].items(),16):
 summary.write(i,0,groups[g]);summary.write_number(i,1,stats['original']['count'])
 for j,key in enumerate(methods,2):
    col=xl_col_to_name(6+2*list(methods).index(key))
    summary.write_formula(i,j,f'=COUNTIFS(\'24组独立验证\'!A5:A28,A{i+1},\'24组独立验证\'!{col}5:{col}28,"<=0.1")',fmt['int'],stats[key]['pass10'])
summary.merge_range(23,0,25,6,f"开发分块对照：模型选中实测最快候选 {a['controlled_top1']}/12；最近模板命中真实分块 {a['nearest_hits']}/24；5模板覆盖真实分块 {a['top5_hits']}/24。事后真实分块只用于诊断，不参与正式预测。",fmt['wrap'])
summary.print_area(0,0,25,6)

dispatch=sheet('分块选择','分块来源与事后执行诊断','真实分块在预测封存、时延测量完成后才提取。固定实际分块诊断使用额外信息，不计入正式预测成绩。',
 ['M×N×K','最近模板','校准mapper','5模板择优','模板来源尺寸','事后实际分块','最近命中','5模板覆盖','实际分块诊断μs','诊断绝对误差','两轮差异'],[27,20,20,20,27,20,13,13,21,17,15])
for i,x in enumerate(a['cases'],4):
 dispatch.write_row(i,0,[tile(x['shape']),tile(x['nearest_tile']),tile(x['automatic_tile']),tile(x['library_min_tile']),tile(x['nearest_source_shape']),tile(x['posthoc']['tile']) if x['posthoc'].get('tile') else '未识别',int(x['nearest_hits_actual']),int(x['top5_contains_actual'])])
 d=x['observed_tile_diagnostic']
 if d:dispatch.write_number(i,8,d['predicted_us'],fmt['num']);dispatch.write_number(i,9,d['absolute_error'],fmt['pct'])
 else:dispatch.write(i,8,'n.a.',fmt['na']);dispatch.write(i,9,'n.a.',fmt['na'])
 dispatch.write_number(i,10,x['round_drift_pct']/100,fmt['pct'])
finish(dispatch,27,10);errfmt(dispatch,9,27)

cs=sheet('12组受控对照','开发集 · 受控分块对照','Triton仅作分块敏感性实验，普遍慢于厂商库；未用于本轮校准。原始输入：controlled_measure.json / controlled_predictions.json。',
 ['M×N×K','执行分块','实测μs','模型μs','绝对相对误差','每线程寄存器','编译Shared B','spill','数值正确','两轮差异'],[27,23,16,16,20,20,20,12,15,16])
predmap={(x['id'],tuple(x['tile'])):x for x in cp}
for i,x in enumerate(control['cases'],4):
 cs.write(i,0,tile(x['shape']));cs.write(i,1,tile(x['tile']) if x['tile'] else '厂商库对照');cs.write_number(i,2,x['measured_us'],fmt['num'])
 if x['tile']:
    v=predmap[x['id'],tuple(x['tile'])]['best']['duration_us'];cs.write_number(i,3,v,fmt['num']);cs.write_formula(i,4,f'=ABS(D{i+1}/C{i+1}-1)',fmt['pct'],abs(v/x['measured_us']-1))
    cs.write_row(i,5,[x['resources']['registers_per_thread'],x['resources']['shared_bytes'],x['resources']['spills']])
 else:
    for j in range(3,8):cs.write(i,j,'n.a.',fmt['na'])
 cs.write(i,8,int(x['correctness']['passed']));cs.write_number(i,9,abs(x['rounds'][0]['p50_us']-x['rounds'][1]['p50_us'])/x['measured_us'],fmt['pct'])
finish(cs,75,9);errfmt(cs,4,75)

library=sheet('开发方案库','78组历史开发数据 · 模板来源','仅提取历史粗粒度分块，不导入实测时延。旧验收/留出均作为本轮开发集。1条Split-K记录不纳入当前单kernel模板。',
 ['历史ID','M','N','K','bm','bn','bk','可用模板','历史kernel数'],[48,11,11,11,11,11,11,15,18])
for i,x in enumerate(p['development_cases'],4):
 library.write_row(i,0,[x['id'],*x['shape']]);library.write_row(i,4,x['tile'] if x['supported'] else ['n.a.']*3);library.write_row(i,7,[int(x['supported']),x['kernel_count']])
finish(library,81,8)
hs=sheet('硬件约束检查','同一模板迁移到新硬件 · 可行性检查','仅模型资源敏感性，不是新硬件实测验证；形状1024×8192×8192；效率1，无BW1100校准启动项。',
 ['硬件变更','模板bm×bn×bk','可行映射数量','可行','模型时延μs','Shared B','寄存器B','L2 B','gm×gn'],[25,23,19,10,18,18,18,18,15])
names={'reference':'原硬件','shared_16KiB':'Shared=16 KiB','registers_128KiB':'寄存器=128 KiB','cores_36':'CU=36','L2_512KiB':'L2=512 KiB'}
for i,x in enumerate(hw['cases'],4):
 hs.write_row(i,0,[names[x['variant']],tile(x['tile']),x['feasible'],int(x['best'] is not None)])
 b=x['best']
 if b:
    hs.write_number(i,4,b['duration_us'],fmt['num']);hs.write_row(i,5,[b['capacity']['shared_bytes'],b['capacity']['register_bytes'],b['capacity']['l2_bytes']]);hs.write(i,8,f"{b['mapping']['gm']}×{b['mapping']['gn']}",fmt['center'])
 else:
    for j in range(4,9):hs.write(i,j,'n.a.',fmt['na'])
finish(hs,88,8)
book.close()
print('Report and workbook authored:',out)
