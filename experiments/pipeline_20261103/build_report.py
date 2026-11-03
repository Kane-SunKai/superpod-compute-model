"""Reproducible report. XLSX fallback after verified artifact-tool native crash."""
import json,pathlib,statistics,collections
import xlsxwriter
r=pathlib.Path(__file__).resolve().parent;o=r/'outputs/pipeline_20261103';o.mkdir(parents=True,exist_ok=True)
v=json.loads((r/'variant_results.json').read_text());hold=json.loads((r/'holdout_results.json').read_text());cal=json.loads((r/'frozen_calibration.json').read_text());meta={x['id']:x for x in json.loads((r/'fixed_dispatch_all.json').read_text())}
acc=[x for x in v['cases'] if x['role']=='acceptance'];calrows=[x for x in v['cases'] if x['role']=='calibration']
groups={'decode':'Decode','prefill':'Prefill','square':'方阵','long_k':'长K','aspect':'高瘦/宽扁'}
versions=['V0','D_observed_MT','V2_read_bw','V3_effective_compute']
titles=['原模型','固定观测分块','再修正输入带宽','再校准持续计算']
def group_summary(rows):
 out=[]
 for key,label in groups.items():
  ss=[x for x in rows if x['group']==key];a=sum(x['versions']['V0']['pass10'] for x in ss);b=sum(bool(x['versions']['V3_effective_compute'] and x['versions']['V3_effective_compute']['pass10']) for x in ss)
  out.append((label,len(ss),a,b))
 return out
group_table='\n'.join(f'|{g}|{n}|{a}|{b}|' for g,n,a,b in group_summary(acc))
bad=[x for x in acc if not x['versions']['V3_effective_compute'] or not x['versions']['V3_effective_compute']['pass10']]
bad_table=[]
for x in bad:
 q=x['versions']['V3_effective_compute'];pred=f"{q['predicted_us']:.3f}" if q else '不支持';err=f"{q['signed_error_pct']:+.2f}%" if q else 'Split-K=8，另有归约kernel'
 bad_table.append(f"|{'×'.join(map(str,x['shape']))}|{x['measured_us']:.3f}|{pred}|{err}|")
failed_hold=[x for x in hold['cases'] if not x.get('pass10')]
ht='\n'.join(f"|{'×'.join(map(str,x['shape']))}|{x['measured_us']:.3f}|{x['result']['best']['duration_us']:.3f}|{x['signed_error_pct']:+.2f}%|" for x in failed_hold)
summary_rows='\n'.join(f"|{name}|{v['summary']['acceptance'][key]['pass10']}/44|{v['summary']['acceptance'][key]['supported']}/44|{v['summary']['acceptance'][key]['mape']:.2f}%|" for key,name in zip(versions,titles))
text=f'''# BW1100 分块、数据搬运与MMAC测试优化报告

日期：2026-11-03。基线：10.29实现，回退提交74e5522，对应84147e4。矩阵维度统一使用M、N、K，C[M,N]=A[M,K]×B[K,N]。

## 结论与适用范围

44组验收中，原模型15组误差≤10%。加入观测到的输出分块/K步长，再用独立实验校准输入带宽和持续计算效率，实验模式提高到34/44。仍有9组超差、1组执行方案未支持。**未达到全部≤10%的目标。**

这34组的结果包含额外的执行计划输入，不能宣称仅凭M/N/K和硬件参数的原模型已达到该精度。实际kernel只用于提取粗粒度MT/K步长，不使用被验收算例的时延作查表输入。buffer数量、CU分组和数据复用仍由现有模型选择，因此它也不是完整的真实kernel回放。

默认硬件文件、8-cycle MMAC参数和默认预测策略未替换。生产代码只增加输入HBM带宽的可覆盖接口，默认返回原值。50组历史预测重新计算完全一致，最大差为0 μs；62项单元测试通过。

## 测试集合与测量方式

|集合/实验|规模|用途|
|---|---:|---|
|既有验收|44组，排除6组边界尺寸|使用此前冻结的两轮Graph实测，逐项回归|
|新增独立校准|22组，全部两轮|11 decode、7 prefill、4方阵，与验收不重合|
|新增冻结后留出验证|12组，全部两轮|参数冻结后再测，与前两集合不重合|
|纯MMAC|54条件×2轮|改变并行波数、独立累加链和循环长度|
|LDS读取|18条件×2轮|独立检查Shared读取服务|
|读取/拷贝和并发扫描|84条结果|数据量、任务数、分段宽度和读/写方向对照|
|连续读取饱和扫描|24条结果|用不同字节量的耗时斜率估计输入带宽|
|MMAC与LDS混合|96条结果|纯计算、纯LDS、独立交错、依赖交错对照|
|有效缓存对照|5形状×两种工作集×2轮|热地址Graph与超过32MiB的轮换工作集|

GEMM统一使用FP16、连续A[M,K]/B[K,N]、torch.matmul(A,B,out=C)、预分配输出。先预热30次，Graph内64次大GEMM或256次小GEMM，预热回放6次，再测2轮×25次Graph GPU Event计时。每轮取p50并除以Graph内调用数，最终平均两轮。精度开关与历史实验相同。校准和留出均通过数值检查。

服务器沿用sunkai_compute，/workspace映射到/root/sk。全部新输入、结果、编译产物、日志和缓存写在/root/sk/pipeline_20261103内。没有改动驱动、功率或时钟设置；没有运行256 MAC/cycle后端。

## 发现1：增加bm=1并没有改善本次decode误差

已完成固定bn/bk/缓冲设置的单变量对照，以及加入bm=1/2/4/8的候选搜索。逻辑小分块仍保留16行MMAC覆盖和物理资源预算。12个decode的最优预测与原模型相同，最大变化0 μs。

原因是HBM/L2已经按有效输入行数计量；补齐后的MMAC和Shared操作数服务没有因逻辑bm变小而消失。某些等价映射会选更小bm，但这不等于时延优化。不能据此给小M直接打折。

## 发现2：大权重decode的实际分块与模型最优分块不同

|M,N,K|模型原分块bm,bn|库内核名称中的MT|原预测μs|仅固定观测分块μs|实测μs|
|---|---|---|---:|---:|---:|
|1,8192,8192|16,128|128×16×32|81.321|81.462|70.210|
|1,28672,8192|16,256|256×256×32|276.793|409.499|455.329|

当前连续row-major调用由库以转置输出视角执行，名称MT的第一维映射至N、第二维映射至M。此解释仅用于本次观测到的内核家族；不是所有库或布局的通用规则。

大权重decode所选MT256×256意味着矩阵执行覆盖范围明显大于有效M。当前MMAC模型按该覆盖计费后，预测已经从277升至约410 μs。这支持“实际执行计划不同是主要误差来源之一”。没有采集硬件利用率计数器，不能把它写成已直接测得的实际计算占比。

**需要修正此前推断：两种后端预测接近，只证明原模型路径对计算吞吐不敏感，不证明库内核同样完全受HBM限制。**

## 发现3：MMAC约8 cycle仍有依据，混合执行不能无条件按max重叠

使用4096/16384循环的差值消除固定成本。在72个工作组、4/8个wave、8条累加链条件下，每引擎持续服务约8.02～8.04个clock64 tick/MMAC，Event斜率与1.3GHz换算也接近该值。没有证据把物理8 cycle直接改成约9或10 cycle。

混合微基准中，独立LDS读取与MMAC交错的时间约为max(单独LDS,单独计算)的1.02～1.83倍，依赖读取还包括操作数转换成本。这说明理想重叠不是普遍成立的。但该微基准有自身循环、向量累加和指令组织，不能把这个倍数直接套入所有GEMM，也不能认定为纯硬件争用系数。

实验V3用7个独立prefill样本确定有效matrix_efficiency={cal['matrix_efficiency']:.2f}，保留8-cycle硬件服务。它是算子级持续效率校准，不是MMAC指令延迟测量。留一校准得到0.86～0.92，说明参数具有一定样本依赖。已有搬运阶段仍单独计费，剩余效率可能吸收未表示的调度/供给成本，尚未完成严格机制分解。

## 发现4：搬运的分段与并发确实重要，但微基准不能直接充当GEMM带宽表

以读取301.99MB、K=6144/N=24576为例，同一读取kernel从36个工作组增加到288个，第二轮耗时约864.00→173.12 μs。同样完整读取一次，按列面板切分后还会随面板宽度与K分组变化。因此字节数相同不等于时间相同。

该读取微基准包含加载和校验累加，并不是真实GEMM的完整访问序列。只将高并发下三种数据量的耗时斜率用于估计连续读取服务上限：六个斜率估计的中位数为{cal['hbm_bandwidth_Bps']/1e12:.6f} TB/s。微基准截距不搬进GEMM固定开销。输出HBM写服务保持1.72 TB/s，L2读写、Shared读写保持原配置。

这一步对既有decode有帮助，但独立校准集中部分decode反而变差。说明统一有效带宽仍不能覆盖访问模式差异，不能把1.923 TB/s当作所有GEMM必然达到的实际速度。

## 发现5：Graph热缓存不是本次大权重误差的充分解释

缓存诊断最终采用GPU连续执行的Graph，比较重复同一输入与轮换总工作集超过32MiB的输入。原先单个Graph前后由CPU逐个记录Event的小算例出现提交空隙，结果已排除；HIP不支持的外部捕获Event尝试也未作为数据使用。

256³热/轮换约5.14/5.32 μs；1024³约17.16/18.82 μs；1536³约41.30/42.14 μs。缓存会影响结果，但不能解释大权重decode约40%的原始偏差。该对照用于诊断，44组验收口径仍保持原同地址Graph。

## 分阶段回归结果

|版本|≤10%通过数|可预测数量|平均绝对相对误差|
|---|---:|---:|---:|
{summary_rows}

V1（小bm候选）与V0结果相同。含实际分块的三个版本有1组不支持，表中平均误差只统计43个可预测算例；通过率分母仍为44，未支持不能算通过。若对该组退回原模型，V3整44组混合路径平均误差约{(v['summary']['acceptance']['V3_effective_compute']['mape']*43+next(x for x in acc if not x['versions']['V3_effective_compute'])['versions']['V0']['absolute_error_pct'])/44:.2f}%，但那不是纯执行计划模式。

|类别|组数|原模型≤10%|V3≤10%|
|---|---:|---:|---:|
{group_table}

## 失败项保留

|M×N×K|实测μs|V3预测μs|有符号误差/状态|
|---|---:|---:|---|
{chr(10).join(bad_table)}

2048×28672×8192与2048×8192×28672实际调用同一个MT256×256×16内核。固定MT和统一效率仍未解释全部差异。前者约21%的残差需要继续检查任务分批、访问复用和流水交错。不能再只靠放大统一MMAC周期。

512×768×16384实际包含GSU8及PostGSU归约kernel。现有resident-C完整K模式没有表达这条执行链，故明确不支持，没有把额外归约漏掉后伪装成预测。

## 冻结参数后的12组新尺寸

通过{hold['summary']['pass10']}/12，平均绝对误差{hold['summary']['mape']:.2f}%。参数文件SHA256：`{hold['parameters_sha256']}`，与测试前计划一致，未依据留出结果再调参。

|失败留出M×N×K|实测μs|预测μs|有符号误差|
|---|---:|---:|---:|
{ht}

新留出中的896³仍有较大误差，说明方案还不能宣称已普遍收敛。旧验收decode12/12通过，也不代表任意decode都通过。

## 代码改动与复现

生产代码：`superpod_compute/operators.py`只增加`input_hbm_bandwidth()`扩展点，默认行为不变。实验目录：`experiments/pipeline_20261103`。`experimental_model.py`实现独立输入读服务；`predict_with_plan.py`提供离线可选执行计划入口；硬件、校准值、策略、执行计划和代码指纹全部进入实验缓存键。

现有默认CLI仍使用10.29策略。实验入口的用法见同目录`USER_GUIDE.md`。实验参数不是新GPU的通用参数，不应不经校准套到其他硬件。

证据：calibration_measure.json、small_m_results.json、mmac_probe_round1/2.log、lds_probe_round1/2.log、memory_probe.log、memory_saturation.log、mixed_probe.log、metadata_all.json、cache_rotation.json、frozen_calibration.json、variant_results.json、holdout_results.json。原始数据均保留，已失败的诊断路径没有混入最终统计。

## 下一步需要决定的方案

建议保留默认模型，把本次实验模式作为可选分支。若继续追求所有规则尺寸≤10%，优先补充两个具体能力：①输出任务实际分批、任务并发与缓存复用的对应关系；②明确的Split-K及归约。对于896³等例子先做受控验证，避免再按单个尺寸加补丁。

这些会涉及现有执行组织的扩展，尚未实施。当前证据不支持直接用统一偏置或继续调整8-cycle硬件值解决全部剩余问题。
'''
(o/'BW1100_分块与搬运优化测试报告_20261103.md').write_text(text,encoding='utf-8')

wb=xlsxwriter.Workbook(o/'BW1100_分块与搬运优化测试_20261103.xlsx')
def fmt(**kw):return wb.add_format({'font_name':'Microsoft YaHei','font_size':10,'valign':'vcenter',**kw})
normal=fmt();title=fmt(font_size=17,bold=True,font_color='#17324D');sub=fmt(font_color='#536B83');hdr=fmt(bold=True,font_color='white',bg_color='#17324D',text_wrap=True)
num=fmt(num_format='0.000');pct=fmt(num_format='0.00%');signed=fmt(num_format='+0.00%;-0.00%;0.00%');integer=fmt(num_format='#,##0')
badfmt=fmt(bg_color='#FBE8E7',font_color='#B52323',bold=True);amber=fmt(bg_color='#FFF3D8',font_color='#9C6A00');bold=fmt(bold=True,bg_color='#EFF4F8');wrap=fmt(text_wrap=True)
def setup(ws,last):
 ws.hide_gridlines(2);ws.set_default_row(22);ws.set_column(0,0,8,integer);ws.set_column(1,1,15);ws.set_column(2,4,11,integer);ws.set_column(5,last,15,num);ws.freeze_panes(7,5);ws.set_zoom(85);ws.set_landscape();ws.set_paper(8);ws.fit_to_pages(1,0);ws.repeat_rows(6);ws.set_margins(.25,.25,.3,.3)
for name,rows in [('44组验收',acc),('22组独立校准',calrows)]:
 ws=wb.add_worksheet(name);setup(ws,17);ws.set_column(16,16,22);ws.set_column(17,17,24,wrap)
 ws.write(0,0,'BW1100 分块与搬运优化测试',title)
 ws.write(1,0,'FP16 torch.matmul；同地址Graph；M/N/K；物理MMAC保留8 cycle；时间单位μs',sub)
 ws.write(2,0,'D/V2/V3增加观测分块输入。V3有效效率0.87由独立校准得到，未按验收逐组拟合。',sub)
 source='variant_results.json；验收实测来源 graph50-recheck-20261103/measurements.json' if rows is acc else 'variant_results.json；实测来源 calibration_measure.json；本表不是独立验收成绩'
 ws.write(3,0,source,sub)
 end=7+len(rows)
 ws.write(4,0,'原≤10%',bold);ws.write_formula(4,1,f'=COUNTIFS(H8:H{end},"<=0.1")',integer,sum(x['versions']['V0']['pass10'] for x in rows))
 ws.write(4,3,'V3≤10%',bold);ws.write_formula(4,4,f'=COUNTIFS(P8:P{end},"通过")',integer,sum(bool(x['versions']['V3_effective_compute'] and x['versions']['V3_effective_compute']['pass10']) for x in rows))
 ws.write(4,6,'V3未支持',bold);ws.write_formula(4,7,f'=COUNTIFS(P8:P{end},"未支持")',integer,sum(x['versions']['V3_effective_compute'] is None for x in rows))
 ws.write(4,9,'V3平均误差',bold);ws.write_formula(4,10,f'=AVERAGE(O8:O{end})',pct,statistics.mean(x['versions']['V3_effective_compute']['absolute_error_pct']/100 for x in rows if x['versions']['V3_effective_compute']))
 headers=['编号','类别','M','N','K','实测μs','V0预测μs','V0绝对误差','D固定分块μs','D绝对误差','V2输入带宽μs','V2绝对误差','V3持续计算μs','V3有符号误差','V3绝对误差','V3误差≤10%','观测bm×bn×bk','说明']
 for j,h in enumerate(headers):ws.write(6,j,h,hdr)
 ws.set_row(6,38)
 for i,x in enumerate(rows,7):
  er=i+1;ws.write(i,0,i-6);ws.write(i,1,groups[x['group']]);ws.write_row(i,2,x['shape']);ws.write(i,5,x['measured_us'],num)
  for key,pc,ec in [('V0',6,7),('D_observed_MT',8,9),('V2_read_bw',10,11)]:
   z=x['versions'][key]
   if z:ws.write(i,pc,z['predicted_us'],num);ws.write_formula(i,ec,f'=ABS({xlsxwriter.utility.xl_col_to_name(pc)}{er}/F{er}-1)',pct,z['absolute_error_pct']/100)
   else:ws.write(i,pc,'未支持');ws.write(i,ec,'未支持')
  z=x['versions']['V3_effective_compute']
  if z:
   ws.write(i,12,z['predicted_us'],num);ws.write_formula(i,13,f'=M{er}/F{er}-1',signed,z['signed_error_pct']/100);ws.write_formula(i,14,f'=ABS(N{er})',pct,z['absolute_error_pct']/100);ws.write_formula(i,15,f'=IF(O{er}<=0.1,"通过","超差")',normal,'通过' if z['pass10'] else '超差')
  else:
   for j in range(12,16):ws.write(i,j,'未支持')
  mt=meta[x['id']];ws.write(i,16,f"{mt['bm']}×{mt['bn']}×{mt['bk']}" if mt.get('bm') else 'GSU8 + 归约')
  ws.write(i,17,'需Split-K与额外归约' if z is None else '',wrap)
 ws.add_table(6,0,end-1,17,{'name':'Acceptance' if rows is acc else 'Calibration','columns':[{'header':h,'header_format':hdr} for h in headers],'style':'Table Style Medium 2'})
 for col in (7,9,11,14):
  ws.conditional_format(7,col,end-1,col,{'type':'cell','criteria':'>','value':.15,'format':badfmt})
  ws.conditional_format(7,col,end-1,col,{'type':'cell','criteria':'between','minimum':.1,'maximum':.15,'format':amber})
 if rows is acc:
  st=end+3;ws.write_row(st,1,['类别','样本数','原≤10%','V3≤10%'],hdr)
  for j,(g,n,a,b) in enumerate(group_summary(acc),st+1):
   ws.write(j,1,g);ws.write_formula(j,2,f'=COUNTIFS(B8:B{end},B{j+1})',integer,n);ws.write_formula(j,3,f'=COUNTIFS(B8:B{end},B{j+1},H8:H{end},"<=0.1")',integer,a);ws.write_formula(j,4,f'=COUNTIFS(B8:B{end},B{j+1},P8:P{end},"通过")',integer,b)
  ws.write(st+8,1,'V3输入读带宽：1.923451 TB/s；输出写带宽：1.72 TB/s；固定服务：3.121245 μs。',sub)
  ws.write(st+9,1,'V3平均误差统计43个可预测算例；通过数分母仍为44。没有满足全部≤10%。',sub)
 ws.print_area(0,0,end+14 if rows is acc else end,17)
ws=wb.add_worksheet('12组冻结后留出');setup(ws,11);ws.set_column(10,10,24);ws.set_column(11,11,20)
ws.write(0,0,'冻结参数后的新尺寸验证',title);ws.write(1,0,'12个形状与校准/验收均不重复；参数冻结后再实测；没有根据本表重新调参。',sub);ws.write(2,0,'额外输入：观测MT/K步长；来源 holdout_measure.json、holdout_metadata.json、holdout_results.json',sub)
ws.write(4,0,'≤10%',bold);ws.write_formula(4,1,'=COUNTIFS(J8:J19,"通过")',integer,hold['summary']['pass10']);ws.write(4,3,'平均误差',bold);ws.write_formula(4,4,'=AVERAGE(I8:I19)',pct,hold['summary']['mape']/100)
heads=['编号','类别','M','N','K','实测μs','V3预测μs','有符号误差','绝对误差','误差≤10%','分块bm×bn×bk','备注']
for j,h in enumerate(heads):ws.write(6,j,h,hdr)
ws.set_row(6,36)
for i,x in enumerate(hold['cases'],7):
 e=i+1;ws.write(i,0,i-6);ws.write(i,1,groups[x['group']]);ws.write_row(i,2,x['shape']);ws.write(i,5,x['measured_us'],num)
 if x['supported']:
  ws.write(i,6,x['result']['best']['duration_us'],num);ws.write_formula(i,7,f'=G{e}/F{e}-1',signed,x['signed_error_pct']/100);ws.write_formula(i,8,f'=ABS(H{e})',pct,x['absolute_error_pct']/100);ws.write_formula(i,9,f'=IF(I{e}<=0.1,"通过","超差")',normal,'通过' if x['pass10'] else '超差');m=x['result']['best']['mapping'];ws.write(i,10,f"{m['bm']}×{m['bn']}×{m['bk']}")
 else:ws.write(i,9,'未支持');ws.write(i,11,x['reason'])
ws.add_table(6,0,18,11,{'name':'Holdout','columns':[{'header':h,'header_format':hdr} for h in heads],'style':'Table Style Medium 2'})
ws.conditional_format('I8:I19',{'type':'cell','criteria':'>','value':.15,'format':badfmt});ws.conditional_format('I8:I19',{'type':'cell','criteria':'between','minimum':.1,'maximum':.15,'format':amber});ws.print_area(0,0,19,11)
wb.close()
print(o)
