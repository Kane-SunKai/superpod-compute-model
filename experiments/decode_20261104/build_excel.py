"""XLSX fallback: bundled artifact-tool native engine is unavailable on Windows.
The workbook is subsequently recalculated and rendered with native Excel.
"""
import json,pathlib,statistics
import xlsxwriter
r=pathlib.Path(__file__).resolve().parent
out=r/'outputs/decode_20261104';out.mkdir(parents=True,exist_ok=True)
freeze=json.loads((r/'freeze_final.json').read_text());raw=json.loads((r/'final_measure.json').read_text())
analysis=json.loads((r/'final_analysis.json').read_text())
metadata={x['id']:x for x in json.loads((r/'final_metadata.json').read_text())}
observed={x['id']:x for x in raw['cases']};predictions=freeze['predictions']
assert raw['completed'] and len(predictions)==len(observed)==36
file=out/'BW1100_Decode优化验证_20261104.xlsx'
w=xlsxwriter.Workbook(file);w.set_properties(dict(title='BW1100 Decode 计算模型验证',author='sunkai',comments='One frozen model, 36 independent cases; dispatch checked AFTER timings and predictions.'))
s=w.add_worksheet('独立验证结果');d=w.add_worksheet('GPU原始计时')
base=dict(font_name='Microsoft YaHei',font_size=10,font_color='#203448',valign='vcenter')
fmt=lambda **kw:w.add_format({**base,**kw})
title=fmt(font_size=16,bold=True);note=fmt(font_size=10,font_color='#52687A')
header=fmt(bold=True,font_color='white',bg_color='#203D57',align='center',text_wrap=True,bottom=1,bottom_color='white')
label=fmt(bold=True);integer=fmt(num_format='#,##0');number=fmt(num_format='0.000');pct=fmt(num_format='0.00%');signed=fmt(num_format='+0.00%;-0.00%;0.00%')
text=fmt();inputnum=fmt(num_format='0.000',font_color='#245B8C');warn=fmt(bg_color='#FCE9E7',font_color='#B3261E');blue=fmt(bg_color='#EDF3F8',bold=True)
for sh in (s,d):
 sh.hide_gridlines(2);sh.set_default_row(22);sh.set_landscape();sh.set_paper(8);sh.fit_to_pages(1,0)
 sh.set_margins(.28,.28,.35,.35)
s.set_tab_color('#203D57');s.set_zoom(85);d.set_zoom(90)
s.set_column('A:C',8);s.set_column('D:G',14);s.set_column('H:I',12);s.set_column('J:J',10);s.set_column('K:L',19);s.set_column('M:M',13);s.set_column('N:N',28)
s.set_row(0,8);s.set_row(1,28);s.set_row(2,23);s.set_row(9,38)
s.write('A2','BW1100 Decode 计算模型验证',title)
s.write('A3','FP16，torch.matmul(A,B,out=C)，MMAC 8-cycle，单卡 Graph 稳态 GPU Event 计时。',note)
s.write('A5','独立样例',label);s.write_formula('B5','=COUNT(A11:A46)',integer,36)
s.write('D5','误差≤10%',label);s.write_formula('E5','=COUNTIF(J11:J46,"通过")',integer,analysis['passed'])
s.write('G5','平均绝对误差',label);s.write_formula('H5','=AVERAGE(I11:I46)',pct,analysis['mape'])
s.write('J5','最大绝对误差',label);s.write_formula('K5','=MAX(I11:I46)',pct,analysis['max_error'])
s.write('A7','124组开发回归121组通过；本页36组来自6个全新N/K组合，预测与参数均在实测前冻结。',note)
s.write('A8','误差＝(预测−实测)/实测；实测取两轮p50均值。红色标出绝对误差超过10%的样例。',note)
headers=['M','N','K','第1轮p50 (μs)','第2轮p50 (μs)','实测均值 (μs)','预测时延 (μs)','有符号误差','绝对误差','≤10%','预测 bm×bn×bk','实际 bm×bn×bk','预取级别\n预测→实际','后验核对（未参与预测）']
s.write_row(9,0,headers,header)
d.write_row(0,0,['样例ID','M','N','K','轮次','样本序号','每次GEMM (μs)'],header)
d.set_row(0,30);d.set_column('A:A',36);d.set_column('B:F',11);d.set_column('G:G',21);d.freeze_panes(1,1)
rawrow=1
for i,p in enumerate(predictions):
 row=10+i;excelrow=row+1;x=observed[p['id']];m,n,k=p['shape'];medians=[]
 for rd,round_data in enumerate(x['rounds'],1):
  start=rawrow+1
  for j,t in enumerate(round_data['samples_us'],1):
   d.write_row(rawrow,0,[p['id'],m,n,k,rd,j],text);d.write_number(rawrow,6,t,inputnum);rawrow+=1
  end=rawrow
  medians.append(statistics.median(round_data['samples_us']))
  s.write_formula(row,rd+2,f"=MEDIAN('GPU原始计时'!G{start}:G{end})",number,medians[-1])
 for col,v in enumerate((m,n,k)):s.write_number(row,col,v,integer)
 actual=statistics.mean(medians);err=p['duration_us']/actual-1
 s.write_formula(row,5,f'=AVERAGE(D{excelrow}:E{excelrow})',number,actual)
 s.write_number(row,6,p['duration_us'],number)
 s.write_formula(row,7,f'=(G{excelrow}-F{excelrow})/F{excelrow}',signed,err)
 s.write_formula(row,8,f'=ABS(H{excelrow})',pct,abs(err))
 s.write_formula(row,9,f'=IF(I{excelrow}<=10%,"通过","未通过")',text,'通过' if abs(err)<=.1 else '未通过')
 actual_meta=metadata[p['id']]
 source=next(x for x in freeze['profile']['library'] if x['id']==p['dispatch_source'])
 s.write(row,10,'×'.join(map(str,p['tile'])),text)
 s.write(row,11,'×'.join(map(str,actual_meta['tile'])),text)
 s.write(row,12,f"{source['prefetch_global']} → {actual_meta['prefetch_global']}",text)
 same=p['tile']==actual_meta['tile'] and source['prefetch_global']==actual_meta['prefetch_global']
 diagnostic='分块/预取选择不同' if not same else '方案一致；检查成本估计' if abs(err)>.1 else '方案一致，误差达标'
 s.write(row,13,diagnostic,text)
 s.write_comment(row,10,'自动检索来源：'+p['dispatch_source']+'；每CU驻留上限：'+str(p['resident_tasks_per_cu']))
 s.set_row(row,24)
s.autofilter('A10:N46');s.freeze_panes(10,3)
s.conditional_format('H11:J46',dict(type='formula',criteria='=$I11>10%',format=warn))
d.autofilter(0,0,rawrow-1,6)
s.write('A49','模型与结论',label)
s.write('A50','未达到全部样例误差≤10%的目标。误差通过与执行方案选对是两项不同检查，请同时看J列与N列。',note)
s.write('A51','输入沿K累加、双缓冲；4个SIMD各192 KiB约束寄存器；预取1计尾批，预取2按资源允许驻留。',note)
s.write('A52','HBM有效输入读1.83 TB/s，Shared有效读57.5 Byte/cycle/CU；固定开销3.121 μs；MMAC仍为8 cycle。',note)
s.write('A53','软件方案取自开发记录；64×128×16的碎片读取/Shared端口串行等为开发数据推定的简化配置。',note)
s.write('A54','有效服务参数与软件配置均是当前BW1100的校准输入，未更改硬件标称峰值。',note)
s.write('A56','测量口径与复现',label)
s.write('A57','预热30次；Graph内64或256次GEMM；预热回放6次；两轮各25个GPU Event样本；36组数值检查均通过。',note)
s.write('A58','同地址、预分配输出；Event包围Graph再除以调用次数。原始数据：final_measure.json；冻结预测：freeze_final.json。',note)
s.write('A59','当前入口：python tools/predict_decode.py --m 2 --n 8192 --k 8192；项目：D:\\01codex_work\\03_super node\\compute_model。',note)
s.write('A60','本页实际分块由验收完成后的分析取得，未传入预测。原始计时在下一页，均值与误差使用Excel公式计算。',note)
s.print_area('A1:N60');s.repeat_rows(9,9)
d.print_area(0,0,rawrow-1,6);d.repeat_rows(0,0)
s.activate();s.set_first_sheet();w.close()
print(file)
