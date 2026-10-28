"""Self-contained, editable SVG explanations of the GEMM mapping parameters."""
from html import escape

def text(x,y,s,size=17,bold=False,anchor='start',extra=''):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="#183b55" {extra}>{escape(str(s))}</text>'
def box(x,y,w,h,fill='#eef4f8',stroke='#b5cad9',extra=''):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}" {extra}/>'
def line(x,y,x2,y2,color='#7091a7',dash=False):
    attr='stroke-dasharray="4 4"' if dash else ''
    return f'<path d="M{x},{y} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="2" {attr}/>'
def begin(h,title):return f'<svg viewBox="0 0 1200 {h}" role="img" aria-label="{title}" xmlns="http://www.w3.org/2000/svg">'
def hdim(x,x2,y,label):
    return line(x,y,x2,y)+line(x,y-5,x,y+5)+line(x2,y-5,x2,y+5)+text((x+x2)/2,y-10,label,16,True,'middle')
def vdim(x,y,y2,label):
    return line(x,y,x,y2)+line(x-5,y,x+5,y)+line(x-5,y2,x+5,y2)+f'<text transform="translate({x-12},{(y+y2)/2}) rotate(-90)" text-anchor="middle" font-size="16" font-weight="700" fill="#183b55">{escape(label)}</text>'

def render():
    s=begin(380,'步骤1：M、N和K在A、B、C三个矩阵上的位置')
    s+=text(26,33,'① M / N / K：先认清完整矩阵的三条尺寸',23,True)
    for x,w,h,name,top,left,color in [(85,245,170,'A','K = 1024 列','M = 3072 行','#deebfd'),(470,245,170,'B','N = 2304 列','K = 1024 行','#ddefe7'),(850,245,170,'C','N = 2304 列','M = 3072 行','#fff0d6')]:
        s+=box(x,113,w,h,color)+hdim(x,x+w,90,top)+vdim(x-21,113,283,left)+text(x+w/2,211,name,42,True,'middle')
    s+=text(383,210,'×',35,True)+text(766,210,'=',35,True)
    s+=text(26,326,'M：C有多少行；N：C有多少列；K：计算每个C元素时，需要乘加多少对A/B元素。',18)
    s+=text(26,356,'C[m,n] = A[m,0]B[0,n] + A[m,1]B[1,n] + … + A[m,1023]B[1023,n]',17)+'</svg>'
    fig1=s

    s=begin(535,'步骤2：bm、bn定义单个CU负责的输出块大小')
    s+=text(26,34,'② bm / bn：每个CU负责多大的C子块',23,True)
    x,y,cw,ch=95,112,43,25
    s+=hdim(x,x+6*cw,88,'完整 C：N = 2304')+vdim(62,y,y+12*ch,'完整 C：M = 3072')
    for r in range(12):
        for c in range(6):
            active=r==2 and c==1
            s+=box(x+c*cw,y+r*ch,cw-2,ch-2,'#e4af50' if active else '#fff0d6',extra=f'data-c-tile="{r},{c}" style="cursor:pointer"')
    s+=text(95,440,'整张C：12行 × 6列 = 72个输出块',17,True)
    s+=text(95,469,'点击小方格，或用上方下拉框选择一个CU任务',14)
    s+=line(379,220,497,220)+text(433,204,'放大',16,False,'middle')
    s+=box(545,153,267,179,'#fff0d6','#d0a45f')+hdim(545,812,129,'bn = 384 列')+vdim(518,153,332,'bm = 256 行')
    s+=text(678,223,'C(r,c)',28,True,'middle')+text(678,262,'一个CU的输出块',18,False,'middle')
    s+=text(860,166,'bm：block M',20,True)+text(860,198,'沿M方向取256行',17)
    s+=text(860,252,'bn：block N',20,True)+text(860,284,'沿N方向取384列',17)
    s+=text(520,382,'当前选择：r=2，c=1，逻辑任务13',18,True,extra='id="tile-selection"')
    s+=text(520,420,'C[512:768, 384:768]',20,True,extra='id="tile-global-range"')
    s+=text(520,454,'行号512–767，列号384–767；范围采用左闭右开',15,extra='id="tile-range-note"')
    s+=text(26,511,'bm / bn 的单位是“元素个数”。分块尺寸是mapper选择的执行参数，不是硬件固定规格。',16)+'</svg>'
    fig2=s

    s=begin(420,'步骤3：gm、gn定义一批并行任务覆盖的CU分组和C范围')
    s+=text(26,34,'③ gm / gn：这一批安排多少个CU，并覆盖多大的C区域',23,True)
    x,y,cw,ch=92,105,33,18
    s+=hdim(x,x+6*cw,82,'gn = 6 列CU任务')+vdim(64,y,y+12*ch,'gm = 12 行CU任务')
    for r in range(12):
        for c in range(6):s+=box(x+c*cw,y+r*ch,cw-2,ch-2,'#cfe0ef')
    s+=text(340,110,'一批并行任务数：gm × gn = 12 × 6 = 72',20,True)
    s+=text(340,154,'覆盖C的行数：gm × bm = 12 × 256 = 3072',18)
    s+=text(340,193,'覆盖C的列数：gn × bn = 6 × 384 = 2304',18)
    s+=text(340,235,'本例恰好一批覆盖完整C，所以72个CU只需这一批。',17)
    s+=text(340,276,'12×6描述任务分组；物理CU不必按这个矩形排列。',16)
    s+=text(340,310,'gm / gn 是“本批使用数量”，更大的C可以分成多批。',17,True)
    s+=text(26,365,'若 N 改为4608：C共有12×12个块；每批仍用12×6=72个CU，顺序执行两批。',17)
    s+=box(26,380,525,25,'#cfe0ef')+text(288,398,'第0批：C[:, 0:2304]',14,True,'middle')
    s+=box(579,380,525,25,'#ddefe7')+text(841,398,'第1批：C[:, 2304:4608]',14,True,'middle')+'</svg>'
    fig3=s

    s=begin(590,'步骤4：bk沿K划分输入，单CU反复更新同一C块')
    s+=text(26,33,'④ bk：一个CU每次沿K读取多少对A/B元素',23,True)
    s+=text(26,65,'本CU使用A的bm行与B的bn列（都跨完整K）；实际按bk逐段搬入，完整条带无需同时驻留Shared。',16)
    # Local input matrices have 64 real panels, so highlight is position-correct.
    x,y,w,h=85,144,320,135
    s+=hdim(x,x+w,122,'完整 K = 1024')+vdim(57,y,y+h,'bm = 256 行')
    for i in range(64):s+=box(x+i*5,y,4.8,h,'#89b3e5' if i==0 else '#e1ecfa','#c1d3e6',extra=f'data-ak="{i}"')
    s+=text(245,313,'A_r = A的第r个行条带',17,True,'middle')
    s+=text(245,342,'从其中取bk=16列',17,False,'middle')
    x,y,w,h=515,144,170,256
    s+=hdim(x,x+w,122,'bn = 384 列')+vdim(487,y,y+h,'完整 K = 1024')
    for i in range(64):s+=box(x,y+i*4,w,3.8,'#77bba3' if i==0 else '#e0f0e9','#bed9cc',extra=f'data-bk="{i}"')
    s+=text(600,435,'B_c = B的第c个列条带',17,True,'middle')
    s+=text(600,464,'从其中取bk=16行',17,False,'middle')
    s+=box(762,138,405,327,'#f7fafc')+text(783,170,'这一段真正交给该CU的输入',20,True)
    s+=box(788,194,72,119,'#deebfd')+text(824,229,'aᵢ',24,True,'middle')+text(824,264,'256×16',14,True,'middle')
    s+=text(882,263,'×',24,True)
    s+=box(923,231,213,52,'#ddefe7')+text(1029,262,'bᵢ：16×384',18,True,'middle')
    s+=text(784,347,'C(r,c) ← C(r,c) + aᵢ × bᵢ',22,True)
    s+=text(784,385,'只更新同一块256×384的C。',17)
    s+=text(784,420,'64段K结束，才得到完整结果。',17,True)
    s+=text(26,512,'当前 i=0：a₀=A[512:768, 0:16]；b₀=B[0:16, 384:768]',18,True,extra='id="k-global-range"')
    s+=text(26,546,'第0段负责k=0…15；下一段k=16…31。C部分和保留，输入a/b换成下一段。',16,extra='id="k-step-note"')
    s+=text(26,578,'bk=16 → K/bk=1024/16=64段。每段的输入大小改变，最终C块的bm×bn尺寸始终不变。',16)+'</svg>'
    fig4=s

    s=begin(465,'步骤5：CU输出块分给四矩阵单元，再分成16x16指令输出片')
    s+=text(26,34,'⑤ CU内部再划分：输出子区 → 16×16指令片',23,True)
    s+=hdim(83,357,100,'本CU：bn = 384列')+vdim(56,125,327,'本CU：bm = 256行')
    for r in range(2):
        for c in range(2):
            x,y=83+c*137,125+r*101
            s+=box(x,y,134,98,['#f8dfb2','#fceacb','#fceacb','#f8dfb2'][r*2+c])
            s+=text(x+67,y+38,f'单元 {r*2+c}',18,True,'middle')+text(x+67,y+66,'128 × 192',17,False,'middle')
    s+=text(83,362,'2×2个矩阵单元，分别负责独立C子区',15)
    s+=line(378,214,450,214)+text(414,196,'放大',15,False,'middle')
    x,y,cw,ch=510,125,25,25
    s+=hdim(x,x+12*cw,100,'192列 ÷ 16 = 12列小片')+vdim(482,y,y+8*ch,'128行 ÷ 16 = 8行小片')
    for r in range(8):
        for c in range(12):s+=box(x+c*cw,y+r*ch,cw-2,ch-2,'#d6a655' if r==c==0 else '#fceacb')
    s+=text(660,362,'一个单元：8×12=96个16×16输出片',16,True,'middle')
    s+=box(850,125,322,202,'#f6f9fc')+text(872,162,'每个小片做一次',20,True)+text(872,202,'[16×16] × [16×16]',22,True)+text(872,239,'累加到[16×16]的C片',18)
    s+=text(872,282,'指令形状：16×16×16',19,True)
    s+=text(26,409,'本例bk=16，正好等于一条指令的K长度：每单元96条指令 × 8周期 = 768周期/面板。',17)
    s+=text(26,442,'bm/bn/bk是软件分块；16×16×16是硬件指令粒度。若bk改成32，每个输出小片需要2个K步。',16)+'</svg>'
    fig5=s

    table='''<div id="tiling-params" class="table-scroll"><table><thead><tr><th>参数</th><th>白话含义</th><th>本例</th><th>对应图</th></tr></thead><tbody>
<tr><td>M / N / K</td><td>完整C的行数 / 列数 / 每个输出元素的乘加长度</td><td>3072 / 2304 / 1024</td><td>①</td></tr>
<tr><td>bm</td><td>一个CU负责的C子块有多少行，单位：元素</td><td>256行</td><td>②</td></tr>
<tr><td>bn</td><td>一个CU负责的C子块有多少列，单位：元素</td><td>384列</td><td>②</td></tr>
<tr><td>gm</td><td>一批任务在M方向安排多少个CU</td><td>12个</td><td>③</td></tr>
<tr><td>gn</td><td>一批任务在N方向安排多少个CU</td><td>6个</td><td>③</td></tr>
<tr><td>bk</td><td>每次沿K处理多长：A取几列、B同时取几行</td><td>16；总共64段</td><td>④</td></tr></tbody></table></div>'''
    controls='''<div class="toolbar" id="tilingControls"><b>跟踪一个CU任务：</b><label>行组r <select id="tileR"></select></label><label>列组c <select id="tileC"></select></label><label>K面板i <select id="tileK"></select></label><span class="small">从0开始编号；改变选择后，②和④的高亮及全局索引同步更新。</span></div>'''
    output=table+f'<div class="diagram" id="tiling-dimensions">{fig1}</div>'+controls
    output+=f'<div class="diagram" id="tiling-output">{fig2}</div><div class="diagram" id="tiling-group">{fig3}</div><div class="diagram" id="tiling-k">{fig4}</div><div class="diagram" id="tiling-engine">{fig5}</div>'
    output+='''<div class="formula"><b>把五步连起来：</b>完整C → 按bm×bn划分输出块 → 每批安排gm×gn个CU → 每CU沿K以bk为步长读入aᵢ/bᵢ → CU内部矩阵单元更新同一C块 → 全部K结束后写回。<br>本例：每CU 256×384；一批12×6=72个CU；每次K=16，共64次；最终拼成3072×2304的C。</div><p class="small">图形为分块关系示意，长宽不按真实元素比例。CU(r,c)是任务组中的逻辑位置；实际硬件调度不要求相同的物理排布。索引[start:end]包含start、不包含end。</p>'''
    return output

JS=r'''
(()=>{
 const r=document.getElementById('tileR'),c=document.getElementById('tileC'),k=document.getElementById('tileK');
 [[r,12,2],[c,6,1],[k,64,0]].forEach(([e,n,v])=>{e.innerHTML=Array.from({length:n},(_,i)=>`<option value="${i}">${i}</option>`).join('');e.value=String(v)});
 function update(){const rr=+r.value,cc=+c.value,ii=+k.value,rs=rr*256,cs=cc*384,ks=ii*16;
  document.querySelectorAll('[data-c-tile]').forEach(t=>{const on=t.dataset.cTile===`${rr},${cc}`;t.setAttribute('fill',on?'#e4af50':'#fff0d6');t.setAttribute('stroke',on?'#a57521':'#b5cad9');t.setAttribute('stroke-width',on?'2.5':'1')});
  document.querySelectorAll('[data-ak]').forEach(t=>t.setAttribute('fill',+t.dataset.ak===ii?'#4386d1':'#e1ecfa'));
  document.querySelectorAll('[data-bk]').forEach(t=>t.setAttribute('fill',+t.dataset.bk===ii?'#24886c':'#e0f0e9'));
  document.getElementById('tile-selection').textContent=`当前选择：r=${rr}，c=${cc}，逻辑任务${rr*6+cc}`;
  document.getElementById('tile-global-range').textContent=`C[${rs}:${rs+256}, ${cs}:${cs+384}]`;
  document.getElementById('tile-range-note').textContent=`行号${rs}–${rs+255}，列号${cs}–${cs+383}；范围采用左闭右开`;
  document.getElementById('k-global-range').textContent=`当前 i=${ii}：a${ii}=A[${rs}:${rs+256}, ${ks}:${ks+16}]；b${ii}=B[${ks}:${ks+16}, ${cs}:${cs+384}]`;
  document.getElementById('k-step-note').textContent=`第${ii}段负责k=${ks}…${ks+15}；`+(ii<63?`下一段k=${ks+16}…${ks+31}。C部分和保留，输入a/b换成下一段。`:'这是最后一段；乘加后完成完整K，随后按输出片写回C。');
 }
 [r,c,k].forEach(e=>e.onchange=update);
 document.querySelectorAll('[data-c-tile]').forEach(t=>t.onclick=()=>{const [rr,cc]=t.dataset.cTile.split(',');r.value=rr;c.value=cc;update()});
 update();
})();
'''
