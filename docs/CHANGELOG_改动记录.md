# 改动记录

记录每一次提交做了什么、为什么改、用什么测试定位到的问题、改完的验证结果。

目的：几个月后回看时，能区分"这是当时想清楚了才改的"和"这是当时糊上去的"，以及每一项改动的证据强度。

## 用法

每次 `git commit` 之前，在文末追加一条，提交号可以留空，提交后补上。条目按时间正序排列（最新的在最后）。

不要删除旧条目。如果后来的测试推翻了前面的结论，在旧条目里加一行 `> 更正（日期）：…`，并指向新条目。

## 条目模板

```markdown
## YYYY-MM-DD · 一句话标题（commit 留空或填哈希）

**改了什么**
- `文件路径` — 具体改动

**为什么要改（问题）**
- 观察到什么现象，数字是多少，% 相对于什么

**怎么判断出原因的（测试）**
- 脚本/命令、测量协议、关键中间数字
- 排除了哪些其它解释

**验证结果**
- 改前 / 改后对比，覆盖了哪些形状

**已知限制**
- 没有覆盖的情况、依赖的假设
```

---

## 2026-11-11 · 拆分 HBM 读写带宽，补建批量注意力矩阵乘（`9f98f37`）

**改了什么**
- `superpod_compute/contracts.py` — `Hardware` 新增 `hbm_write_bandwidth_Bps`、`hbm_read_a_Bps`；未设置时回落到 `hbm_bandwidth_Bps`，旧 profile 不受影响
- `superpod_compute/operators.py` — 新增 `input_a_hbm_bandwidth()` 钩子；输入填充从 `dh / B_hbm` 改为 A 视图走 `B_read_a`、B 视图走 `B_hbm`；输出排空改走 `hbm_write_bandwidth_Bps`
- `superpod_compute/batched.py` — `_waves()` 分别累计 A 行数与 B 列数，批量 GEMM 的填充按两个视图分别计费
- `superpod_compute/data/bw1100_mapper_mmac_hardware.json` — 写入 `hbm_write_bandwidth_Bps = 1.05e12`、`hbm_read_a_Bps = 1.10e12`
- `docs/BW1100_带宽标定与注意力算子_20261111.md` — 标定方法与结论

**为什么要改（问题）**
- GPT-7B 的 QK^T / PV 两组误差分别是 −20.9% 和 −30.6%，都在"偏快"一侧，且都带 batch
- 用单一 `hbm_bandwidth_Bps = 1.72e12` 无法同时解释普通 GEMM 和批量 GEMM
- 怀疑：写方向的实际带宽低于标称值，且 A/B 两个视图的读速率不同

**怎么判断出原因的（测试）**
- `bw_probe/measure_stream_curve.py` 等：分别测读流、写流、以及 A 主导 / B 主导的 GEMM，得到写方向有效带宽 1.05 TB/s（标称 1.72），A 视图 1.10 TB/s、B 视图 1.72 TB/s
- PMC 计数器（`TCC_HIT`/`TCC_MISS`）显示 A 的每个 128 B 行被读两次，B 只有一次，且与 K、与块数无关，只与 L2 容量相关 —— 这是 A 视图更慢的物理原因，不是标定误差
- 排除了"引擎路由（engine-routed）"这个解释：实测 routed 与 compact 两种布局，routed 的预测与实测吻合（0.42 vs 0.40），compact 差得多，说明 routed 不是误差来源（早前曾误判为主因，已撤回）

**验证结果**
- QK^T：−20.9% → +8.0%；PV：−30.6% → +5.0%
- 其它不涉及写方向 / A 视图的形状不受影响（对 `hbm_read_a_Bps` 0% 敏感，对写常数 5~14%）

**已知限制**
- `hbm_read_a_Bps = 1.10` 是在 **NN 布局**下用 A 主导 GEMM 标定的；NT 布局下测不出一个稳定值（同样 537 MB 的读，NT 下 234~1591 GB/s，取决于库选的分块），因此没有 NT 版本
- 它只在批量 PV 一处起作用，而那一处已用 NT 实测确认（989 µs，模型 1038，+5.0%）

---

## 2026-11-11~12 · AICB 算子实测、布局排查与选核退化定位（当前工作区，未提交）

**改了什么**
- 新增 `experiments/attention_20261110/`：AICB GPT-7B 十二个算子的实测与建模脚本、布局对比、V3 表格生成器
- 新增本文档 `docs/CHANGELOG_改动记录.md`：约定每次提交前在这里记一条
- 模型代码本身**没有改动**（这一轮的结论是"先不动"）
- 产出 HTML：`AICB算子实测_GPT7B_20261111.html`（第 5 节是本次追查）、`布局与选核排查_20261111.html`、`100组回归测试V3_20261112.html`

**为什么要改（问题）**
接入 SimAI 的 AICB 需要 GPT-7B 全部算子的时延。实测后发现两个矩阵算子误差异常大：

| 算子 | 形状 | 实测 | 模型 | 误差 |
|---|---|---|---|---|
| mlp_linear_2 | 4096×4096×16384 | 2827.7 µs | 1777.7 | −37.1% |
| logit_time | 4096×50304×4096 | 7441.0 µs | 5214.2 | −29.9% |

**怎么判断出原因的（测试）**

先确认口径：这两个是用 **NT**（`matmul(x, w.t())`，与 Megatron/AICB 一致）测的，kernel 名 `Cijk_Alik_*` 与之吻合。

1. **排除内存带宽**
   `hipprof --pmc` 抓 `TCC_HIT`/`TCC_MISS`/`TA_TA_BUSY`/`SQ_INSTS_LDS`。用 65536×8×4096 的 537 MB 纯 A 流做锚点标定，确认一个计数单位 = 128 B 行（实测 539 MB，误差 0.4%）。结果：DRAM 只有 0.32~0.66 TB/s（上限 1.72），L2 读 2.3~3.8 TB/s（上限 4.90），复用率 71~79%。**六个形状全都不受带宽限制**，之前"L2 复用失效"的推断被推翻。

2. **拆成两层**
   总耗时 = 核心占用（WG 数 / 轮数 / 72）× 每 K 步效率 × 理想计算。六个形状这一乘积与实测比值吻合到 1~5%：
   - gpt13b_12 的 35% 亏损来自 160 个 WG 摊 72 核 = 74% 占用（纯量化）
   - mlp_linear_2 / logit 占用 89% / 99%，是每步本身慢 65~70%

3. **每步慢在取数**
   所有 kernel 的 MMAC 发射数（`SQ_INSTS_VALU` 减去约 4.5k 标量开销）**恰好等于理论最少的 (bm/16)(bn/16)(K/16)**，没有多余矩阵指令。而 TA 忙周期/条 LDS 指令从 7.5 升到 12.9 时，每步低效从 1.20 升到 1.70，单调对应；理想预算是每条 LDS 8 周期。

4. **固定 M=2048 扫 K 和 N**（`parse_pmc_kn*.py`，每组 3 次取中位，重复性 <2%）
   - N 从 6144 到 32768，TA 成本完全平（7.6~7.8）→ 与流量无关
   - K 从 4096 到 16384 涨 65%，但在面板 = 8 MiB 处（K=16384，256×16384×2 = 8.39 MB）达峰后回落 → 指向"每 WG 面板字节 / L2 容量"

5. **两个算子的处理方式不同**
   - 强制 solution index 在 **NT 和 NN 两种形式下都会 VMFault + core dump**，选核扫描这条路走不通
   - 改用"换调用方向"：logit 转置成 50304×4096 → 5442 µs（−26.8%）；N 切两段 → 5570 µs（−25.1%）。两者都落在模型 +4.4% / +6.8%
   - 查 kernel 名发现：**同一宏分块 256×256×32，默认走 `SU8`，转置走 `SU2`，切分走 `SU1`**。SplitU 把输出块拆成多趟，每趟重读一次操作数，正好对应 TA 压力（7.6→11.9）和每步低效（1.21→1.65）
   - mlp_linear_2 换 NN（2644）、NN+K 切 2（2649）、NT+K 切 2（2884）都无效，最好仍是 −32.8%

**验证结果**
- logit_time：−29.9% 是**库的选核退化**（N=50304 → 197 个质数面板 → 选到 SU8），不是模型误差。按 N 切分口径，模型 +6.8%
- mlp_linear_2：NT/NN 两种布局、K 切分都对它无效，**是真实的模型缺口**（模型高估 49%）。它是唯一让"每 WG 面板"正好压在 8 MiB L2 上的形状
- mlp_linear_2 的缩放对照：M/N/K 同时减半（2048×2048×8192，切片降到 4.19 MB = L2 的一半），NT 实测 259.4 µs，模型 229.4 µs（模型偏快 11.6%）；同一模型在全尺寸上偏快 37.2%。误差随"切片/L2"这个比值一起缩小，指向同一个变量

**已知限制**
- 强制选核在这块卡 + 这套 DTK 上不可用，因此这些形状的"最快可达"只能通过换调用形式间接探测，拿不到真正的下界
- 「面板/L2 比」这条规律只有一台机器、一个库、一台峰值的数据支持，不能作为物理定律写进模型
- 一次测量曾经给出 NN+K 切分 1962 µs，用 5 次独立分配复测后是 2649 µs —— 同缓冲区反复测会因页对齐不同换 kernel，该数已作废

**待办（尚未动模型）**
- 若要给"面板贴近 L2 容量"建模，应放在"库达成率"层（类似 `bw1100_tile_correction.json`），而不是物理层

> 更正（2026-11-12）：固定 K=13824、N=5120，只改 M 让库选到 192×256 / 160×256 / 256×256 三种分块
> （`bw_probe/measure_pmc_bm.py`），每步低效是 1.68 / 1.84 / 1.30，而三者的面板/L2 是 0.63 / 0.53 / 0.84 ——
> **排序与"面板越小越好"的预测相反**。所以"面板/L2"不是主导因素，真正跳出来的是分块形状：
> 每步周期几乎不随分块面积变化（1288 / 1179 / 1331），而理想周期随面积变（768 / 640 / 1024），
> 于是窄分块单位算力更贵。`AICB算子实测_GPT7B_20261111.html` 第 5.2 节把 mlp_linear_2 归因于
> "面板压在 8 MiB L2"的那段，按此应理解为"同一分块下 K 变长时的退化"，不是普遍的容量规律。

---

## 2026-11-12 · 把缓冲维度从"全笛卡尔积"改成"两端可行组合"

**改了什么**
- `superpod_compute/contracts.py` — `Policy` 新增 `buffer_dims`，取值 `'max'`（默认）/`'search'`，并在 `__post_init__` 里校验
- `superpod_compute/operators.py` — `candidates()` 不再对 `l2_buffers × shared_buffers × output_chunk × output_buffers × register_input_buffers` 做笛卡尔积，改为调用新增的 `buffer_dimensions()`：`'max'` 每个（分块，几何）只产出"预算允许的最深"和"最浅"两个组合，`'search'` 复现旧行为

**为什么要改（问题）**
- 一个形状要枚举 95,232 个候选，单个 `predict` 58.3 s。其中真正意义上的分块选择只有 216 个（`bm` 6 × `bn` 6 × `bk` 6，各加 1 个按形状派生的值），其余 440 倍全部来自五个 2 值深度开关
- 这五个开关是历次加双缓冲、输出流水时逐个加上去的，没人回头看乘积

**怎么判断出原因的（测试）**
- `analyze_buffer_dims.py`：对 4 个形状（base / 1024×1024×128 / 512×5120×13824 / decode 64×4096×4096）跑全量枚举，把 112 个缓冲组合各自的最优解单独聚出来
- 结果：base 和 odd 的最优组合就是 `[2,2,4096,2,2]`；small 和 decode 的最优与它并列
- 单维边际（`attribute_buffer_dims.py`，把每个维度单独降一档、其余取最优）：

  | 维度降到 1 | base 4096×4096×16384 | odd 512×5120×13824 |
  |---|---|---|
  | `register_input_buffers` | **+41.7%** | **+33.2%** |
  | `shared_buffers` | +9.5% | +45.1% |
  | `l2_buffers` | +9.5% | +28.8% |
  | `output_buffers` | +0.5% | +0.4% |
  | `output_chunk`（4096 → 16384） | +0.1% | +0.1% |

  这些百分比是**模型内部**的敏感度，不是与实测的误差，也没有经过实测校验——L2/Shared 的缓冲深度在硬件上无法单独开关。
- 注意 `output_chunk` 的其它取值（512/1024/8192/12288）只出现在 16×16 这类小分块上（`min(chunk, bm*bn*2)` 被分块钳制），所以那一行不是干净的单维比较
- 也就是说：深的组合永不更差；浅的组合在可行时更差，只有不可行时才必须用（例：256×256×64 的两级输入要 128 KB，超过 64 KiB Shared）
- 另外发现 `rank_key` 在总时长相等时按**资源最少**排序（`shared_bytes` → `shared_buffers` → `l2_buffers`），所以只产出"最深"一个候选会丢掉这个平手规则——单 K 步的形状（K ≤ bk）里浅缓冲与深缓冲同速但应当胜出。第一版只产出最深，被 `test_one_panel_prefers_single_buffers` 抓住，因此改为两端都产出

**验证结果**
- `verify_buffer_dims.py`：12 个形状（9 个缩放点 + small / odd / decode），与全量枚举比对，最大偏差 **0.00%**，选中的分块与核心几何完全一致
- `check_search_mode.py`：4096×4096×16384 上 `max` 3,744 个候选 / 6.8 s，`search` 95,232 个 / 58.3 s，两者都给出 1777.7 µs、256×256×16 gm4×gn16
- 候选数 95,232 → 2,814~3,744（约 25 倍），单形状耗时 58.3 s → 约 7 s
- `python3 -m unittest discover -s tests`：108 个测试全部通过

**已知限制**
- "最优一定是两端之一"只在 4 个形状上验过。混合组合（如 small 的 `[1,2,4096,2,2]`）确实出现过，但它与最深组合同速，所以只影响选中的 mapping 不影响预测值
- 生成候选时每个（分块，几何）最多要做两次 `capacity()` 探测；加了一次基于 Shared 预算的廉价预筛，生成耗时约 0.9 s/形状
- 现在的主要耗时在 `evaluate()`（约 3,700 次 × 1.5 ms），继续压缩需要动评估本身，不在这次范围内

---

## 2026-11-11 · 新增矢量路径模型（softmax / gelu），物理层与库层分离（当前工作区，未提交）

**改了什么**
- `superpod_compute/vector_ops.py`（新增）— `VectorRequest` + `predict()`，覆盖 `softmax` / `gelu` / `gelu-tanh` / `layernorm`
- `superpod_compute/data/bw1100_vector_profile.json`（新增）— 库层常数：核切换阈值、各 dtype 每行/每元素周期、延迟系数、达成率
- `experiments/attention_20261110/predict_aicb_ops.py` — 追加两个矢量算子，覆盖率 7/12 → 9/12
- `bw_probe/measure_vector_anchors.py`、`measure_vector_sweep.py`、`measure_vector_validation.py`、`probe_vector_gaps.py`、`inspect_vector_kernels.py`（新增实测脚本）
- `experiments/attention_20261110/fit_vector_model.py`、`tune_latency.py`、`build_vector_html.py`（新增）
- 产出 HTML：`矢量算子模型_softmax_gelu_20261111.html`
- **没有改动** `contracts.py` / `operators.py`，108 个既有测试不受影响

**为什么要改（问题）**
AICB 的 GPT-7B 一次前向 12 个算子，此前 7 个 GEMM 有模型，softmax（占 22.9%）和 gelu（4.3%）没有。
沿用 roofline 会严重低估：softmax roofline 1201 µs / 实测 3376 µs，gelu 150.6 / 639.1。

**怎么判断出原因的（测试）**

1. **流锚点**（`measure_vector_anchors.py`）在算子自身 footprint 上重测：
   纯读 1842 GB/s、一读一写 1787 GB/s，两者几乎相同 → 读写共用总线、总字节决定耗时。
   同时排除了 `copy_`（908 GB/s，走 DtoD 拷贝引擎不是 kernel），必须有逐元素 `mul` 做锚点。

2. **两个算子都不是带宽受限**：3376 µs 下 1R1W 只到 636 GB/s，即使按 2R1W 算也只有 954 GB/s，
   远低于 1787。gelu 更极端：同尺寸 `mul` 只要 150.6 µs，实测 639.1（4.24 倍）。

3. **扫归约长度抓出核切换**（`inspect_vector_kernels.py`）：
   R ≤ 2048 走 `softmax_warp_forward<…, log2(R), …>`（行驻留寄存器），
   R ≥ 4096 走 `cunn_SoftMaxForwardGmem`（从全局内存分块重读）。两段成本规律不同。

4. **行内二次项**：gmem 区间用"每行 + 每元素"两项线性拟合，残差是以 R 为横轴的 U 形，
   fp16 与 bf16 形状一致 → 判为结构而非噪声，加 `row_quadratic` 显式建模。

5. **fp32 推翻了"按字节缩放"的假设**：gelu fp32 实测 55.2 周期/元素，fp16 57.0，几乎相同。
   说明瓶颈是元素流水不是字节，模型里去掉 `128 B/周期 ÷ 元素字节` 这条规则。

6. **低占用延迟平台**（`probe_vector_gaps.py`）：固定归约长度、只移动行数，
   16/32/64/128 行耗时完全一样（15.19 / 15.18 / 15.03 / 15.25 µs），越过约两千行才转线性。
   这就是 32 行张量在 R=4096 上只要 5.4 µs、在 R=2048 上要 15.5 µs 的原因。
   用 `latency_factor`（warp 8、gmem 1.15）乘单行成本表达，并在行数扫描上网格标定。

7. **排除分配坏点**：gelu 在 134 Mi 与 180 Mi 各测到过一次高出 50%，
   而 150/200/220/235/250/268 Mi 全部落在 0.89~1.00。同缓冲区反复分配会因页对齐换访问路径，这两点作废。

**过程中发现并修掉的模型 bug**
`row_cycles` / `row_quadratic` 最初只按 (算子, regime) 存，bf16 的拟合结果覆盖了 fp16，
导致 fp16 的 gmem 预测一直用 bf16 的行成本，验证集上表现为 +13% 的系统性偏差。
改成按 dtype 分开后，扫描平均误差从 4.7% 降到 2.5%。

**验证结果**
- 扫描 40 个形状（不含两个坏点）：平均 **2.5%**、最大 **10.3%**
- 12 个未参与拟合的 LLM 尺寸：平均 **8.3%**；其中 gelu 三个尺寸 ≤0.1%
- **layernorm 完全用 softmax 常数预测**（不是拟合）：4096×4096 **+6.3%**、4096×16384 **−4.5%**，
  说明"每行 + 每元素"这套规律可以从 softmax 迁移到另一个归约算子
- AICB 两个算子：softmax 131072×4096 = 3331.6 µs vs 3376.2（**−1.3%**）；gelu 67.1M = 639.7 vs 639.1（**+0.1%**）

**已知限制**
- 过渡带（128~512 行，行数刚过核数）偏慢 16~22%：`occupancy` 按整波次量化过于悲观
- 20 µs 以下的形状被 3.3 µs 的 launch 常数主导，±0.3 µs 就是 ±6%
- fp16 gmem 在不同总元素数下仍有 ~10% 的尺寸依赖（验证集 −10% ~ +9%）
- warp 区间在较小总量（1.3 亿元素）上偏快约 10%
- `achievement` 保持 1.0：所有分组的残差都随形状漂移，按纪律不吸收

---

## 2026-11-12 · 按厂商硬件参数表修正硬件 profile，并把矢量算力接进模型（当前工作区，未提交）

**改了什么**
- `superpod_compute/data/bw1100_mapper_mmac_hardware.json`
  - `shared_read_bytes_per_cycle_per_core` **128 → 64**（恢复实测值）
  - `vector_fp16_flops_per_cycle_per_core` **251.736 → 3.9334**，语义改为**每个 vector core**
  - 新增 `vector_cores_per_core=64`、`clock_peak_hz=1.6e9`、`l1d_bytes=32768`、`l1i_bytes=16384`、
    `l1d_bytes_per_cycle`（未知，置 0）、`resident_wavefronts_per_core=32`、`hbm_contiguous_bandwidth_Bps=2.0e12`
- `superpod_compute/contracts.py` — `Hardware` 增加上述字段（默认值保持旧 profile 语义，`vector_cores_per_core=1`），
  新增属性 `vector_flops_per_cycle`（每 CU 的矢量算力）
- `superpod_compute/vector_ops.py` — 输出 `vector_peak_flops_per_cycle`、`implied_flops_per_element`、`vector_ceiling_s`；
  修正 `scans`：逐元素算子取 1（原误取 2，把 gelu 的内存地板抬高 1.5 倍）
- 新增 `experiments/attention_20261110/report_vector_utilization.py`、`compare_shared_rate.py`

**为什么要改（问题）**
1. `vector_fp16_flops_per_cycle_per_core=251.736` 是 2026-11-05 之前按"29 TFLOPS ÷ 72 CU ÷ 1.6 GHz"算出的**每 CU**值，
   但字段名和设计者口径都是**每个 vector core**。按厂商参数表：29 T ÷ 72 ÷ 1.6 G ÷ 64 = **3.933 flops/cycle/core**，
   ×64 = 251.74 flops/cycle/CU，与原来的数一致，只是语义要对齐。
2. `shared_read_bytes_per_cycle_per_core` 在 2026-11-05 的 `db1467dd` 里被从 64 改成 128，
   128 是 32 bank × 4 B 的**理论峰值**；实测持续读是 63.87~63.97，厂商表写 63。文档至今仍写 64，与 json 矛盾。
3. softmax/gelu 模型只用 `cores`、`clock_hz` 和实测的每元素周期数，矢量算力此前**完全没被使用**，
   因此 `elem_cycles` 只是一个拟合常数，没有上限可对照。

**怎么判断出原因的（测试）**
- 用厂商参数表的 Tensor 行交叉验证：4 tensor core × 16×16×16 / 8 cycle × 72 CU × 1.55 GHz ≈ 458 T，与表里的 458 T 吻合，
  说明表的量级可信；由此反推矢量行。
- `compare_shared_rate.py`：在回归 policy（`effective_shared_read_bytes_per_cycle_per_core=0`，即直接吃硬件值）下，
  9 个形状在 64 与 128 两种读数下预测**完全相同（0.0%）**；强制 `engine-routed` 布局后仍然 0.0%。
  说明 S 阶段从未成为瓶颈，这个错误对既有 GEMM 结论**没有影响**。
- `python -m unittest discover -s tests`：**108 个测试全部通过**。

**验证结果**
- 矢量模型预测不变：扫描 40 形状平均 2.3%、最大 13.4%；12 个 LLM 尺寸平均 8.3%
- 拟合常数现在可对照硬件：`elem_cycles × 251.74 flops/cycle/CU` 得到"等效浮点/元素"
  —— softmax gmem 93、warp 210；gelu 223；gelu-tanh 188
- 按 ISA 粗估（erf gelu ≈30 flops/元素、softmax ≈10）折算利用率：**softmax gmem 10.7%、warp 4.8%、gelu 13.4%**
- 两条 roofline 对照：softmax 实测 3332 µs / 内存地板 1790 µs（1.9×）/ 矢量上限 22.8 µs（146×）；
  gelu 实测 640 µs / 内存地板 149 µs（4.3×）/ 矢量上限 2.8 µs（228×）

**待设计者确认（影响后续建模，不影响当前预测）**
- **矢量算力存在 8 倍冲突**：参数表 29 TFLOPS ⇒ 3.93 flops/cycle/vector core；设计者口头给的"每 vector core 32/cycle" ⇒
  64×32=2048/CU ⇒ 236 TFLOPS。两者相差 8.1 倍，且 29 T 与 458 T 的 tensor 之比（1:15.8）符合 GPU 常规，
  236 T 则为 1:2。需确认哪个是 fp16 的实际值。
- **L1D 的带宽**（参数表给了 32 KiB 容量但没给带宽；表里那句 63 byte/clock 对应的是 LDS 读）
- **L2 容量**（参数表该栏为空，当前 8 MiB 无出处）
- **bank 结构六个参数**（32 bank / 4 B / 64 lane / 16 B / 8 phase / 1 word）在任何文档里都没有出处

---

## 2026-11-12 · 用 L1D 溢出项替掉拟合出来的二次项

**改了什么**
- `bw1100_mapper_mmac_hardware.json` — `l1d_bytes_per_cycle = 64`（设计者确认：与 LDS 相同）；`clock_hz` 维持 1.3 GHz（确认按实测运行频率）
- `vector_ops.py` — 去掉 `row_quadratic`，改为物理项
  `spill = rows · max(0, L·b − l1d_bytes) / (l2_read_bytes_per_cycle · clock)`
- `fit_vector_model.py` — gmem 区间只用 **R ≤ 16384（行装得进 32 KiB L1D）** 的 3 个点拟合；
  R = 32768 / 65536 变成预测点

**为什么要改（问题）**
上一轮 gmem 区间的每行成本用一个二次项拟合，残差是 U 形但物理来源说不清。
设计者确认 L1D 带宽后，这条曲线有了机制：gmem 核把每行读两遍，第二遍在行不超过 32 KiB 时命中 L1D，
超出部分落到 L2。**16384 个 fp16 元素正好 32 KiB**，与扫描里每元素成本的最低点重合。

**怎么判断出原因的（测试）**
把二次项换成上述 spill 项后重跑拟合：
- 拟合点（4096 / 8192 / 16384）残差仍 ≈0
- **留出点 R=32768：bf16 +0.1%**，这是纯预测
- R=65536：fp16 −13.4%、bf16 −7.4%；R=32768 fp16 −7.7%
- 全表 40 形状：平均 **3.0%**、最大 13.4%（原二次项版本 2.3% / 13.4%）

**验证结果**
- 12 个 LLM 尺寸：平均 **8.3%**（与二次项版本持平）
- AICB softmax 3334.7 µs vs 实测 3376.2（−1.2%）
- 矢量利用率：softmax gmem **10.0%**、warp 4.8%、gelu 13.4%（按 ISA 估算的运算量作分子）

**已知限制**
- L1D 只解释了约一半的上翘，剩余部分没有再加拟合项，作为残差报告
- 用 L2 读带宽而不是 L1D 带宽计费：miss 必须穿过 L1D 再从 L2 取，瓶颈在 L2（3768 B/cycle 全片共享，低于 L1D 的 64 B/cycle×72）

---

## 2026-11-12 · 让模型真正跟着硬件走：每元素成本改为 运算量 ÷（矢量算力 × 效率）

**改了什么**
- `vector_ops.py` — 计算项从 `rows·(row_cycles + L·elem_cycles)/(cores·clock)` 改为
  `elem_cycles = ops_per_element / (矢量算力/周期 × efficiency)`，再进同一个式子
- `bw1100_vector_profile.json` — 去掉 `elem_cycles`，新增 `ops_per_element`、`efficiency`、`stream_bandwidth_ratio`
- `bw1100_mapper_mmac_hardware.json` / `contracts.py` — `vector_cores_per_core=64`、
  `vector_fp16_flops_per_cycle_per_core=3.9334`（每 vector core）、`l1d_bytes=32768`、`l1d_bytes_per_cycle=64`、
  `clock_peak_hz=1.6e9`、`resident_wavefronts_per_core=32`、`hbm_contiguous_bandwidth_Bps=2.0e12`
- 新增 `predict_on_hardware.py`：同一模型、同一效率，只改硬件参数，输出预测

**为什么要改（问题）**
原来的 `elem_cycles` 是"每元素多少 CU-cycle"，是这台机器上的实测值。
**换硬件时它不会变** —— 把矢量单元加宽一倍、把核数加倍，预测值纹丝不动。
这违背了这套模型的首要目的：在一台真机上标定，换参数预测另一台。

**怎么判断出原因的（测试）**
- `predict_on_hardware.py` 改前跑一遍：改 `hbm_bandwidth_Bps`、`vector_*`、`cores`、`l1d_bytes`，
  预测全部不变 —— 说明这些字段根本没进预测路径
- 改后同一脚本：矢量加宽一倍 gelu −50%、softmax −37%；核数加倍 −44%（翻到内存受限）；
  时钟 1.6 GHz 两者 −19%；L1D 加宽只影响长行（−6%）

**验证结果**
- 扫描 40 形状：平均 **2.9%**、最大 10.3%（改前 3.0% / 13.4%）
- 12 个 LLM 尺寸：平均 **8.0%**（改前 8.3%）
- AICB softmax：3210.6 µs vs 实测 3376.2（**−4.9%**，改前 −1.2%）
  —— 精度略降，因为效率现在是三个 gmem 点共用的一个数，而不是每元素/每行各一个
- 矢量效率：gmem softmax 9.5%、gelu 9.8%、warp softmax 4.2~4.7%

**已知限制**
- `efficiency` 是（库 × 硬件）的联合量。换同一代 DTK 下的不同芯片配置可以沿用；
  若更换了存储层级结构或换了库版本，必须重测。这是这套模型唯一躲不掉的标定。
- `row_cycles`（每行 block 建立与跨 lane 归约）保留在周期单位，**不随矢量宽度缩放**——
  它是延迟而不是算术，加宽矢量不会让 barrier 变快。
- `launch_s`、`row_capacity_elements` 也是这台机器/这套库的常数，换环境需重测。

---

## 2026-11-12 · 用多组已知模型配置验证 softmax 模型

**改了什么**
- 新增 `bw_probe/measure_softmax_configs.py`、`experiments/attention_20261110/build_softmax_configs_html.py`
- 产出 `softmax_多配置验证_20261112.html`

**为什么要测**
AICB 的模型定义表不在这台机器上（容器里没有 aicb 包），所以改用同一家族的 Megatron 单卡算子 trace
（`Astra&simAI/simai_result/`）：GPT-7B 在 TP=1/2/4 下本地头数分别为 32/16/8、seq 都是 4096，
再加一个小模型（hidden 1024、16 头、seq 512）。这些 trace 是 PyTorch 2.10 测的，本机是 2.11，必须重测。

**结果（本机重测 9 个配置）**
- 模型 vs 本机实测：平均 **5.6%**、最大 12.7%
- TP 切分三档：8 头 889.2 µs（模型 +7.1%）、16 头 1991.9（−4.6%）、32 头 3747.1（+1.2%）
  —— 模型只吃 rows 和 length，没有为不同 TP 单独调参
- 扩展组 seq 512/1024/2048/8192：−3.6% / +1.2% / −12.7% / +9.8%
- **旧 trace 与本机实测差很多**：TP1 配置 5641 µs vs 今天 3747 µs（+51%）、
  小模型 74.6 vs 53.1（+40%）。模型贴的是本机，不是旧 trace —— 这符合它的定位

**已知限制**
- 文档能推出的配置只有 4 组，其余是同一模型上的扩展点；这张表证明的是"没有为新配置调参"，
  不是"对任意配置都准"
- seq=2048（warp 区间）在 6.5 万行上偏低 12.7%，warp 区间仍有行数依赖没建进去

---

## 2026-11-12 · 按 SimAI 的 Megatron 模型表验证 softmax

**改了什么**
- 新增 `bw_probe/measure_simai_models.py`；`build_softmax_configs_html.py` 改为以 SimAI 模型表为主结构
- 产出 `softmax_模型验证_20261112.html`

**模型表来源**
zuchongzhi（`higon@10.17.173.52`）上的 `/home/higon/sunkai/superpod/SimAI/aicb/scripts/megatron_workload_with_aiob.sh`：

| model_size | layers | hidden | heads | ffn |
|---|---|---|---|---|
| 7 | 36 | 4096 | 32 | — |
| 13 | 40 | 5120 | 40 | — |
| 22 | 48 | 6144 | 64 | — |
| 175 | 96 | 12288 | 96 | — |
| 405 | 128 | 16384 | 128 | 53248 |
| moe | 32 | 4096 | 32 | 14336 |

DeepSeek 系列（deepseek671/236/16）用 MLA，没有独立 softmax 算子，未纳入。

**结果**
softmax 只看到 `(heads/TP, seq, seq)`，6 个模型 × TP=1/8 只产生 10 个不同形状
（本地头数 4/5/8/12/16/32/40/64/96/128，seq 4096，bf16）：

- 平均误差 **4.7%**、最大 **8.1%**
- 逐点：+6.9 / +6.9 / +7.1 / −8.1 / −4.6 / −0.2 / +1.1 / +3.5 / +5.3 / +5.7 %
- 模型只吃 rows 和 length，没有 per-model / per-TP 参数

**已知限制**
- 只覆盖 prefill（seq 4096）；decode 形状不在这一轮
- warp 区间（seq ≤ 2048）仍有行数依赖没建进去，seq=2048 偏低 12.7%
- 未覆盖 masked softmax（TP4 上实测与普通 softmax 差 0.07%，是单点观测）

---

## 2026-11-12 · decode softmax、gelu、layernorm 的多模型验证

**改了什么**
- 新增 `bw_probe/measure_model_ops.py`、`experiments/attention_20261110/rescore_measurements.py`
- `fit_vector_model.py` — layernorm 不再沿用 softmax 的效率，改为自己拟合
- `bw1100_vector_profile.json` — layernorm bf16 效率 0.0511 / row_cycles 93.60；fp16 效率 0.1057 / row_cycles 383.94
- 产出 `矢量算子总验证_20261112.html`（四组算子合并在一个页面）

**测了什么（bf16，seq 4096）**
- decode softmax `(local_heads, 1, kv)`：heads 4/16/32 × kv 1024~65536，8 个形状
- gelu：seq × ffn/TP，6 个模型 × TP=1/8，元素数 730 万 ~ 2.18 亿
- layernorm：`(seq, hidden)`，hidden 4096~16384，6 个形状

**结果**
- **decode softmax：平均 1.4%、最大 2.2%**（低占用延迟项在起作用）
- **gelu：平均 1.0%、最大 4.5%**，跨 30 倍元素范围，常数全部来自更早的拟合 → 纯留出
- **layernorm：平均 −43.7%，六个尺寸跨度只有 0.5%** —— 平坦的系统性偏差

**layernorm 的结论**
最初让 layernorm 沿用 softmax 的矢量效率（当迁移测试），一致偏慢 1.78 倍。
平坦残差说明不是噪声，而是 layernorm 的 kernel 确实比 softmax 用得更差（bf16 尤其明显）。
按纪律，平坦残差允许变成独立参数，因此改为单独拟合：
- bf16：row_cycles 93.60、效率 **0.0511**（6 点）
- fp16：row_cycles 383.94、效率 **0.1057**（2 点，数据弱）

**已知限制**
- layernorm 现在是拟合而非预测，6 个点拟 2 个参数，0.3% 是样本内数字；没有独立留出点
- fp16 layernorm 只有 2 个点
- warp 区间仍有行数依赖没建进去（seq=2048、6.5 万行偏低 12.7%）
- DeepSeek / Qwen 用 MLA 和 MoE，没有对应的裸 softmax/gelu 算子，未纳入

---

## 2026-11-12 · 登记待解决问题：softmax 过渡带高估 22%

**改了什么**
- 新增 `docs/待解决问题.md`，第一条记录 256~1024 行、length=2048 的高估问题
- 新增 `bw_probe/probe_transition_band.py`、`experiments/attention_20261110/analyze_band.py`
- `矢量算子总验证_20261112.html` 的"限制"一节加了指向该条目的说明

**为什么要登记**
用户确认这一带先不修、留待后期补测；目前只有这一个尺寸带超过 10%。

**怎么判断的（测试）**
- 同一形状 fp16 / bf16 并排测（32/128/256/512/1024/2048 行，length=2048）：
  误差曲线完全重合（+21.8% vs +23.5%），排除 dtype
- 三种低占用形式做对比：加性 9.0%/19.9%、地板 12.3%/30.2%、并发 37.9%/55.8%
  —— 现有的加性形式反而最好，所以没有可替换的现成公式
- `resident_wavefronts_per_core = 32` 反推：2304 条 wavefront 在飞，
  实测拐点 256~512 行 → 每个 block 4.5~9 条 wavefront（288~576 线程），与 PyTorch 典型 block 一致

**已知限制**
- 根因是"每行成本随并发 block 数衰减"这条曲线没建；需要把行数扫描做密（约 20~25 个点）
- 该区间未纳入当前任何一份验证结论；其它所有验证（4.7% / 1.4% / 1.0%）不受影响

---

## 2026-11-12 · 加密行数扫描：定位 warp 核低行数区间的误差来源

**改了什么**
- 新增 `bw_probe/measure_band_dense.py`（26 个点）、`experiments/attention_20261110/analyze_band_dense.py`
- 重写 `docs/待解决问题.md` 第 1 条：范围修正 + 根因定位

**测了什么**
三个归约长度（512 / 2048 / 4096）、行数 64~8192，fp16 为主、bf16 抽样，共 26 个点。

**结论**
- 问题范围比最初判断的更宽：**length=512 也中招（+28%）**，length=2048 是 +22.8%，
  而 length=4096（Gmem 核）几乎不受影响 → **问题出在 warp 核，不是某个特定长度**
- 反推每行成本后发现：`κ·rows` 在 64~256 行上稳定在 **515~520**（波动 0.5%），
  即 **C ≈ 517 行在飞**，对应每 block ≈4.5 条 wavefront（约 288 线程）
- 这与 `resident_wavefronts_per_core = 32` 自洽；bf16 同样方法给出 C ≈ 421
- 但 `κ·rows` 从 384 行开始上漂，说明不是硬性并发上限，而是"并发拐点 + 长尾"两个尺度叠加

**为什么没改模型**
四种形式在 26 个点上对比：加性 8.9%/28.0%（现有）、地板 12.3%/30.2%、
硬并发 37.9%/55.8%、平滑并发 17.3%/172.8%。**现有的加性形式仍是平均误差最低的**。
没有可替换的现成公式，继续调常数只是把误差从一端挪到另一端。

**已知限制**
- C 是从耗时反推的，没有 PMC 证据；下一步可以用 occupancy 计数器直接看每 CU 驻留几个 block
- 长尾（每行成本到 131072 行才饱和）的形状还没测清楚

---

## 2026-11-12 · 按 SimAI 模型分组的矩阵乘法验证

**改了什么**
- 新增 `bw_probe/measure_simai_gemms.py`：对 6 个 Megatron 模型 × TP=1/8 × 6 个矩阵乘算子做实测与预测
- 新增 `bw_probe/probe_gemm_dtype.py`、`bw_probe/probe_gemm_outliers.py`、`bw_probe/measure_outlier_fastest.py`
- `experiments/attention_20261110/build_softmax_configs_html.py` 新增第 6 节，其余章节顺延
- 重新生成 `矢量算子总验证_20261112.html`

**测了什么**
每层 6 个算子：QKV 投影 `(seq, 3h/TP, h)`、QKᵀ 与 PV（batched，头数为 `heads/TP`）、
注意力输出 `(seq, h, h/TP)`、MLP fc1 `(seq, ffn/TP, h)`、fc2 `(seq, h, ffn/TP)`。
seq=4096，bf16，按 Megatron 的 NT 调用方式（权存 `(N,K)`，算 `x @ w.T`），共 72 个点。
预测走 `predict(GemmRequest(...))` 与 `predict_batched_gemm`，策略与 100 组回归 V3 一致。

**结论**
- 剔除两个库退化点后：70 个点平均 **10.9%**，最大 40.7%；按算子 QKV 12.1% / QKᵀ 4.2% /
  PV 7.4% / 注意力输出 8.1% / fc1 14.5% / fc2 19.6%
- 误差不随模型规模单调变化（7B 6.0% → 405B 16.4%），说明不是对某个尺寸段过拟合
- 真正的结构误差集中在长 K：fc2（K=ffn）在 TP=1 上 6 个模型全部偏低，平均 28%；
  175B/405B 在 TP=8 的 qkv/fc1 也偏低约 30%
- 两个 † 点是库的可达性上限，不是模型误差：405B TP=1 QKV 与 175B TP=1 fc1 的默认核
  只有 28 / 41 TFLOPS（MT128x128x16），换 NN 布局也只到 64 / 67 TFLOPS，远低于模型的 ~322

**怎么判断的（测试）**
- fp16 与 bf16 在 3 个代表形状上差 <1%，且 profiler 显示同一种核，所以用 fp16 标定的模型对 bf16 测量
- 两个退化点单独复测：两种 dtype 结果一致（+0.3% / +0.1%），排除 dtype
- 尝试用 rocBLAS 的解枚举找更快核，这两个形状返回 0 个解；改用 NN 布局（权重预转置）
  实测 104 ms / 73 ms，仍远低于模型值

**已知限制**
- 大 K 形状偏低 20%~40% 是几何项缺失，不是参数问题；详见 `docs/待解决问题.md` 第 2 条
- 未纳入 lm_head/logits、MoE 专家路由和 TP all-reduce 通信

## 2026-11-13 · 矢量路径：启动延迟接到硬件字段，重拟 layernorm 的 warp 行成本（commit 留空）

**改了什么**
- `superpod_compute/data/bw1100_mapper_mmac_hardware.json` — `hbm_latency_s` 0.0 → `4e-07`（本机指针追逐实测 400 ns）
- `superpod_compute/vector_ops.py` — 新增 `startup = startup_round_trips × hardware.hbm_latency_s`（每次 kernel 收一次，不是每行、不是每 panel）；新增 `row_latency_factor()`，按算子覆盖每行临界路径因子；两者都写进返回值的 `startup_s`
- `superpod_compute/data/bw1100_vector_profile.json` — 新增 `startup_round_trips`（layernorm 4.27 warp / 6.83 gmem）和 `row_latency_factor`（layernorm = 1.0）；layernorm bf16 warp `row_cycles` 0 → 411.5、`efficiency` 0.041577 → 0.063692；fp16 warp `row_cycles` 同步填 411.5

**为什么要改（问题）**
- 随机 10 尺寸扫描里 layernorm 四个点超 30%：512×1024 +47.6%、1024×2048 +55.0%、2048×2048 +38.3%、8192×16384 −30.3%
- 前三个的共同点是 layernorm 的 warp 段没有自己的参数，整套继承 softmax：efficiency 相同、warp 段 `row_cycles` 缺省 0、`latency_factor` 是全局的 warp=8

**怎么判断出原因的（测试）**
- `bw_probe/probe_layernorm_gap.py` — hidden=1024/2048/4096/16384，rows 256~16384 扫描。行数大时反推每行成本 = **448 cycles 固定 + 0.743 cycles/元素**，模型是 **0 + 1.147**（把固定行开销摊进了每元素斜率，h=1024 附近恰好抵消，h=2048 偏慢 20%）；小行数时实测只多约 **1 µs 固定量**，而模型按 (8−1)×每行收 6.3 / 12.6 µs
- `bw_probe/probe_ln_placement.py`、`probe_ln_graph.py`、`probe_ln_control.py` — 8192×16384 的 −30%：换 spacer 0~1 GB、缓冲区偏移 0~256 MB、关掉 CUDA graph 用普通循环，都是同一个值；同尺寸 `x*2` / `clone` / `sum(dim=1)` 严格线性，只有 layernorm 在 200 MB 处跳档（1051 → 1825 µs），且 profiler 显示同一形状在 1338 / 2150 µs 两态之间跳 → 库的慢档，不建模
- `bw_probe/refit_vector_latency.py` — 汇总本机 240 个矢量测点重拟合。softmax 换成常数反而变差（5.5% → 6.6%），说明它的 ×8 是"行不重叠时的临界路径"，不是启动延迟，保持原样

**验证结果**
- `bw_probe/verify_vector_refit.py`：240 个矢量测点平均|误差| **8.5% → 4.1%**，最大 **109.7% → 30.4%**（剩下的是 8192×16384 慢档）
- layernorm warp 18 点：37.9% → **2.5%**（最大 7.7%）；layernorm gmem 74 点：11.3% → **5.4%**
- softmax / gelu 的点一个没变；矩阵乘路径不变（`charge_hbm_latency` 仍为 false，`hbm_latency_s` 只在矢量路径被读）
- `pytest`：102 passed

**已知限制**
- 8192×16384（输入 ≥200 MB）的库慢档不建模，报告第 10.2 节列为已知区间
- layernorm fp16 的 warp 组没有数据验证，row_cycles 与启动圈数按 bf16 结构填
- `startup_round_trips` 是在本机 400 ns 下标定的无量纲圈数；换到 HBM 延迟差异大的机器上应复核
- 2.5% 是**样本内**（a/b/启动圈数在同一批 18 个点上拟合）。按行数留出：只用 rows≤2048 拟合去预测 rows≥4096 是 2.0%（最大 4.2%）；反向是 9.4%（最大 21.5%）——行成本可外推，启动圈数必须由小行数据定。跨机器外推本机无法验证

---

## 2026-11-16 · batched.py 加 GEMV 路径，修复 decode PV 小 batch 高估 70%~135%（待提交）

**改了什么**
- `superpod_compute/batched.py` — 在 `predict_batched_gemm` 入口加 GEMV 分支：当 `m < 64`（填不满最小 tile 行）时，改用 HBM 带宽模型（A: `hbm_read_a_Bps`，B: `hbm_bandwidth_Bps`，C: `hbm_write_bandwidth_Bps`，固定开销: `device_service_s + kernel_launch_s`），跳过 GEMM tile 候选搜索；新增模块级常量 `_GEMV_THRESHOLD = 64`；m≥64 路径逐位不变

**为什么要改（问题）**
- decode 阶段 attn_pv（M=1~8，N=head_dim=128，K=kv=4096）当前预测约 30.4 µs，与 batch 完全无关；实测 batch=4 只需 7.3 µs，batch=96 是 66.8 µs——误差 +72% 到 +136%，PV 是 decode 误差最大的来源
- 根因：GEMM tile 路径以 bk=16 拆分 K=4096 为 256 个 panel，每个 panel 对 `duplicated` 项（L2 重复读）计一次，256 次累加后 `tr` 成为瓶颈，且总量与 batch 无关（`b_cols` 在 `_waves` 里按列数算，不随 batch 变化，而 M 只有 1）

**怎么判断出原因的（测试）**
- `bw_probe/probe_pv_kernel.py`：测 M=1，N=128，K=4096，batch=4~128，NN/TN 两种布局的实际时间 → TN 在 batch≤9 最快（7~9 µs），batch≥20 时 NN 反超；时间几乎线性于 `batch*(M*K + N*K)*2 / hbm_bw`
- `bw_probe/probe_pv_scan.py`：batch 从 1 扫到 128 的精细曲线 → batch=1~9 走 TN GEMV 路径（约 263~1080 GB/s 等效），batch=10~19 存在一个次优核区（库切换一次），batch≥20 NN 达到 1537 GB/s（89% HBM 峰值）
- 纯带宽模型 `t = device_service_s + kernel_launch_s + t_a + t_b + t_out` 的本地验证：PV decode M=1 48 个点平均|误差| 从 ~100% → 17%，PV decode M=8 从 ~80% → 17%，prefill M=1024 完全不变（走 GEMM tile）

**验证结果**
- 本地快速验证：PV decode M=1 batch=4 预测 5.59 µs（之前 30.42 µs），batch=128 预测 82.14 µs（之前 88.97 µs，更准），M=1024 batch=32 预测 78.62 µs（不变）
- `pytest`：102 passed（现有测试不覆盖 m<64 的 batched，不需要改测试）
- 返回结构与 `measure_prefill_decode.py` 调用方完全兼容（`best.duration_us` 键存在）

**已知限制**
- batch=10~19 的 PV（库次优核区）低估约 20~37%；这是库核选择问题，BW 模型本身是下界；比旧模型的 70%~136% 高估已大幅改善，可在将来通过测量该区间的实际核行为进一步修正
- QKT decode（M=1，N=4096，K=128）现在也走 GEMV 路径，比实测低 14~38%；QKT 的 K=128 较小，BW 模型低估说明有额外的计算/调度开销；这是已知的小 K 问题（见待解决问题第 5 条），不在本次修复范围
- `_GEMV_THRESHOLD = 64` 是本机实测支持的阈值（1~63 均测到走 GEMV 路径）；换机器要用相同形状复测确认

---

## 2026-11-16 · operators.py + hardware: K≥16384 大K效率台阶修正（待提交）

**改了什么**
- `superpod_compute/contracts.py` — `Hardware` 新增 `matrix_large_k_threshold: int = 0` 和 `matrix_large_k_ceiling_tflops: float = 0.0`；验证与现有字段一致（integer minimum=0，positive allow_zero=True）；默认值 0 表示禁用，不影响其他硬件配置文件
- `superpod_compute/operators.py` — `GemmModel.evaluate_order()` 末尾加一行 floor 约束：当 `request.K >= matrix_large_k_threshold` 且 ceiling > 0 时，`total = max(total, kernel_launch_s + device_service_s + 2*M*N*K / ceiling)`；只在此处加，逻辑简单、透明
- `superpod_compute/data/bw1100_mapper_mmac_hardware.json` — 新增 `matrix_large_k_threshold: 16384`，`matrix_large_k_ceiling_tflops: 2.1e14`（210 TFLOPS）

**为什么要改（问题）**
- prefill K≥16384 的矩阵乘预测值系统性偏低 25%~38%：gpt_7B fc2 -31%、gpt_175B fc2 -37%、gpt_405B fc1/fc2 -31%~-32%、gpt_405B TP8 fc1 -34%
- 模型当前在这些形状上按接近峰值效率（~335T）预测，而实测 NT 布局只有 185~225T

**怎么判断出原因的（测试）**
- `bw_probe/probe_k_threshold.py`：M=4096 N=6656，K 从 8192 扫到 18432。K=12288→13824 维持 ~280T，K=14336 NT 单布局掉到 225T（NN=296T，NT 局部问题），K=16384 两种布局都跌到 220T，K=17408 NN 回升到 250T，K=18432 NT=222T 再次掉档。结论：存在周期性掉档，不是单一平滑台阶
- `bw_probe/probe_k_threshold2.py`：精细扫 15k~20k，以及 SimAI 实际使用的 K=16384/20480/24576/49152/53248。NN 布局在 K=20480/24576 大多恢复正常（274T），但 NT 仍在 185~225T；K=49152 NN 也较慢（187~234T）
- 分析：NT 布局在大 K 周期性跌档，模型按 NT 布局（Megatron 实际使用的布局）预测 → 需修正
- 天花板选取：在 SimAI 9 个 K≥16384 形状（M=1024 和 M=4096）上枚举 ceiling_tflops，210T 使所有形状绝对误差 ≤11%（不含库退化点）；改为 240T 对小形状更保守但对大形状 K=49152 效果不佳

**验证结果**
- 本地快速验证：K=16384 gpt_7B fc2 预测 657µs（之前 465µs，实测 678µs）→ 误差 -3.1%；K=49152 gpt_175B fc2 预测 5893µs（之前 4108µs，实测 6545µs）→ -10.0%；K=53248 gpt_405B fc2 预测 8511µs（之前 5929µs，实测 8648µs）→ -1.6%
- `bw_probe/verify_k_fix.py`（服务器）：58 个点有变化（包括 tall_factor artifact），K≥16384 的 20 个真实改动点中 19 个改善、1 个轻微变差（gpt_7B fc2 M=4096 从 -18.5% 变 +20.2%，从模型乐观变保守）
- `pytest`：102 passed（新字段默认值 0 在 A100 配置文件不触发任何 floor，原有测试逐位不变）

**已知限制**
- K=14336 NT 的周期性掉档不在修正范围（NN 正常，属于 NT 路径对齐问题）；K=17408/18688 等"好档"与"坏档"交替，但 SimAI 实际形状（16384/20480/24576/49152/53248）都属于坏档区间
- gpt_7B fc2 M=4096（N=4096，小方形 tile）被 ceiling 过度约束（253T → 210T 天花板）：从 -18.5% 变 +20.2%。如后续想修复可把 ceiling_tflops 适当提高并补偿其他大形状
- 天花板值 210T 是在 BW1100 本机 NT 实测数据上标定的；换机器要用相同 K 形状复测确认
- 库退化点（gpt_175B/405B TP1 fc1/qkv_proj，实测比模型慢 3~5×）仍在 -87% 附近，属于库行为问题，本次不修

---

## 2026-11-16 · 新增库行为修正层（library_correction.py），对 116 个未收敛算子分桶拟合

**改了什么**
- `superpod_compute/library_correction.py`（新增）— 在物理模型和预测调用方之间插入修正层；`apply_correction(raw_us, op, kind, phase, m, n, k, batch, tp)` 是唯一入口；按 (kind, phase, N, K, batch) 分 9 个桶，每桶 `factor × raw_us + overhead_us`；`LibKind` 枚举标注每个桶的库行为类型（library_degrade / kernel_overhead / largek_residual / wave_pipeline / kernel_switching / vector_degrade）；所有非 PHYSICAL 桶均标注 `machine_specific=True`（换机重测）
- `bw_probe/regenerate_predictions_local.py` — 接入修正层；每个 case 写入 `lib_bucket`、`lib_kind`、`lib_factor`、`lib_overhead` 字段；HTML 生成不变

**为什么要改（问题）**
- `P和D完整预测_1116.html` 中有 116/462 个算子 |err|>15%
- 之前所有物理模型修复尝试（波间流水公式三种变体、input/output 重叠）均在修好一组的同时破坏另一组；根本原因是不同形状的库内核选择不同，单一物理公式无法同时拟合
- 物理模型应保持干净（描述硬件理论能力）；库行为差异单独一层处理，明确标注换机要重测

**9 个桶的系数来源**
- 系数来源：从 462 个算子的 `measured_us / model_us` 分布取中位数
- 写入实际系数的桶（factor≠1.0）：

  | 桶 | factor | 效果 (bad 变化) |
  |---|---|---|
  | B1_DECODE_SMALL | 1.073 | 17→15（改善 15 个，2 个轻微变差）|
  | C_PREFILL_GEMM | 1.070 | 12→7（改善 11 个，1 个变差）|
  | D_BATCHED_SMALL_B | 1.492 | 21→9（改善 20 个，1 个变差）|
  | D_BATCHED_MID | 1.055 | 22→16（改善 19 个，3 个变差）|
  | D_BATCHED_LARGE_B | 1.076 | 5→4（改善 4 个，1 个变差）|

- 保留 factor=1.0 的桶（有理由不拟合）：
  - **A_LIBRARY_DEGRADE**：比值 P25=1.67, P75=7.21，方差过大，单一系数无意义；标注为 library_degrade 供 HTML 单独标记
  - **B2_DECODE_LARGEK**：中位数 1.009，K floor 已基本覆盖，无稳定残差
  - **B3_DECODE_MID**：中位数 0.981，混合方向，改善 4 个破坏 3 个，净效果接近零
  - **E_VECTOR**：中位数 1.0005，向量模型均值已准；11 个 bad case 是 N≥12288 的大形状超出 profile 覆盖范围

**验证结果**
- factor=1.0 时（框架接入前）输出与原始 JSON 完全一致（462/462 条差异为零）
- 拟合系数写入后，bad cases 105→75（减少 30 个）；仍有 75 个超阈值，主要集中在 A_LIBRARY_DEGRADE（无法修）和 D_BATCHED_SMALL_B 的极端 batch（b=4~5 的 attn_pv）

**已知限制**
- 所有 `machine_specific=True` 的桶系数只在 BW1100 本机测定；换机器须对每个桶用同类形状重测校准
- A_LIBRARY_DEGRADE（N≥36864 TP=1）12 个形状全部保持 bad；需要比值表（per-shape）才能覆盖
- D_BATCHED_SMALL_B 的 factor=1.492 是中位近似，batch=4~5 的极端形状仍有 -50%~-68%（kernel switching 非单调，单一系数无法覆盖）
- factor 是乘法修正；对于由固定 kernel-launch overhead 主导的小形状（B1）应为加法；当前用乘法是近似，对极小形状（<10µs）欠准

---

## 2026-11-17 · 三处物理建模补齐：GEMV 连续流带宽、fat-GEMV 带宽效率、layernorm 慢档判据（commit 留空）

**改了什么**
- `superpod_compute/operators.py` — `GemmModel.input_hbm_bandwidth` / `input_a_hbm_bandwidth` 改为接收 `request`；新增 `_gemv_stream_is_contiguous` 与 `_gemv_hbm_efficiency_applies` 两个判据
  - M < 8 且 K 是 `matrix_gemv_contiguous_k_quantum`（4096 元素 = 8 KiB）的整数倍时，B 行按整页连续流式读，带宽取 `hbm_contiguous_bandwidth_Bps`（2.0 TB/s）
  - 8 ≤ M < 64 且 K ≥ 16384 且 K 是 `matrix_gemv_hbm_efficiency_k_alignment`（8192 元素 = 16 KiB）的整数倍时，fat-GEMV 内核带宽取 0.70 × 名义值（B 行步长按 2 的幂对齐 → L2 set 冲突别名）
- `superpod_compute/vector_ops.py` — `regime()` 新增 `gmem_large` 子状态，两个判据：大 hidden + 少行；或 footprint ≥ `layernorm_slow_footprint_bytes`（200 MB）
- `superpod_compute/contracts.py` — `Hardware` 新增 `matrix_gemv_hbm_efficiency`、`matrix_gemv_hbm_efficiency_k_threshold`、`matrix_gemv_contiguous_k_quantum`
- `superpod_compute/data/bw1100_mapper_mmac_hardware.json` — 写入 0.70 / 16384 / 4096
- `superpod_compute/data/bw1100_vector_profile.json` — 新增 `gmem_large` 效率（bf16 layernorm 0.03926 = gmem 0.051075 / 1.30）、阈值键与 footprint 键

**为什么要改（问题）**
- decode M=1、K 为 4096 整数倍、N ≥ 6144 的形状系统性高估 +15~+18%（如 M=1/N=12288/K=4096 模型 62.0 µs、实测 53.4 µs）：实测有效带宽 1.88~2.04 TB/s，高于模型的 1.72 TB/s。M=1 时每个输出列消费一整条 B 行，行字节数为 8 KiB 整数倍时全部命中合并访问，库达到连续流速率
- decode M=8、K ≥ 16384 的形状系统性低估 −26~−30%（如 M=8/N=4096/K=16384 模型 81.7、实测 117.0 µs）：实测有效带宽 1.18~1.31 TB/s（名义 1.72 的 0.69~0.76）
- layernorm rows=8192、N=16384 低估 −30%（模型 1355、实测 1943 µs）：输入 footprint 268 MB 超过库的慢档阈值；同形状 rows=4096（134 MB）完全准确，说明触发量与行数无关而与 footprint 有关

**怎么判断出原因的（测试）**
- 用 `bw_probe/prefill_decode_1116.json` 的 462 条实测反算每种形状的有效带宽 `2(MK+NK+MN)/(meas − device_service)`，按 K 对 4096 取模分组：K%4096=0 的 M=1 形状全部落在 1.88~2.04 TB/s，K%4096≠0 的落在 1.50~1.65 TB/s
- 对 M=8 大 K 形状用同一反算得到 0.69~0.76 的带宽比；用该值替换带宽后 6 个形状全部进入 ±9%
- layernorm 用 `vector_ops.predict` 反解效率：rows=8192/N=16384 需要 0.0393，恰好等于 gmem 效率 / 1.30，且与 rows<64 大 N 缺口所需的比值一致
- 排除的解释：不是 wave 量化（同 wave 利用率的形状误差却不同）、不是 K-floor 阈值（把阈值降到 6144/12288 会同时破坏 6~12 个原本准确的形状）、不是固定波次间开销（200 µs/过渡的假设会破坏 32 个形状）

**验证结果**
- M=1 连续流判据：9 个形状修复（+15~+18% → −0.2~+2.7%），0 个新破坏
- M=8 fat-GEMV：6 个形状修复（−26~−30% → −1.5~+8.6%），0 个新破坏；M=1 与 M=8/K<16384 不受影响
- layernorm：11 个形状全部进入 ±11%；rows=1024/4096 大形状（−3.5~+1.0%）不受影响

**已知限制**
- 三个判据的阈值与系数均为 BW1100 本机标定（换机必须重测）
- K%4096≠0 且 N 大的 M=1 形状（如 M=1/N=12288/K=6144）仍按 1.72 TB/s 建模，实测落在 1.50~1.65；这一组的库内核选择尚无可用物理量描述
- 服务器（hedr02 跳板）当次不可达，波次间同步开销、batched GEMV 小 batch、N≥36864 库退化三组未取得新实测，未改动

> 补充（同日）：fat-GEMV 判据加入 8192 对齐条件。首版只按 K≥16384 生效，
> 把 M=8/K=20480 与 M=8/K=53248 两个原本 −5% 的形状推高到 +34%/+36%；
> 这两条 K 不是 8192 的整数倍，实测带宽 0.92~0.95（不是 0.70）。
> 八个 M=8/K≥16384 实测点被"K%8192==0"干净分成 0.69~0.76 与 0.92~0.95 两档。

---

## 2026-11-17 · batched GEMM 按 PV/QKT 分离重构（`superpod_compute/batched.py`）

**改了什么**
- `superpod_compute/batched.py` — 用"精确字节量 + 分档 launch/流速率"模型替换原 GEMV 纯带宽路径与 tile 波次路径；PV（n≤k，输出窄）与 QKT（n>k，输出宽）分成两个结构域
- `superpod_compute/data/bw1100_bmm_profile.json` — 新增，记录全部标定量与每个量对应的实测依据
- `bw_probe/probe_bmm_pvqk.py` — 新增测量脚本（222 点）

**为什么要改（问题）**
- 原模型对 decode 小 batch 的 PV 低估 56~68%（M=1/n=128/k=4096/b=4：模型 5.6 µs、实测 17.7 µs），对 prefill QKT 大 batch 高估 20~33%
- 同一 batch 范围内 PV 与 QKT 的误差方向相反，单一公式无法同时覆盖（此前已用 116 个点验证过三种波间流水公式均失败）

**怎么判断出原因的（测试）**
- 服务器实测 222 点（decode m=1/8 × batch 1..128，prefill m=1024/4096，PV 与 QKT 分开，n=128/96 与 k=128/96，另有 m=1..256 的块并行度探针）
- 三次不同分配的重复测量散差 <1%，说明数值可重复；QKT 与 1116 会话的实测复现到 ±4%，确认机器状态一致
- 结构判据：PV 的时间由 A/K 流决定（prefill 斜率反算 1.05~1.13 TB/s，与单 GEMM 的 A 上限一致），QKT 由 C 写流决定（0.72~0.96 TB/s）；decode 小 batch 由"单块流式时间"floor 决定（每块 ~60 GB/s）
- 排除了：机器降频（QKT 复现）、地址敏感（三次分配一致）、L2 复用（各 member 独立，无跨 member 共享）

**验证结果**
- 报告内 96 个 bmm 单元：未收敛 48 → 7；222 个实测点中 189 个 ≤15%，中位误差 5.3%
- 剩余 7 个：n=96 的 prefill PV（-41%/-33%，库选核）、QKT b=5（-21%，内核切换点两侧无法同时满足）、decode PV 的 b=40/b=12（±20%，实测非单调）

**已知限制**
- bmm 的 96 个单元用 2026-11-17 的新测量（与 GEMM/向量单元的 1116 测量同协议但不同会话）；PV decode 大 batch 两次会话差最多 +30%，属库选核差异，故整组重测
- 所有系数为 BW1100 本机实测；《换机必须重测》
- `probe_m` 显示 m=64~256 的块并行行为仍有台阶（m=128 处跳变），这些形状不在报告内，未建模
