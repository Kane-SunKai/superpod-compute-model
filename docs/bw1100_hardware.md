# BW1100硬件配置：来源、换算和使用方法

配置文件：`superpod_compute/data/bw1100.json`。

这是兼容当前`Hardware`接口的**BW1100等效硬件配置**。直接来源是用户提供的两张硬件参数表；L2容量和当前卡可用显存补充自`real_bw1100_fp16_v2.json`。没有用三组GEMM实测时延反向拟合参数，没有修改模型算法，也没有把默认A100的启动开销带过来。

## 1. 如何运行

在项目根目录，用已有Python环境执行：

```powershell
cd 'D:\03_Project_Work\02_WORK_Super node\compute_model'

.\run.ps1 gemm --m 3072 --n 2304 --k 1024 `
  --dtype fp16 `
  --hardware superpod_compute/data/bw1100.json `
  --output results/bw1100_regular.json
```

原有默认配置仍是A100，必须传`--hardware`才能选择BW1100。要一次预测原来的三组矩阵：

```powershell
$py = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py tools/benchmark_gemm.py predict `
  --cases examples/benchmark_gemms.json `
  --hardware superpod_compute/data/bw1100.json `
  --output results/model_bw1100.json
```

启动开销等未给定的费用暂未计入，因此输出还不能视为经过真机校准的完整调用时间。这里的“配置能运行”和“模型预测准确”是两回事。

## 2. 逐项配置

附图中的KB容量按KiB（1024字节）解释；带宽中的TB/s按十进制1e12 B/s解释。容量预算与真实容量的区别在表中注明。

| 模型字段 | 写入值 | 来源或处理方法 |
|---|---:|---|
| `name` | BW1100-72CU-1.35GHz-equivalent-v1 | 明确标识工作频率与等效配置 |
| `cores` | 72 | 附图72 CU，实测设备信息同样报告72 |
| `clock_hz` | 1350000000 | 采用附图“主频估计1.35 GHz”；并非本次实测期间的监控频率 |
| `arrays_per_core` | 4 | 附图每CU有4个Tensor Core；模型将其表达为4个等效阵列 |
| `array_rows` / `array_cols` | 16 / 16 | 使用16×16作为等效阵列形状；不是由三维指令尺寸证实的物理阵列 |
| `mac_per_cell_per_cycle` | 1.9412570529513888 | 用整卡FP16/BF16标称458 TFLOP/s及1.6 GHz反算的等效吞吐，见下节 |
| `matrix_k_quantum` | 16 | 以附图FP16的16×16×16中的K=16作为模型K补齐粒度 |
| `vector_fp16_flops_per_cycle_per_core` | 251.73611111111111 | 以Vector FP16 29 TFLOP/s反算；当前GEMM模型未使用此字段计费 |
| `hbm_bandwidth_Bps` | 1720000000000 | 采用附图HipStream 1.72 TB/s；作为当前统一读写服务带宽，不代表所有流量模式的实测带宽 |
| `hbm_capacity_bytes` | 154602045440 | 当前实测卡实际报告值，约143.984 GiB；避免混用图中144GB/96GB两种规格 |
| `hbm_latency_s` | 0 | 附图未给出，明确不计固定服务延迟；不表示物理延迟为零 |
| `l2_bytes` | 8388608 | 实测文件报告8 MiB，附图容量一栏未填写 |
| `l2_bytes_per_cycle` | 2990 | 原表读3768、写2990；接口仅有一个字段，采用两者中较小的2990用于读写，见第4节 |
| `source_sram_bytes_per_core` | 65536 | 在本接口中填可用于显式shared暂存的LDS 64 KiB，不把L1D/L1I加进来 |
| `shared_budget_bytes_per_core` | 65536 | 模型允许单个驻留任务使用每CU全部64 KiB LDS的预算；双缓冲合计也必须在64 KiB内 |
| `register_budget_bytes_per_core` | 786432 | 按附图VGPR 768 KiB作为每CU聚合预算；当前模型没有每线程/每波前寄存器分配约束 |
| `register_reserve_bytes_per_core` | 32768 | 沿用项目的32 KiB工作寄存器预留策略；这不是附图给定的BW1100硬件规格 |
| `kernel_launch_s` | 0 | 没有提供BW1100启动开销，暂不计费；没有沿用A100的21 µs，也未从host-event差值推断 |

## 3. 为什么矩阵吞吐是一个非整数

当前模型计算峰值的公式是：

```text
整卡FLOP/s = Core数 × 阵列数/Core × 行数 × 列数
           × MAC/(阵列单元·cycle) × 2 FLOP/MAC × 频率Hz
```

按附图表头72 CU、1.6 GHz，将458T解释为458e12 FLOP/s：

```text
等效MAC/(cell·cycle)
= 458e12 / (72 × 4 × 16 × 16 × 2 × 1.6e9)
= 1.9412570529513888
```

这只是让现有抽象的峰值与所给算力对齐，不表示真实物理单元每周期执行了非整数次指令。小数位来自公式计算，不表示原始458T有同样的测量精度。

`16×16×16`描述的矩阵指令形状不提供发射间隔/执行周期，不能直接推出每周期吞吐，也不能把最后一个16当成`mac_per_cell_per_cycle=16`。这里保留16×16等效形状，利用用户给出的整体吞吐约束服务速度。

Vector同理：

```text
29e12 / (72 × 1.6e9) = 251.73611111111111 FLOP/(CU·cycle)
```

本配置采用1.35 GHz工作频率，假定每周期吞吐不变，故：

```text
矩阵FP16/BF16等效峰值 = 458 × (1.35/1.6) = 386.4375 TFLOP/s
Vector FP16等效峰值 = 29 × (1.35/1.6) = 24.46875 TFLOP/s
```

这是频率线性缩放假设，不是新增实测值。若确认真机运行在1.6 GHz，只需改`clock_hz`为1600000000并修改name标签，矩阵峰值会回到458 TFLOP/s，不要再额外改MAC系数。换频率时模型的L2 B/s也跟随变化；HBM的独立B/s字段保持不变。附图没有给出各时钟域关系，当前是模型的统一时钟约定。

该等效系数在现有阵列后端中还会缩放周期表，包括填充/排空代价。因此对齐峰值不等于验证了BW1100各tile的真实阵列时延。后续需用设备kernel或微基准验证，不能把原LLMCompass阵列周期表当作BW1100实测表。

## 4. L2读写带宽怎样处理

原始参数是整卡共享L2的：

```text
只读：3768 B/cycle
只写：2990 B/cycle
```

按模型1.35 GHz换算，分别是5.0868 TB/s和4.0365 TB/s。当前`Hardware`只有`l2_bytes_per_cycle`，`operators.py`用它同时计算L2→Core和Core→L2时间。因此本配置选较小的2990：没有把读写速率相加，也没有把整个设备的L2速率乘72。

这是兼容接口的保守服务率选择。它会把原表只读服务时间放大约26%，并不代表真实L2读写带宽相同；也不构成完整模型时延的严格上界。要精确表示原表，需要扩展读写两个速率字段及相应计费代码，还要明确读写是否共享端口，不能仅增加JSON字段就生效。

## 5. 哪些原始信息暂时无法在当前模型生效

| 附图数据 | 当前处理 |
|---|---|
| L1D 32 KiB、L1I 16 KiB | 当前模型无独立L1命中率/流量模型；没有并入shared容量 |
| LDS 64 KiB | 已作为shared容量约束生效 |
| `63 byte/clock` | 附图位置靠近LDS，但未明确是LDS何种读写方向、每CU还是其他统计范围；当前接口也没有独立shared带宽，未擅自映射为L2带宽 |
| VGPR 768 KiB | 已用于每CU聚合寄存器容量；不能据此保证任意大tile在真实线程分配下均可实现 |
| 每CU 64个Vector Core | 原始信息保留于本文，当前接口用聚合Vector吞吐表示，不另乘一次64 |
| FP64/FP32/TF32、FP8/INT8算力与FP8 16×16×32 | 首版只支持FP16/BF16 GEMM，不能借此JSON启用其他dtype |
| 连续地址2.0 TB/s | 保留为另一个带宽场景；需比较时复制配置并把`hbm_bandwidth_Bps`改为2000000000000，不与1.72 TB/s叠加 |
| HBM 96GB规格 | 当前测量的是约144 GiB卡，不采用96GB值 |

尤其是`63 byte/clock`：如果之后确认它是每CU的LDS服务带宽，必须新增shared→寄存器/计算单元的流量与时延建模，才能体现这一限制。当前流水线只显式计HBM↔L2和L2↔Core服务，**不能声称已经覆盖附图全部存储层级的带宽瓶颈**。

## 6. 本配置如何理解

已完成的是“把有来源的数据映射到当前模型、明确缺失参数和接口近似”。不是宣称BW1100模型已经完成精度校准。

特别是启动延迟、固定HBM延迟、输出转换/整理和LDS内部服务并未全部建模。配置中的0代表省略计费，不能解释为真实硬件零开销。与实测对比时应保留这些边界，不通过偷偷保留A100默认值来补齐。

配置附带接口验证记录`docs/bw1100_profile_validation.json`：检查字段加载、峰值换算、预算可行性，并运行三组算例的mapper。记录只用于确认接口可运行，不用于声明真机准确率。
