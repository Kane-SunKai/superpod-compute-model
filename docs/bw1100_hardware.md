# BW1100硬件配置：来源、换算和使用方法

配置文件：`superpod_compute/data/bw1100.json`。

这是兼容当前`Hardware`接口的**BW1100硬件配置（16×16物理阵列结构约定、每单元每周期1 MAC）**。按用户明确指定，本方案将每个Tensor Core视为一个16×16物理阵列，每个CU有4个这样的阵列；这项结构作为本方案的建模前提。直接来源是用户提供的两张硬件参数表；L2容量和当前卡可用显存补充自`real_bw1100_fp16_v2.json`。没有用三组GEMM实测时延反向拟合参数，没有修改模型算法，也没有把默认A100的启动开销带过来。

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
| `name` | BW1100-72CU-1.35GHz-mac1-v2 | 明确标识工作频率与MAC=1配置 |
| `cores` | 72 | 附图72 CU，实测设备信息同样报告72 |
| `clock_hz` | 1350000000 | 采用附图“主频估计1.35 GHz”；并非本次实测期间的监控频率 |
| `arrays_per_core` | 4 | 附图每CU有4个Tensor Core；按用户指定，每个Tensor Core作为一个16×16物理阵列建模 |
| `array_rows` / `array_cols` | 16 / 16 | 按用户指定，作为物理阵列的16行、16列，共256个阵列单元 |
| `mac_per_cell_per_cycle` | 1.0 | 按用户2026-10-28的明确要求，每阵列单元每周期1 MAC；不再按458 TFLOP/s反算 |
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

## 3. MAC=1对应的峰值和含义

按用户2026-10-28的明确要求，当前配置将`mac_per_cell_per_cycle`改成1。
每CU有4个16×16阵列，每单元每周期1 MAC，一次MAC按2 FLOP计：

```text
1.35 GHz：72 × 4 × 16 × 16 × 1 × 2 × 1.35e9
         = 199.0656 TFLOP/s
1.60 GHz：72 × 4 × 16 × 16 × 1 × 2 × 1.60e9
         = 235.9296 TFLOP/s
```

这两个值不等于附图的458 TFLOP/s。本次按用户指定采用MAC=1，不修改阵列数量、阵列尺寸或其他效率参数去凑458T。这个差异应保留为硬件描述与模型抽象之间的待解释项。

旧版`equivalent-v1`曾用458e12/(72×4×16×16×2×1.6e9)=1.9412570529513888反算等效吞吐；该旧值已停用。Git及诊断记录保留旧配置用于单变量比较。

`16×16×16`的前两个16按用户要求视为物理阵列行列，第三个16为一次矩阵运算的K长度；一次16×16乘16×16包含4096次MAC。流水填充、排空及实际指令时间由后端估算，不能仅用4096/256就代替完整kernel时间。

Vector字段仍由29e12/(72×1.6e9)=251.73611111111111 FLOP/(CU·cycle)获得；当前GEMM不使用它计费。本次只改变MAC参数和配置标签。

`clock_hz`仍为图中估计的1.35 GHz，不是实测监控到的频率。模型把L2 B/cycle乘该频率得到B/s；HBM的独立B/s字段不随之自动变化。

原阵列公式/周期表没有被宣称为BW1100实测表。修改MAC后必须重新搜索分块；硬件参数进入缓存键，旧缓存不会直接复用。

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
