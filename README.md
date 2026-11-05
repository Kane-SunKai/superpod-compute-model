# Superpod Compute Model

> 当前统一开发基线为 **3.4**：自有 mapper＋MMAC8，显式操作数布局同时约束 Shared 容量、输入写入、操作数寄存器和 bank 读取服务。BW1100 使用 `bw1100_mapper_mmac_hardware.json` 与 `bw1100_mapper_mmac_policy.json`。正常预测不需要实机分块。13组普通 Prefill 独立验证的 MAPE 从8.77%降至5.12%，≤10%通过数8→9；20组Decode仍为5/20，GPT主GEMM3/4，注意力BMM尚未收敛。没有宣称所有算子达到10%。[当前模型与运行说明](docs/BW1100_当前模型与运行说明.md)；[可选实机诊断](docs/自有Mapper与实机诊断_使用说明.md)。本轮没有将按验证案例拟合的修正系数加入模型。

> 历史版本2026-10-29 **V3.2**：已补充寄存器→Shared输出写服务、L2读3768/写2990 Byte/cycle；Shared输出读回不计时。两种算力方案同口径对照：MMAC规则组41/50、256 MAC/cycle规则组34/50（≤15%），默认继续MMAC。详见 [V3.2修改与实验报告](docs/BW1100_V32_输出流水与算力对照.md)。

> 新版学习入口：[总体方案与完整流水线](docs/单卡计算建模_总体方案与流水线.html) · [简明User Guide](docs/计算模型_User_Guide.md)。主图包含Shared→寄存器、C写回和HBM/L2复用，可切换对照。

> 历史版本2026-10-28 **V3.1**：BW1100通过独立MMAC/LDS微基准核实，采用8周期指令服务和64 B/CU/cycle Shared读取；加入Shared→A/B寄存器流水及输入寄存器容量检查。原60组回归41组误差≤15%，中位误差7.29%，平均16.55%，19组仍未达标。详见 [V3.1依据与结果](docs/BW1100_V31_指令与LDS核实.md)。A100默认继续使用array后端。下文V2及V1描述为历史背景。

> 2026-10-28 V2：已实现C分片输出、容量受限的A/B输入保留、分段流水线及显式设备固定服务项。72组实测（12校准+60验证）中，35/60验证误差≤15%，尚未达到任意尺寸≤15%的目标。详见 [V2验证报告](docs/BW1100_V2_72组验证.md) 与 [交互设计说明](docs/计算建模总体方案_K分块与双缓冲Mapper.html)。后文原有冷输入及整块C容量说明为V1历史描述，V2规则以报告为准。


独立的单GPU计算建模模块，首版实现 **GEMM：M/N分配独立输出，K分批输入，C驻留到完整K结束**。没有修改原LLMCompass、ASTRA-sim或SimAI项目。

默认使用A100配置及阵列计算后端，联合搜索分块、Core二维分组、L2/shared单缓冲或双缓冲。无需GPU，运行时只用Python标准库（Python 3.10+）。Softmax和LayerNorm尚未实现，已提供算子插件及通用事件/资源接口。

## 1. 直接运行

在Windows PowerShell中：

```powershell
cd 'D:\01codex_work\03_super node\compute_model'

# 第一次：搜索；再次执行同一命令：精确查表
.\run.ps1 gemm --m 3072 --n 2304 --k 1024 --output results/array_search.json

# 固定教学分块，导出代表性波的前8个K面板事件
.\run.ps1 evaluate --m 3072 --n 2304 --k 1024 `
  --mapping 128,128,128,12,9,2,2 --trace-panels 8 `
  --output results/fixed_mapping.json

# 非整除尺寸
.\run.ps1 gemm --m 130 --n 97 --k 270 --output results/tail.json

# 峰值公式对照后端；不会与阵列后端误用同一缓存
.\run.ps1 gemm --m 3072 --n 2304 --k 1024 --backend peak `
  --output results/peak_search.json
```

`run.ps1`依次查找`SUPERPOD_PYTHON`、PATH中的Python、本机已有的Codex Python运行时。可显式指定你自己的解释器：

```powershell
$env:SUPERPOD_PYTHON = 'C:\path\to\python.exe'
```

在Linux/WSL或已有Python环境中，从仓库根目录运行：

```bash
python -m superpod_compute gemm --m 3072 --n 2304 --k 1024 --output results/array_search.json
python -m unittest discover -s tests -v
```

也可将本项目安装为包：`python -m pip install .`，之后使用`superpod-compute`命令。本机没有自动修改全局Python环境。

## 2. 输入、输出与使用边界

输入是**本卡局部算子**的M/N/K，既不自动做TP切分，也不包含跨rank通信。支持FP16/BF16输入与输出、FP32累加、C=AB。`--layout`可描述普通连续B或连续权重W的转置视图；尚未完整建模行距/地址相关的访问效率。`--dtype bf16`使用当前相同16位服务配置并区分缓存身份。任意strided布局、bias/激活融合、稀疏与量化尚未实现。

输出JSON的重要字段：

| 字段 | 含义 |
|---|---|
| `best.mapping` | bm/bn/bk、gm/gn及两级缓冲数 |
| `best.duration_s` / `duration_us` | 当前模型的预测时间；单位明确分开 |
| `best.capacity` | shared输入/输出峰值、FP32 C与工作寄存器、L2预算 |
| `best.wave_classes` | 完整波/边界波的数量、活跃Core、K尾段、计算来源和阶段时间 |
| `hbm_read_bytes` / `l2_to_core_bytes` | 唯一HBM输入与送至各Core的副本流量，分别统计 |
| `output_bytes` | 最终C只写一次；不包含中间C写回 |
| `array_sources_wave_panels` | 多少个波面板来自公式、LUT或OS未命中估算 |
| `searched_candidates` / `feasible_candidates` | 枚举数、资源合法数；不是任意程序空间的全局最优 |
| `cache_hit` / `signature_sha256` | 是否精确查表及其身份 |
| `scope_notes` | 当前预测省略项、资源模式与精度边界 |

默认时间包含kernel启动、冷波输入搬运、块内计算、最终L2/HBM写回。**输出转换/布局整理的独立费用默认是0，尚未标定；结果不是GPU实测，也不是已验证准确的完整CUDA kernel时延。** 可在策略中设置`epilogue_s_per_wave`。所有算法测试不能替代硬件误差验证。

## 3. 硬件参数与策略怎么改

```powershell
.\run.ps1 hardware --output examples/my_hardware.json
.\run.ps1 gemm --m 3072 --n 2304 --k 1024 `
  --hardware examples/my_hardware.json --policy examples/policy.json `
  --output results/custom.json
```

默认profile：`superpod_compute/data/a100.json`。来源与自己的使用预算分开：

| 项目 | 默认值 |
|---|---:|
| Core数 / 频率 | 108 / 1.41 GHz |
| 每Core矩阵阵列 | 4个16×16，每单元1 MAC/cycle |
| HBM带宽 / 容量 | 2039 GB/s / 80 GB，按十进制 |
| L2容量 / 服务率 | 40 MiB / 5120 B/cycle |
| 原版抽象SRAM | 192 KiB/Core，仅保留来源值 |
| 本方案shared预算 | 128 KiB/Core |
| 本方案寄存器预算 / 工作预留 | 128 KiB / 32 KiB |
| C累加 / 输出 | FP32 4 B / 16位 2 B |
| GEMM启动开销 | 21 µs，来自原版overhead配置 |

`examples/policy.json`定义候选集合、效率、传输模式、额外时延与Top-K数量。`transfer_resource_mode`可以取`independent`或`shared-port`：后一种使H/R事件争用同一传输端口，不能无条件重叠。它仍是聚合服务模型，不模拟L2 bank和具体SM端口结构。

M/N尾块只搬有效元素，但按所选输出矩阵块的槽位执行；K尾块按16的矩阵粒度补零计算。M=1允许部分Core空闲，并保留padding损失，不偷用整卡峰值；尚未提供GEMV专用优化。A+B+C超过HBM容量会拒绝，未实现out-of-core。

## 4. 阵列后端做了什么

默认`--backend array`：

1. 把一个Core的矩阵块按M/N分给其矩阵阵列；阵列间K划分固定为1，避免额外归约。
2. 对各合法M/N阵列分配比较周期，取最快者。
3. 按LLMCompass的快速利用率条件估算；否则查捆绑周期表。
4. 查表未命中则使用独立OS折叠波前公式，并标为`own-os-wavefront-estimate`。
5. 周期除以频率得到秒，放入C计算事件。HBM/L2搬运在外面计费，不重复计入。

原版来源、许可证及转换说明见[THIRD_PARTY.md](THIRD_PARTY.md)。未命中公式与捆绑的1,049个不同查表键逐条一致，但这不是任何阵列/数据流下的SCALE-Sim等价证明。项目不调用SCALE-Sim，不把离线模型周期称为实测。

`--backend peak`仅用于对照：按每Core峰值与效率计算。默认阵列后端的tile形状与启动/排空代价会影响mapper的选择，因此不保证与HTML峰值演示选出相同bk。

## 5. Python接口

```python
from superpod_compute import GemmRequest, Hardware, Policy, predict

request = GemmRequest(M=3072, N=2304, K=1024, dtype="fp16")
result = predict(request, Hardware.load(),
                 policy=Policy(), backend="array",
                 cache_path=".cache/mappings.sqlite3")
print(result["best"]["mapping"])
print(result["best"]["duration_s"])
```

`cache_path=None`时库接口不写缓存。CLI默认使用仓库工作目录的`.cache/mappings.sqlite3`；`--no-cache`关闭，`--force`强制重搜。`--output`是可选项，不会每次自动创建一整套结果目录。

缓存键包含算子/尺寸/精度、完整硬件配置、搜索策略、阵列LUT内容哈希、后端及模型版本。相似尺寸不直接复用最优结果。命中后再检查mapping是否合法及其评分；损坏记录不会按正确结果返回。

```powershell
.\run.ps1 cache count
.\run.ps1 cache clear
```

## 6. 如何继续支持Softmax、LayerNorm

`operators.py::OperatorModel`定义插件契约。新增算子负责请求校验、合法算法与分块候选、容量账本、事件生成/评分及映射序列化。`EventScheduler`负责资源与依赖，通用搜索与SQLite缓存不依赖GEMM字段。测试中用一个非GEMM映射插件验证了这一点；生产环境目前只注册GEMM。

后续Softmax应显式构造max/exp/sum/normalize等事件，LayerNorm应构造统计量与归一化事件，再接入向量、特殊函数和归约后端。不能把矩阵阵列吞吐直接拿去算exp。详细接口和数据流见[docs/architecture.md](docs/architecture.md)。

## 7. 验证、结果与Git

真机对比使用新增的`tools/benchmark_gemm.py`：`measure`在真实GPU上运行三组GEMM，`predict`运行计算模型，`compare`读取实测文件并生成JSON/CSV对比表。具体命令、计时范围、算子尺寸和硬件配置修改方法见[真机测试指南](docs/real_gpu_benchmark.md)。

```powershell
$py = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -m unittest discover -s tests -v
& $py tools/validate.py
```

测试覆盖逐元素分块数学、全部来源LUT、阵列分配、容量、四种缓冲组合、共享端口、加速流水与独立DAG对照、长K快进、非整除、少量活跃Core、搜索最小值、缓存命中/失效/损坏处理和算子扩展接口。`tools/validate.py`会生成可复现的案例报告；不会启动GPU或网络任务。

已完成的数值和测试记录见[验证报告](docs/validation.md)。可单独运行`python tools/audit_lut.py`核查捆绑的阵列周期表；无需保留原LLMCompass目录。

本项目是独立本地Git仓库，不设置远端。代码、配置、必要来源数据、测试和验证摘要会提交；`.cache`、临时结果、Python缓存和虚拟环境不提交。可以运行`git log --oneline`查看备份点。当前机器没有Git用户身份配置，因此自动生成的备份提交使用`Codex Local Backup <codex@localhost>`，未修改全局Git配置。
