# 三组GEMM的真机测试与模型对比

测试入口：`tools/benchmark_gemm.py`。算例配置：`examples/benchmark_gemms.json`。

这里的“三种”指三组不同形状的矩阵乘法，都执行 **C[M,N] = A[M,K] × B[K,N]**，不含bias、激活、反向传播或多卡通信。它们与首版计算模型的三个验证算例一致。

| 名称 | M | N | K | A形状 | B形状 | C形状 | 要观察什么 |
|---|---:|---:|---:|---|---|---|---|
| regular | 3072 | 2304 | 1024 | 3072×1024 | 1024×2304 | 3072×2304 | 常规分块、多个Core和多波输出的综合表现 |
| non_multiple | 130 | 97 | 270 | 130×270 | 270×97 | 130×97 | 尾块、补齐、低并行度，以及小算子的启动开销 |
| single_row | 1 | 8192 | 8192 | 1×8192 | 8192×8192 | 1×8192 | 类似decode线性层的单行形状；不是完整decode模型 |

默认FP16输入/输出。在支持的PyTorch版本中关闭半精度中间归约及FP16累加选项，朝FP32累加口径对齐；实际指令由设备库决定。BF16可通过参数切换。矩阵连续存储，输出预分配，执行`torch.mm(a, b, out=c)`。

## 1. 在真实机器上怎么运行

使用你之前能运行AIOB的Python环境，先确认设备可用：

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.version.cuda, 'HIP:', torch.version.hip); print('GPU available:', torch.cuda.is_available())"
```

CUDA版和HIP版PyTorch都使用`torch.cuda`接口；HIP设备也写`cuda:0`，不是`hip:0`。BW1100等设备需使用厂家提供的兼容PyTorch环境，本脚本不会安装或替换环境。[PyTorch HIP说明](https://docs.pytorch.org/docs/main/notes/hip.html)

将项目复制到真实机器。**如果只做实测，只需复制`tools/benchmark_gemm.py`和`examples/benchmark_gemms.json`，不依赖计算模型包。** 下面命令在项目根目录执行；如果放到了其他目录，修改文件路径即可。

```bash
python tools/benchmark_gemm.py measure \
  --cases examples/benchmark_gemms.json \
  --device 0 \
  --dtype fp16 \
  --warmup 30 \
  --iterations 100 \
  --cache-mode both \
  --flush-mib 256 \
  --output results/real_gpu_fp16.json
```

| 参数 | 含义 |
|---|---|
| `--cases` | 矩阵尺寸配置；不传则使用上述三组内置算例 |
| `--device 0` | 当前进程可见的第0张GPU；一次只在一张卡上算 |
| `--dtype fp16` | 覆盖配置文件的精度；可换成`bf16`，需要真机支持 |
| `--warmup 30` | 每个算例、每种缓存模式先执行30次，避开首次初始化等影响 |
| `--iterations 100` | 正式计时100次，每次是一次完整GEMM，不是100次时间相加 |
| `--cache-mode both` | 分别测试warm和evict两种模式，也可以只指定其中一种 |
| `--flush-mib 256` | evict模式每次计时前访问256 MiB的独立缓冲区，冲刷缓存；该操作不计入GEMM时间 |
| `--output` | 保存单个JSON文件；相同路径再次运行会覆盖，父目录自动创建 |

`warm`重复使用相同A/B/C地址，不主动清缓存；数据大于缓存时仍有HBM访问。`evict`在每次GEMM前写入大缓冲区以降低旧数据驻留的影响，是**近似冷缓存**，不是硬件强制失效指令。缓冲区建议至少大于实际L2容量的2倍；256 MiB适用于这里的40 MiB A100 L2假设。如果目标GPU L2更大，应增大该参数，显存不足则按真实L2容量调整。

如果只复制一个脚本，也能使用内置三组算例：

```bash
python benchmark_gemm.py measure --dtype bf16 --output real_bw1100_bf16.json
```

脚本不退回CPU计时。没有可用GPU或实测精度不受设备支持时，会报错退出。采样正确性检查失败也会退出，并保存诊断数据。

## 2. 计时过程是什么

每个算例依次执行：

```text
在GPU分配A/B/C → 随机初始化A/B → 运行一次GEMM
  → 抽取最多8行×8列，在CPU上以FP64沿完整K计算参考值
  → 校验输出 → 预热30次 → 同步GPU
  → 正式计时100次 → 保存每次时间和统计值
```

正式计时的一次流程：

```text
evict模式才做：写缓存冲刷缓冲区 → 等待写操作完成
记录GPU起始事件
    torch.mm(A, B, out=C)
记录GPU结束事件 → 等待结束事件完成
读取两个事件之间的毫秒数，乘1000转换为微秒
```

随机数生成、A/B/C分配、CPU/GPU输入复制、预热、正确性检查、缓存冲刷不计入GEMM事件时间。矩阵库内部若有必要的转换或辅助kernel，只要在同一调用的事件区间内，就会计入。PyTorch可自行选择矩阵库、kernel和分块，不会按我们的mapper结果执行。[torch.mm接口](https://docs.pytorch.org/docs/stable/generated/torch.mm.html)、[GPU事件计时](https://docs.pytorch.org/docs/stable/generated/torch.cuda.streams.Event.html)

JSON保存两种时间，单位都是µs：

| 字段 | 含义 | 如何使用 |
|---|---|---|
| `gpu_event_us.p50` | 100次GPU事件区间的中位数 | 主要观察指标；含可能的提交间隙，尤其小算子不能视为纯指令执行时间 |
| `gpu_event_us.p90` / `stdev` | 较慢分位数与波动 | 判断机器负载、频率或计时噪声是否影响结果 |
| `host_observed_us.p50` | CPU侧从调用前至等待结束事件返回的耗时 | 包含Python、运行时、结束事件记录和同步，不等于纯GPU时间 |
| `gpu_event_samples_us` | 所有原始GPU计时样本 | 复核分布，不仅保留一个平均值 |
| `effective_tflops` | 有效运算量`2MNK`除以事件中位时间 | 衡量该形状的有效吞吐，不是GPU硬件峰值 |

同一文件还保存GPU名称、设备可见编号、显存容量、PyTorch/CUDA/HIP版本、实际精度开关、脚本哈希、参数与采样数值校验。不会自动锁频或改变功耗设置；正式对比应尽量在GPU空闲且温度/频率稳定时执行，并重复一次观察波动。

## 3. 在本机预测，再拿实测文件自动对比

预测不需要PyTorch，也不需要GPU。Windows PowerShell中：

```powershell
cd 'D:\01codex_work\03_super node\compute_model'
$py = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'

& $py tools/benchmark_gemm.py predict `
  --cases examples/benchmark_gemms.json `
  --hardware superpod_compute/data/a100.json `
  --output results/model_a100_fp16.json
```

把服务器生成的JSON复制到本机`results/real_gpu_fp16.json`，然后：

```powershell
& $py tools/benchmark_gemm.py compare `
  --measured results/real_gpu_fp16.json `
  --hardware superpod_compute/data/a100.json `
  --output results/comparison_a100.json
```

得到`comparison_a100.json`和可用Excel打开的`comparison_a100.csv`。compare直接读取**实测文件里的M/N/K和dtype**来预测，避免手工重复输入时尺寸或精度不一致。使用其他GPU时必须换成对应的硬件JSON；实测设备信息不会自动转换成完整硬件模型。

表里同时保留：

1. GPU事件实测时间和CPU观察时间。
2. 模型总时间，包含硬件配置里的`kernel_launch_s`（默认21 µs）。
3. 模型总时间减去这项配置启动开销后的时间。
4. 两种模型时间分别相对于GPU事件时间的原始百分比差异：`(预测/实测−1)×100%`；正数表示预测偏慢。

**第3项是排查计时范围的辅助列，不是自动修正后的“正确预测”。** 事件时间、Python调用时间和配置启动开销不是完全相同的概念，不能把21 µs无条件加到实测或从实测中减去。

另外，模型假定每个输出波重新从HBM读入，真机evict只在整个GEMM之前冲刷缓存，实际kernel内部仍可能跨波复用；真机矩阵库也可能采用GEMV或split-K等优化。因此，这次对比是在检查“我们的模型对实际矩阵库性能有多好的预测能力”，不是验证真机严格执行同一条分块流水线。若差异较大，先检查形状、精度、硬件、缓存和启动范围，再分析映射策略，不要仅调一个效率系数把单个点强行拟合。

## 4. 如何修改算子尺寸

修改`examples/benchmark_gemms.json`，例如只保留一组：

```json
{
  "dtype": "bf16",
  "cases": [
    {"name": "my_projection", "M": 4096, "N": 3072, "K": 4096}
  ]
}
```

表示A为4096×4096、B为4096×3072、C为4096×3072。`M/N/K`必须是正整数，各组name不得重复。可增加任意多组，不限三组。

M通常可对应`batch×序列长度`，K是输入特征数，N是输出特征数。如果模拟TP切分，填**分给这一张卡的局部形状**；脚本不自动按GPU数量切分。

`--dtype`命令参数优先于JSON，例如命令中保留`--dtype fp16`会覆盖文件里的`bf16`。换矩阵形状/精度后应重新实测，不复用旧时间。当前只支持上述二维GEMM，换成Softmax、LayerNorm或batch matmul需另加测试与建模实现，不能只修改name。

## 5. 如何修改硬件配置

有两件不同的事：

- **真机测试选择GPU：** 修改`--device`或设备可见环境变量；运行的仍是真实设备，JSON无法将其变成另一种GPU。
- **预测模型更换硬件：** 复制A100配置为自己的JSON，修改硬件参数，并通过`--hardware`传入。

例如在本机先复制：

```powershell
Copy-Item superpod_compute/data/a100.json examples/my_gpu.json
```

编辑`examples/my_gpu.json`后运行：

```powershell
& $py tools/benchmark_gemm.py predict `
  --cases examples/benchmark_gemms.json `
  --hardware examples/my_gpu.json `
  --output results/model_my_gpu.json
```

硬件字段含义如下，所有带`bytes`的容量都是字节，带`_s`的时间都是秒：

| 字段 | 默认值 | 修改时依据 |
|---|---:|---|
| `name` | A100… | 配置标签，用于记录；改名字不改变性能 |
| `cores` | 108 | 当前建模设备的Core/SM数；需统一CU与本模型Core的映射 |
| `clock_hz` | 1.41e9 | 建模频率，单位Hz，优先与测试稳定工作频率一致 |
| `arrays_per_core` | 4 | 每Core等效矩阵阵列数 |
| `array_rows` / `array_cols` | 16 / 16 | 等效二维阵列尺寸，不是三维矩阵指令的M/N/K直接照搬 |
| `mac_per_cell_per_cycle` | 1 | 每阵列单元每周期的乘加次数；一次MAC按2 FLOP计 |
| `matrix_k_quantum` | 16 | 模型按多大的K粒度补齐尾块 |
| `hbm_bandwidth_Bps` | 2039e9 | HBM字节/秒，可用有效带宽；不是Gb/s |
| `hbm_capacity_bytes` | 80000000000 | 可容纳A/B/C的HBM容量 |
| `hbm_latency_s` | 1e-6 | HBM服务固定延迟；默认策略不逐面板计此项，启用见下节 |
| `l2_bytes` | 41943040 | L2容量，40 MiB |
| `l2_bytes_per_cycle` | 5120 | 整卡聚合L2服务率，B/cycle；`带宽B/s ÷ clock_hz`可转换 |
| `source_sram_bytes_per_core` | 196608 | 每Core原始片上SRAM容量，192 KiB |
| `shared_budget_bytes_per_core` | 131072 | 本模型允许使用的shared预算，128 KiB；不能超过source_sram |
| `register_budget_bytes_per_core` | 131072 | 本模型给驻留任务分配的寄存器字节预算；不是随意填写整卡容量 |
| `register_reserve_bytes_per_core` | 32768 | 给非C数据/工作状态预留32 KiB；剩余空间容纳FP32累加C |
| `kernel_launch_s` | 21e-6 | 配置的单算子启动成本；不应不加验证地沿用到其他设备 |
| `vector_fp16_flops_per_cycle_per_core` | 512 | 后续向量算子预留字段，当前GEMM没有直接使用它计费 |

当前等效整卡矩阵峰值由下式决定：

```text
cores × arrays_per_core × array_rows × array_cols
      × mac_per_cell_per_cycle × 2 × clock_hz   [FLOP/s]
```

改GPU时要同时核对这个峰值和容量/带宽，不能只改Core数量。三维的`16×16×16`一般描述一次矩阵指令处理的M/N/K，并不直接给出二维物理阵列及周期吞吐；缺少指令周期等信息时，不应凭尺寸把等效阵列参数填满。

寄存器和shared这里是**建模使用预算**，不等于某个厂商kernel一定使用的资源。换到不同架构后，阵列填充/排空公式也需要验证，不保证只替换JSON就获得同样准确率。

## 6. 分块和双缓冲策略在哪改

策略在`examples/policy.json`，与硬件参数分开。例如`bm_values/bn_values/bk_values`是搜索哪些分块尺寸，`buffer_counts:[1,2]`允许单/双缓冲，`matrix_efficiency`是明确设置的阵列效率系数，`epilogue_s_per_wave`是每波额外输出整理成本，`charge_hbm_latency`决定是否额外计配置的HBM固定延迟。

```powershell
& $py tools/benchmark_gemm.py compare `
  --measured results/real_gpu_fp16.json `
  --hardware examples/my_gpu.json `
  --policy examples/policy.json `
  --cache .cache/real_gpu_comparison.sqlite3 `
  --output results/comparison_custom.json
```

传`--cache`后，相同算子、硬件和策略可复用搜索结果；改变配置会形成新缓存键。模型策略只影响预测，不会控制PyTorch矩阵库的内部tile。脚本只修改明确指定的结果文件，不修改GPU设置或原始模型配置。

## 7. 本次交付验证范围

本机当前Python环境没有安装PyTorch，因此没有生成或冒充真机时间。已验证配置读取、统计/对比计算、三组模型预测入口，项目39项单元测试通过。三组默认模型总时间依次为79.972、21.335、86.945 µs；它们不是实测数据。需要你在有可用CUDA/HIP版PyTorch的真实设备上运行`measure`，才能得到真实的实测文件。
