# 单卡计算模型 · 简明 User Guide

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
