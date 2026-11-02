# 单卡计算模型 · 简明 User Guide

当前实现：V3.2。代码目录：`D:\03_Project_Work\02_WORK_Super node\compute_model`。
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

预期打印`duration_us=50.182891`，1批输出；其中固定设备服务约3.121245 µs、64个K面板输入/计算流水38.702586 µs、分片输出8.359060 µs。此数值是模型估计。相同形状已有Graph实测64.709 µs，误差约−22.45%，尚未达到15%。

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

把固定分块命令中的`--n 2304`改为`--n 4608`，会计算两批C。当前策略下A+B不能全放入L2，所以初始仍为冷输入；模型会选择保留A的跨批复用，预期`duration_us=97.187383`、HBM输入15 MiB。

如要做严格无复用对照：复制策略文件，在新文件里设`initial_cache_state`为`cold`、`input_reuse`为`false`。用相同映射重新evaluate，预期97.244536 µs、HBM输入21 MiB。搬运减少6 MiB，但本例计算与搬运高度重叠，总时间只减少约0.057153 µs。

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
& $py -B tools/verify_v32.py
```

60项实现测试通过；完整对照结果见 `results/validation_20261029_v32/comparison_modes.json`。同一组验证样例上分别搜索，不重拟合固定开销。

## 8. 切换为每个 Tensor Core 256 MAC/cycle

将前述命令的硬件文件换为 `superpod_compute/data/bw1100_mac256.json`，策略换为 `examples/bw1100_mac256_comparison.json`；保持 `--backend auto`。
默认硬件仍为MMAC模型。两个文件中的L2、Shared、HBM和设备固定服务相同，只切换矩阵计算服务。256 MAC/cycle指每个矩阵服务单元；每CU四个、全卡72CU。

## 9. 重跑本次两模型对照

```powershell
& $py -B tools/compare_compute_modes.py --measurements results/validation_20261028/final_measurements.json --output-dir results/validation_20261029_v32 --workers 4
```

脚本复用同一组实测，不连接GPU；已有同名报告可断点续跑。换源码、硬件或策略后应使用新的输出目录，避免复用旧实验。
输出包含144条预测（72组×2），按12组校准和60组验证分别汇总，阈值为绝对相对误差≤15%。

## 10. 新增硬件参数

- `l2_read_bytes_per_cycle=3768`：全卡L2读。
- `l2_write_bytes_per_cycle=2990`：全卡L2写。
- `shared_write_bytes_per_cycle_per_core`：本次对齐向量写实测两轮均值59.35200341023225 Byte/cycle/CU。
- `shared_read_bytes_per_cycle_per_core=64`：之前Shared输入读取的实测。

新字段默认0时表示兼容旧行为：L2读写回退到旧统一字段；Shared输出写不计费。当前BW1100配置已明确填入新值。
输出沿 W(寄存器→Shared) → L(L2写) → D(L2读/HBM写) 调度；Shared槽在L结束释放，L2槽在D结束释放。Shared输出读回不单独计时。
两模型沿用原固定服务3.121245 µs，未重新拟合。硬件指纹已更新，含义是显式迁移同一固定服务用于受控比较，不是重新校准。

## 11. 显式指定C输出路径（实验接口）

策略JSON现在支持 `output_route`：

- `via-shared`：原路径，累加寄存器→Shared→L2→HBM。省略字段时仍使用此项。
- `direct-register`：寄存器直接写L2，再排出到HBM；不分配输出Shared槽，不计算寄存器→Shared写时间。

应该按目标内核实际的数据流选择，不能在两种路径中挑更接近实测的一种当校准。当前仅在两个已检查的BW1100 GEMM内核中确认了直接输出；其他内核需要独立核查。直接输出的转换/短写指令服务仍未完善，不能认为切换后就已对齐真实内核。

本机演示一次指定分块的直接输出预测：

```powershell
cd 'D:\01codex_work\03_super node\compute_model'
& $py -m superpod_compute evaluate `
  --m 3072 --n 2304 --k 1024 `
  --hardware superpod_compute/data/bw1100.json `
  --policy examples/bw1100_graph_direct_output.json `
  --mapping 384,256,16,8,9,1,3,4096,2,1 `
  --output results/direct_output_example.json
```

`$py`沿用前文Python路径。此命令演示输出路径接口，不代表输入访存、wave划分、尾部代码都已复刻厂商内核。`evaluate`从策略读取输出路径；Python直接调用`evaluate_mapping`时，以`Mapping(..., output_route='direct-register')`明确指定路径。搜索接口`predict`则从`Policy.output_route`生成对应的映射。

结果中的 `mapping.output_route` 和每类工作波的 `output_stream.output_route` 会显示所用路径。直接输出的 `shared_output_limit_per_core=0`，流水线只含 `register-to-L2`、`L2-read/HBM-drain`。两种路径的输入搬运、矩阵算力和总输出字节数不因该字段而改变。缓存签名包含该选项，不混用两条路径的搜索结果。

复现本次固定分块的输出路径对照：

```powershell
& $py tools/diagnostics/compare_output_routes.py `
  --output results/boundary_20261102/output_route_comparison_rerun.json
```

该脚本沿用已归档实测和映射，修改的只有输出路径，不重新选择映射、拟合算力或使用GEMM结果调参。本次差异仅0.053～0.106 µs，未提高15%阈值内的样本数量，具体见诊断文档第18节。
