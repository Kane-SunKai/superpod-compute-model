# BW1100 可选执行计划实验模式

本目录记录2026-11-03的独立测试。默认模型仍保持10.29预测行为。本模式额外接收库实际选择的输出分块和K步长，不能把它的精度理解成只输入M/N/K即可达到。

## 一次离线预测

在本机PowerShell执行：

```powershell
cd 'D:\01codex_work\03_super node\compute_model'
$py = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py experiments/pipeline_20261103/predict_with_plan.py `
  --M 1 --N 28672 --K 8192 `
  --execution-plan experiments/pipeline_20261103/example_execution_plan.json `
  --calibration experiments/pipeline_20261103/frozen_calibration.json `
  --cache-dir results/pipeline_20261103/cache `
  --output results/pipeline_20261103/example.json
```

这条命令只在CPU上进行预测，不连接GPU，也不读取这个尺寸的实测时延。例子的预测约470.039 μs。已有实测455.329 μs，误差约+3.23%。

`--M/--N/--K`表示A[M,K]×B[K,N]。执行计划JSON的`bm/bn/bk`分别表示一个任务的输出行数、输出列数和每次K推进长度。`split_k`目前必须为1，`layout`必须为`row-major-contiguous`。

示例分块来自BW1100上这一形状的库调用，不能不经核实复制给所有尺寸。即便M=1，实际库也可能选择bm=256的执行覆盖范围。物理计算按覆盖范围计费，输入输出的有效数据量另算。

## 输入与输出

|输入|来源|
|---|---|
|M、N、K|应用模型|
|FP16、连续布局、FP32累加|当前实验范围固定|
|硬件|默认superpod_compute/data/bw1100.json|
|执行计划bm/bn/bk/split_k|应用层提供，或同库同设备上一次性提取|
|输入读带宽|独立微基准斜率的中位数，约1.923451 TB/s|
|有效矩阵服务效率|7个独立prefill校准得到0.87|
|固定设备服务|原有3.121245 μs，未重新按验收样本拟合|

输出JSON包含总duration、分块、容量检查、各层数据量、流水阶段、是否命中缓存与CPU侧预测耗时。硬件MMAC仍为8 cycle，0.87只表示本实验的有效服务校准。

只有输入HBM读取使用新带宽。输出HBM写仍用原1.72 TB/s。L2读写和Shared读写参数保持原配置。

实验缓存的键包括形状、执行计划、完整硬件/策略/校准和代码指纹。不同输入不会复用旧结果。默认不启用缓存，只有传入`--cache-dir`才保存。

## 查看本次证据

- `small_m_results.json`：小bm固定映射与候选搜索。
- `calibration_measure.json`、`calibration_predictions.json`：22组独立校准。
- `frozen_calibration.json`：冻结参数、全部效率候选、留一校准结果。
- `variant_results.json`：原模型、观测分块、输入带宽、持续效率逐项回归。
- `holdout_results.json`：参数冻结后的12组未见尺寸。
- `outputs/pipeline_20261103`：中文报告和Excel。

## 重做分析

```powershell
& $py experiments/pipeline_20261103/analyze_probes.py
& $py experiments/pipeline_20261103/calibrate_variants.py
& $py experiments/pipeline_20261103/evaluate_holdout.py
```

上面重新分析的是已有独立数据，不会重新跑GPU。保持原始文件不变时会得到相同参数。若更换校准数据，应另建实验目录、冻结新的留出计划，不覆盖本次结果。

## 真机复现

服务器新文件均在`/root/sk/pipeline_20261103`，容器内对应`/workspace/pipeline_20261103`。`run_server.sh`及各driver记录了编译、环境变量和调用顺序。原脚本对已有测量输出有防覆盖检查。

如需重新实测，应换一个/root/sk下的新目录并同步修改脚本路径，不能直接覆盖本次原始JSON。运行前检查同卡是否有其他任务。不得修改驱动、时钟、功率或其他人的容器。

## 已知限制

44组中34组达标，9组超差，1组Split-K未支持。冻结后12组中9组达标，896³的误差仍较大。不同矩阵库、GPU、布局、dtype及非Graph口径不能直接复用0.87。

本次只做了受控的轻量改进，没有增加按验收尺寸查时延、逐组修正系数或指令级仿真。实验模式保留为可选入口，未替换默认预测策略。
