# 分块方案库实验入口

此目录为独立实验，不改变默认 `superpod_compute.api.predict`。MMAC仍按16×16×16、8 cycle建模。

## 输入与运行

在 `D:\01codex_work\03_super node\compute_model` 下使用安装本项目依赖的Python：

```powershell
python experiments/mapper_library_20261103/mapper.py --M 1 --N 10240 --K 5120 --output results/mapper_demo.json
```

输入对应 `A[M,K] @ B[K,N]`，连续row-major FP16，C为FP32累加、FP16写回。当前不含bias、转置布局、Split-K或非GEMM算子。

输出 `methods` 包含原mapper、校准mapper、最近模板、五模板择优、合并候选五组结果。每组的 `best` 有duration、完整mapping、资源预算、输入读字节量、波次与流水说明。`entries` 给出候选模板来源；`cache_hit` 表示精确结果缓存是否命中。

首次查找需要搜索；再次同样调用读 `cache/`。缓存键包含尺寸、硬件、策略、开发库、输入读带宽与计算源码摘要。开发库仅含历史分块信息，不含实测时延。

新硬件可加：

```powershell
python experiments/mapper_library_20261103/mapper.py --M 1024 --N 8192 --K 8192 --hardware superpod_compute/data/你的硬件.json --output results/new_hardware.json
```

必须符合项目Hardware字段定义。只要硬件与冻结BW1100配置不一致，就停用BW1100特定效率、输入读带宽和device service，重新检查容量并估时。模板可作候选，不能保证其在新硬件可行或最优，也不保证新硬件精度。

## 本轮冻结与实测

- `frozen_plan.json`：78组历史开发记录、24组独立验证、参数与检索规则。
- `disjointness_audit.json`：180份历史文件去重依据；最终与358种已知尺寸无重合。
- `predictions.json` / `prediction_seal.json`：测量前已封存的预测与SHA256，只保存在本地实验目录。
- `validation_measure.json`：两轮Graph GPU Event原始数据。
- `posthoc_metadata.json`：测量后才采集的实际分块，仅用于解释误差。
- `controlled_plan.json`：12个开发形状×5个显式分块及库对照；与24组验证隔离。
- `analysis.json`：统计结果；`build_report.py`生成报告和Excel；`verify_excel.ps1`检查重算和版面。

已有实测文件不覆盖。复测时在服务器 `/root/sk` 下建立新的实验目录，将 `run_server.sh` 的路径整体改为新目录，然后调用：

```bash
bash /root/sk/你的新实验目录/run_server.sh measure_graph.py \
  --plan frozen_plan.json --output validation_measure_repeat.json
```

这个启动脚本将运行时缓存、临时文件、Torch/Triton/HIP日志全部放在同一实验目录内。不要省略这些环境设置。

## 怎样解释本轮结果

最近模板14/24误差≤10%，其中Prefill 8/8；但总体平均误差未改善，长K显著退化。它尚不是默认替代方案。

本轮仅增加可选候选与检索实验，没有依据验证结果反调规则。若下一轮使用这24组来改规则，必须把它们转为开发数据，再另外冻结新验证集。
