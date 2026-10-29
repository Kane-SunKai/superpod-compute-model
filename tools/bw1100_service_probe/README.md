# BW1100独立硬件服务核实

2026-10-29补充：`lds_write_probe.cpp`、`run_lds_write.py`测量寄存器→Shared写入。沿用下方隔离包装，执行 `bash run_task.sh python3 -B run_lds_write.py`。两轮数据在`lds_write_round1.json`和`lds_write_round2.json`；有效写服务取72 blocks、8 waves/block两轮斜率均值59.35200341023225 Byte/cycle/CU。之前64是读带宽。两轮源码仅增加显式`return;`以消除编译器warning，数值算法不变。当前模型对照用`tools/compare_compute_modes.py`，下文V3.1命令是历史复现入口，需要对应历史代码。

这些脚本保留本次服务器路径与容器名，以便复现。服务器所有写入都限制在`/root/sk/compute_mmac_20261028`（容器`/workspace/compute_mmac_20261028`）。通过`run_task.sh`设置日志和缓存路径；不要绕过它运行GPU测试。

把本目录文件复制到服务器上述目录，再从宿主机执行：

```bash
cd /root/sk/compute_mmac_20261028
bash run_mmac.sh
bash run_task.sh python3 -B run_lds.py
bash run_task.sh python3 -B verify_lds.py
bash run_task.sh python3 -B register_probe.py
```

`run_mmac.sh`编译厂商MMAC intrinsic，输出54组9次计时的汇总与频率状态统计；`run_lds.py`输出18组Shared读微基准汇总；`verify_lds.py`核对`src_shared_base`和group-segment元数据；`register_probe.py`只编译，不在GPU执行。编译产生的静态汇编用于核实指令，未记录完整运行trace。

`kernel_metadata.py`需要既有`/workspace/benchmark_gemm.py`与`/workspace/compute_validation_20261028/final_measurements.json`，通过`bash run_task.sh python3 -B kernel_metadata.py`采集60组实际kernel名称；仅存名称汇总，不把Profiler耗时作为精度评测数据。

本机原72组GEMM数据保存在`results/validation_20261028/final_measurements.json`。在项目根目录重跑预测：

```powershell
python -B tools/validation_20261028/compare_suite.py --version v31 --workers 4
```

默认只重算模型，不连接GPU。结果写入`results/validation_20261028_v31`。`--version v3`选取未启用Shared读阶段的对照硬件文件；历史V1/V2须用`--source`明确指向相应旧代码。比较脚本重新计算固定服务项并写入比较报告；若换了硬件或测量环境，不应复用已有校准策略，应重新校准并更新硬件指纹。
