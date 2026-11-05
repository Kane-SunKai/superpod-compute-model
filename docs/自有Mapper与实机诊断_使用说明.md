# 自有 mapper 与实机诊断：使用说明

正式计算模型保持 3.3 的分块与资源约束方案。新增的诊断模块用于固定一个变量、检查阶段服务，不改变正常预测路径和已有缓存身份。

## 1. 正式预测不需要真实库参数

在项目根目录运行，以下示例使用 BW1100 和 MMAC 后端：

```powershell
python -m superpod_compute gemm --m 2 --n 110592 --k 1536 `
  --hardware superpod_compute/data/bw1100_mapper_mmac_hardware.json `
  --policy superpod_compute/data/bw1100_mapper_mmac_policy.json `
  --backend mmac --no-cache `
  --output results/own_mapper_prediction.json
```

输入是本卡算子的 M/N/K、精度、硬件配置和 mapper 策略。应用层无需提供库内核名称、实机分块或真实计算时延。模型自动选择分块和缓冲方式。

如果系统未配置 `python`，使用自己的 Python 解释器绝对路径执行同样的参数。在当前 Windows 环境中可用：

```powershell
$py = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -m superpod_compute gemm --m 2 --n 110592 --k 1536 `
  --hardware superpod_compute/data/bw1100_mapper_mmac_hardware.json `
  --policy superpod_compute/data/bw1100_mapper_mmac_policy.json `
  --backend mmac --no-cache --output results/own_mapper_prediction.json
```

更换未来 GPU 时，修改硬件 JSON 和有依据的软件策略。mapper 会根据新容量、带宽和计算资源重新搜索，不能直接把当前 BW1100 库的一个分块当作新硬件的最佳分块。

## 2. 可选：只固定实机观察到的分块做诊断

如果真机观察到该算子使用 `bm=128、bn=128、bk=16`，可以另外运行：

```powershell
python tools/diagnose_gemm.py --m 2 --n 110592 --k 1536 `
  --observed-tile 128,128,16 `
  --hardware superpod_compute/data/bw1100_mapper_mmac_hardware.json `
  --policy superpod_compute/data/bw1100_mapper_mmac_policy.json `
  --trace-panels 8 --output results/tile_diagnostic.json
```

这一步只固定 bm/bn/bk。CU 分组、输入/输出缓冲及复用顺序仍由原成本模型选择。其结果明确标记为 `purpose=diagnostic`、`alignment=tile-only`，不会写入正式 mapper 的查表缓存。

**相同分块还不等于相同执行程序。**向量读宽、Shared 地址排列、编译器资源分配和预取/同步顺序仍可能不同。诊断输出列出这些尚未对齐的项目，避免把它称为“实机内核的精确预测”。

## 3. 如何看阶段账本

| 字段 | 含义 | 时间口径 |
|---|---|---|
| `ledger.service_demand_s.H` | HBM 输入与 L2 填充的端点服务 | 每波整卡传输工作量的服务时间累计 |
| `ledger.service_demand_s.R` | L2 读取与 Shared 输入写入的端点服务 | 每波的瓶颈端点服务累计 |
| `ledger.service_demand_s.S` | Shared 向操作数寄存器读取 | 每波最慢 CU 的服务累计 |
| `ledger.service_demand_s.C` | MMAC 与已有 K 循环控制服务 | 每波最慢 CU 的服务累计 |
| `ledger.service_demand_s.output` | 原输出流水的耗时 | 含累加寄存器写 Shared、L2 写、HBM 排空 |
| `ledger.elapsed_s` | 原模型依据依赖、缓冲组织得到的完整时间 | 与该分块预测时间一致 |
| `ledger.all_input_stages_serial_s` | 假设所有输入阶段串行时的敏感性结果 | 仅用于检查重叠的影响，不是另一套已验证模型 |

阶段服务量会重叠，不能将 H/R/S/C 的数字直接相加后称为真实时延，也不能将其视为硬件计数器测得的整卡忙碌时间。

例如，固定分块之后仍显著偏快，接下来要检查该程序实际搬了多少数据、采用什么地址排列、允许哪些活动重叠。不能继续把差异全部归因于 mapper。

## 4. Python 接口

```python
from superpod_compute import GemmRequest, Hardware, Policy, predict
from superpod_compute.diagnostics import ObservedTile, diagnose_tile
import json

h = Hardware.load('superpod_compute/data/bw1100_mapper_mmac_hardware.json')
p = Policy(**json.load(open('superpod_compute/data/bw1100_mapper_mmac_policy.json')))
request = GemmRequest(M=2, N=110592, K=1536)

normal = predict(request, h, policy=p)  # 正式路径：自动选择分块
diagnostic = diagnose_tile(request, ObservedTile(128, 128, 16), h, policy=p)
```

两个入口均不接受 GEMM 实测时延作为计算参数。实测数据在模型输出后用于计算误差。

## 5. 本轮实施范围

- 已实现独立诊断入口和有单位的阶段账本，保留原有完整预测路径。
- 已用独立探针核查 Shared 读取的地址模式和并发效应，确认统一有效服务率存在适用条件。
- 真实库描述继续作为分析依据。未经验证的“某个库标志对应某种服务率”尚未写入正式规则。
- 本轮未改变正式时延公式，因此正常自动选核的通过率没有被诊断通过率替换。

## 6. 新增 Shared 地址资源子模型

`superpod_compute/shared_memory.py` 增加了可选的 bank 服务计算。bank 是 Shared 内可并行工作的存储单元；不同地址落入同一个 bank 时可能需要分多次服务。

它输入的是一个计算方案计划访问的地址，不需要 GEMM 的实测时间。将来应由 mapper 内部的数据布局和读操作组织生成这些地址，应用层仍只提供算子信息。目前该组件单独用于诊断，尚未接入正式 GEMM 候选评分。

### 公式与单位

- `B`：bank 数量，无单位。
- `w`：每个 bank word 的字节数，Byte。
- `P`：每个 bank 每周期可服务的不同 word 数，word/cycle。
- `addr`：计划读取的起始地址，Byte。
- `bank(addr) = floor(addr / w) mod B`。
- 对一组可以同时发起的读取，统计每个 bank 需要服务的不同 word 数 `U_b`。相同 word 的读取在该组内按广播处理。
- 该组服务周期 `c = max_b ceil(U_b / P)`，单位 cycle。
- 一次完整读取分成若干服务组，`C_read = Σ c`。服务分组由硬件访问规则给出，不是自由拟合的系数。
- 多次读取按计划的地址求服务总量，`T_service = C_total / frequency`，单位秒。完整算子仍需另行组织计算与搬运的重叠。

公开参考采用 [AMD 的 LDS bank 访问说明](https://rocm.docs.amd.com/projects/composable_kernel/en/docs-7.2.1/conceptual/ck_tile/hardware/lds_bank_conflicts.html) 中的 128-bit 读取分组。BW1100 适用性通过独立实测核查，不将 AMD 文档直接当作 BW1100 的官方规格。

使用示例：

```python
from superpod_compute.shared_memory import cdna_b128_read_banks

banks = cdna_b128_read_banks()
contiguous = tuple(lane * 16 for lane in range(64))
strided = tuple(((lane % 16) * 32 + lane // 16) % 512 * 16
                for lane in range(64))

print(banks.estimate(contiguous).cycles)  # 8 cycles
print(banks.estimate(strided).cycles)    # 32 cycles
```

两种请求的逻辑字节数相同，地址排列不同使服务周期相差4倍。若目标硬件的 bank 数、word 宽度或服务分组不同，应创建相应的 `SharedReadBanks` 配置并独立验证，不能沿用这个预置名称。
