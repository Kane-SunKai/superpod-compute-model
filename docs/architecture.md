# 实现结构与模型口径

## 模块与职责

```text
GemmRequest / Hardware / Policy
        │
        ▼
OperatorRegistry → GemmModel
        │           ├─ candidates：bm/bn/bk、Core分组、两级缓冲
        │           ├─ capacity：shared / register / L2 / Core
        │           └─ evaluate：数据流量、尾块、波次、后端服务
        ▼
api.predict：精确查表或遍历候选
        │
        ├─ ArrayBackend / PeakBackend → 每Core矩阵服务秒数
        ├─ pipeline → H/R/C、缓冲释放、传输端口
        └─ MappingCache → SQLite、完整签名、完整性校验
        │
        ▼
JSON：最佳mapping、Top-K、时延、字节账本、约束、来源、范围
```

核心文件：

| 文件 | 作用 |
|---|---|
| `contracts.py` | 不可变请求、硬件、GEMM策略和映射；输入合法性 |
| `backends.py` | 阵列快公式、LUT、独立OS未命中估算、峰值对照 |
| `scheduler.py` | 通用DAG资源调度，以及H/R/C专用精确快进 |
| `operators.py` | 插件协议、GEMM候选和代价生成、容量判断 |
| `api.py` | 与算子字段解耦的搜索、选优、命中复核 |
| `cache.py` | SQLite保存JSON，SHA-256身份与内容校验 |
| `cli.py` | 命令行、配置加载、摘要和可选JSON文件 |

## 计算与搬运的分界

矩阵后端输入为一个Core的输出块与本次K长度，输出cycle/seconds。按M/N把任务分到本Core的多个阵列；不在阵列间切K。它不包含HBM、L2、kernel启动，外层不会重复计这些费用。

后端来源字段：

- `llmcompass-fast-0.99`、`llmcompass-fast-0.98`：来源快速公式。
- `llmcompass-narrow-fast`：保留来源窄块利用率公式，不把它解释成实测效率。
- `llmcompass-lut`、`llmcompass-lut-transposed`：离线模型周期；转置查表只允许方形阵列。
- `own-os-wavefront-estimate`：独立OS折叠波前估算，未运行SCALE-Sim。
- `peak`：每Core峰值公式，仅用于对照。

每个来源在结果中可追溯。后端类自己保留有界、实例内的tile缓存；整算子mapping则进入SQLite，二者不是同一张表。

## 资源与字节数

设波输出为有效m×n，一个Core槽位为bm×bn，本次有效K长度为k。

```text
rm = ceil(m/bm)，rn = ceil(n/bn)，活跃Core = rm*rn
H字节 = (m+n)*k*2
R字节 = (rn*m + rm*n)*k*2
最终C字节 = m*n*2
每Core矩阵槽位FLOP = 2*bm*bn*align(k,16)
```

H是HBM→L2的唯一输入，R是L2→多个Core的逻辑副本，不假设多播。资源预算检查使用分配的完整块，而实际搬运只按有效元素；M/N尾块保留槽位padding，K尾块以16对齐计算。未将补零后的越界地址计为真实HBM读取。

shared峰值取 `max(S*(bm+bn)*bk*2, bm*bn*2)`。输入阶段和输出整理阶段分开，不能相加。寄存器占用为 `bm*bn*4 + reserve`，C从初始化一直保留到完整K结束。L2使用保守上界 `L*(GM+GN)*bk*2 + GM*GN*2`；即使尾波较小也按分配大小过滤。

只让每个Core同时负责一个输出块；较小shared没有自动换算为更高occupancy。跨波串行、跨波冷读；波内采用唯一面板驻留服务。没有把192 KiB来源SRAM当作独立可用shared。

## 缓冲与调度

每个输入面板包括三个事件H、R、C：

```text
H_i.start = max(H服务可用, L2槽(i mod L)释放)
R_i.start = max(H_i.end, R服务可用, shared槽(i mod S)释放)
C_i.start = max(R_i.end, 上次C.end)
L2槽释放 = R_i.end
shared槽释放 = C_i.end
```

H服务按整卡HBM带宽计，R按整卡L2带宽计。C按单Core矩阵能力计，而非每个Core都用整卡算力；波计算取最慢的活跃Core。`shared-port`模式额外让H、R占同一个端口，避免同时使用同一受限资源。该配置没有模拟每个L2 bank、NoC路由或真实shared端口。

事件与周期使用秒。`pipeline_events()`可以展开同样的DAG，交给通用`EventScheduler`独立对照；自动测试比较这两种实现。长K情况下，状态向量包含H/R/C资源时刻和两类槽释放时刻，只有相同槽位相位、全部状态等量平移时才快进。尾面板仍正常推进。

完整波与边界波合并为最多4类。每类独立评分乘出现次数，不把边界波错误地当作满波，也不凭空认为所有Core活跃。

最后显式加输出转换/整理、Core→L2和L2→HBM；当前输出转换/整理配置默认为0并在结果警示。第一次输入填充、最后阶段排空、输出写回和kernel启动不会被直接一个max抹掉。

## 搜索与查表

固定策略下枚举有限候选集合，容量不合法即过滤。评分后按时间、shared峰值、缓冲数量和稳定分块顺序排序；返回前5名。所谓最优是本候选空间、本模型下的最小值。没有使用黑箱优化器，也没有跳过未评分候选后声称全枚举。

签名包括request、完整hardware、policy、后端版本和LUT哈希、算子模型版本、搜索版本、初始状态。SHA-256为索引，原始签名仍保留核验。SQLite结果JSON也有独立checksum；不是pickle。命中后重新检查mapping候选资格、容量和评分。模型逻辑改变时必须提升版本，尤其不能仅靠“获选mapping时延没变”认为旧最优排名仍有效。

输入布局、融合等暂不支持的行为被请求类型拒绝。将来支持时，新增字段必须进入请求签名。平台并发竞争改变可用带宽时应改硬件有效资源配置再搜索/重放，不直接复用无竞争的duration。

## 添加新算子

定义自己的Request和Policy（都提供`to_dict()`），实现`OperatorModel`协议。`mapping_from_dict`和`rank_key`由算子自己提供，因此Softmax不需要冒用GEMM的bm/bn/bk字段。用`OperatorRegistry.register()`注册；给`predict()`传入对应计算服务后端对象。

例如Softmax需要定义沿哪一维归约、整行是否驻留、是否两遍读取、哪些中间数据保留，再构造load/max/exp/sum/normalize/store事件和容量约束。向量、特殊函数及归约的时延模型另行实现。通用`Event`允许多个resource ID和依赖；C的GEMM驻留规则不能原样套到Softmax。

首版没有ASTRA-sim、SimAI或网络模型接入。平台可以通过库接口传入本rank算子，接收duration与scope；通信不在本模块计费。

## 尚未校准的部分

默认使用来源的A100抽象阵列和带宽；硬件指令布局、SM占用率、bank冲突、寄存器溢出、转换kernel与缓存真实替换行为未实测。软件验证证明逻辑约束和参考数据一致，不能外推为真实GPU误差保证。资源预算与后端/策略可修改，正是为后续校准保留接口。
