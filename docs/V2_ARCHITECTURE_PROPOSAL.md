# V2 架构提案：先验证状态可辨识，再扩大模型

日期：2026-10-16。本文是设计，不是已实现或已训练的模型。完整公式见 [数学推导](V2_MATHEMATICAL_DERIVATION.md)，证据和晋级规则见 [研究决策](V2_RESEARCH_DECISION.md)。硬件预算为 RTX 4060 8GB；本轮不改变现有 inference、checkpoint 或 submission。

**执行顺序修订（2026-10-16）**：用户新指令优先。先做独立评分契约、冻结强模型的低维State–库存修正，再做双向GP与小型Queue事件对照；完整LTM–MHE必须通过观测映射与动力学前检。当前可执行模块位于`src/trafficflow/v2`，结果见 [执行报告](../reports/v2/RESULTS.md)。连续随机状态与离散事件分开；匝道唯一记账；未知边界按固定仿射分支与合法协方差消元。无需重新建设全域数字孪生来满足研究目录形式。

## 1. 现有架构审计与真实增量

| 仓库模块 | 实际结构 | 已有价值 | V2 必须新增的内容 |
|---|---|---|---|
| State LightGBM 与相关特征 | 广域时空/统计特征的监督回归 | 高效、强观测重构；全量标签覆盖 | 不能仅以更小标签集 NN 作公平替代 |
| `src/trafficflow/adaptive_graph_state.py` | 时间/图模块与全局稠密 attention；速度/流量或密度残差 | 非局部统计关联、remask | 不是仅局部 GAT；新增物理接口通量/边界后验才有结构差异 |
| `src/trafficflow/state_assimilation.py` | 冻结预测上的图/时间二次残差平滑 | SPD 小问题，低成本 | 不含潜在需求、容量或 CTM；MHE 应与它直接对照 |
| `src/trafficflow/physics.py` | 可观测局部物理代理 | 诊断 FD 与局部一致性 | 缺少隐藏 flux，不能报告官方 Physics |
| `src/trafficflow/joint_graph_queue.py` | 13 步因果历史、速度/流量/logit 多头、可选未来 GRU | 历史掩码合法，已有联合监督实验 | 多头或 GRU 本身不构成守恒状态空间模型 |
| `src/trafficflow/expected_iou_queue.py` | 已有概率/集合决策研究 | 避免把 .5 当普遍最优阈值 | 单纯新 IoU 解码不应再单独立项 |
| `src/trafficflow/bayesian_od.py` | 高斯协方差先验、计数后验；必要时非负优化 | 显式不确定性模型 | 非负解后的高斯方差并非截断后验；先验错配风险已被正式失败暴露 |
| `src/trafficflow/digital_twin.py` | 自建 CTM、已知模拟输入/种子、名义 FD 观测 | 算法正确性和失效压力测试 | 不能充当组织方隐藏生成器或晋级真值 |

物理图与 attention 图必须分离。A128 中 I210_W 测量节点 53/81 在物理邻接中孤立，而 I5_N 为 0/258；全局 attention 仍能通信。用里程近邻填图可以是统计特征，但不能伪称已知输运连接。

## 2. 统一模块契约

建议后续新建独立研究包，先不接入正式预测入口。下列名字仅表示拟议接口。

```text
PublicObservationBundle
  times, station_ids, observed[v,q], channel_mask, quality
  role: state_smoother | queue_causal
  cutoff, target_keys, permitted_covariates
NetworkContract
  cells[L, lanes, vf, cap, kj], certified_edges, junctions
  unresolved_boundaries, sensor_map, aggregation_hypothesis
StatePosterior
  time, k_samples, boundary_samples, capacity_modes, event_age
  detector_bias, weights, identifiability_flags, fit_diagnostics

build_network(public_static) -> NetworkContract
fit_parameters(train_only, fixed_protocol) -> PhysicalParameters
assimilate(bundle, network, parameters, arrival) -> StatePosterior
forecast(posterior_at_T, public_condition, horizons=6) -> TrajectorySamples
decode_state(posterior, target_keys) -> observed_vq_predictions
decode_queue(trajectories, candidate_sets) -> binary_template_predictions
diagnose_physics(anchored_predictions) -> fd + labeled_proxy_only
diagnose_od(A,c,b) -> rank, row/null decomposition, prior_sensitivity
```

所有转换记录单位、来源与排序。State/Queue 不同信息集由接口强制检查，不能仅靠调用者自觉。Queue cutoff 后的值、quality、mask 改写不应改变预测。State 目标必须严格与模板匹配。没有任何接口接收真实事故参数、组织方隐藏 flux 或私有随机种子。

预测 posterior 与决策输出分离：State 重构观测均值，Queue 对轨迹集合优化期望 IoU，Physics 消费同一组 State 预测。禁止额外提交一个更“物理”的隐藏状态去冒充官方 State 输入；目标位置外的锚点按官方规则保留。

## 3. 模型一：边界不确定性下的 CTM–MHE / Smoother

**技术目标**：检验“合法时空观测 + 方向性守恒传播”是否真的提供比已有树/attention/图平滑更多的信息。首先改善状态与时间密度增量，不以自建 Physics 总分作为终点。

### 3.1 状态与控制维度

- 主状态：每个可靠单元密度；外部入口储车队列仅在数据能约束时保留。
- 未知输入：边界/匝道需求的少量样条系数或 AR(1) 因子，5–15 分钟节点；不是每个 link/time 一个自由源项。
- 容量：首版固定名义 FD；第二消融才加入一处或少数瓶颈门及持续先验。
- 偏差：低维 detector 流量/速度偏差，带强收缩；不允许逐点偏差解释全部数据。
- 参数训练与轨迹同化分开：物理/噪声参数仅从训练角色拟合；验证/私有仅同化当时合法已发布观测，不拟合被遮真值。

### 3.2 求解流程

1. 检查拓扑引用、单位、FD 与几何；把无法闭合连接显式标为边界。
2. 在可靠链上做保守聚合并记录 sensor mapping；验证细粗网格误差，不按显存随意改长度。
3. 用历史统计或合法强模型预测初始化密度；用过去窗口形成 arrival。比较自由/拥堵两个初始分支，记录最终目标和活跃集。
4. 固定事件，线性化观测与活跃通量，解带信赖域的稀疏 QP；回到真实 CTM 检查下降与可行性。
5. 以 profile likelihood 或局部 ensemble 对边界 nuisance 积分/敏感性分析，而非只保留一个“最优边界真值”。
6. State 双向窗口做离线平滑；Queue 用截至 T 的窗口初始化六步预测。对无观测支持状态输出宽区间和可识别性标志。

可比较窗口 12/24/36 个五分钟间隔；最长约三小时。先固定短窗，不把几十组窗口搜索当结构研究。重叠窗口以精度加权融合只是数值规则，不能将相关窗口视为独立样本。

### 3.3 凸近似与失败处理

固定活跃分支下的 QP 是局部近似。节点 min、\(q/k\)、容量切换使整体非凸；不存在本设计可引用的全局最优保证。若多初值结果分歧，应报告不确定性，不挑验证分最高的局部解。若边界系数可补偿任意错误密度，缩小边界维度并检查式 (21) 的有效信息矩阵。

观测似然更好但 held-out 重构更差，优先怀疑过拟合边界/偏差，而不是继续增加迭代。若对任意温和先验都不稳定，停用该区域物理修正，保留现有预测作为明示回退。

### 3.4 为什么不是重跑旧局部物理约束

旧平滑只在预测残差上施加图/时间二次项；本方案显式表示车辆库存、接口输运、隐藏边界的后验与可观测性。它能在成立条件下利用延迟到达的观测，但也更容易受闭合错误影响。因此必须对照旧平滑及“相同参数量但无守恒”的状态模型，确认增益来自守恒而非额外平滑自由度。

## 4. 模型二：半马尔可夫事件驱动的保守残差状态空间模型

**技术目标**：改善 onset 位置/规模、ongoing 前沿与恢复，并保留 State 急剧降速。它不是给现有速度回归再加一个 BCE 头。

### 4.1 离散事件与连续状态

用少数候选瓶颈位置、容量门、恢复时间作为离散/低维随机状态；用 CTM 密度作为连续状态。事件持续时间用 hazard 或显式半马尔可夫分布，避免 HMM 的几何持续时间成为未经验证的固定假设。

推断分两层：16–32 个事件/需求情景；每个情景用低秩/局部连续估计（首版每情景 MAP，不先叠加大量 EnKF 粒子）。观测更新用真实速度/流量似然；事件权重采用归一化 log-weight，并监测有效样本量。必要时合并近似等价情景，不能把同一场事故的数千 MC 样本当作数千独立事件。

onset 的公共选窗条件限制前五步拥堵集合为空，重点不再是重复优化发生时刻，而是 T+30 的位置与范围。真实事故发生时刻仍未知。对 ongoing 则联合预测前沿移动、消散和新瓶颈，不强制持续。

### 4.2 神经残差的分工

先做固定物理模型和低维 hazard，再添加共享小网络（隐藏维度 16–32、1–2 层）学习**接口通量修正或瓶颈门残差**。单接口可写

\[
\tilde f_e=\Pi_{[0,\min(D,R)]}(f_e^{CTM}+r_\phi(\text{局部状态、合法历史})).
\]

在节点处必须投影整个流向量到供需/转向多面体，不能逐接口独立裁剪后超出下游总容量。同一修正流在上游流出与下游流入使用相反符号，保持全网守恒。此投影表示允许非最大通量的模型误差，因此需要小残差惩罚与停流诊断；它不是对标准 CTM 的无误差重写。

物理部分负责守恒、方向、容量范围和波传播；残差负责低阶闭合、聚合和有限的滞后误差。观测 decoder 另外负责 noisy State 的偏差。不能让 residual、需求、容量、decoder 同时无约束自由，否则物理变量不可解释。

模型一若无法通过观测闭合门槛，模型二先只实现低维事件/前沿模型，不把失败的 CTM 硬接入；是否继续保守残差版本需重新决策。两候选不是必须连续投入的大项目。

### 4.3 联合训练与推断

训练目标采用数学文档式 (16)/(18)，加适度 State 监督和事件代理监督；各项按样本/通道量纲归一。Queue 真实 latent 标签未公开，公开速度阈值仅为 noisy proxy。阈值附近可用噪声模型软标签或区间标签，并报告误标敏感性，不把伪标签称真值。

共享动态状态，分别使用 State smoother 与 Queue causal filter。为保护 State，先冻结已验证观测 decoder/主预测，再评估小残差是否改善误差；允许最终实验结论是“只作 Queue 后验模块”，而非强制一套点输出同时取代全部任务。

训练不直接离散化 IoU 求梯度。先校准轨迹分布，再在冻结的 calibration 集选集合规则；assessment 不反选。与已有概率解码使用相同样本数、候选集合数和调参预算，避免把搜索量当模型收益。

## 5. 结构化概率 OD：研究接口与暂缓事项

保持现有正式 Ridge。新诊断分四类，不作为本轮实现前二名：

| 方法 | 可落地形式 | 能回答的问题 | 不能解决的问题 |
|---|---|---|---|
| 经验贝叶斯 | 最大化 `N(c;Ab,AΣA'+R)` 边际似然 | 可观测协方差尺度是否合理 | 纯零空间方差不可由计数学习 |
| 最大熵 | `min KL(f||b)`，计数约束/软损失 | 相对先验最小改动的唯一选择 | 选择唯一不等于真值唯一 |
| 层次需求 | origin/destination 随机效应 + 路径残差 | 参数共享、总量/吸引量不确定性 | Jacobian 亏秩与规范自由度仍在 |
| 稳健优化 | Huber/重尾计数似然、噪声集合、多先验最坏风险 | 限制计数异常和先验外推 | 无证据的 ambiguity set 同样可错 |

统一输出 row-space/null-space 变化、总需求、prior deviation、destination shares 和边界解比例。模拟改进只能否决脆弱方法，不能证明正式晋级；特别禁止再次以同族生成分布中的 posterior 优势覆盖 K01 失败证据。

## 6. 复杂度、稀疏性与 8GB 预算

设 \(n\) 单元、\(e\) 接口、\(W\) 观测间隔、\(m\) 每间隔子步、\(r\) ensemble/情景数、\(h\) 神经隐宽、\(J\) 外迭代。

| 运算 | 粗略成本 | 限制 |
|---|---|---|
| CTM 前向 | `O(W m (n+e))` | 子步 m 可能超过 1000 |
| EnKF 预报 | `O(r W m (n+e))` | 全局协方差 `O(n²)` 应避免；局部更新成本依邻域/观测数 |
| MHE JVP/梯度 | 每次约 `O(W m (n+e))` | 迭代次数、活跃集改变会放大常数 |
| 稀疏 QP | 依图填充、ordering 和求解容限 | 不能仅因局部动力学就宣称总体线性复杂度 |
| 稠密时间块消元 | 最坏 `O(W n³)` | 不应构造完整 `(Wn)²` Hessian |
| 残差网络 | 约 `O(W m e h²)`，按调用频率可降 | 可每观测间隔更新门参数，子步只做物理核 |
| 事件情景 | 前述成本乘情景数 | 避免“粒子 × ensemble × 子步”无预算相乘 |

静态检查结果（公开 FD 的 `cap/(kj-cap/vf)` 与 link 长度，CFL .8；这是稳定性估算，未运行模型）：

| 方向 | FD/link 行数 | 最短 km | 最小 CFL 上限秒 | 每五分钟子步上界要求 |
|---|---:|---:|---:|---:|
| D12_I405_N | 89 | .010 | .3120 | 1203 |
| D12_I405_S | 44 | .042 | 1.3440 | 280 |
| D12_I5_N | 258 | .010 | .3064 | 1224 |
| D12_I5_S | 100 | .041 | 1.2198 | 308 |
| D7_I10_E | 146 | .010 | .3186 | 1178 |
| D7_I10_W | 107 | .011 | .3286 | 1142 |
| D7_I210_E | 297 | .010 | .3172 | 1183 |
| D7_I210_W | 81 | .014 | .4257 | 881 |
| D7_I405_N | 352 | .010 | .3177 | 1181 |
| D7_I405_S | 109 | .018 | .5290 | 709 |

计算式为 `ceil(300 / (0.8 * min_i(3600*L_i/max(vf_i,w_i))))`。表中“子步上界要求”指使步长不超过上限所需的最少整数子步数，不是保证精度所需的最大步数。所有行数都是表的行数，不代表没有外部拓扑引用或完整内部路网。

内存算术示例（非实测）：64 ensemble × 200 cells × 8 浮点状态 × 4 bytes = 409,600 bytes；保存 36 时刻约 14.1 MiB。但 36×200=7200 维稠密 double Hessian 单矩阵已约 395.5 MiB，分解/工作区更多；神经 autograd 还会保存每子步的激活，不能据小状态数组推断显存足够。

首轮在 CPU 解稀疏 MHE；GPU 仅用于后续小残差批次。限定实际测量 GPU 峰值不超过 6 GiB，保留系统余量；超出时减少情景/分块/使用检查点，不能改变评估域。严格记录 wall time、CPU RSS、CUDA allocated/reserved；本轮没有任何新模型运行时间或 GPU 峰值。

## 7. 实施次序与停止条件

1. 契约与数值核：合法信息、拓扑、单位、保正、全网守恒、CFL、分支外有限差分梯度。
2. 模型一最小同化：固定 FD、低维边界，比较旧图平滑；不先训练神经网络。
3. 观测算子/边界 profile：评估 latent 与 detector 层的区分是否必要，检查先验敏感性。
4. 模型二事件模块：低维 hazard/持续/前沿，与同输入树基线比较。
5. 只有前述门槛通过，才加入保守残差并评估跨方向泛化。

任一环节发现未来泄漏、几何无法解释、容量后验完全由先验决定、State 显著受损或事件增益仅来自更弱对照，停止扩大。是否产生正式候选与 Kaggle 验证属于后续另行授权任务，本轮不执行。
