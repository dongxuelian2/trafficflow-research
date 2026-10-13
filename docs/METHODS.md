# 方法和验证设计

`src/trafficflow` 为独立实现，官方代码只用于对照复现。

## 数据契约

训练标签只能来自公开 train unmasked observation。validation/private 的 raw reader 显式拒绝访问。
Task 1 targets 严格由模板枚举；额外空值不会自动变成监督标签。
ID 保留字符串，时间文本保留 UTC 原拼写；同一天不拼接不同 mask views。
资格由 `pct_observed >= 75` 和 speed/flow finite 联合决定。
局部观测不足不会被解释为零；对于结构上无 ramp 的连接，零才是合法值。

## 状态重建

按日期切分：前约六分之一（最少七天）建立 profile，之后到 65% 用于拟合，到 82% 用于 calibration，剩余日期为 test。
profile 固定于最早日期，训练样本不通过自己的未遮挡标签生成插值/邻域特征。
历史 profile 按 link/weekday/5-min slot 汇总；按 link 均值和网络参数分级 fallback。
时空插值加入观测年龄，超过 60 min 的时间证据弃用；空间位置来自 milepost。
上下游特征根据 released topology 搜索最近的已测链接，允许穿过 latent mainline links。
Kalman 对 profile 残差做随机游走滤波和 RTS 平滑，仅用于 offline Task 1。
树模型学习插值残差，speed 用 25 km/h、flow/lane 用 600 vph/lane 归一化；每个日期限定样本数以控制内存。
V2 特征先在 total-link-flow 空间插值，再转换为目标 link 的每车道流量，处理车道数变化。加入 density 插值、走廊当天流量尺度、拥堵占比和上下游流量变化。
联合密度模型另预测 q/v，再按误差尺度投影 speed/flow；D12 calibration 消融未显示收益，本轮采用投影权重 0。

calibration 比较历史、插值、Kalman、LightGBM、CatBoost，再比较所选基础模型的物理调整。
选择仅依赖 calibration；test 只做最终诊断，不允许根据 test 表现重新选择。
最终生产模型固定已选方法后使用后续公开 train 日期重拟合，不能把该模型的 train 拟合分数当作 holdout 结果。

## 队列预测

训练代理窗来自公开 train 观测，history 为 `[T−60,T)`，targets 为 `T+5 … T+30`。
当前公开包 competition_contract v2.0.0 将 ongoing 定义为 history 中同一 link 至少两次观测 queue，onset 为其余情形；各 condition 的 origin 至少相隔 360 min，跨日保留 spacing。
仅选择代理 horizon 有 queue 且 history coverage >= 70% 的窗口，模拟官方抽样条件。
训练 horizon 的观测标签合法，但它与正式 underlying-state 标签存在测量噪声差异。
feature builder 的输入验证拒绝 origin 及之后的观测，forecast 只使用窗口 history。
当前速度、过去 30 min slope、60 min 均值、coverage、flow/lane、density、上下游当前/趋势、时间和 lead 特征用于预测概率。
波模型以 20 km/h 作为初始 backward-wave 参数，并通过 sigmoid 表达到达时间不确定性。
增强特征使用公开 topology 的多跳距离和 FD 反向波速，描述最近拥堵前缘的距离/到达时间、5/15/30 分钟传播范围的下游速度、全走廊拥堵比例和 flow/capacity。全部由 history 和公开几何计算。
LightGBM 直接学习代理概率；calibration 上用 13 个候选阈值优化按 condition 等权的空间时间 IoU。
记录 Brier 作为代理概率诊断，不声称概率已针对真实隐藏标签校准。
后续校准在 0.05–0.80 的 31 点网格比较全局/condition 阈值，比较 400 iterations/23 leaves 与 600/63。公开契约保证每窗 horizon 非空，可比较空预测时选择最大概率单元的先验；只有 calibration 改善才采用。所有 window 条件均为已公开元数据，未读取 horizon 值。

## 物理约束

density 按 `q/max(v,1)` 派生，accumulation 按 density*length 派生。
只在完整连续 5-min transition、所有边界邻居有观测/预测、所需 ramps 全部有效时构造守恒稀疏算子。
latent 或外部边界无可用流时跳过，并报告覆盖率；不会用当前-link flow 伪造边界。
候选 flow 优化以原预测为先验，对可观测守恒施加弱惩罚，约束 flow >= 0；speed 固定于初值。
可选 1.05 倍 free-speed/capacity bounds 是普通边界，不强行施加 50 vph floor。
所有调整必须通过 Task 1 calibration 消融；不通过本地 Task 3 fallback 数值选择模型。
第二份候选在 calibration State 损失≤0.004 的预设预算内，比较 strength 0/1e-5/1e-4/1e-3，并选较低可观测 RMS。当前仅五个 N/E 方向存在严格可观测转换，S/W 方向维持零调整。这是权衡实验，正式 Physics/Total 仍未知。
目前是观测覆盖受限的局部约束和密度耦合实验，没有重建不可观测边界。公开数据上的约束消融与 State 精度一同记录。

## OD 反演

矩阵用 CSR 仅保留公开 counted links，未测 connector 不是零 counts。
比较 `||Af-c||²+λ||f-b||²`、`||Af-c||²+λ|| (f-b)/max(b,10) ||²` 及 counts 加权残差+KL prior。
正 λ 的前两者用独立的 measurement-space block-pivot NNLS：Woodbury 将自由集线性系统缩至 counted-links × counted-links，非负活跃集检查 KKT 条件。λ=0 回退到稀疏 bounded least squares；KL 用先验尺度预条件 L-BFGS-B。
16 个随机问题与独立 SciPy dense NNLS 的 path-flow 最大差 <1e-6；正式 D12 数据与旧稀疏求解目标一致，记录实际耗时和 KKT 残差。
记录收敛、时间、S_link、prior L1 变化和 sensor holdout；失败候选记为失败，不影响已成功的默认方案。
S_link 不能识别 underdetermined null space，默认保留 λ=.05 ridge；不为追求 perfect link-fit 自动选择弱正则。
sensor holdout 的空间诊断不能替代时间验证或真实 path/OD 标签。

## 泛化边界

十个走廊使用同一日期分组，并分别报告方向、mask R1/R2/R3、拥堵/非拥堵观测和 ramp 可观测覆盖。D12_I5_N 是用于研究模型结构的开发走廊，其 test 已多次查看；其余走廊使用预先固定的候选族和参数，calibration 选择后再评 test。
多个 rolling folds、跨走廊完全不训练的 transfer holdout 和真实事故类型标签验证仍属于后续实验。
没有公开 incident labels 时只能把急剧降速标为 observation-event proxy，不得称作真实事故分组。
人工 fixture 仅测试实现和日志系统，不能证明上述泛化能力，也不能用其最优参数作为获奖证据。
