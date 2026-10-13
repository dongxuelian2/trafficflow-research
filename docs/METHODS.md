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

calibration 比较历史、插值、Kalman、LightGBM、CatBoost，再比较所选基础模型的物理调整。
选择仅依赖 calibration；test 只做最终诊断，不允许根据 test 表现重新选择。
最终生产模型固定已选方法后使用后续公开 train 日期重拟合，不能把该模型的 train 拟合分数当作 holdout 结果。

## 队列预测

训练代理窗来自公开 train 观测，history 为 `[T−60,T)`，targets 为 `T+5 … T+30`。
onset 的整个 history 无观测 queue，ongoing 在最后观测时刻有 queue；各 condition 的 origin 至少相隔 360 min。
仅选择代理 horizon 有 queue 且 history coverage >= 70% 的窗口，模拟官方抽样条件。
训练 horizon 的观测标签合法，但它与正式 underlying-state 标签存在测量噪声差异。
feature builder 的输入验证拒绝 origin 及之后的观测，forecast 只使用窗口 history。
当前速度、过去 30 min slope、60 min 均值、coverage、flow/lane、density、上下游当前/趋势、时间和 lead 特征用于预测概率。
波模型以 20 km/h 作为初始 backward-wave 参数，并通过 sigmoid 表达到达时间不确定性。
LightGBM 直接学习代理概率；calibration 上用 13 个候选阈值优化按 condition 等权的空间时间 IoU。
记录 Brier 作为代理概率诊断，不声称概率已针对真实隐藏标签校准。

## 物理约束

density 按 `q/max(v,1)` 派生，accumulation 按 density*length 派生。
只在完整连续 5-min transition、所有边界邻居有观测/预测、所需 ramps 全部有效时构造守恒稀疏算子。
latent 或外部边界无可用流时跳过，并报告覆盖率；不会用当前-link flow 伪造边界。
候选 flow 优化以原预测为先验，对可观测守恒施加弱惩罚，约束 flow >= 0；speed 固定于初值。
可选 1.05 倍 free-speed/capacity bounds 是普通边界，不强行施加 50 vph floor。
所有调整必须通过 Task 1 calibration 消融；不通过本地 Task 3 fallback 数值选择模型。
此阶段不是完整 CTM 或 joint nonlinear speed/flow state estimator，后续需在实际官方数据上检验收益。

## OD 反演

矩阵用 CSR 仅保留公开 counted links，未测 connector 不是零 counts。
比较 `||Af-c||²+λ||f-b||²`、`||Af-c||²+λ|| (f-b)/max(b,10) ||²` 及 counts 加权残差+KL prior。
前两者用稀疏 bounded least squares，KL 用先验尺度预条件 L-BFGS-B。
记录收敛、时间、S_link、prior L1 变化和 sensor holdout；失败候选记为失败，不影响已成功的默认方案。
S_link 不能识别 underdetermined null space，默认保留 λ=.05 ridge；不为追求 perfect link-fit 自动选择弱正则。
sensor holdout 的空间诊断不能替代时间验证或真实 path/OD 标签。

## 尚待真实数据的验证

十个走廊的完整日期切分、多个 rolling folds、跨走廊 holdout、不同事故类型、mask R1/R2/R3、以及稀疏 ramp coverage。
没有公开 incident labels 时只能把急剧降速标为 observation-event proxy，不得称作真实事故分组。
人工 fixture 仅测试实现和日志系统，不能证明上述泛化能力，也不能用其最优参数作为获奖证据。
