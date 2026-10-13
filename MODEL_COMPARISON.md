# 可复现模型对照

正式基准为 Kaggle ref **57138874 / 0.82940**，SHA-256 `43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1`。
完整文件与原模型保存在 `artifacts/frozen_baselines/57138874`。下面指标来源不同，不相加为正式 Total。

| 模型 | 证据集 | State / OD 指标 | Queue 指标 | Physics 证据 | 状态 |
|---|---|---:|---:|---|---|
| 首轮 LGB + refined Queue + Ridge .05 | 首轮单次时间留出 | State 0.935606 | 观测代理 0.867664 | incomplete public diagnostic | 已正式提交，总分 0.82940 |
| 首轮弱物理后处理 | 同首轮留出 | State 0.934427 | 同基准 | 无正式 Physics 增益证据 | 本地保存，未正式提交 |
| CTM latent oracle | B01 独立模拟 | observation State 0.952006 | 未评估 | 两种显式模拟 flux contract | 诊断控制组 |
| 时空插值 | B01 同场景 | observation State 0.887547 | 未评估 | 模拟物理 | 压力基线 |
| RTS | B01 同场景 | observation State 0.848378 | 未评估 | 模拟物理 | 更差，不能据此推广 |
| 贝叶斯 OD | D02 独立模拟 | JSON 中的 simulated_S_ODME | 不适用 | 不适用 | 有离线增益，待受控正式对照 |
| 事件 Queue | C01 同滚动划分 | 不适用 | 运行中 | 不适用 | 未晋级 |
| 图时空残差模型 | 新 rolling folds | 尚未测量 | 不适用 | 联合损失待评估 | 实现中 |

模型晋级需要相同划分、多个拟合种子、按天或事件配对不确定性分析和分方向退化检查。行级随机拆分不可作为依据。跨走廊 transfer 必须整组排除一个 family 的训练标签；当前逐走廊训练不冒充 transfer。
