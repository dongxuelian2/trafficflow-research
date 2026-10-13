# 可复现模型对照

正式基准为 Kaggle ref **57138874 / 0.82940**，SHA-256 `43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1`。
完整文件与原模型保存在 `artifacts/frozen_baselines/57138874`。下面指标来源不同，不相加为正式 Total。

组织方公开公告 742350 明确 onset 只在最后一步出现 queue。首轮 proxy 及 C01–C22 未匹配此采样属性，表内历史 Queue 数值均是一般事件压力测试。C24 起对齐公开时序；带噪观测仍不等于 latent truth。

| 模型 | 证据集 | State / OD 指标 | Queue 指标 | Physics 证据 | 状态 |
|---|---|---:|---:|---|---|
| 首轮 LGB + refined Queue + Ridge .05 | 首轮单次时间留出 | State 0.935606 | 观测代理 0.867664 | incomplete public diagnostic | 已正式提交，总分 0.82940 |
| 首轮弱物理后处理 | 同首轮留出 | State 0.934427 | 同基准 | 无正式 Physics 增益证据 | 本地保存，未正式提交 |
| CTM latent oracle | B01 独立模拟 | observation State 0.952006 | 未评估 | 两种显式模拟 flux contract | 诊断控制组 |
| 时空插值 | B01 同场景 | observation State 0.887547 | 未评估 | 模拟物理 | 压力基线 |
| RTS | B01 同场景 | observation State 0.848378 | 未评估 | 模拟物理 | 更差，不能据此推广 |
| 贝叶斯 OD | D02 独立模拟 / K01 正式 | simulated_ODME 0.870706 vs Ridge 0.825047 | 不适用 | 不适用 | **正式 Total 0.76474，显著退化，拒绝晋级** |
| 事件 Queue | C01/C02/C03 同滚动划分 | 不适用 | 随机共享模型 0.558105 / 0.781109 / 0.657088 | 不适用 | 分事件模型没有稳定独立增益，不晋级 |
| 图时空 + 300k 标签 LGB α=.5 | A01 rolling_autumn | assessment State 0.940205 vs 同划分 LGB 0.936655 | 不适用 | FD 联合损失没有 selection 增益 | 被 A14 全标签 LGB 0.942340 超过；标签曝光量不同 |
| 图时空 + LGB α=.5 | A02 rolling_winter | assessment State 0.942887 vs 同划分 LGB 0.940342 | 不适用 | 无正式 Physics 测量 | NN seeds 20261029/20261107 也有本地增益 |
| 图模型跨 family 迁移 | A05，目标 family 无监督标签 | 半数插值组合 0.927536 vs 插值 0.928998 | 不适用 | 无正式 Physics 测量 | 迁移失败；不晋级统一全走廊模型 |
| 全标签 LGB | A14 rolling_autumn，1,804,908 fit labels | State 0.942340 vs 300k LGB 0.936655 | 不适用 | target-pair density L1 0.771083 | 强比较基线；191.10 秒 CPU |
| 全标签 LGB + graph FD α=.25 | A15，同 A14 | State 0.943417；增益 0.001077 | 不适用 | target-pair density L1 0.767690 | 只由 selection 选定；低于晋级门槛 |
| 图 density-increment loss λ=.01 | A10，同 A01 的 300k 比较组 | State 0.938577 | 不适用 | target-pair L1 0.824392 → 0.786185 | 代理物理改善；未取得正式 Physics 证据 |
| 图 density-increment loss λ=.1 | A11，同 A01 | selection 选择 α=0 | 不适用 | 无正式 Physics 测量 | 强约束失败，不晋级 |
| 逐方向图专家 | A12 / A13 autumn | I405_N 0.917777 → 0.920775；I10_W 无增益 | 不适用 | I405_N 选 FD 联合损失 | 不支持统一部署；比较组仍仅 300k |
| 合法 at-T Queue | C05/C06 同窗口秋季 | 不适用 | at-T 600/63 0.612418 vs before-T 0.540552；全候选 cal 选中 0.526387 的对照 | 不适用 | 小 cal 选错模型；保留失败，不按 assessment 改选 |
| 合法 at-T Queue | C07 I10_W / C08 I5_N 冬季 | 不适用 | cal 选中 0.802211 / 0.697673；同容量 before-T 0.765435 / 0.670006 | 不适用 | 仅带噪代理，cal 分别只有 6 / 10 个窗口 |
| 全标签 LGB + graph FD | A17 winter / A20 I405_N autumn | .947299 vs .946854 / .922655 vs .921455 | 不适用 | 无正式 Physics | 同强基线比较，收益均低于 .002 |
| occupancy Queue | C12/C14 I5_N 秋/冬；C13 I10_W | 不适用 | 相对 cal 选定 at-T：去重增益 .062191 / .048073 / **−.050232** | 不适用 | 两个 I5_N 区间为正，但跨方向退化，不全局晋级 |
| 公告对齐 at-T Queue | C24 I5_N 秋 / C26 冬，45 cal / 30 assessment | 不适用 | 去重 .897422 vs .871047 / .868240 vs .876610 | 不适用 | 两季 CI 均跨零，冬季退化；未晋级 |
| 公告对齐 occupancy Queue | C25 I10_W 秋，45 cal / 30 assessment | 不适用 | cal 选 .628012 vs before-T .606343，CI [−.064953,.082322] | 不适用 | 只有 5 个独立 onset；不晋级 |
| 时序修正 K02 | 原正式预测，仅 5 个早期 Queue 值 | 全部保留 | 未取得正式 Queue | 全部保留 | private 完全不变；完整候选本地保留，未上传 |

模型晋级需要相同划分、多个拟合种子、按天或事件配对不确定性分析和分方向退化检查。行级随机拆分不可作为依据。跨走廊 transfer 必须整组排除一个 family 的训练标签；当前逐走廊训练不冒充 transfer。

图网络与树模型的日期相同不意味着标签预算相同。后续 State 架构结论必须以可用全部 fit 标签或明确的大预算 LGB 为比较组。Queue 抽样 seed 可能重复同一事件；按唯一 origin 和日期分析，不能把重复 seed 当独立试验。
