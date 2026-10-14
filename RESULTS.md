# RESULTS — 0.950+ 高强度研究

**当前已核验的正式 Kaggle Total：0.82940。目标 0.95000+ 尚未达到。**

本人提交 ref `57138874`，2026-10-13 09:53:40 UTC，状态 COMPLETE。回下载文件与本地 `artifacts/submission_best/submission.csv` SHA-256 完全相同：`43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1`。四个正式分项仍未知，不能从 Total 唯一反推。

基准提交、模型、配置和代码 `15036c8` 已独立冻结，共 106 文件、712,127,414 bytes。详见 `reports/research/baseline_57138874_frozen.json`。本地最佳候选仍保留原文件；任何新候选另存目录。

新实验与失败结果见 `EXPERIMENTS.md`；候选对照见 `MODEL_COMPARISON.md`；优先路线见 `RESEARCH_ROADMAP.md`。
登记协议 SHA：`91e48a8f2f86775466c8c9b5ad6bdbe693ea25554ef9b18d9e12cd80d3256143`。

**最新复核：State 的监督收益扩展到两季、两走廊族；正式 Total 仍为 .82940。** I5_N 第三 NN/LGB seed 的 A41 仅按 cal 选纯 attention、α=.75，assessment **.942330 → .950416**，gain **.008086**，14 日配对 CI **[.006530,.008567]**。I405_N 冬季 A43 为 **.925958 → .933986**，gain **.008028**、CI **[.007056,.009472]**，increment 代理 L1 **.765975 → .726403**。六组匹配消融中，attention-vs-geometry 的日均 CI 均正；density 的独立 State 收益仍不稳定。A41/A43 standalone 各 14 日独立 cal 重建/预测通过；均是公开 train 开发测量。

**A44 固定三种子组合通过独立数值复跑。** 三个原 cal 封存成员等权，比较组也是同三个 full-label LGB 的等权预测；assessment State **.942742 → .951168**，gain **.008426**，日均 CI **[.007304,.008681]**，increment 代理 L1 **.756319 → .671424**。平均三个 seed 的日分数后再 bootstrap，gain CI **[.006253,.007463]**，仍只有 14 个共享日期。组合 wrapper 的预测和保存/重载在一个完整 assessment 日核验，各成员在全部 cal/assessment 日复现。模型 SHA `37b3f219a7264786abfc4c7b96ea5578e735fc8c058055b0823fc6135bf64be3`。三倍成员推断、单季组合尚无跨季节或正式收益证明，暂不扩展生产组合。

![六组强 State 对照与已封存架构消融](reports/research/adaptive_matched_comparison_v4.png)

**Queue 的概率修正和波速邻域仍未达到晋级标准。** C31 固定 C30 checkpoint，减去 log(4) 修正理想加权 BCE 概率偏移；graph cal-domain Brier **.015419 → .012059**，但组合仍选 α=0。C32/C34 控制同输入、同预算、同校准规则的 direct / 15 分钟 / 30 分钟波速图，两季 cal 均选 30 分钟、α=.25；去重 proxy IoU 秋 **.897422 → .904028**、冬 **.868240 → .879943**，gain **.006606 / .011703**，day-block CI **[−.004692,.018938] / [−.001776,.032738]**，均跨零。未选中的 15 分钟图秋季 gain .014705、CI [.001545,.027350]；不能按 assessment 反选。C33 普通因果图冬季 gain .008485、CI [−.006698,.028666]，亦未晋级。波速图只是公开几何的表示，不是严格传播约束；真实 latent Queue 分数未知。

**OD 熵反演通过模拟筛选，正式恢复精度未知。** D06 保留 69 次优化失败和未通过开发最差场景门槛的结果；D07 将原 69 例恢复，但新增一个未收敛案例；D08 按浮点误差预算与 Newton decrement 接受残差下降步，全量重跑 **0 failures**，最优性容差仍 **2e−6**。Ridge 的全部原开发分数完全一致，既有成功 entropy 分数最大变化仅 2.32e−9。开发数据选定 entropy **τ=.05、prior floor=20**；十方向、七种独立已知真值分布的 assessment 模拟 ODME **.691063 → .747273**，family/seed 配对 gain CI **[.043844,.068733]**。数据、种子、选择门槛未改变；这是数值回归和已查看模拟分布，不是新的盲测。

Entropy 在公开 train 的 S_link-only 约 **.987973–.998652**，相对 Ridge 的改变量约 99% 位于计数零空间，总需求比率约 .920–1.084；这些只能描述稳定性，不能判定隐藏路径真值。已正式失败的固定 Bayes 在同一模拟里仍得 .728094，高于 Ridge，说明模拟增益不足以保证赛事迁移。保留原正式 OD 与 0.82940 最佳文件；没有新平台动作。

A40 冬季图缓存完成（153 fit / 14 cal / 14 assessment）；A42 full-label LGB 用 **3,985,807** 个标签，500k/full **.923899 / .925958**。当前研究环境 **43 tests passed**，CPU **38 passed / 3 optional-Torch module skips**；完整候选 K03/K04 的此前全行核验仍有效。

**A45 在第三个走廊族的监督对照也改善。** I10_W、107 sensors，cal 选纯 attention α=.75，assessment State **.951973 → .955810**，gain **.003837**，14 日日均 CI **[.003496,.004039]**；increment 代理 L1 **.719817 → .701798**。401.02 秒 GPU，359.92 MiB，14 日 cal 独立部署复跑通过。geometry 的 cal α=.25，attention/density 为 .75，因此各自校准后的模型差异不能全部归因于单一模块。未选中的 density assessment .955828 略高，保留原 pure-attention 选择。A37 的零样本失败仍有效：A45 使用了目标走廊公开 train 标签，不能把它称为零样本泛化。

**A33 取得强基线上的本地收益，尚未正式提交。** I5_N 秋季，仅按 calibration 选 attention+density、α=.75，assessment State **.942340 → .949192**，gain **.006852**；14 天配对日均 CI **[.006348,.007157]**。increment 代理 L1 **.771083 → .687617**，不能当作正式 Physics。完整实验 489.27 秒，GPU 峰值 **845.07 MiB**。几何图对照 / 注意力 / 注意力+密度损失控制相同 NN 宽度、patch RNG、优化器和预算，cal 组合 State 为 .956208 / .957598 / .958978；不是正式总分。

A34 冬季 cal 选纯 attention、α=.75，assessment State **.946854 → .954329**，gain **.007475**，日均 CI **[.006259,.007608]**；increment 代理 L1 **.735824 → .682571**。A35 第二个 NN/LGB seed 20261029，cal 选 attention+density、α=.75，State **.942295 → .948993**，gain **.006698**，CI **[.005566,.006983]**；代理 L1 **.769809 → .686074**。两次并行训练各约 1140 秒，单进程峰值 845.07 MiB。两季、两种子仍只覆盖 I5_N，assessment 是已查看的公开 train 开发日期，不是正式分项或最终盲测。

A33–A35 standalone 各在 14 天 calibration 独立重建输入并预测，逐通道数组精确一致、指标匹配封存选择。部署模型 SHA 分别为 `d5067934dec3facdf66ca1f4d22089aa69e646c9329a63ddbaeabfabc0d83cef`、`bb052c476ba09717363e4d29efba4fb2fd16a1e4446a475a0cf64535d009534a`、`2468fac035e1967db1a6f8dc079cbe55ffe687fa1e192384e2ef903805bfd41d`。

**匹配消融支持注意力，暂不支持密度损失的稳定独立收益。** 三组几何图 / attention / attention+density 的 assessment State 依次为秋 seed613 **.946929 / .948837 / .949192**、冬 seed613 **.952141 / .954329 / .953338**、秋 seed629 **.947936 / .949257 / .948993**；各架构使用原 calibration 封存的 checkpoint 与 α=.75，没有按 assessment 改选。注意力相对几何图的日均增益区间三组均为正；密度损失相对纯注意力的区间两组跨零、冬季全负 **[−.001629,−.000314]**。A35 未选中的纯 attention 更高，保留原选择。

![强基线上的神经网络匹配对照](reports/research/adaptive_matched_comparison_v3.png)

A38 按同 station/link 且相隔恰好五分钟的公开观测划分场景，避免混合共址站点。三组 R3 的 State gain 为 **.009450 / .008030 / .008431**。严重降速 ≥30 km/h 的诊断 channel-score gain 为秋 seed613 **+.158233（213 targets）**、冬 **+.036653（181）**、秋 seed629 **+.199325（同 213）**；样本少且种子共享日期，不能当作真实事故或独立事件重复。三个恢复分组与拥堵观测分组也改善，具体 RMSE、计数、按日区间见 A38 报告。

**A36 在第二个走廊族的监督训练也改善。** I405_N、352 sensors，cal 选纯 attention α=.75；强 LGB **.921455 → .930130**，gain **.008676**，14 日配对 CI **[.006482,.009089]**，R3 gain **.010771**。耗时 1212.19 秒，GPU 峰值 1180.95 MiB，standalone 14 日复跑通过。几何图 / attention / density 的原 cal 固定组合为 **.926662 / .930130 / .929638**；attention-vs-geometry 日均 CI [.003216,.003735]，density-vs-attention 区间跨零。A39 严重降速诊断 gain +.115282（275 targets），恢复 +.033326（641）；这些仍是带噪观测诊断。第二个方向仅一个季节/seed，尚不满足全网晋级。

**A37 零样本迁移失败。** I5_N → I10_W 不用目标历史标签或校准，固定半数组合 **.922352 vs 插值 .928998**，gain **−.006646**，日均 CI **[−.008168,−.005565]**；纯神经模型仅 .884405。A36 是使用该方向公开 train 标签的监督模型，不能抵消 A37 的零样本失败。

P01 已完成公开 train 生产准备：228 日期、4,409,727 个 required fit labels，全标签 LGB 与 NN 缓存保存，耗时 444.90 秒；没有生产留出成绩。P02 固定两个 attention / attention+density 假设、24 epochs、α=.75，已训练完成，891.96 秒、GPU 峰值 844.85 MiB。生产使用先前开发日期的公开标签是合法拟合，不能再把它们称作生产 holdout。按用户最新指示，先做模型，暂不处理平台发布。

**K03/K04 两份完整候选已生成，均未上传。** 各 **6,980,503 rows**，严格自然 key 覆盖核验与独立完整 streaming diff 均通过；仅 I5_N 的 **1,131,560 State rows** 改变（validation 588,468、private 543,092），Queue/OD 源文件 SHA 与原基准相同，其他 State 方向字段逐行相同。Task 3 使用这些 State 输入，真实 Physics 改变未知。K03 SHA **`a4645b6a5100b71b138afc37ca89bf167dd0026e7243398bba29e69d00ffd694`**；K04 SHA **`404e059f8169aa71c0a5387a40a544c346b889c98711ea03299def4d7a8560b6`**。候选各在约 243 秒完成预测/合并，保存在 `artifacts/research/K03_I5_N_attention_State_Physics` 与 `K04_I5_N_attention_density_State_Physics`；正式最佳仍是原 0.82940 文件。

C30 因果 Queue 序列模型首轮未晋级：十三个 T 及此前历史 slot 预测六个未来 horizon，时间卷积与有向图使用相同输入/预算/事件，图消息是两者唯一结构差异。相对原 LGB，NN 还使用合法 occupancy 与公开 condition，因此树对照不能单独归因架构。calibration 中 LGB **.898170**，纯 temporal **.456449**、纯 graph **.775337**，两种 NN 的最优组合权重都为 **0**；原树继续获选。assessment 去重代理 .897422，与原树相同，不能称 NN 改善；未测的纯 NN assessment 不作推断。253.87 秒 GPU，峰值 110.75 MiB。future-input 拒绝、质量过滤、无观测 fallback 与 onset-loss domain 测试通过；全套研究环境 **35 passed**，CPU **30 passed / 3 optional-Torch module skips**。

![匹配的十方向 State 与公告对齐 Queue 对照](reports/research/matched_state_queue_comparison_v3.png)

**十方向预算对照 A29–A31 已完成。** 秋季按五个走廊族等权，500k / full State 为 **.937382 / .938659**，增益 **.001277**。同一 14 天的跨方向配对日均 gain 为 .001075，block CI [.000823,.001327]；低于全局 .002 门槛，且只有 I5_N 有三个 fit seed，不晋级全网候选。I5_N 冬季 A32 为 **.941587 → .946854**，gain .005268、日均 CI [.004170,.005316]。D12_I405_S 的 500k 组已覆盖全部 fit 标签，其零增益不代表增加数据无效。

**C27–C29 的最后一步 onset 专家未晋级。** 相同事件上，calibration 均选专家阈值模型；去重代理 gain I5_N 秋 / I10_W 秋 / I5_N 冬为 **−.007850 / +.034186 / +.013551**，day-block CI [−.045468,.018747] / [−.003698,.105194] / [−.001254,.032408]，全部跨零。I10_W 未选中的 expected-IoU 在 assessment 更好，不能反选。三个共享对照从公开帧独立重建和拟合，树文本、阈值、逐窗 IoU 与 C24–C26 全部一致；不是正式全量复跑或 latent-truth 验证。

B03/B04 无训练的 masked temporal-density 投影，State 选择器两季都选基准；joint 秋 .25 / 冬 .1，State **.937994 / .946067**、increment L1 **.757656 / .729182**，存在明确 State 代价。B05/B06 L1 密度回归的 joint 都选 .5：秋 **.939019 / .732209**，不如 A19 同 blend 的 L2 **.940472 / .727239**；冬 **.944348 / .707246**，相对基准 State 损失 .002506、代理 L1 改善 .028578。selection 预先允许最多 .005 State 损失。这些权衡不证明正式 Physics 增益；新实现 **31 tests passed**。

**Queue 采样审计修正了验证范围。** 组织方在 2026-09-21 的[公开公告](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742350)确认：已发布的 onset 窗口在 T+5…T+25 没有拥堵，只有 T+30 可以出现拥堵，并明确允许使用此属性。公告作者已通过官方公开论坛 API 核验。C23 独立检查 40 个公开 train onset 窗口：39 个带噪观测在 T+30 首次出现队列，1 个未观测到队列。这不是 underlying-state 标签或正式 Queue 成绩。此前首轮代理与 C01–C22 的随机 onset 分布没有匹配这一属性，以下历史数值只作为一般事件压力测试保留，不能作为比赛采样下的晋级证据。

C24–C26 改为仅抽取公开 train 观测代理在最后一步首次拥堵的 onset，保留 45 天 calibration / 30 天 assessment 和相同候选窗口，所有候选执行公告时序约束。唯一 origin 去重、condition 等权后的 at-T 对 before-T 增益为 I5_N 秋季 **+.026375**、I10_W 秋季 **+.020772**、I5_N 冬季 **−.008370**；95% day-block CI 分别为 [−.007666,.065189] / [−.059299,.077061] / [−.032518,.009781]。三组区间全部跨零，尚不晋级。I5_N 秋季 occupancy 对 at-T 反而退化 .029563，CI [−.071365,−.001998]。真实状态与带噪代理仍有差异，正式 selector 的其他细节仍未知。

时序修正控制 K02 已本地生成并核验 6,980,503 行，仅两个 validation 窗口的 5 个 Queue 值由 1 改 0，private 完全不变。SHA 为 `cf31d26f1f75b95a3363ee59e89d5d9e392a91df479c288b147f1241ebfaf93c`，**未上传、正式成绩未知**。按完整官方聚合权重，每个改变窗口 IoU 最多增加 1，validation Total 增益安全上界为 .0075，private 为 0；假设所有旧预测单元都计分时，更紧的条件上界为 .001369，不能忽略未知 eligibility 将它当作无条件上界。保留候选，优先将正式比较用于可能改善 private 的模型。

**State 强基线复核改变了结论。** A14 将同划分、同参数 LightGBM 的标签预算从 300,000 增至全部 1,804,908，秋季 assessment State 从 **0.936655** 升至 **0.942340**，超过此前图组合的 0.940205。A15 在此强基线上，仅用 selection 选择 graph+FD、α=.25，assessment 为 **0.943417**，增益 **0.001077**。14 天日均配对增益的 block CI 为 [0.000717,0.001161]；这是开发诊断，低于登记的 0.002 晋级门槛，未成为正式候选。此前多个 NN seed 的收益只证明相对 300k 标签比较组的改善。

Queue 的规则允许使用已发布且 timestamp ≤T 的观测。C04 测得各方向 T 时刻可用观测占比约 44%–59%；新代码只以 Parquet 时间谓词读取 T，不读取未来窗口值。C05–C08 的同窗口观测代理实验存在正负结果：D12_I5_N 秋季 at-T 600/63 为 0.612418，before-T 同模型为 0.540552；但加入 bookkeeping 对照后，小 selection 集选中了 assessment 更差的 0.526387。D7_I10_W 的 at-T 400/23 也退化。不能按 assessment 反选获胜模型；较大 calibration、跨方向和不同 seed 的实验继续进行。真实 Queue 分项仍未知。

C09–C11 扩大可得校准窗口后，唯一事件去重、按日 block 的 at-T 增益区间在三组实验中全部跨零，尚不能晋级。A16 冬季全标签 LGB 为 **0.946854**，同样超过此前冬季图组合；标签预算影响具有跨季节证据。当前继续研究的版本不覆盖正式最佳提交。

C12–C14 增加双方共同发布的 occupancy 历史特征。相对各自由 selection 选定的 at-T 模型，I5_N 秋/冬唯一事件代理 IoU 增益为 **.062191 / .048073**，按日 block 95% CI 分别 [.031133,.098928] / [.011601,.082537]；I10_W 增益 **−.050232**，CI [−.151396,.041897]。方向退化阻止全局晋级；更长 45 天 calibration / 30 天 assessment 与事件均衡损失正在复核。occupancy 不被命名为真实潜在拥堵标签。

A17 冬季强 LGB 的 graph FD 组合为 **.947299 vs .946854**，日均增益 CI 跨零。A18 的 I405_N 全标签 LGB 为 **.921455**，A20 图组合为 **.922655**，增益约 .00120。三组强基线对照都低于 .002 登记门槛。

A19 的联合密度候选在 selection 上按“State 不低于 baseline −.002，最小化 density-increment 代理”选定 blend=.5；assessment State 为 **.940472 vs .942340**，代理 L1 **.727239 vs .771083**。这只是 State–物理代理的取舍，不证明正式 Physics 增益。A23b 的 occupancy/density State 特征在相同 1,804,908 个标签上得到 **.942621 vs .942340**，改善只有 .000281，未晋级。

更长校准/评估的 C15–C17 未确认 occupancy 的独立收益。增加训练事件的 C18/C20 在 I5_N 秋/冬获得代理改善，cal 选定候选相对 before-T 的去重 gain 为 **.038203 / .035234**，95% day-block CI [.015500,.061067] / [.007808,.062894]；I10_W C19 为 **−.023223**，CI 跨零。公开 condition 对照 C21/C22 没有稳定改善，保留原选择。

A24–A28 的三种子、同日期 500k/full 对照平均 pooled State gain 为 **.003896**；先平均 fit seeds 后，14 天日均配对 gain 的 block CI 为 [.002445,.003846]。这支持扩大训练标签预算，不支持宣称正式分项或 Total 已改善。首轮冻结提交 SHA 再核验一致。

B02 的学习密度增量同化，joint strength=.1，assessment State **.942160**、target-pair increment L1 **.743408**；强基线分别为 .942340 / .771083。State 选择器仍选 strength=0，不能宣称 State 提升；正式 Physics 未知。A29/A30 六个其他方向已完成 500k/full 对照，pooled State 增益：I5_S .000838、I10_E .001078、I10_W .000059、I210_E .000873、I210_W .000937、I405_S .001565，均低于 .002。训练前源码快照已保存；尚未完成全部十方向和多 seed 晋级检查。

![强 State 比较基线与跨方向 Queue 一般事件代理不确定性](reports/research/strong_baseline_comparison_v2.png)

图中 Queue 为公告对齐前的一般事件压力测试；不能与 C24–C26 的不同采样集直接比较增益。

A10 的 density-increment 联合候选秋季 State 为 0.938577，target-pair 相对 L1 为 0.786185（300k 标签基线 0.824392）；该目标只覆盖有观测且涉及 target 的相邻时刻，**不等同正式 Physics**。λ=.1 的 A11 未通过 selection。A12 的 D7_I405_N 图组合为 0.920775 vs 300k LGB 0.917777；A13 的 D7_I10_W 选择 α=0，无神经网络增益。跨 family 零样本迁移 A05 失败。

D02 自建 OD 模拟复合分数为 **0.870706**，Ridge 为 **0.825047**；D04 扩大 prior 失配分布后，missing-prior-path 场景下 Bayes 仍退化。这些模拟测量不能覆盖下面的正式失败。

OD-only 正式控制候选 ref **57140644** 已评分：**0.76474**，比基准下降 **0.06466**，拒绝晋级。当前最佳正式分数仍为 **0.82940**。候选 SHA：`4976e6bc1fe45207b6ea05dda2031d7885c9b38e1d607efc62c97887ba1e9055`。这次失败说明自建 OD 分布对正式数据的迁移假设不可靠；模拟增益不能抵消正式负面结果。证据见 `reports/research/formal_scores.json`。

![独立模拟与观测代理对照](reports/research/research_v2_comparison.png)

独立 CTM 的 54 个压力场景已运行，车辆守恒误差 ≤6.76e-13。其 latent oracle 对 noisy observation 的 State 为 0.952006，时空插值为 0.887547，RTS 为 0.848378。这些是自建模拟测量，不是比赛分数。
公开拓扑审计否定了“部分方向无可评分关系”的猜想：十方向均有可评分连接。原本地诊断错误地排除了外部连接，已修复；仍不能计算正式 Physics。

以下保留首轮交付报告作为历史实验说明。其“尚无正式分数”与 0.940 目标只描述首轮交付时状态，当前状态以本节为准。原报告也已进入冻结快照。

---
# RESULTS — 2026-10-13 官方数据首轮交付

**当前官方数据已完成下载和验证；尚无正式公开榜成绩，不能声称达到 0.940+ 或取得奖项。**
已完成十走廊四任务基线复现、独立改进、时间分组验证、模型冻结，以及 **6,980,503 行**完整提交。
公开 train 时间留出 State = **0.935606**；八走廊拥堵观测代理 IoU = **0.867664**。
真实 Queue、Physics、ODME 和正式 Total 均未知；这些代理/诊断不能替代正式分项。

## 正式指标与研发目标

| 分项 | 官方 BASELINES validation 参考 | 研发目标 | 本项目正式成绩 | 可验证的本地测量 |
|---|---:|---:|---|---|
| State | 0.6929 | 0.960 | 未知 | train 时间留出 0.935606 |
| Queue | 0.3017 | 0.890 | 未知 | 观测代理 IoU 0.867664 |
| Physics | 0.3549 | 0.940 | 未知 | 可观测守恒 RMS、FD residual、coverage |
| ODME | 0.8359 | 0.980 | 未知 | train S_link 0.999935；其他项未知 |
| Total | 0.5534 | 0.940+ | **未知** | 不将 proxy/diagnostics/S_link 拼成正式总分 |

正式规则为 `0.35 State + 0.30 Queue + 0.15 Physics + 0.20 ODME`。
参考文档固定在官方 commit `c88cddf533bbf0afa4ff1fc6c031d760a08e7a31`；参考 validation 成绩不是本项目复现成绩。
README 的 Queue persistence 0.2518 与 BASELINES 0.3017 不同，真实标签未公开，差异未能独立消除。

## 数据、环境与基线复现

- 仅使用 Kaggle 2026-09-11 重建的当前公开包，共 9,688 文件，归档 8,335,874,386 bytes，解压 12,387,782,323 bytes。
- ZIP 每项 CRC、官方列表大小和完整逐文件 SHA-256 均通过；归档 SHA-256 为 `ab2767c1c83f55d357ae16f4754d014b2bff8a6a1c28cbf9fc4876e522fb936f`。
- 数据是合成记录，使用真实走廊参数校准；合成日期不能匹配外部真实交通记录。
- Python 3.13.1，60 个锁定依赖，干净环境 `pip check` 通过，14 项测试通过，未使用 GPU。
- 官方 33 个 vendored Git blob 核验一致；原版 Task 1 builder/scorer、Task 2 persistence、Task 3 anchored evaluator、Task 4 builder/link scorer均在当前官方包实际运行。
- 原版 Task 1 十走廊 train 的 State = 0.732028，D12_I5_N = 0.625461，与 RUN_LOCAL 的 0.6252 接近。
- 原版 Task 2 生成 train+validation 174,000 rows 和 private 87,000 rows。真实标签未公开，无法复现其正式 IoU。
- 原版 Task 3 无 organizer boundary flux，只生成 `NOT_FORMAL` fallback 诊断；原版 Task 4 仅能测量 S_link。没有把 public_reference 当 path-flow 真值。

原版 train State 的日期/训练范围与下文 holdout 不同，不直接计算跨表增益。
当前包 train 的原版评分计数与模板一致，missing=0；人工 fixture 中额外 horizon blanks 导致的反例未在当前官方 train 观察到。
证据：[数据校验](reports/official/download_verification.json)、[全文件指纹](reports/official/release_manifest.json)、[原版 State](reports/official/official_state_baselines_train.csv)、[原版执行](reports/official/official_diagnostic_runs.json)。

## 时间验证和泛化

每走廊 273 个 train 日期：profile 2030-06-01–07-15（45 天），fit 07-16–11-24（132 天），calibration 11-25–2031-01-09（46 天），test 01-10–02-28（50 天）。
State 标签来自公开 train eligible observation；特征来自 masked 当日，profile 来自更早日期。每走廊最多 300,000 fitting rows，种子 2026。
模型/阈值/物理设置只按 calibration 选择，最终模型再使用后续公开 train 重拟合；其训练拟合值不当作 holdout。
D12_I5_N 是架构开发走廊，其 test 已多次查看；其余走廊使用预先固定候选与参数。Queue refinement 又在同一时间组做校准，因此这是一轮固定时间验证，不能称为完整 nested cross-validation。
已按方向、R1/R2/R3、拥堵/非拥堵观测和可用边界覆盖分组；更多 rolling folds、完全跨走廊 transfer holdout 及真实事故标签验证尚待完成。

## State：十走廊时间留出

| panel | 历史 profile test | 选定模型 calibration | 选定模型 test |
| --- | --- | --- | --- |
| D12_I405_N | 0.808033 | 0.952752 | 0.955335 |
| D12_I405_S | 0.752518 | 0.936967 | 0.934236 |
| D12_I5_N | 0.576658 | 0.939746 | 0.939808 |
| D12_I5_S | 0.633672 | 0.931664 | 0.932476 |
| D7_I10_E | 0.698937 | 0.933615 | 0.936318 |
| D7_I10_W | 0.798172 | 0.950247 | 0.947425 |
| D7_I210_E | 0.878762 | 0.936805 | 0.937946 |
| D7_I210_W | 0.753999 | 0.924223 | 0.924132 |
| D7_I405_N | 0.495627 | 0.920657 | 0.920987 |
| D7_I405_S | 0.610480 | 0.928402 | 0.927400 |

按五个走廊族等权，历史 profile test = 0.700686，当前模型 = 0.935606，增益 +0.234921。
所有方向选中 V2 LightGBM 与 geometry bounds；400 iterations、63 leaves，4 threads。V2 在 total-flow 空间插值后转换为目标每车道值，增加 density/走廊流量尺度/多跳上下游信息。
各 mask regime 的 RMSE、拥堵/非拥堵分组见 [state_strata.csv](reports/official/state_strata.csv)。

D12 开发消融（同一日期分组）：V1 插值 test 0.889830，Kalman 0.848687，LightGBM 150/31 0.930848，CatBoost 150 0.926643。
400/63 V1 calibration 0.935305；V2 calibration 0.939690。联合 density projection 的 blend 0.5/1.0 calibration 为 0.938578/0.934725，本轮采用 blend 0，未为复杂度保留无收益投影。
最终 bounds 后 D12 test 0.939808。完整原始记录在 `artifacts/official_experiments/D12_I5_N` 与 `artifacts/state_physics/D12_I5_N`。

历史急降速诊断曾报告 D12 降速 ≥30 km/h 的 647 个目标 State=0.404768、恢复 ≥15 km/h 的 1,083 个目标=0.497832、稳定变化 <5 km/h 的 712,397 个目标=0.950816。审计发现原脚本只按 link 分组，对于共址 station 不能保证正确的五分钟差分；已改为 station AND link 并对原模型、原 50 日复跑，当前 I5_N 的全部分组计数与指标仍一致。新复跑另存 `reports/research/first_delivery_scenario_station_corrected.json`，未覆盖冻结记录；其日期和模型与 A38/A39 不同，不用于直接计算新模型增益。
这些是公开观测分组，不能命名为真实事故类型；详见 [scenario_D12_I5_N.json](reports/official/scenario_D12_I5_N.json)。后续优先研究拥堵前缘和突变，而非仅增加树数量。

## Queue：合法观测代理，真实 S_queue 未知

| panel | Persistence test proxy | 400/23 test proxy | 最终 test proxy |
| --- | --- | --- | --- |
| D12_I5_N | 0.225440 | 0.822997 | 0.824194 |
| D12_I5_S | 0.238417 | 0.747631 | 0.781689 |
| D7_I10_E | 0.325386 | 0.918080 | 0.932298 |
| D7_I10_W | 0.277531 | 0.769697 | 0.774242 |
| D7_I210_E | 0.382534 | 0.983416 | 0.981557 |
| D7_I210_W | 0.316930 | 0.895659 | 0.913964 |
| D7_I405_N | 0.308792 | 0.866200 | 0.880950 |
| D7_I405_S | 0.297483 | 0.817518 | 0.852420 |

按四个被评分走廊族等权，Persistence proxy=0.296564，最终 proxy=0.867664。
特征只取发布的 `[T−60,T)` history 和公开 topology/FD 几何；不读取 horizon、未来 ramp 或 buffer 后观测。
代理 horizon 标签来自 train eligible speed≤released cutoff，与 underlying-state 真标签不同。代理窗要求 horizon 非空，condition 按当前 contract v2 的同 link 至少两次 history queue 分类，跨日保留 360 min spacing。
比较 persistence、物理波、400/23、600/63 LightGBM；增加最近拥堵前缘距离、波到达时间、5/15/30 min 下游范围速度、走廊拥堵比例。
31 点阈值网格比较 global/condition threshold 和空预测时最大概率单元的公开非空先验；calibration 决定设置，test 仅报告。参数、阈值、Brier、onset/ongoing 和每窗 IoU 见 [完整摘要](reports/official/campaign_summary.json)。
各走廊 calibration/test 落差仍明显，且正式每 split 仅十窗；proxy 方差和噪声阈值差异是主要过拟合风险。

## Physics：约束权衡，不是正式成绩

初轮 V1 D12 calibration：未调整 State=0.930799、平均每日可观测 RMS=15.188 vehicles；强度 0.001 为 0.928000/10.261，强度 0.01 为 0.901778/9.854。
强约束的 State 损失明显。本轮另保留弱约束候选：在 calibration State 损失≤0.004 的范围内选最小可观测 RMS，比较 0、1e-5、1e-4、1e-3，再报告 test。它是 Pareto 研发启发式，不能称为综合最优。
缺失 ramp、latent/external boundary 不伪造为零；只修改 target cells。没有 50 vph floor，没有将 fallback 数值作为正式 Physics。

| panel | strength | State test | State change | RMS baseline veh | RMS candidate veh | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| D12_I405_N | 0.001000 | 0.954048 | -0.001287 | 11.092905 | 7.836726 | 0.028025 |
| D12_I405_S | 0.000000 | 0.934236 | 0.000000 | 未知 | 未知 | 0.000000 |
| D12_I5_N | 0.001000 | 0.936338 | -0.003470 | 14.312918 | 10.862186 | 0.073844 |
| D12_I5_S | 0.000000 | 0.932476 | 0.000000 | 未知 | 未知 | 0.000000 |
| D7_I10_E | 0.001000 | 0.935699 | -0.000620 | 10.653004 | 8.483930 | 0.033630 |
| D7_I10_W | 0.000000 | 0.947425 | 0.000000 | 未知 | 未知 | 0.000000 |
| D7_I210_E | 0.001000 | 0.935038 | -0.002907 | 10.538233 | 8.276224 | 0.138891 |
| D7_I210_W | 0.000000 | 0.924132 | 0.000000 | 未知 | 未知 | 0.000000 |
| D7_I405_N | 0.001000 | 0.917475 | -0.003512 | 15.623260 | 12.522234 | 0.130428 |
| D7_I405_S | 0.000000 | 0.927400 | 0.000000 | 未知 | 未知 | 0.000000 |

五个南向/西向走廊在严格边界条件下没有可观测转换，保留 strength=0；其余方向 test coverage 约 3%–14%。RMS 下降无法推出真实 S_physics 提升。正式边界流未公开，所有 Physics/Total 字段继续为 null。

## OD：独立稀疏反演和收敛核验

比较 prior、ridge λ=0.05/0.005、relative-prior regularization、count-weighted residual+KL prior；逐走廊 S_link、sensor holdout、prior L1 change、失败和耗时见 [odme_ablation.csv](reports/official/odme_ablation.csv)。
默认 λ=0.05；较弱正则更贴 counts，不能证明更准的隐藏 path flows。S_od、S_dev、S_attr、S_ODME 全部未知。
matrix 仅包含 counted links，未测 connector 不当作零。正 λ 的独立 block-pivot NNLS 使用 Woodbury，把自由系统缩至 measured-links 维度，并检查非负与 KKT。
16 个随机问题与独立 dense SciPy NNLS 最大 path-flow 差<1e-6。正式十走廊与原版 dense NNLS 的最大差为 3.1575036e-05，相对最大 reference flow 为 1.61e-08；详见 [parity](reports/official/od_all_panel_parity.json)。
D12 validation 同一 ridge 目标从旧独立稀疏迭代的 461.38 秒降至 0.86 秒，S_link 几乎相同；这是非隔离运行测量，不是与原版 dense NNLS 的测速比较。
例如 D12_I5_N 的 257 counts、4,278 paths，nullity 至少 4,021；高 S_link 不能识别全部 OD。

## 完整性、复现与资源

完整主文件 `artifacts/submission_best/submission.csv`：State 6,735,795、Queue 174,000、OD 70,708，总 6,980,503 rows。
严格 SQLite 合并检查 master key 全覆盖、自然键和 ID 的跨 chunk 重复、finite/nonnegative、binary queue 以及额外 rows；全部通过。
主文件 SHA-256：`43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1`。Task 3 自动使用同一 State，无额外行。
弱约束候选 `artifacts/submission_physics/submission.csv` 也包含相同完整行集；其独立合并验证和 SHA-256 见 `reports/official/submission_physics_verification.json`。Queue 与 OD 输出保持同一候选，State 的弱约束单独比较。
另一个环境重新推理 D12_I405_S private 的 107,416 rows，与原结果所有字段完全相同，最大 speed/flow 差为 0；不据此声称整个七百万行已独立重新训练验证。
Kalman/邻域向量化在三个官方日期的全部 features/base 上数值完全一致，单日约 0.8→0.37 秒；证据 [repeat](reports/official/repeat_inference.json)、[向量化](reports/official/state_vectorization_parity.json)。

本机 Windows 11、16 logical CPU，无 GPU；主 campaign 两个进程，每模型四线程，Queue refinement 一个进程。以下是并发环境下的实际 wall time，峰值为进程 lifetime，不是隔离单模型显存/内存：

| panel | 实验秒 | 最终拟合秒 | 预测秒 | 实验进程峰值 MiB |
| --- | --- | --- | --- | --- |
| D12_I405_N | 231.126631 | 92.734506 | 48.707085 | 955.617188 |
| D12_I405_S | 133.550475 | 56.351370 | 28.640754 | 688.000000 |
| D12_I5_N | 700.513921 | 335.447589 | 125.887040 | 1528.636719 |
| D12_I5_S | 318.454820 | 141.117297 | 52.194141 | 899.128906 |
| D7_I10_E | 494.086567 | 206.480912 | 77.959379 | 1025.539062 |
| D7_I10_W | 348.351104 | 126.027731 | 59.482775 | 946.445312 |
| D7_I210_E | 884.210458 | 373.671629 | 122.817722 | 1867.675781 |
| D7_I210_W | 327.749926 | 145.549618 | 48.432643 | 837.722656 |
| D7_I405_N | 754.294966 | 301.940250 | 93.178795 | 2101.527344 |
| D7_I405_S | 283.190253 | 115.610826 | 49.979213 | 943.757812 |

完整合并耗时 194.0 秒、进程峰值 266.9 MiB。依赖、随机种子、数据/模型 SHA、命令和测量协议均保存。
复现入口：`scripts/reproduce.ps1`；加 `-PhysicsCandidate` 同时生成弱约束候选。源码与报告在 Git，大数据和模型在本机 artifacts，未纳入仓库。

## 剩余瓶颈与下一轮

1. 取得正式 Kaggle 反馈后验证四任务的综合迁移性；当前未上传，真实 Total 未知。
2. State 重点为急降速/恢复及 R3 缺失区间；研究合法 occupancy/density 辅助观测、变化点和拥堵前缘，先消融再扩展。
3. Queue 重点为低分方向、噪声阈值与少窗口方差；额外 timestamp≤T 历史/ramp 仍未消融。禁止任何 timestamp>T 输入。
4. Physics 检查弱约束对正式分数的影响；局部 RMS 优化可能无法迁移到未观察边界，不能以本地不完整评分代替正式验证。
5. OD 继续检验 prior/regularization 稳定性；只有 path truth 的正式评测才能判断其准确率。
6. 更多 rolling folds、跨走廊 transfer 和独立全量复现；准备奖项注册、最终代码和技术 PDF。登记与最终包分别截至北京时间 10 月 26 日/11 月 11 日 19:59；当前 Kaggle API deadline 为 11 月 7 日 14:55，每日最多五次。团队信息及注册尚未完成，未发送邮件。

人工 fixture 的旧轮结果单独保存在 `reports/fixture/RESULTS_PRE_OFFICIAL.md`，不能作为比赛性能。
第三方仅阅读公开架构，无代码/数据/预测导入；公开资产和许可证见 `docs/EXTERNAL_ASSETS.md`。没有使用隐藏标签、旧泄露包或组织方私有资源。
目标尚未达到不会被当作技术结论；本轮交付建立了可以继续做合法实验和正式验证的完整系统。

