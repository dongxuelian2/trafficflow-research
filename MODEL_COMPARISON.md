# 可复现模型对照

正式基准为 Kaggle ref **57138874 / 0.82940**，SHA-256 `43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1`。
完整文件与原模型保存在 `artifacts/frozen_baselines/57138874`。下面指标来源不同，不相加为正式 Total。

组织方公开公告 742350 明确 onset 只在最后一步出现 queue。首轮 proxy 及 C01–C22 未匹配此采样属性，表内历史 Queue 数值均是一般事件压力测试。C24 起对齐公开时序；带噪观测仍不等于 latent truth。

| 模型 | 证据集 | State / OD 指标 | Queue 指标 | Physics 证据 | 状态 |
|---|---|---:|---:|---|---|
| D12_I405_S 监督图模型 | A47 秋，同 full-LGB，cal 选 density α=.75 | .941872 vs .941500；gain .000371，日均 CI [−.000654,.004659] | 不适用 | increment proxy .741879 vs .758759 | 第四走廊族小收益未过 gate；14 日 standalone 核验通过 |
| 固定半数图 State 混合 | B07 自主 CTM，三 seed/六条件/三缺失率 | half-density .864132 vs interpolation .853404；mild −.010945 | 未评估 | CTM-flux .332950 vs .358359 | 全 54 条件独立进程数值相同；不晋级生产 |
| 固定纯图 State | 同 B07；geometry-only profile、无 occupancy | attention .848350；density .855957；插值 .853404 | 未评估 | density CTM-flux .327728 | 急降速改善、恢复速度恶化；不视为正式 Physics |
| 熵点估计新失配 | D10/D11 两种真实新增失配，设置不变 | sparse-active .572888 vs Ridge .648198；两类 macro CI [−.016780,.051299] | 不适用 | 不适用 | 原 route-choice 标签实际为 demand 同类复核；稀疏失败，不晋级 |
| 熵 Hessian/经验区间 | D10 210 开发案例校准、300 assessment 案例 | sparse raw coverage .03994，扩张 6.4025 后 .97659，宽度/平均真流量 3.051 | 不适用 | 不适用 | 仅边际经验覆盖；无 simultaneous/shift/posterior 保证 |
| 三种子固定 State 组合 | A44 I5_N 秋；同三 seed full-LGB 组合对照 | **.951168 vs .942742**，日均 CI [.007304,.008681] | 不适用 | increment proxy .671424 vs .756319 | 14 个共享日；一个完整日 wrapper/load 核验；未扩展生产 |
| 第三 seed 图注意力 | A41 I5_N 秋，seed 20261107 | **.950416 vs .942330**；日均 CI [.006530,.008567] | 不适用 | 观测增量代理，非正式 Physics | 14 日 cal standalone 复跑通过 |
| 冬季图注意力 | A43 I405_N 冬，seed 20261013 | **.933986 vs .925958**；日均 CI [.007056,.009472] | 不适用 | increment proxy .726403 vs .765975 | 第二方向两季复核；尚无全网 gate |
| I10_W 监督图注意力 | A45 秋，seed 20261013，full-LGB 对照 | **.955810 vs .951973**；日均 CI [.003496,.004039] | 不适用 | proxy .701798 vs .719817 | 第三走廊族监督收益；14 日部署复跑通过；不是零样本 |
| class-weight 概率修正 | C31 固定 checkpoint | 不适用 | graph Brier .015419 → .012059；组合 α=0 | 不适用 | 概率误差下降未转为 IoU 增益 |
| 公开波速图邻域 | C32 秋 / C34 冬，cal 均选 30min α=.25 | 不适用 | 去重 gain .006606 / .011703；两季 CI 跨零 | 表示性邻域，不是严格传播锥 | 不晋级；未反选更好的 15min assessment |
| 熵正则双空间 Newton | D08 七种独立已知真值分布 | **模拟 ODME .747273 vs Ridge .691063**；cluster CI [.043844,.068733] | 不适用 | 不适用 | 合法模拟 gate 通过；真实 OD 未知，正式最佳保持 |
| 熵数值失败与修复 | D06 / D07 / D08 同数据/门槛 | 69 failures → 1 新失败 → 0；Ridge 分数逐项一致 | 不适用 | 不适用 | tol 2e−6 保持；原记录保留；不是新的盲测 |
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
| 全标签 LGB，十方向 | A29–A31 同秋季/参数 | family State .938659 vs .937382，gain .001277 | 不适用 | 无正式 Physics | 低于全局 .002 门槛，不全网晋级 |
| 全标签 LGB，冬季 | A32 I5_N，同 500k 对照 | .946854 vs .941587，gain .005268 | 不适用 | 无正式 Physics | 单方向预算收益跨季节成立 |
| onset / ongoing 专家 | C27–C29，对齐时序，cal 选 threshold | 不适用 | 去重 gain −.007850 / +.034186 / +.013551；所有 CI 跨零 | 不适用 | 不晋级；不按 assessment 改选 expected-IoU |
| temporal-density 投影 | B03 / B04 强比较组 | joint State .937994 / .946067，均低于基准 | 不适用 | increment L1 .757656 / .729182 | State 保留 baseline，未证明正式 Physics |
| L1 密度回归 | B05 / B06 强比较组 | joint State .939019 / .944348 | 不适用 | increment L1 .732209 / .707246 | 秋季不如 A19 L2，State 两季保留 baseline |
| 图注意力 + 密度损失 α=.75 | A33 I5_N autumn，强 LGB 比较 | **.949192 vs .942340**；日均 CI [.006348,.007157] | 不适用 | proxy increment L1 **.687617 vs .771083** | standalone 已验证；真实 Physics 未知 |
| 图注意力 α=.75 | A34 I5_N winter，强 LGB 比较 | **.954329 vs .946854**；日均 CI [.006259,.007608] | 不适用 | proxy increment L1 **.682571 vs .735824** | 两季单方向开发证据；standalone 通过 |
| 图注意力 α=.75 | A36 I405_N autumn，强 LGB 比较 | **.930130 vs .921455**；日均 CI [.006482,.009089] | 不适用 | increment proxy 仍非正式 Physics | 第二方向监督模型；仅单季单 seed；standalone 通过 |
| 图注意力 + 密度损失 α=.75 | A35 I5_N autumn，NN/LGB seed 20261029 | **.948993 vs .942295**；日均 CI [.005566,.006983] | 不适用 | proxy increment L1 **.686074 vs .769809** | 与 A33 共享日期；不反选更高的纯 attention |
| 匹配几何图 / attention / density 消融 | A33–A35，原 cal checkpoints/α 固定 | attention 的三组配对 CI 均正；density 两组跨零、冬季全负 | 不适用 | 独立 density State 收益未确认 | 原赢家保留；开发消融不是新最终测试 |
| 跨 family 零样本 attention | A37 I5_N → I10_W | 半数组合 **.922352 vs 插值 .928998**；纯 NN .884405 | 不适用 | 不适用 | CI 全负，迁移失败；不部署共享全网模型 |
| 因果 Queue temporal / graph | C30，matched C27 事件/标签 | 不适用 | cal 纯 NN .456449 / .775337 vs 树 .898170；组合都选 α=0 | 没有 State 未来信息输入 | 253.87 秒 GPU；不晋级；选定 assessment 与原树相同 |
| 生产 attention / attention+density | P02，I5_N 全公开 train | **无生产 holdout 分数** | 保留原 Queue | 真正 Physics 未知 | K03/K04 全任务覆盖及限定方向全行 diff 通过；未上传 |

模型晋级需要相同划分、多个拟合种子、按天或事件配对不确定性分析和分方向退化检查。行级随机拆分不可作为依据。跨走廊 transfer 必须整组排除一个 family 的训练标签；当前逐走廊训练不冒充 transfer。

图网络与树模型的日期相同不意味着标签预算相同。后续 State 架构结论必须以可用全部 fit 标签或明确的大预算 LGB 为比较组。Queue 抽样 seed 可能重复同一事件；按唯一 origin 和日期分析，不能把重复 seed 当独立试验。
