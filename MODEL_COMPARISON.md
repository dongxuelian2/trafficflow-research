# 可复现模型对照

正式基准为 Kaggle ref **57138874 / 0.82940**，SHA-256 `43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1`。
完整文件与原模型保存在 `artifacts/frozen_baselines/57138874`。下面指标来源不同，不相加为正式 Total。

最新State两fold及固定权重二月复核已完成。下面首表为I210_W秋季同一14日公开train后留集，全部checkpoint/alpha/primary先由cal封存。

| 模型 | 本地State | 对原incumbent |
|---|---:|---:|
| full-LGB | .927102 | −.006476 |
| 原attention/LGB incumbent | .933577 | 参考 |
| 新匹配TCN两seed均值 | .933279 | −.000298 |
| 新TCN+GRU两seed均值，cal primary | .935491 | +.001914 |

两个seed的GRU−TCN为+.001321/+.002452，day-mean CI分别[.000407,.001110]/[.000078,.001479]；不能把day-mean区间当pooled gain的CI。相同seed共享原权重、所有采样位置与实际监督数组，均覆盖517171格点；GRU额外21313参数，优化预算相同而参数预算不同。原incumbent的有限context单独效应−.000015。

GRU相对原模型State略好且旧增量代理.716127→.710947；相对匹配TCN均值的代理.705201却更差，正式Physics未知。观测急降速仅54目标，3h缺失目标0，不声称已解决全部长时缺失。

| 同一冻结策略的其他公开train评估 | 原incumbent | TCN两seed均值 | GRU两seed均值 | GRU对原incumbent |
|---|---:|---:|---:|---:|
| I5_N冬季原14日 | .954329 | .955886 | .956818 | +.002488 |
| I210_W二月28日 | .934295 | .934220 | .935606 | +.001311 |
| I5_N二月28日 | .954551 | .956094 | .956542 | +.001992 |
| I210_W连续28日次要汇总 | .933915 | .934681 | .935949 | +.002033 |

两原fold平均gain+.002201通过预登记的本地State gate；二月平均+.001651。所有四个原匹配seed的GRU−TCN pooled gain为+.001321/+.002452/+.001062/+.000810，day-mean区间均正，但GRU多21,313参数，不是等容量对照。原cal/assessment共560次完整独立role-day重放，最大差4.55e−13 lane units；二月560指标重算加8次直接推断，不混为560次新预测。

| 独立合成失效人群 | 原incumbent | GRU均值 | GRU−incumbent |
|---|---:|---:|---:|
| 3h / 30%检测器 | .735689 | .737391 | +.001701 |
| 3h / 70%检测器 | .697971 | .698035 | +.000064 |
| 6h / 30%检测器 | .638638 | .635694 | −.002945 |
| 6h / 70%检测器 | .621071 | .618787 | −.002284 |

合成分数与上表模板State人群不同，不能横向比较绝对分数。42个中心共享168个嵌套case，两个6h条件名义区间均负、3h/70%跨零；没有按压力结果重新选模型。保留正式最佳，尚无新全网训练或完整submission；见 `docs/PUBLIC_OUTAGE_STRESS_RESEARCH.md`。

最新共享需求中心+路径残差，在development与**固定强entropy**比较，两轮都未过gate；旧选择封存不被assessment改变。

| 模型 / 条件 | 相对entropy模拟ODME gain | 区间 / 决策 |
|---|---:|---|
| alpha=.25/tau=.05，九同类新seed | +.019200 | [.007648,.032899]；development missing-prior负增益，未晋级 |
| 同模型，六新压力法律 | +.008855 | [−.000775,.020439]；分布迁移优势未确立 |
| 同模型，第二批九同类 | +.018551 | [.007499,.031124]；保持原development拒绝 |
| 同模型，第二批六新类 | +.009706 | [.000547,.020791]；次要研究复核 |
| 同模型，合并六seed九同类 / 六新类 | +.018876 / +.009280 | [.007580,.031924] / [.000110,.020543]；不改变gate，非正式区间 |
| floor20/alpha=.25，development | −.000844 | 最差−.010670；未选中 |
| floor20/alpha=.5，development | −.002507 | 最差−.021614；未选中 |

第一轮同类平均路径L1 .551727→.501416，但平方L2 .369977→.389611；六新类L1 .586731→.566634，平方L2 .753916→.808060。尾部代价与平均误差需并列；两种loss的排序不等同正式OD/Total。原floor2 API和alpha0强参考保持精确复现，新的prior-center floor也不构成预测下界。详见 `docs/DEMAND_RESIDUAL_OD_RESEARCH.md`。

第二批同类平方L2 .410714→.449764，新类 .716385→.776173，代价再次出现。两批900个fresh assessment问题、六seed与五固定族仍基于自主法律；合并CI是次要诊断，没有预注册的普适保证或多重比较校正。1350开发控制显式复用，不增加独立数据量。

最新 OD 配对比较为九种自主已知真值条件、十方向、三研究 seed；同一270个问题，模型选择先由 development 封存。D17为保持原选择的数值修复重跑，评估 seed 已查看，不能当新的盲测。

| OD 模型 | macro 模拟 S_ODME | 对 Ridge gain / CI | 对 entropy gain / CI |
|---|---:|---|---|
| 原 Ridge λ=.05 | .664500 | 参考 | — |
| 固定 entropy τ=.05/floor20 | .707992 | +.043492 / [.028484,.058255] | 参考 |
| 正值需求因子 sd=.2/noise=.02/floor2 | .724308 | +.059808 / [.038371,.076213] | +.016316 / [−.001239,.037405] |

新模型仍输给entropy五种条件，missing-prior-paths差−.046655，不能依据平均分替换强参考；K07仅为完整本地OD-only控制。数值修复使7条失败记录恢复（4个唯一solver问题），原参考1080记录预测/分数不变，成功分数最大漂移4.48e−7。选中参数独立oracle6/6一致，全部配置16/18；局部多解和低rank未解决。统计区间只针对五个固定公共族/三个seed及选定模拟法律，**不是正式OD/Total或后验区间**。详见D19/D20与 `docs/LOG_DEMAND_OD_RESEARCH.md`。

最新完整域校准（冻结fit权重，参数只由cal选择）：

| fold / 模型 | tree proxy IoU | selected mixture proxy IoU | 去重gain CI | before/after timing差异 |
|---|---:|---:|---|---|
| I5_N秋 / wave30 α=.25 | .872422 | .879028 | [−.004692,.018938] | 0；两者force=false |
| I5_N冬 / wave30 α=.25 | .868240 | .879943 | [−.001776,.032738] | 0；两者force=false |

两季均未通过Queue晋级门槛，C39可部署wrapper及独立220个cal-window复跑完成。K06 LOCAL完整文件只改变I5_N的415个Queue值，其他全行一致；正式分数未知，不替代.82940最佳。C40模拟仅检验采样/未来eligibility假设，所有模拟IoU不得填入本表的真实Queue列。

最新同初始化/采样的density-loss对照（每个checkpoint/α由原cal封存）：

| panel / fold | State-only | legacy density | coverage-target density | full-denominator density | primary选择 |
|---|---:|---:|---:|---:|---|
| I210_W / 秋 | .933577 | .932989 | .931394 | .932287 | coverage，未因assessment退化反选 |
| I5_N / 冬 | .954329 | .953338 | .954403 | .954407 | State-only，保留 |

公开coverage proxy L1对应秋.404858/.397383/.396416/.399691、冬.373316/.364022/.365023/.365453；不同于旧target-touching proxy，不是正式Physics。秋追加30日cal-choice State再退化.000480，CI全负；冬State小改善区间跨零。λ=.01在完整分母下有效强度约减半，因此不是等梯度幅度对照。

Queue未来评分域修正：C27原expert输出未变，但相对force=true基准的gain从−.007850改为+.017150、CI仍跨零；C28/C29的expert/baseline都是pointwise force=false，主比较不变。C32秋corrected baseline/wave30为.872422/.879028，原绝对.897422/.904028只保留为filtered-domain历史数；gain.006606与CI不变。C34冬force=false主比较不变。所有旧报告SHA保留，新旧模型权重不变；下一轮集合规则只在完整公共forecast域校准。

最新冻结先验同化对照（全部cal封存，旧14日公开train assessment）：

| panel / fold | 原神经/LGB组合 | 普通残差同化 | 遮观测中心同化 | State决策 |
|---|---:|---:|---:|---|
| I210_W / 秋 | .933577 | .933577（α=0） | .933577（α=0） | 全部cal淘汰，保留先验 |
| I5_N / 冬 | .954329 | .954406 | .954453 | 最大gain .000124，未过.002 |
| I405_N / 冬 | .933986 | .933992 | .934099 | 最大gain .000112，未过.002 |

5组CF会额外遮掉可见观测再预测其创新中心，不重新拟合先验。三组全部28日原prior/观测/graph一致，solver网格相同；cal所选参数不同，比较的是策略。I5_N CF的旧increment L1 .679941比普通 .676296更差，不能宣称同时改善State与Physics。所有正式分项未知。A65新标签缓存尚无训练结果，且本fold额外buffer锚点数为0；主要纠正Mode A coverage和分母，不当模型收益。

最新质量/密度模型采用相同初始化、采样和强 LGB；下面为公开 train 开发对照，所有模型/α 先在 cal 封存。A52 的 strict12 权重复现 A49，不作为新 seed。

| 冻结模型 | 原14日 State | 追加30日 State | A55 v2 projected increment L1 | 结论 |
|---|---:|---:|---:|---|
| full-label LGB | .927102 | .927004 | .419699 | 同分割强基线 |
| 原 strict12 attention | .933577 | .932585 | .404858 | 保留匹配 State 控制 |
| quality18 attention | .932965 | .932977 | .398451 | 追加 gain +.000392、日均 CI 跨零；不晋级 |
| density coordinates + λ=.01 | .932951 | .932583 | .389999 | State 无独立增益；仅物理代理 challenger |

proxy 改善不是正式 Physics；A55 用 raw train 的非目标 evaluator anchors、Mode A 和全部有效相邻对，旧 density-target-pair 指标口径不同。追加30日含10/13/7个R1/R2/R3日期，无调参或选择，质量日均 CI [−.000510,.000797]、密度 [−.001017,.000476]。正式 .82940、原最佳 SHA 与全任务候选均不改动。

组织方公开公告 742350 明确 onset 只在最后一步出现 queue。首轮 proxy 及 C01–C22 未匹配此采样属性，表内历史 Queue 数值均是一般事件压力测试。C24 起对齐公开时序；带噪观测仍不等于 latent truth。

| 模型 | 证据集 | State / OD 指标 | Queue 指标 | Physics 证据 | 状态 |
|---|---|---:|---:|---|---|
| D7_I210_W 图注意力 | A49 秋，同 full-LGB，cal α=.5 | .933577 vs .927102；日均 CI [.004342,.006415] | 不适用 | increment proxy .716127 vs .740967 | 第五 family 监督复核；独立 14 日 cal 重建；全网未晋级 |
| prior 浓度条件 Gaussian | D12 九模拟分布 | .666372 vs Ridge .662287；CI [.000466,.007706] | 不适用 | 不适用 | 全部 30 份赛事输入 inactive；无当前算法收益 |
| entropy / 非负 MAP 专家 | D13 新 research seeds，九分布 | .732618 vs .663559；CI [.052272,.086461] | 不适用 | 不适用 | 非 final；540 method cases 0 failures；正式未知 |
| K05 完整 LOCAL 控制 | 当前 validation/private 输入，20 split-panel | S_link-only .986901–.998681；真实 ODME 未知 | 原预测逐行相同 | 原 State 逐行相同 | 当前实际为 entropy OD-only；仅 70,708 OD 值改变，未上传 |
| CTM 加权基线 / history 消融 | B09 144 独立配置，原72 control复现 | 120min 插值 .118663→.227091；45min prior-only +1h严重退化 | 未评估 | 两种模拟 flux，不是正式指标 | 基线适配器修正；不算生产算法新收益 |
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
# C42–C48 配对窗口 IoU 损失：尚无稳定增益

| period / optimizer seed | BCE cal-selected | soft-IoU cal-selected | Lovász cal-selected | 按 cal 保留 |
|---|---:|---:|---:|---|
| 秋 / 20261013 | .879028 | .842850 | .872422 (α0) | BCE |
| 秋 / 20261029 | .872422 (α0) | .872422 (α0) | .872422 (α0) | tree |
| 冬 / 20261013 | .879943 | .878782 | .857370 | BCE |
| 冬 / 20261029 | .881507 | .878744 | .878723 | Lovász |

均为同一公开 TRAIN 观测事件的去重 condition-macro proxy，不是正式 Queue。冬新 seed 的 BCE 虽在 assessment 更高，cal 选择已封存，不能反选。原 seed 的 BCE 参数与旧 wave30 完全相同；新 seed 检验初始化/优化与 batch 变化，事件集合仍相同。12 个导出模型及 raw probability 660 次 cal-window 独立复现通过。

等 alpha 的 C46 保留次要正例：冬原 seed Lovász α.25 .883696 vs BCE .879943，gain+.003753，名义 CI 正；cal 原选 α.5，另一 seed α.25 的 gain−.002783、CI 跨零。纯网络最高 .830712 仍弱于强 tree .868240。16 个次要比较未校正多重性，不以某一正区间替换 primary。真实分项与综合分未知，K06 及正式最佳保持冻结。
# C49 跨种子集成：策略对照未晋级

| policy | 秋 proxy IoU / vs旧BCE | 冬 proxy IoU / vs旧BCE |
|---|---|---|
| 旧cal BCE α.25 | .879028 / 0 | .879943 / 0 |
| cal-selected mean-BCE | .880013 / +.000985 CI跨零 | .858806 / −.021137 CI[−.040511,−.004430] |
| 全候选cal argmax | 旧BCE，无改变 | mean-BCE α.5，同上 |
| 最简单 within1 paired-gap SE | tree .872422 / −.006606 | tree .868240 / −.011703 |

21个候选/季共用已选checkpoint和cal labels，bootstrap只检查选择敏感性；不能提供独立覆盖率。冬平均、α和threshold一起变化；不是固定决策下仅平均的因果对照。C53全部10个导出角色独立历史重建，score/binary各550次精确；这些角色和windows重复，不是10个新训练模型或550个独立事件。无新走廊或正式增益，不覆盖K06/正式最佳。

C51/C52仅测试两种history-only基线的已知latent/观测标签契约，不是将公开模型在数字孪生上验证通过。σ8的onset persistence mean-scene latent IoU0 vsobserved .029412说明指标差异，不能宣称模型改善；只有3个独立seed，场景/overlap依赖保留。
# C55–C61 五阶段同协议对照

| direction / fit_end | decision-selected参考及最终选择 | assessment noisy proxy | 固定三seed graph_mean |
|---|---|---:|---:|
| I5_S /2030-10-15 | rich600/63 tree | .781151 | .788512 |
| I5_S /2030-11-30 | plain600/63 tree | .797255 | .789825 |
| I10_W /2030-10-15 | rich600/63 tree | .557659 | .604862 |
| I10_W /2030-11-30 | rich600/63 tree | .814942 | .815249 |

graph_mean均未由decision选中，四个paired day/origin CI跨零，不反选assessment赢家。三种子图只与同划分最强plain/rich参照比較，表示完整预测策略，不能把神经原序列与树summary的差异全归因图结构。后段selection没有非零challenger，本轮无法评价gate对实际替换的效力。

新版条件均衡采样是代理的采样机制变化。C55原版 .840835与新版.781151的raw score不可比较；同新版事件上原selected .781706 vs新selected .781151，差−.000555/CI跨零。所有模型通过独立history和完整域复核，共608重复role/window调用；不等于608独立测试样本。三optimizer seeds共享事件，公开开发日历跨fold重叠，正式分项仍未知，无新晋级。
