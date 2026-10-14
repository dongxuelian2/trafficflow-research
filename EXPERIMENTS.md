# 实验台账 — 0.950+ 攻关

所有模拟指标、观测代理与正式 Kaggle 指标分别记录。失败实验不删除，既有模型与提交不覆盖。

本轮完成 A48–A50、B08/B09、D12/D13，以及完整 LOCAL K05。正式 Total 仍 .82940，没有上传或替换正式最佳。

| 编号 | 假设 / 对照 | 实测结果 | 资源 / 决策 |
|---|---|---|---|
| A48 | D7_I210_W 同秋季图缓存 | 92 fit / 14 cal / 14 assessment、81 nodes | 75.55 秒 CPU，输入与强基线 profile 匹配 |
| A49 | 三架构与 full-label LGB 匹配 | cal 选 attention α=.5；State .927102→.933577；日均 CI [.004342,.006415] | 311.71 秒 GPU、288.65 MiB；14 日 cal standalone 重建通过 |
| A50 | 公开 train 缺失段长度 | 三 cache R3 input p99=60min、target p99=25min | 仅 availability/mask；跨 panel 共 360 个文件不是独立 360 日 |
| B08 | 四种自主连续缺失 × 六条件 × 三 seed | 插值 State：uniform .832553、半走廊 .711935、事故 120min .118663、全网45min .576647 | 4.86 秒；独立复跑 5.99 秒，全 72 case 精确相同 |
| B09 | prior × interpolation 2×2，并检验 +1h 时钟 | 加权插值使 120min 缺失 .118663→.227091；prior-only 时钟偏移严重退化 | 26.59 秒 GPU；144 case、576 adapter 配置；无模型选择 |
| D12 | 六 MAP/Huber 配置与可见 prior 规则 | 选相对 Gaussian cv=.25、η=0、threshold .75；旧 seed gain .004085，CI 为正 | 749.94 秒 CPU；额外 sparse penalty/Huber 未选；30 个赛事输入全部退回 Ridge |
| D13 | entropy fallback + MAP 专家，开发缓存选定后开三个新 research seed | 九类模拟 .663559→.732618；gain CI [.052272,.086461]；最差类均值也正 | 89.52 秒 CPU；540 method cases 无失败；非 final / 正式测量 |
| K05 | 完整 OD-only 本地控制 | 6,980,503 行；仅 70,708 OD 值改变；所有 State/Queue 字段不变 | 146.60 秒构建；全行独立 diff 通过；未上传，真实成绩未知 |

D12 的 standalone 相对 Gaussian 还会在 missing-prior 与 group-biased 场景退化，不能按平均分直接替换所有方向。D13 只用开发案例选择了 absolute cv=.75/η=1 的稀疏分支；新 seed 平均 sparse ODME .653270→.765436，但当前 validation/private prior 浓度都不触发，K05 实际全为 entropy。旧 K01 正式失败仍是反证，新的模拟并不消除该风险。

D13 新种子中14/270案例为负，最差 count_outliers .555405→.527118（−.028287），属于D7_I405_N、seed7001、entropy分支。误差审计另存，模型不根据已看案例改选；下一步应从开发端检验可见计数不一致特征及稳健 likelihood，而不是只优化均值。

B08/B09 比较严格分离 target population：不同 missing pattern/时钟有不同 targets，不把它们的绝对分数差当同标签算法增益。B09 同场景 weighted vs temporal-first 是配对适配器对照；该加权方式本来就在生产 StateFeatures 中。历史 profile 仅使用 dev seeds 411/829/1201 的 eligible masked-observation 字段，全局按 288 slots/cell 聚合；六个 profile 元素没有观测而回退几何。场景共享噪声 RNG，不能把 36 个合成日当 36 个独立样本。

本轮完成 A46/A47、B07 与独立进程复跑、D09 失败记录/D10 全量修复与覆盖率复核。没有模型晋级或新增正式成绩。开发和 assessment 的身份保留，最后独立模拟 seeds 仍未开启。

| 编号 | 假设 / 对照 | 实测结果 | 资源 / 决策 |
|---|---|---|---|
| A46 | D12_I405_S 同 autumn fold 的 masked graph cache | 92 fit / 14 cal / 14 assessment，44 nodes；full-LGB profile 和输入重建匹配 | 57.63 秒 CPU；缓存及 start snapshot 保留 |
| A47 | 同强 LGB 的 geometry/attention/density 控制 | cal 选 density α=.75；State .941500 → .941872，gain .000371；日均 CI [−.000654,.004659] | 203.31 秒 GPU，179.63 MiB；未过 .002 gate，14 日 cal standalone 核验通过 |
| B07 | A33 固定 checkpoint 在独立 CTM 的 54 条件 | interpolation .853404，pure attention .848350，density .855957，half-density .864132 | 7.79 秒 GPU、46.59 MiB；mild 退化、CTM-flux 诊断下降，不晋级 |
| B07 repeat | 相同全部条件的独立进程复跑 | 所有 registered/input SHA/候选值/summary 完全一致 | 9.04 秒；3 seeds 而非 54 独立样本 |
| D09 | 冻结点估计 + Hessian 区间/新失配 | 校准倍数 6.402476；汇总时 metadata 覆盖模型身份，assessment aggregate 未保存 | 失败源快照、seal、210 校准数组保留；不接受缺失/NaN 成绩 |
| D10 | 仅修复 D09 元数据隔离/阶段保存 | 同 210 数组和倍数；旧 420 point scores 完全相同；600 assessment method cases、0 failures | 167.91 秒 CPU；模拟 sparse-active −.075310，原三登记条件 CI 跨零，不晋级 |
| D11 | 先核对路径选择是否存在于公开 path set | 十方向全部 OD 对单路径；D10 route-choice 标签实际为既有 demand 扰动 | 2.24 秒 CPU；两类真正新分布 macro CI [−.016780,.051299]，不晋级 |

D11 更正 D09/D10 生成标签的含义：`route_choice_shift` 是历史名称，singleton OD 下 preference 归一化抵消，因此不能称为新的路径选择分布。上表原“三个新分布”指三个登记条件；实际为两类新失配与一次既有 demand 分布复核。原指标未修改，九种实质生成分布对应十个历史标签。以后路径选择候选必须先证明公开资产有相同 OD 的可替代路径，或明示自主生成的替代路径实验范围。

A47 消融保留原 cal 权重：geometry .940372、attention .941929、density .941872；未因 assessment 中 attention 略高而改选。B07 density 的急剧降速 target latent speed RMSE 28.969 → 18.243，但恢复 speed RMSE 22.827 → 25.351；795 / 788 是共享缺失率下的重复 target 计数，不是独立事件数。事故片段收益不能抵消 mild 或恢复失误。

D10 empirical interval 的 raw coverage 在七个旧分布为 .3048–.7801，校准后 .9498–.9996；新区间宽度约真实平均路径流量的 2.54–3.45 倍。校准只改变诊断区间，不改变预测点。Sparse-active 的 .9766 边际覆盖率没有转成准确路径恢复或任何全路径覆盖保证。精确范围、所有 case 与失败原因见原 JSON；独立 dense Hessian inverse、equal-case calibration tests 核验数值。

本轮完成 A40–A44、C31–C34、D06–D08。I5_N 三 seed 与 I405_N 两季的注意力收益均相对 full-label LGB 复核；三种子等权组合保留为开发模型。Queue 两季 cal-selected 波速图 CI 均跨零，不晋级。熵反演在同一生成分布的数值修复后通过模拟门槛，真实 OD 仍未知；原正式最佳不改变。

| 编号 | 固定操作 / 对照 | 测量 | 资源 / 决策 |
|---|---|---|---|
| A40/A42 | I405_N 冬季图缓存、500k/full 同参数控制 | 153/14/14 日；full 3,985,807 labels；State .923899 → .925958 | 459.01 秒准备、497.09 秒 CPU 控制；非正式分项 |
| A41 | I5_N 第三 NN/LGB seed 20261107 | cal 选 attention α=.75；State .942330 → .950416；日均 CI [.006530,.008567] | 504.70 秒 GPU，845.07 MiB；14 日 standalone cal 复跑通过 |
| A43 | I405_N 冬季，同 NN 协议与 full-label 强基线 | State .925958 → .933986；CI [.007056,.009472]；increment proxy .765975 → .726403 | 705.18 秒 GPU，1180.95 MiB；14 日 standalone 复跑通过 |
| A44 | 三个原 cal 成员等权 vs 同三 LGB 等权 | State .942742 → .951168；CI [.007304,.008681]；proxy L1 .756319 → .671424 | 108.25 秒；共享 14 日，不冒充 42 日；一个整日 wrapper/save/reload 核验 |
| A45 | I10_W 监督注意力 vs 同日期 full-LGB | State .951973 → .955810；CI [.003496,.004039]；proxy L1 .719817 → .701798 | 401.02 秒 GPU，359.92 MiB；14 日部署复跑通过；原 cal 选择保持 |
| C31 | 固定 checkpoint 的 class-weight 概率修正 | graph Brier .015419 → .012059；IoU 组合仍选 α=0 | 187.20 秒；概率误差下降不代表 Queue IoU 增益 |
| C32 | 秋季 direct/15min/30min 波速图范围对照 | cal 选 30min α=.25；去重 gain .006606，CI [−.004692,.018938] | 176.90 秒 GPU；不反选 assessment 更好的 15min |
| C33 | 冬季普通因果 temporal/graph，正确加权概率 | cal 选 graph；去重 gain .008485，CI [−.006698,.028666] | 219.89 秒 GPU；不晋级 |
| C34 | 冬季同 C32 的三范围对照 | cal 选 30min α=.25；gain .011703，CI [−.001776,.032738] | 185.75 秒 GPU，111.27 MiB；跨季区间仍跨零 |
| D06 | 七种已知真值分布，熵双空间 Newton vs Ridge/Bayes | 69 次优化失败；有效配置未过最差开发场景门槛，选 Ridge | 769.35 秒；不删除失败 |
| D07 | 浮点舍入预算内接受残差下降步 | 原 69 例全部恢复；新增 1 例失败；dev 选 τ=.05/floor20 | 356.32 秒；该失败不在选定配置，继续数值修复 |
| D08 | 用 Newton decrement 判定浮点精度附近步骤 | 全量 0 failures；Ridge 分数完全相同；既有成功 entropy 最大漂移 2.32e−9 | 341.75 秒；tol 2e−6 不变，全部生成/选择条件保持 |

D08 的独立生成模拟 ODME：Ridge .691063、固定旧 Bayes .728094、开发选定 entropy .747273。family/seed bootstrap 的 entropy gain CI [.043844,.068733]，只适用于这七种生成分布与三个 assessment seed；final seeds 尚未生成。旧 Bayes 在模拟上更好却已正式失败，不能把新模拟收益当正式增益。公开 train 计数只能做稳定性诊断；entropy 约 99% 的变化在零空间。

本轮源代码与模型结果均另存；A45 监督复核完成，不能抵消 A37 零样本失败。平台工作按用户指示暂缓，只有本地 Git 归档。43 research-env tests、CPU 38 passed / 3 optional-Torch module skips；数值修复还由 D06→D08、D07→D08 同场景比较核验。

采样审计 C23 发现组织方公告 742350 已明确 onset 的最后一步属性。首轮代理和 C01–C22 不匹配该时序，全部保留为一般事件压力测试。只有 C24 起的 `--onset-last-step` 对照近似这一公开属性；仍未取得 underlying-state queue truth。

| 编号 | 假设 / 操作 | 数据与划分 | 结果 | 资源 / 决策 |
|---|---|---|---|---|
| F00 | 冻结首轮正式提交 | 本人 Kaggle 回下载，与本地文件 SHA 比对 | 正式 Total **0.82940**，ref 57138874；四项正式分数未知 | 106 个冻结文件，712,127,414 bytes；保留基准 |
| P00 | 公共拓扑是否导致部分方向无法评分 | 公开拓扑、原版 scorer、人工理想两时刻状态 | 十方向全部有可评分连接，否定结构性 0.950 上限猜想 | 本地诊断漏掉外部连接；修复并测试 |
| B01 | CTM、缺失、事故和噪声压力实验 | 独立 54 个场景，3 seed × 6 条件 × R1/R2/R3 | 守恒误差 ≤6.76e-13 vehicles；State observation oracle 1.0，latent oracle 0.952006，时空插值 0.887547，RTS 0.848378 | 7.60 秒 CPU；模拟数据不等于正式生成器 |
| D01 | 分层贝叶斯 OD 对比 Ridge | 两走廊、独立已知真值模拟 | 初步增益，但 zone set 的迭代顺序依赖 Python hash seed | **不作为可复现结论**；保留记录，改为 sorted 后重跑 D02 |
| D02 | 固定组顺序的分层贝叶斯 OD | 十走廊，每方向 3 生成分布 × 3 assessment seed；selection seed 独立 | 模拟 ODME Ridge 0.825047 → Bayes 0.870706；少数 case 退化，最差 −0.003804 | 72.95 秒 CPU；public_reference 从未读取 |
| D03 | 不同 Python hash seed 独立进程复跑 | D12_I5_N / D7_I210_W | 除耗时和运行指纹外，全部数值一致 | 支持重复性，不证明隐藏真值准确 |
| C01 | 随机事件采样、onset/ongoing 分别建模 | D12_I5_N rolling_autumn；带噪观测代理 | old joint 0.426710 → random joint 0.558105 / separate 0.570598 | 同 14 天抽样种子重叠；非独立样本 |
| C02 | 跨方向复核事件模型 | D7_I10_W rolling_autumn | old joint 0.770357 / random joint 0.781109 / separate 0.745426 | **独立事件模型退化**，不晋级 |
| C03 | 跨季节复核事件模型 | D12_I5_N rolling_winter | old joint 0.596339 / random joint 0.657088 / separate 0.646768 | separate 不如共享模型，不晋级 |
| A01 | 缺失感知时间卷积 / 图扩散 / 图+FD 联合损失 | D12_I5_N rolling_autumn，同 92 天 fit / 14 天 selection / 14 天 assessment | selection 选择 graph+LGB α=.5；assessment State 0.936655 → 0.940205；日均配对增益 0.002739，95% block CI [0.001963,0.003462] | GPU 56.43 秒，峰值 344.41 MiB；只有单走廊单 seed，不晋级 |
| A01-FD | FD auxiliary weight .01 | 同 A01 | selection 低于纯 graph；未按 assessment 反选 | 保留负面消融 |
| A02 | 冬季 rolling 对照 | D12_I5_N rolling_winter | State 0.940342 → 0.942887 | 同架构、同参数，α=.5 |
| A03/A04 | 第二个 NN fit seed 20261029 | 秋季 / 冬季，同固定 LGB 比较 | selected State 0.939143 / 0.944371 | 增益继续存在；LGB fit seed 固定 |
| A07/A08 | 第三个 NN fit seed 20261107 | 秋季 / 冬季，同固定 LGB 比较 | selected State 0.939584 / 0.943671 | 多 seed 仍只覆盖单方向 |
| A05 | 零样本跨 family transfer | D12_I5 → D7_I10_W，目标 family 标签不进输入 / 无目标 calibration | 插值 0.928998，纯 graph 0.913238，固定半数组合 0.927536；日均 CI 完全负 | **迁移失败**，不能据此部署共享全走廊模型 |
| A06 | 第三个 rolling 时间段 | D12_I5_N rolling_august | 300k LGB 0.944993 → graph α=.25 0.946550 | 52.92 秒 GPU；增益低于 .002 |
| A09 | public-train accumulation increment 联合损失 | 同 A01；有观测节点固定，仅预测 target | λ=.01；按 State 仍选择普通 graph，未选 auxiliary | 71.17 秒 GPU；不将 density increment proxy 称正式 Physics |
| A10 | 由 cal density 指标选联合候选 | 同 A01；State 至少不低于 baseline 的约束 | State 0.938577；target-pair L1 .824392 → .786185 | 单方向、单 seed、300k 比较组；不是正式 Physics |
| A11 | density-increment 权重 .1 | 同 A01 | selection 选择 α=0，无 State 增益 | 24.66 秒 GPU；强约束失败 |
| A12 | I405_N 逐方向图专家 | D7_I405_N autumn，352 sensors | 300k LGB .917777 → graph FD α=.5 .920775 | 78.50 秒 GPU；仍待强基线预算对照 |
| A13 | I10_W 逐方向图专家 | D7_I10_W autumn | cal 选择 α=0，assessment 保持 .951806 | 86.63 秒 GPU；无独立增益 |
| A14 | 同参数 LGB 标签预算控制 | D12_I5_N autumn，300k → 全部 1,804,908 标签 | State .936655 → **.942340**，超过原 graph 组合 .940205 | 191.10 秒 CPU；修正架构收益解释 |
| A15 | 强 LGB 上重做图组合 | 同 A14，仅 cal 选 graph FD α=.25 | assessment **.943417**，比强 LGB +.001077；日均 block CI [.000717,.001161] | 低于 .002 晋级门槛；无正式 State/Physics 测量 |
| C04 | 发布 T 时刻数据的合法可用率 | validation/private 的 at-T 时间谓词扫描 | 平均可用约 44%–59%；不读未来数值 | Rules 允许 ≤T，原模型只用 <T |
| C05 | 同窗口 at-T 消融 | I5_N autumn | before-T 600/63 .540552 → at-T .612418，cal 选择 at-T | 73.45 秒 CPU；10 个 cal 窗口 |
| C06 | 分离 latest-valid bookkeeping 作用 | 同 C05，增加两个 before-T 对照 | cal 选中 before-T-latest 600/63，assessment **.526387** | 98.76 秒 CPU；选择失稳，不能按 test 改选 .612418 |
| C07 | at-T 跨方向检查 | I10_W autumn | cal 选 at-T 600/63 .802211 vs before-T .765435；at-T 400/23 退化 | 42.22 秒 CPU；仅 6 个 cal 窗口 |
| C08 | at-T 跨季节检查 | I5_N winter | cal 选 at-T 400/23 .697673 vs before-T .670006 | 132.37 秒 CPU；仅 10 个 cal 窗口 |
| D04 | 扩大独立 prior 失配分布 | 十方向、4 种 reverse noisy prior、3 assessment seed | Bayes 在 missing-prior-path 模拟下 .606048 vs Ridge .611359，负增益 | 50.28 秒 CPU；未解释正式全量退化，不以此调隐藏反馈 |
| A16 | 冬季全标签 LGB 控制 | I5_N winter，全部 3,042,465 个标签 | State **.946854**，超过此前最佳冬季图组合 .944371 | 290.08 秒 CPU；强基线结论跨季节成立 |
| C09–C11 | 扩大事件 cal 配额至每条件 30，assessment 至 20 | I5_N 秋/冬、I10_W 秋，实际配额由可用事件限制 | cal 选定候选未稳定改善；唯一 origin 的 at-T 配对 CI 均跨零 | 不晋级；保留 65 / 14 / 59 个唯一 origin 的按日 block 诊断 |
| A17 | 冬季强 LGB 上重做图组合 | 同 A16，cal 选 graph FD α=.25 | State .946854 → .947299；日均增益 CI [−.000010,.000610] | 低于 .002 门槛，CI 跨零 |
| A18 | 较大 I405_N 的全标签 LGB | 2,438,669 fitting labels，同 A12 | assessment State **.921455**，超过原图组合 .920775 | 245.22 秒 CPU |
| A20 | I405_N 强 LGB 上重做图组合 | 同 A18，cal 选 graph FD α=.25 | assessment State **.922655**，增益 .001200 | 低于 .002 门槛 |
| C12–C14 | occupancy 当前值/趋势/上游下游/速度密度一致性 | 同 C09–C11，全部特征 timestamp ≤T | I5_N 秋/冬去重代理增益 .062191 / .048073，CI 正；I10_W **−.050232** | 双方历史只用共同发布 occupancy，不使用 window_history 缺失的 density 字段 |
| C15–C17 | 45 天 cal / 30 天 assessment；事件条件/正负类均衡损失 | I5_N 秋/冬、I10_W 秋；独立新开发划分 | cal 均选 occupancy 600/63；assessment .680572 / .658247 / .574586；I10_W before-T 400/23 为 .622312 | 更长 cal 未消除跨方向退化；事件权重候选未由 cal 选中 |
| A19 | 全标签 direct density expectation 与 q/v 联合投影 | 同 A14；复用原 State 权重，仅拟合密度回归器 | cal State 选 .1，assessment .942174；joint 选 .5，State .940472、density L1 .727239 vs .771083 | 195.49 秒 CPU，峰值 2619 MiB；没有正式 Physics |
| A23/A23b | 发布 occupancy/derived-density 的 State 树特征 | 同 A14，全 1,804,908 labels | A23 在拟合前因只读数组失败，保留；显式 copy 后 A23b State **.942621** vs .942340 | 219.06 秒 CPU；微小增益不晋级 |
| C18/C19/C20 | 扩大合法训练事件预算 | 同 C15/C16/C17 的日期；fit 配额 500，spacing 30 分钟；assessment 仍六小时 | cal 选 occupancy，assessment I5_N 秋 .747888 / 冬 .725320；I10_W .657542 vs before-T .681712，仍退化 | 不将增加训练窗口视为增加独立事件样本 |
| D05 | 坐标可识别性与完整网络负载诊断 | 公开网络/计数/本队两份 OD 预测 | 所有计数列模式均不同，**无重复列**；仍没有一般线性可识别的单独路径，max leverage 平均 .02779；K01 未计数链路负载漂移约 8.2% | 否定重复列猜想；大零空间来自线性组合，不读取隐藏 path truth |
| A24/A25 | 与正式 500k 预算对应的强基线 / 第二个 full fit seed | I5_N autumn，同日期与模型参数 | 500k State .938184；full seed 20261029 .942295；seed 20261013 full .942340 | 145.94 / 200.76 秒 CPU；其余 seed 控制继续进行 |
| A24–A28 汇总 | 三种子 500k vs 全标签配对 | I5_N autumn，同 14 assessment days | full 平均 pooled State 增益 **.003896**；平均跨 seed 的日均 gain .003146，block CI [.002445,.003846] | 三个 fit seed 不冒充 42 个独立日期；满足这一开发方向的 .002 门槛 |
| C21/C22 | 合法公开 condition 与观测推导 condition 对照 | 同 C18/C19，所有窗口/标签匹配 | cal 仍选普通 occupancy；published-condition 候选没有稳定独立收益 | 保留失败，模型部署支持该显式可选标记但不默认开启 |
| B02 | 学习 Δdensity 并固定可见观测的三对角同化 | 同 A14 全标签基线；public-train target-pair 标签、geometry 权重 | State 选 strength=0；joint 选 .1，assessment State .942160 vs .942340、density L1 .743408 vs .771083 | 317.45 秒 CPU；没有正式 Physics，可见锚硬固定 |
| A29/A30 | 六个其他方向的 500k/full 对照 | autumn，同早期 profile / fit / selection / assessment | full pooled State gain：I5_S .000838 / I10_E .001078 / I10_W .000059 / I210_E .000873 / I210_W .000937 / I405_S .001565 | 全部低于 .002；124.28 / 207.56 / 141.96 / 358.97 / 115.91 / 119.43 秒 CPU；训练前源码 ZIP |
| C23 | 公开 onset 公告与采样审计 | 40 个公开 train 窗；本队 val/private 预测与公开窗口 metadata | 39 个带噪 train 首次 queue at T+30，1 个无观测 queue；本队仅 5 个早期正预测 | 组织方允许时序属性；不读 val/private 未来观测，不视为真实标签评分 |
| C24 | 与公告 onset 时序对齐的 at-T / occupancy 对照 | I5_N autumn，45 天 cal / 30 天 assessment | 109 个去重 origin；before-T .871047 → at-T .897422，gain .026375，CI [−.007666,.065189] | cal 选 at-T；366.66 秒 CPU；occupancy 显著退化，不晋级 |
| C25 | 公告对齐的跨方向检查 | I10_W autumn，同 C24 方法 | 26 个去重 origin；before-T .606343 → cal 选 occupancy .628012；CI [−.064953,.082322] | 80.95 秒 CPU；仅 5 个 onset，证据弱，不晋级 |
| C26 | 公告对齐的跨季节检查 | I5_N winter，同 C24 方法 | 103 个去重 origin；before-T .876610 → cal 选 at-T .868240；gain −.008370，CI [−.032518,.009781] | 401.91 秒 CPU；保留退化，不按 assessment 改选 occupancy |
| K02 | 组织方已公开时序的 Queue-only 修正 | 原正式提交，所有其他值逐行保留 | 完整 diff 仅 5 个 Queue 值，private 完全相同；未上传，正式分未知 | 合并 125.42 秒；保留本地，正式比较优先具有 private 改善空间的模型 |
| A31 | 补齐三方向 500k/full 对照 | 同秋季，D12_I405_N/S、D7_I405_N | gain .000741 / 0 / .002521；I405_S 两组均已覆盖全部 fit 标签 | 145.61 / 89.65 / 392.02 秒 CPU |
| A32 | 冬季 500k/full 对照 | I5_N，同 A16 日期/参数 | .941587 → .946854，gain .005268；日均 CI [.004170,.005316] | 398.53 秒 CPU；full 复跑 pooled State 与 A16 一致 |
| A29–A31 汇总 | 按官方五族权重合并十方向 | 秋季，seed 20261013，同 14 天 | .937382 → .938659，global gain .001277；日均 CI [.000823,.001327] | 52.93 秒重建日统计/汇总；低于全局门槛 |
| C27/C28/C29 | onset 最后一步 / ongoing 六步专家与 expected-IoU | matched C24/C25/C26 事件与合法特征 | cal 都选专家 threshold；去重 gain −.007850 / +.034186 / +.013551，CI 全跨零 | 208.83 / 50.54 / 218.37 秒 CPU；不反选未被 cal 选中的胜者 |
| C24–C29 复跑 | 共享对照从公开帧重建并拟合 | 三组原对照与独立新研究 | 树文本、阈值、逐窗 proxy IoU 完全一致 | 不等于真实 Queue 或正式全量复现 |
| B03/B04 | 无新训练的 masked density 插值投影 | I5_N 秋/冬，强 LGB；cal State 容忍 .005 | State 都选 baseline；joint 秋 .937994/.757656、冬 .946067/.729182 | 39.58 / 37.03 秒 CPU；诊断收益小，State 代价明确 |
| B05/B06 | L1 条件中位数密度回归 | 全 fit labels，强 LGB；cal State 容忍 .005 | State 都选 baseline；joint .5 秋 .939019/.732209、冬 .944348/.707246 | 215.40 / 262.97 秒 CPU；秋季不如 A19 L2；无正式 Physics |
| A33 | 几何图 / attention / attention+density 的强基线匹配对照 | I5_N 秋，seed 20261013，全标签 LGB，cal 选择 | State .942340 → .949192，α=.75；日均 CI [.006348,.007157] | 489.27 秒 GPU，845.07 MiB；standalone 14 日独立复跑通过 |
| A34 | 同协议跨季节 | I5_N 冬，seed 20261013 | State .946854 → .954329，纯 attention α=.75；CI [.006259,.007608] | 1139.94 秒并行 GPU，845.07 MiB；不是正式分项 |
| A35 | 同协议第二 NN/LGB fit seed | I5_N 秋，seed 20261029 | State .942295 → .948993，attention+density α=.75；CI [.005566,.006983] | 1140.11 秒并行 GPU，845.07 MiB；与 A33 共享日期 |
| A36 | 同协议第二个走廊族的监督模型 | I405_N 秋，seed 20261013，352 sensors | State .921455 → .930130，纯 attention α=.75；CI [.006482,.009089]，R3 +.010771 | 1212.19 秒 GPU，1180.95 MiB；14 日 standalone 复跑通过 |
| A33–A35 消融 | 所有架构的原 cal checkpoint / blend 均封存 | same assessment；原赢家保留 | attention 比几何图的三组日均 CI 均正；density 比 attention 两组跨零、冬季全负 | 不反选 A35 的 assessment 胜者；所有原赢家指标复现 |
| A37 | 新 attention 模型跨 family 零样本迁移 | I5_N → I10_W；目标标签仅用于评分 | 插值 .928998 / 半数组合 .922352 / 纯 NN .884405；半数组合 CI 全负 | 31.44 秒；失败，不部署共享模型 |
| A38 | 急降速、恢复、拥堵观测、缺失诊断 | 同 A33–A35，按 station/link 五分钟差 | 三组 R3 均改善；少数急降速目标改善，完整 RMSE/计数/日区间记录 | 诊断 score 跨 regime 按 cell pooled，不冒充正式宏平均或真实事故标签 |
| P01 | 全公开 train 的生产准备 | I5_N，45 日 frozen profile 后 228 日 | 4,409,727 labels，全标签 LGB / NN 缓存覆盖完整 | 444.90 秒 CPU；没有 holdout 或正式成绩 |
| P02 | 固定两种生产 NN 假设 | P01 的全部公开 fit 日期，24 epochs / α=.75 | attention 与 attention+density CPU 权重及 standalone 均保存 | 891.96 秒 GPU，844.85 MiB；仅训练记录，没有 holdout |
| K03/K04 | 两种生产假设的完整本地控制候选 | 只改 I5_N State/Physics 输入；其他任务固定 | 各 6,980,503 行，全行 diff 通过，仅 1,131,560 State rows 改变 | 242.97 / 242.22 秒；未上传，正式成绩未知，原最佳不覆盖 |
| C30 | 因果十三步历史预测六步 Queue | matched C27，同事件/标签；NN 使用合法 occupancy/condition | cal 原树 .898170，纯 temporal .456449、纯 graph .775337；组合都选 α=0 | 253.87 秒 GPU，110.75 MiB；拒绝晋级，原树 assessment .897422 保持 |
| A39 | 较大方向的场景诊断 | 同 A36，station/link 五分钟差 | 严重降速诊断 +.115282（275），恢复 +.033326（641） | 带噪观测，不能视为真实事故标签 |
| K01 | 一次 OD-only 正式控制实验 | 同首轮所有 State / Queue 数值，全量 6,980,503 行 diff | **正式 Total 0.76474，下降 0.06466，拒绝晋级**；ref 57140644 | SHA 4976e6bc1fe45207b6ea05dda2031d7885c9b38e1d607efc62c97887ba1e9055 |
| ENV01 | 独立 CUDA 环境 | Python 3.13.1 / RTX 4060 Laptop 8GB | torch 2.8.0+cu128 已安装；实际 CUDA 运算通过 | 当前沙箱临时目录访问失败，改为 E:\\ieee\\.cache\\build_tmp 后构建成功；原 .venv-repro 不修改 |

复现入口：`scripts/register_research_protocol.py`、`audit_physics_topology.py`、`simulate_stress.py`、`od_probability_experiment.py`、`queue_event_experiment.py`。
正式反馈只读本队提交。禁止隐藏标签、组织方内部资产和其他队伍预测。

A33–A36 已完成，四个 standalone 各 14 日重建/预测验证通过。I5_N 两季两 seed、I405_N 单季单 seed 的开发证据均正；A37 跨 family 迁移失败。各架构的匹配 assessment 消融保留 cal 选择，attention 增益稳定，density 独立 State 收益不稳定；不能按已看的 assessment 改选 A35。P01/P02 完成，K03/K04 各 6,980,503 行覆盖与限定任务/方向的全行 diff 通过；正式成绩未知。C30 未超越树模型，拒绝晋级。两个 GPU 训练进程合计曾观察约 3.2 GiB / 8 GiB。原正式 profile 与开发 profile 的数值最大差约 4e-12，仅浮点舍入，不能据 equals=False 推断日期变化。

当前研究环境 35 tests passed（14.32 秒），CPU 30 passed / 3 optional-Torch module skips（8.73 秒）。首轮场景诊断改为按 station/link 差分后，对原模型与原日期独立复跑，所有分组计数和指标仍与历史 I5_N 记录一致；原冻结结果保留。

当前完整模型/配置快照 `model_artifacts_checkpoint_adaptive_v2.json` 保存 **334 项**已完成 artifact SHA。`K03_K04_local_state_controls.json` 保存完整候选 SHA、大小、覆盖及全行 diff；`adaptive_verification_record.json` 记录实际测试与独立复跑的范围。所有文件仅在本地，重要代码与报告通过本地 Git 提交归档。

下一阶段已启动，尚无成绩：A40 为 I405_N 冬季图缓存准备；A41 为 I5_N 第三 NN/LGB fit seed 20261107 的同协议架构对照；A42 为 I405_N 冬季 500k/full 强基线。运行入口、完成标记与续接动作记录于 `research_checkpoint_adaptive_phase.json`。缓存准备拒绝已存在目录，修改 NumPy 数组时显式 copy，开始前保存源码快照。

专家/密度阶段 31 tests passed。expected-IoU 与小规模独立枚举全部标签、全部预测集合一致；独立 Bernoulli 假设不代表交通单元真实独立。density 投影网格严格核对完整 key，并保留缓存预测不变；测试发现 Pandas 可返回只读 NumPy 视图，已显式 copy 并验证与直接预测一致。新实验在开始前保存源代码与依赖锁。

A01 的 CI 是按天配对的诊断，不是非线性 pooled State 或 Kaggle Total 的置信区间。A14 已证实标签预算混杂：全标签 LGB 超过原神经网络组合；A15 的独立增益低于登记门槛。后续必须继续比较强基线和多个 fit seed。

**K01 的迁移失败优先于 D02 模拟增益。** 其他任务值逐行完全相同，按 0.20 权重，ODME 总体分项差约 −0.32330（显示分数舍入误差约 ±0.00005），绝对分项值仍未知。候选可能存在严重 prior distribution 失配，不能仅因模拟更优继续增加其 ensemble 权重。下一轮需扩大生成分布、反向构造 noisy weak prior、检查 nullspace 漂移与真实可用稳定性；不依照这一个正式反馈细调参数。

K01 public-only 诊断：validation 的 Bayes-vs-Ridge L2 差异平均 99.94% 位于观测矩阵零空间；计数误差几乎看不到这些变化。Bayes 总需求约为 Ridge 的 80.74%，相对 prior 的偏移量约为 Ridge 的 1.56 倍。这里只解释模型的行为，不能证明隐藏 path truth 或精确归因正式损失。

现有首轮 January–February test 已经查看；新 rolling assessment 仍是公开 train 开发证据。独立模拟 final seed 7907/8017/9109 尚未生成，需在架构和参数封存后一次性评估。

C09/C10/C11 的去重、按 condition 等权、日期 block bootstrap 增益分别为 .016495 / .024714 / .014972，95% CI 分别为 [−.012655,.060063] / [−.117752,.076207] / [−.017293,.099297]。比较的 before-T 与 at-T 模型都只按各自 selection 选定，assessment 未参与选择。相同 origin 的重复抽样去重，缺失某 condition 的 bootstrap draw 明确排除；I10_W 的事件特别稀少，因此区间证据较弱。

本轮新增模型与配置 116 项 SHA 已记录于 `reports/research/model_artifacts_checkpoint_20261013_a.json`。正在运行的后续模型不在此快照中。CUDA 环境 21 tests passed；测试临时目录使用工作区 `.cache`，系统沙箱 TEMP 不可写。

公告对齐阶段归档：`model_artifacts_checkpoint_20261013_e.json` 保存 235 项已完成模型/配置 SHA；当前 CUDA 环境 25 tests passed，CPU 24 passed / 1 optional-Torch skipped。K02 完整候选与全行 diff 保存在 `artifacts/research/K02_Queue_only_public_timing`，正式成绩仍未知。安全/条件收益界和不上传的决定由 `scripts/record_onset_control_bounds.py` 生成，证据为 `reports/research/K02_public_timing_decision.json`。
