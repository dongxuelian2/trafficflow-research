# 面向官方随机缺失的State数据增强

正式最佳仍为0.82940，0.95000+尚未达到。本研究从多小时额外断测返回官方Task1目标分布：20/30/50%随机单元mask、质量失效与Queue形状的90min空缺。这里只训练State重建模型，没有新Task2 Queue预测器或正式Physics测量。

## 拟合期输入审计

A96只读取原D7_I210_W的92个fit日期，原45日profile、14日cal、14日assessment保持隔离。原517171个station目标按完整输入同link连续缺失长度分布为：

| 缺失长度 | 目标数 |
|---|---:|
| 5min | 141027 |
| 10min | 126699 |
| 15–25min | 181236 |
| 30–55min | 58688 |
| 60–85min | 8187 |
| 90–175min | 1334 |
| ≥180min | 0 |

86.81%不超过25min，1.84%达到60min。R1/R2/R3分别28/43/21个fit日期，目标99628/230132/187411；全日输入有效grid比例均值为.612126/.535018/.381518，除官方点mask外还受公开质量可用性影响。公开train Queue index在这些fit日期的origin数为0，因此不能把这些较长缺口都标成已知Queue空缺。

## 合法重新mask与标签预算

使用公开raw-train的eligible拟合期观测，独立从头生成同一日regime rate的4份link×time Bernoulli mask，seeds20261203/20261207/20261211/20261213。所有station共享link/time mask；这是自主训练增广，不声称复现官方稳定hash，也不生成/对照任何隐藏答案。

point变体只做随机mask，尊重实际已公开Queue history/guard。queue变体在相同随机坐标基础上，每日每view另以metadata RNG选一个模拟origin：T−60…T历史可见、T+5…T+90全部测量移除，这两个区间都不作为Task1监督目标。模拟origin不按onset/ongoing或未来拥堵筛选，不等于官方赛事window selector。

目标speed/flow/occupancy/派生density/pct_observed一起移除，并更新flags后完整重建StateFeatures、base、插值和strict12。Profile固定为原先45日，不从fit/cal/assessment重新估计。网络只接受改动后的输入，原始观测teacher只进入loss。

这与上一轮“仅增加失效、保留旧目标”的消融不同：增广样本会监督原官方模板之外的新位置，因此不同标签人口是预期的。point/queue还会因模拟history/guard排除而具有不同监督数量；这是完整数据策略比较，不是相同标签数量下的单一机制证明。

A96生成92×4×2=736份视图，耗时131.93秒。A99不调用native_remask helper，独立重建坐标、删除字段、history/guard、loss masks及全部派生输入，736视图逐值一致；原raw teacher网格用独立求和核对，float32归一化标签最大差2.38e−7，固定容差1e−6。首末日期×两变体×四view共16次实际loss-target poison检查输入完全一致，113.30秒，与GPU训练并行，不当隔离测速。

缓存完成后，给后续builder补齐相邻日Queue窗口的history/guard相交选取，helper本来支持传入相邻日origin。新增跨午夜guard测试通过；对当前92fit日逐个核查后，公开origin列表全部仍为0、缓存输入完全不变。原A96启动源码保留，边界修订和逐日审计单独保存，不改旧报告或训练数组。

## 训练与分离验证

A97从原seed20261013/20261029分别训练point、queue两变体，共4个97307参数TCN+GRU，24epoch×120steps、batch4、context128/core64、AdamW lr.001/wd1e−4/clip1不变。使用与原训练相同的regime/day/core序列和初始权重；独立RNG以.5概率选4份新view之一，否则使用原输入及原目标。新视图使用新loss mask和公开raw teacher。

原GRU controls直接复用A87中已经与A72b逐参数相同、经A90独立部署核验的冻结模型。它们不重复训练，计算时间不计作本轮新模型。记录所有新监督实际曝光和额外不同标签位置，不把理论可采样当实际全覆盖；首个point模型实际覆盖原517171目标中的517165个，而新增不同位置792635个，总不同位置1309800。总监督次数15205918接近原control15207467，说明主要变化包含标签多样性。

所有checkpoint每4epoch仅按普通cal纯网络State选择；alpha仅按同一cal从0/.25/.5/.75/1选择，两个seed固定等权均值。primary仅在旧incumbent/LGB/GRU及point/queue均值间按cal选择，整体封存后才读原assessment。该assessment已查看，只作回归。训练前另固定2031-01-09…01-22的14日forward日历，模型/alpha/primary不按新结果重选；这些是公开研究日期，其他历史研究可能查看过，不称final/private。

当前116 research tests通过/9.63秒；CPU100通过、5可选Torch模块跳过/7.04秒。以下登记完整训练与固定新日历实测，不以中途cal得分宣称新最佳。

## 原assessment与固定forward实测

四模型均完成，primary仅按cal选择为queue_seed_mean，cal State .937716，point均值.936926，原GRU .934981。四新模型选中alpha均.75；原control为.5/.75。checkpoint选择与alpha属于同一数据策略的结果，不当固定混合权重下的单一因果实验。A97新训练总wall time887.62秒、allocated峰值725.01MiB；各模型约215–217秒，未将旧control训练计入本轮。

| 同一固定策略评估 | 原GRU均值 | point均值 | queue均值（cal primary） | primary pooled gain / day-mean CI |
|---|---:|---:|---:|---|
| 原14日公开assessment | .935491 | .938770 | .938919 | +.003428 / [.000914,.003166] |
| 预登记一月14日forward | .931933 | .934516 | .934816 | +.002883 / [.000874,.005771] |

对应day-mean增益为+.002040/+.003322；表中区间针对day-mean，不是pooled gain或Kaggle CI。两日历都来自公开train，两个optimizer seed共享数据与四个mask视图，区间未作多重比较校正，不证明private split或其他方向表现。

原后留14日R1/R2/R3分别7/6/1日，8910个R3目标不能当8910个独立样本。新一月R1/R2/R3为2/6/6日，R3 State原GRU **.922288**、point **.926421**、queue **.927472**；primary R3 gain+.005184。R1/R2也略正，但R1只有2日。原后留primary R3 gain+.006432主要来自一个日期，证据强度较弱。

第一seed的point/queue相对对应control，原后留gain+.005082/+.005019，一月+.003888/+.004032，day-mean名义区间均正。第二seed相应为+.001433/+.001487和+.001524/+.001992，但四个day-mean区间均跨零，不宣称所有seed都已统计证实。

queue相对point均值仅+.000149（原后留，CI[−.000730,.000808]）和+.000300（一月，CI[.0000097,.000805]），第二个下界接近零且未校正多重性。两变体的标签位置/数量也不同，尚不足以把整体收益归因模拟Queue空缺。

实际不同标签位置：point两seed **1309800/1310493**，queue **1259133/1259532**；原对照517171。point总曝光15205918/15247813，queue14368985/14336327，原对照15207467/15247793（已按旧A87记录核对）；有意记录多样性和曝光差别，不称等标签人口。原先517171目标在第一seed实际监督517165、第二seed517171，少量随机未抽到的目标不是结构性边界不可训练。

## 误差与物理限制

原后留primary对drop20速度RMSE **17.157→15.304**、恢复 **11.260→10.240**；一月drop20 **25.067→23.691**，恢复反而 **14.664→16.119**。一月60min缺口速度RMSE **3.050→2.742**、拥堵观测 **5.766→4.469**，但拥堵flow RMSE **45.562→46.080**。这些是少量、相关的观测目标，原drop/recover54/62，一月81/96，不是真实事故或独立事件。

旧target-pair increment L1在原后留 **.710947→.693890**、一月 **.785325→.760853**，只是本地不完整物理代理，正式Physics仍未知。普通input中3h连续缺失目标为0，不再用严重额外失效成绩代替Task1增益。A98耗时51.86秒、allocated峰值234.58MiB。

当前是值得扩展的单方向State候选，不是已验证全网最佳：需另一个方向/时间fold及更多seed、同remask数据的强LGB对照，再考虑全273日生产拟合和完整候选。原正式.82940和所有完整submission保持冻结；没有新正式成绩。

## 独立曝光与部署复核

A100不调用训练plan函数，独立重建全部regime/day/core和view RNG，以及四个模型实际loss-mask曝光数组；总曝光、不同标签位置和原模板外新增位置逐项一致，不要求它与旧control标签人口相同。随后从公开masked Parquet和raw evaluator重建所有原14日cal+14日assessment，对12种角色（含导出primary）作336次实际predict调用，最大lane-unit差2.2737e−13，耗时84.89秒。旧control是明确复用的权重，不把重放计为新训练。

A101独立重建一月14日raw teacher、目标顺序、station/link lanes及全部速度变化/缺失诊断mask，重算168份保存role/day指标和regime汇总，最大score差2.22e−16；首末日期×12roles另作24次实际predict调用，预测差0，耗时11.16秒。168是保存指标复算数，不是168次新预测；这些检查不替代跨方向或独立训练复现。

## 复现入口

输出必须为新目录，旧模型、源快照与完整submission不覆盖。

```powershell
.venv-research\Scripts\python.exe scripts\build_native_remask_cache.py --source-study reports\research\A72b_recurrent_I210_W_autumn_full_day.json --output artifacts\research\A96_I210_W_native_remask_fit_views
.venv-research\Scripts\python.exe scripts\native_remask_state_experiment.py --views artifacts\research\A96_I210_W_native_remask_fit_views --output artifacts\research\A97_I210_W_native_remask_ablation
.venv-research\Scripts\python.exe scripts\evaluate_native_remask_forward.py --study reports\research\A97_I210_W_native_remask_ablation.json --output reports\research\A98_I210_W_native_remask_forward_calendar.json
.venv-research\Scripts\python.exe scripts\verify_native_remask_cache.py --study reports\research\A96_I210_W_native_remask_fit_views.json --output reports\research\A99_I210_W_native_remask_cache_independent_replay.json
.venv-research\Scripts\python.exe scripts\verify_native_remask_training.py --study reports\research\A97_I210_W_native_remask_ablation.json --output reports\research\A100_I210_W_native_remask_training_independent_replay.json
.venv-research\Scripts\python.exe scripts\verify_native_remask_forward.py --study reports\research\A98_I210_W_native_remask_forward_calendar.json --output reports\research\A101_I210_W_native_remask_forward_independent_verification.json
```
