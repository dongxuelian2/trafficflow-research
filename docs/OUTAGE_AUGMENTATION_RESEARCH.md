# 只改变可见输入的长断测训练消融

正式最佳仍为0.82940，正式0.95000+目标未达到。本研究回应A81b的六小时断测退步，使用公开TRAIN与自主测量失效；不使用隐藏标签、不计算真实隐藏Physics或正式Total。

## 固定假设与配对

I210_W沿用A72b的45日profile、92日fit、14日cal与14日原assessment。每个optimizer seed20261013/20261029都从头训练两个相同97307参数的TCN+GRU：control使用原输入，augmented以固定概率.5选取额外失效视图。两臂全部初始参数、原regime/day/core采样、原标签及逐格点监督曝光保持相同，24epoch×120steps、batch4、AdamW参数不变。

A86在92个fit日期各构造4份完整输入：3/6h×连续30/70%检测器，seed20261109，几何仅依赖公开timestamp/link/milepost。只读取原masked Parquet及原fit cache，不打开raw teacher。被移除的speed、flow、occupancy、派生density、pct_observed与质量flag一致修改，然后从头构建StateFeatures；不能只在旧12通道张量上清零，旧base可能残留被遮测量。原y/mask不扩充；即使额外遮盖了原先可见位置，也不为它新增监督标签。

四个尺寸同一日共享嵌套中心，均为白天失效；有限静态视图不能代表所有事故、全天任意失效分布。独立augmentation RNG不改变原patch RNG，plan记录每个样本的regime/day/core及是否使用哪个视图。

所有checkpoint仅按普通cal纯神经State、每4epoch选择；每seed/arm的alpha只按普通cal从0/.25/.5/.75/1选择。两个固定等权seed均值与原GRU、incumbent、LGB比较cal后整体封存，再读取assessment。该策略消融包含相同选择算法的结果，不保证各臂最后epoch和alpha相同，不能把全部分数差归因单个固定权重的输入操作。

## 日历边界与新研究评估

旧14日assessment已经查看过，只作回归。新日历在A87训练前固定为2030-11-27…12-10，mask seeds20261111/20261113/20261117，3/6h×30/70%共168个嵌套case；与本模型profile/fit/cal/原assessment均隔离，且不使用A81b的已查看压力日期/seed。其他历史研究仍可能查看过这些公开日期，不能称未开启final/private。

普通评估用原Task1模板目标与公开masked输入。自主失效评估用受影响region内全部raw-train eligible station行，其目标人群不同，分别报告。新评估不用于重新选checkpoint、alpha、primary或mask，保留所有方法结果、负面条件和按日期重采样的未校正区间。两个optimizer seed共享数据，不当独立交通事件。

## 已完成的输入核验

A86生成368份fit-only视图，63.72秒CPU。A89不调用增强移除helper，独立按已登记几何用Pandas移除测量，全部368份strict12与region逐值一致；首末fit日期各4视图的原被移除测量先改成极端值再删除，8次实际数据poison检查一致。独立核验59.10秒，与GPU训练并行，不当隔离吞吐。

新增测试检查真正重建后的base变化及被移除测量不可恢复，并检查augmentation概率变化不改变原patch RNG/标签曝光。当前103 research tests通过（9.08秒），CPU87通过/5可选Torch模块跳过（6.69秒）。两个control checkpoint均与原A72b对应seed的所有参数张量逐值一致。

## 训练与原公开后留结果

A87四个模型均完成，原517171个目标格点全部被实际监督；每seed两个臂的原patch SHA、总曝光和逐格点数组一致。每个增强模型的监督曝光约4.716%/4.661%位于额外失效region；这不表示只有这些输入改变，因为插值和邻居信息也会影响region外。训练892.30秒、GPU allocated峰值725.01MiB，部分时间与CPU缓存核验及测试并行，不当隔离吞吐。

| 原14日公开assessment | control策略State | augmented策略State | pooled增益 / day-mean CI |
|---|---:|---:|---|
| seed20261013 | .933135 | .928237 | −.004898 / [−.005867,−.002787] |
| seed20261029 | .935873 | .933462 | −.002411 / [−.002898,−.001279] |
| 固定两seed均值 | .935491 | .931590 | −.003901 / [−.004510,−.002258] |

Cal primary为原prior_GRU，未按assessment改选。两个control重现原GRU完整策略；其alpha分别.5/.75，augmented为.25/.5，表格是同一选择算法产生的策略比较。区间针对day-mean，不是pooled/Kaggle CI。当前固定额外失效训练损失普通重建性能；不能将更多训练视图当作自动增益。

A87同时复核出旧文档的一处登记错误：原seed20261029的TCN alpha.5而GRU .75，此前误写均.5。原JSON、checkpoint、预测及数值分数一直如此，本轮只纠正文档，原对照应解释为各自cal策略的比较。

## 预先固定的新日期结果

A88保持所有原选择，14个新日期上的普通模板State为LGB **.924732**、incumbent **.931141**、原GRU/control均值 **.933808**、augmented均值 **.929945**。增强相对control **−.003863**，day-mean gain−.003951、CI[−.005632,−.002270]；两个seed分别−.004890/−.002344、名义区间均负。原GRU相对incumbent仍+.002667，日均区间[.001393,.003732]，并非整个日历让所有新模型同时失效。

| 新自主失效条件 | control均值 | augmented均值 | 合成case均值增益 / 配对day-mean CI |
|---|---:|---:|---|
| 3h / 30%检测器 | .691200 | .766464 | +.075265 / [.061554,.088976] |
| 3h / 70%检测器 | .628309 | .741502 | +.113193 / [.098469,.127917] |
| 6h / 30%检测器 | .590348 | .692482 | +.102134 / [.086311,.119540] |
| 6h / 70%检测器 | .545320 | .686003 | +.140683 / [.125846,.155520] |

这是普通任务与严重自主失效之间的取舍：增强对长失效有明显局部收益，不能全局替换原模型。四尺寸共享42个date/seed中心、三个mask seed共享14日，CI按日期均值重采样、未作多重比较校正；均值改善不能当作正式State、Physics或Total增益。A88耗时156.20秒、allocated峰值231.51MiB。

原cal primary保留原GRU。新结果没有用于反选alpha、失效尺寸或checkpoint。本批公开研究日历此后已查看；若探索由可见缺失模式决定的专家门控，应在新的日期/seed、不同失效长度/比例及边界位置评估，避免只对本次3/6h生成器有效。全网生产模型与完整submission没有更新，正式最佳仍0.82940。

## 独立部署与评分复核

A90用独立RNG/core实现重建两个seed的全部训练计划与监督曝光；两臂实际数组一致，所有可用fit目标均被监督，control权重和cal alpha精确复现原GRU。对所有14日cal+14日原assessment的10个保存role（含导出primary）作280次实际predict调用，公开输入、目标索引与raw teacher重新核对，最大lane-unit预测差2.2737e−13；耗时74.52秒。不是独立团队重新训练。

A91重新生成新14日普通teacher、168份metadata-selected失效mask/teacher与全部target indices，独立公式复算126份普通role/day指标和1512份合成分数，最大score差1.11e−16。另在首末普通日期×9roles及首日首mask seed的4cases×9roles作54次实际predict调用，预测差0，耗时19.99秒。1512是保存指标重算数，不能冒充1512次新预测或独立事件。

## 可复现入口

所有输出必须使用新目录，不覆盖既有模型。源码以Git及各实验启动时source_snapshot.zip保存。

```powershell
.venv-research\Scripts\python.exe scripts\build_state_augmentation_cache.py --cache artifacts\research\A48_D7_I210_W_autumn_cache --baseline-file artifacts\research\A30_State_budget_directions_b\D7_I210_W\full.pkl --output artifacts\research\A86_I210_W_fit_outage_views
.venv-research\Scripts\python.exe scripts\outage_augmented_state_experiment.py --source-study reports\research\A72b_recurrent_I210_W_autumn_full_day.json --augmentation-cache artifacts\research\A86_I210_W_fit_outage_views --output artifacts\research\A87_I210_W_outage_augmentation
.venv-research\Scripts\python.exe scripts\evaluate_outage_augmentation.py --study reports\research\A87_I210_W_outage_augmentation.json --output reports\research\A88_I210_W_augmentation_future_outages.json
.venv-research\Scripts\python.exe scripts\verify_state_augmentation_cache.py --study reports\research\A86_I210_W_fit_outage_views.json --output reports\research\A89_I210_W_fit_augmentation_independent_replay.json
.venv-research\Scripts\python.exe scripts\verify_outage_training.py --study reports\research\A87_I210_W_outage_augmentation.json --output reports\research\A90_I210_W_augmentation_training_independent_replay.json
.venv-research\Scripts\python.exe scripts\verify_augmentation_future.py --study reports\research\A88_I210_W_augmentation_future_outages.json --output reports\research\A91_I210_W_augmentation_future_independent_verification.json
```
