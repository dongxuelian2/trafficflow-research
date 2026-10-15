# 同视图LightGBM对照与冬季走廊扩展

正式最佳仍0.82940，0.95000+未达到。I210_W同视图树模型对照、冬季I5_N四模型均已完成；冬季全部选择封存后才读取原后留。后续通道集成见 [A112–A115b](CHANNEL_STATE_RESEARCH.md)。

## 树模型数据与公平性边界

A103使用A96与神经网络相同的原fit日期、原profile、原目标及四份point/Queue-shaped重新mask视图。每个变体训练20261013/20261029两个seed；沿用旧强LightGBM每通道400棵树、63叶、lr.035、reg_lambda5、min_child_samples80、4CPU线程及60维v2表格特征。

Point池有2586005行，其中原目标517171、新视图2068834；Queue-shaped池2363377行，其中新视图1846206。原目标占有效权重一半，四份新view共占另一半；每组内部按regime/date/target均匀赋权，总有效权重固定517171。所有加权数据的feature/base/teacher/row key/weight均保存。

这是同mask和标签源的强参照，不是纯架构单因素实验：树使用60维手工特征、平方残差损失、完整加权数据遍历，NN使用strict12、patch RMSE与随机采样；旧树原本未采用相同重加权。更多标签、分布重加权和模型表达能力的效应不能被这一张表完全分离。

BoundedState仅对旧有非负预测应用公开free-speed/capacity的1.05上限，与原State候选相同，flow始终是总流量单位。新wrapper不改变任何旧模型或提交。

## 完整结果与未晋级的混合

| 固定策略 | 原14日assessment State | 预登记1/23–2/5 State |
|---|---:|---:|
| 旧强LGB | .927102 | .921600 |
| point LGB两seed均值 | .931238 | .924704 |
| Queue-shaped LGB两seed均值 | .930716 | .924905 |
| 原A97 cal选中NN | **.938919** | **.929704** |
| .25 point树 + .75 NN | .938490 | .929993 |
| .25 Queue-shaped树 + .75 NN | .938358 | .930050 |

Cal比较旧LGB、冻结NN、两种树均值及树权重.25/.5/.75的六个混合，仍选择NN。原assessment两个树均值相对NN分别−.007681/−.008202，day-mean名义区间均负。新日期仍低−.005000/−.004799，区间均负。

新日期两个.25混合的pooled gain+.000289/+.000346，但日均区间分别[−.000598,.000607]/[−.000513,.000646]，原assessment还略负；没有根据这一微小次要结果改变cal primary。新forward selected policy与原NN完全相同，不能声称树或混合新增正式收益。

四个树拟合分别86.58/86.78/73.96/81.21秒，数据准备69.22/68.82秒；整个A103500.86秒，纯树训练和cal数组比较不需要GPU。A10932.96秒的前向评估实际使用GPU（含冻结NN推断，evidence标记True），不当CPU测速；所有实验与冬季任务部分并行，不作隔离吞吐比较。

## 独立复核

A110逐项重建两池共920个date/view组的全部输入表、teacher、base、row keys及权重，权重最大差0；对6种纯树角色在28个cal/assessment日期实际predict168次，另4次直接选中策略调用，最大lane-unit差2.27e−13；392份保存role/day指标独立复算。核验182.26秒，不是独立重训。

A111重新读取14日raw teacher、目标/lanes及全部变化/缺失诊断mask，独立复算210份保存指标，最大score差1.11e−16；首末日期×9个已保存角色另作18次predict，预测差0，9.92秒。派生混合也核对与组件数组一致，210不能写成210次新推断。

## 冬季扩展：四模型完成

A102将同一4seed/point/Queue-shaped策略扩至D12_I5_N，profile45日、fit153日（7/16–12/15）、cal12/16–12/29、原assessment12/30–1/12；共1224份fit-only视图，数据准备591.11秒。A105对全部视图独立重建和16次真实loss-target poison检查通过，normalized teacher差2.38e−7在原1e−6容差内，498.93秒。

A104保留相同2optimizer seed、97307参数、24×120/batch4、context128/core64、.5增广概率与cal选择算法。旧control取A73/A75已保存、280role-day独立部署核验的模型；检查三个构造源码与旧start archive完全一致、shared initialization hash精确复现，不重复训练旧控制。角色解析仍兼容旧A97的11种角色。

冬季新forward日期1/23–2/5在训练前冻结，和原profile/fit/cal/assessment均不重叠；它仍是公开研究日期，其他过去研究可能使用过，不称final/private。四模型及全部cal选择完成后整体seal才读原assessment，没有根据中途cal停止或改选。

cal选择Queue-shaped双seed均值，四个新模型alpha均为.75。原assessment的旧GRU/point/Queue-shaped均值State为 **.956818/.957220/.957512**，primary gain **+.000694**、day-mean **+.000624**、名义CI **[−.000218,.001466]**，尚无正区间证据。两个Queue seed pooled gain+.001176/+.000344，但日均区间均跨零；第二seed point为−.000005，保留该负面结果。原assessment的R1/R2/R3为5/5/4日。

primary急降速/恢复速度RMSE从11.172/13.706降至9.983/10.016，但拥堵速度从1.738恶化为2.012、恢复流量从122.310恶化为127.809。旧增量L1从.641245恶化为.650629，**不当正式Physics增益**。180min缺失目标0。point/Queue-shaped原标签之外的新监督位置增加，实际独立监督格点约699.5–700.3万/674.5–675.7万，对照原目标304.2万；同更新预算不是等标签人口。

与I210_W合并，两个方向原assessment primary pooled gain的简单平均+.002061达到原mean门槛，但冬季区间跨零；不能据平均数声称稳定跨方向成功。完整A1042154.28秒，四fit/cal角色约512.56–518.28秒/个，allocated峰值2099.12MiB；含与其他研究部分并行的开销，不当隔离测速。

A106固定1/23–2/5结果：原GRU **.954500**、point **.955275**、封存Queue-shaped primary **.954872**。primary gain+.000372、day-mean CI[−.000220,.001040]；point gain+.000775、次要名义CI[.000122,.001363]，但不根据forward反选。Queue-shaped相对point为−.000403、day-mean CI[−.000524,−.000202]。该日历R1/R2/R3为4/7/3日，耗时140.97秒。primary旧增量L1 **.646055→.659551** 变差；恢复速度/流量改善、拥堵和60min缺失速度退步，详见RESULTS及完整JSON，正式Physics未知。

A107独立重建全部4份实际exposure数组及RNG/patch/view计划，28日×12角色336次保存模型predict最大lane-unit差4.55e−13，277.12秒。A108独立从raw TRAIN重建新14日全部teacher/lanes/indices与诊断mask，168保存指标复算最大差2.22e−16，另24次predict差0，33.47秒。复用控制和重算指标不计作新的独立模型/数据。

加入通道组合后，全套118 research tests通过/9.95秒，CPU102通过、5个可选Torch模块跳过/7.79秒。BoundedState与ChannelState的单位、键顺序和保存重载检查通过。原0.82940和所有完整候选保持冻结；正式Physics/Total仍未知，无新完整submission。

## 复现入口

```powershell
.venv-research\Scripts\python.exe scripts\build_native_remask_cache.py --source-study reports\research\A73_recurrent_I5_N_winter.json --output artifacts\research\A102_I5_N_native_remask_fit_views
.venv-research\Scripts\python.exe scripts\native_remask_state_experiment.py --views artifacts\research\A102_I5_N_native_remask_fit_views --reference-replay reports\research\A75_recurrent_I5_N_independent_replay.json --forward-start 2031-01-23 --forward-end 2031-02-05 --output artifacts\research\A104_I5_N_native_remask_ablation
.venv-research\Scripts\python.exe scripts\remask_lightgbm_experiment.py --views artifacts\research\A96_I210_W_native_remask_fit_views --output artifacts\research\A103_I210_W_native_remask_LGB_control
.venv-research\Scripts\python.exe scripts\verify_remask_tree_study.py --study reports\research\A103_I210_W_native_remask_LGB_control.json --output reports\research\A110_I210_W_remask_tree_independent_verification.json
.venv-research\Scripts\python.exe scripts\evaluate_remask_tree_forward.py --study reports\research\A103_I210_W_native_remask_LGB_control.json --output reports\research\A109_I210_W_remask_tree_forward_calendar.json
.venv-research\Scripts\python.exe scripts\evaluate_native_remask_forward.py --study reports\research\A104_I5_N_native_remask_ablation.json --output reports\research\A106_I5_N_native_remask_forward_calendar.json
.venv-research\Scripts\python.exe scripts\verify_native_remask_training.py --study reports\research\A104_I5_N_native_remask_ablation.json --output reports\research\A107_I5_N_native_remask_training_independent_replay.json
.venv-research\Scripts\python.exe scripts\verify_native_remask_forward.py --study reports\research\A106_I5_N_native_remask_forward_calendar.json --output reports\research\A108_I5_N_native_remask_forward_independent_verification.json
```
