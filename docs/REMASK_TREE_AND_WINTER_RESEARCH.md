# 同视图LightGBM对照与冬季走廊扩展

正式最佳仍0.82940，0.95000+未达到。本轮先完成I210_W同视图树模型对照，冬季I5_N四模型仍按冻结方案训练；不将中途cal分数登记为后留成绩。

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

## 冬季扩展：已启动，后留未知

A102将同一4seed/point/Queue-shaped策略扩至D12_I5_N，profile45日、fit153日（7/16–12/15）、cal12/16–12/29、原assessment12/30–1/12；共1224份fit-only视图，数据准备591.11秒。A105对全部视图独立重建和16次真实loss-target poison检查通过，normalized teacher差2.38e−7在原1e−6容差内，498.93秒。

A104保留相同2optimizer seed、97307参数、24×120/batch4、context128/core64、.5增广概率与cal选择算法。旧control取A73/A75已保存、280role-day独立部署核验的模型；检查三个构造源码与旧start archive完全一致、shared initialization hash精确复现，不重复训练旧控制。角色解析仍兼容旧A97的11种角色。

冬季新forward日期1/23–2/5在训练前冻结，和原profile/fit/cal/assessment均不重叠；它仍是公开研究日期，其他过去研究可能使用过，不称final/private。四新模型没有全部完成前，不读原assessment和forward真值，也不根据第一个seed的cal表现终止或改选。GPU allocated峰值与全部State/Physics结果等待完成后登记。

当前117 research tests通过/13.92秒，CPU101通过、5个可选Torch模块跳过/11.42秒；BoundedState的总流量单位、FD上限、键顺序和保存重载测试通过。原0.82940和所有完整候选保持冻结，无平台操作。

## 复现入口

```powershell
.venv-research\Scripts\python.exe scripts\build_native_remask_cache.py --source-study reports\research\A73_recurrent_I5_N_winter.json --output artifacts\research\A102_I5_N_native_remask_fit_views
.venv-research\Scripts\python.exe scripts\native_remask_state_experiment.py --views artifacts\research\A102_I5_N_native_remask_fit_views --reference-replay reports\research\A75_recurrent_I5_N_independent_replay.json --forward-start 2031-01-23 --forward-end 2031-02-05 --output artifacts\research\A104_I5_N_native_remask_ablation
.venv-research\Scripts\python.exe scripts\remask_lightgbm_experiment.py --views artifacts\research\A96_I210_W_native_remask_fit_views --output artifacts\research\A103_I210_W_native_remask_LGB_control
.venv-research\Scripts\python.exe scripts\verify_remask_tree_study.py --study reports\research\A103_I210_W_native_remask_LGB_control.json --output reports\research\A110_I210_W_remask_tree_independent_verification.json
.venv-research\Scripts\python.exe scripts\evaluate_remask_tree_forward.py --study reports\research\A103_I210_W_native_remask_LGB_control.json --output reports\research\A109_I210_W_remask_tree_forward_calendar.json
```
