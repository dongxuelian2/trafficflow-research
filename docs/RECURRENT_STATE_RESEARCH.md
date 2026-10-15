# 固定上下文的双向时序状态分支

正式最佳仍为 **0.82940**，0.95000+ 目标未达到。本轮是公开train上的State研究，不计算真实隐藏Physics或正式Total；原完整提交、所有已验证模型不覆盖。

## 模型与同条件对照

候选在现有四层图时序卷积、两层空间attention编码器后，新增两层双向GRU，每方向hidden24、合计48维，残差gain初始.1。原输入、geometry、卷积、attention和输出头在同一个optimizer seed下逐参数一致；输出头仍零初始化，两模型初始输出均为插值/profile base。候选额外21,313参数，TCN为75,994、TCN+GRU为97,307；不是等参数预算的单一因果证明。

GRU是学习的潜在时序状态，不声称它就是Kalman、RTS、CTM或真实密度动力学。两输出仍为speed/25、lane-flow/600，最终按公开FD上界裁剪，再与同一full-label LGB混合。只有合法masked观测、availability、历史profile、base、occupancy和时刻周期进入strict12输入；诊断label与mask不进入预测接口。

训练/推断均采用128步context、64步supervision/core。每天按64步分片，边界裁剪context，输出每个格点一次。State允许使用公开的前后观测；这些双向输出不能作为Queue在T时刻不可见的未来输入。

每个方向/时间fold采用20261013与20261029两个optimizer seed，每模型24epoch×120steps、batch4。每seed的TCN/GRU共享所有原始权重初始化和完整采样序列，记录逐格点实际监督次数。loss为同一按patch的通道缩放RMSE，R1/R2/R3均衡抽取。两个神经模型使用相同可用fit标签及更新数；full-LGB使用同日期所有station目标，神经cache为同link×time站点均值，标签表示和重复暴露数不完全相同。

checkpoint仅按cal纯神经State、每4epoch检查；每seed/架构的alpha从0/.25/.5/.75/1中仅按cal选择。另固定等权seed均值。primary仅按cal比较这些策略、原incumbent和full-LGB，之后整体封存，再读取assessment。原incumbent以全日上下文推断，同时另存同权重固定窗口的shadow诊断，避免把上下文变化当架构收益。

## 首次试跑的训练覆盖缺口

A72最初按context左端均匀抽128步、只监督中间64步。精确fit-mask审计发现日内前32/后32步永远不会训练：I210_W有 **115,049 / 517,171（22.2458%）**目标格点，I5_N冬有 **674,637 / 3,042,465（22.1740%）**。两个架构受同一限制，但该设计不能充分覆盖全日重建。

首个TCN完成、GRU训练至epoch20后，session93669经Ctrl-C主动停止，exit1；没有selection seal，也未读取assessment标签。两个checkpoint、TCN导出模型、旧注册、源快照、partial cal日志与stop记录全部保留。不是数值非收敛，也不能据此判断架构优劣。

修订为先均匀抽监督中心，再裁剪core和包含它的context，确保每天所有格点有正采样概率；并记录实际曝光。单元测试覆盖全日首尾、短序列、每格点一次输出，以及改变一个batch样本不能影响另一个样本。新的A72b从头初始化，没有复用受限监督权重。

A78只读fit mask、不读assessment标签。原A77前向评估注册保留，A77b v2明确记录替换试跑的原因，未静默改写原计划。

## 研究日历与误差诊断

第一组为D7_I210_W秋季，沿用A48缓存、A30 full-LGB、A49冻结incumbent；第二组为D12_I5_N冬季，沿用A02缓存、A16 full-LGB、A34冻结incumbent。profile/fit/cal/assessment日期分别隔离，但都是已查看过的公开train开发日期，不是未开启的final/private测试。两个seed共享数据，不当独立测试样本。

预先固定的诊断为：相邻5分钟两端均eligible的观测速度下降/恢复至少20km/h；输入单日连续缺失speed至少12/36步；观测速度低于50km/h。它们是观测代理/缺失诊断，不是真实事故或latent queue标签。legacy target-pair increment单独记录，不是完整Physics评分。

在读取原assessment之前，还登记2031-02-01…02-28的固定权重前向日历复核。只有两组训练、全部原选择seal及独立导出完成，才读这28天标签；必须与原profile/fit/cal/assessment完全不重叠。该批仍是公开train的开发证据，其他历史研究可能已经使用过；不称最终盲测，不按其结果重选alpha/模型。

## 可复现命令

所有输出使用新目录，不能覆盖旧研究。源码与锁定依赖分别保存Git和实验source_snapshot.zip；模型、实际采样暴露、每日预测与输入SHA保存在本机artifacts。

```powershell
.venv-research\Scripts\python.exe scripts\recurrent_state_experiment.py --cache artifacts\research\A48_D7_I210_W_autumn_cache --baseline-file artifacts\research\A30_State_budget_directions_b\D7_I210_W\full.pkl --incumbent-file artifacts\research\A49_adaptive_attention_D7_I210_W\State_selected.pkl --output artifacts\research\A72b_recurrent_I210_W_autumn_full_day
.venv-research\Scripts\python.exe scripts\recurrent_state_experiment.py --cache artifacts\research\A02_D12_I5_N_winter_seed20261013 --baseline-file artifacts\research\A16_winter_full_label_LGB_control\lightgbm_large_budget.pkl --incumbent-file artifacts\research\A34_adaptive_attention_strong_winter\State_selected.pkl --output artifacts\research\A73_recurrent_I5_N_winter
.venv-research\Scripts\python.exe scripts\verify_recurrent_state.py --study reports\research\A72b_recurrent_I210_W_autumn_full_day.json --output reports\research\A74b_recurrent_I210_W_independent_replay.json
.venv-research\Scripts\python.exe scripts\verify_recurrent_state.py --study reports\research\A73_recurrent_I5_N_winter.json --output reports\research\A75_recurrent_I5_N_independent_replay.json
.venv-research\Scripts\python.exe scripts\recurrent_forward_extension.py --output reports\research\A77c_recurrent_forward_calendar_v3.json
```

独立导出核验重建所有cal/assessment的masked Parquet输入及公开图数组，重跑每seed、seed均值、reference和primary等10个role。总flow与lane-flow单位中求均值可能产生舍入差，预测绝对容差1e−8 lane units；报告实际最大误差，不冒充位级相同。14日/28日moving-block区间仅是day-mean指标的条件诊断，不是非线性pooled State或Kaggle置信区间。

## 两方向实测与固定策略复核

I210_W的full-LGB/原incumbent/新TCN均值/新GRU均值后留State分别 **.927102/.933577/.933279/.935491**。Cal选中的primary为GRU均值，原incumbent至primary的pooled gain **+.001914**；14日day-mean gain+.002111、CI[.000849,.003374]。每个seed的GRU相对匹配TCN为 **+.001321/+.002452**，日均区间均正。两个seed在TCN/GRU中分别都选epoch20/24，alpha均.5，checkpoint/混合政策在本方向恰好一致。

原incumbent改用有限窗口本身只有assessment State−.000015。两个GRU的R3 gain相对匹配TCN为+.002491/+.006832；drop20共54目标、recover20共62、missing60min共758，missing180min为0。急降速速度RMSE分别改善2.17/4.00kmh，但恢复及flow收益不一致，不能当独立事件样本或全面物理改善。旧增量代理L1为incumbent .716127、TCN均值 .705201、GRU均值 .710947，正式Physics未知。

四个模型各监督全部517171目标格点，同seed曝光数组exact。训练完整wall time755.46秒GPU，allocated峰值725.33MiB；每模型TCN约171.9秒，GRU约193.0秒，包含cal检查，不是隔离吞吐。GRU是额外容量，收益不完全证明其物理/递归机制。

A74首次导出因JSON str未转换Path退出，预测尚未开始，旧源快照/注册/失败记录保留。A74b修复后28日×10roles共280次独立预测，最大差2.2737e−13 lane units、输入和图数组exact；耗时74.78秒。A77c v3仅修订已批准本地评估的export路径，不改日期或模型。

I5_N冬季A73四模型均完成，每个实际监督全部3,042,465格点，同seed曝光exact。原incumbent/TCN均值/GRU均值后留State为 **.954329/.955886/.956818**；cal仍选GRU均值，pooled gain **+.002488**，day-mean gain+.002399、CI[.002050,.002749]。两个seed的GRU−TCN为 **+.001062/+.000810**，日均区间均正。R1/R2/R3为.961192/.956763/.952499，原incumbent为.959366/.954079/.949542。drop20/recover20/missing60min目标220/272/5133，missing180min仍0。

冬季总wall time1970.18秒、进程allocated峰值2100.55MiB，期间与独立失效研究部分并行，不是隔离测速。旧target-pair增量代理L1 incumbent/TCN均值/GRU均值为.682571/.661477/.641245，不能代替正式Physics。A75独立28日×10role重放最大差4.5475e−13 lane units，input/graph exact、219.53秒。两原fold合计560次role-day预测复核。

A76验证全部四对初始权重、采样SHA、曝光数组匹配；原两fold primary平均pooled State增益 **+.002201**，满足原v2本地State增益门槛.002、正day-mean区间及无超限panel损失。GRU参数更多，只有两方向/已查看公共日期；这是原本地gate的判断，不是正式模型晋级或全网收益。

## 二月前向日历

A77c在两组cal/assessment封存且导出核验后执行，固定2031-02-01…02-28全部28日、原checkpoint/alpha/primary。二月日期与每组原profile/fit/cal/assessment完全隔离；没有重新训练或按二月反选。

| 二月公开train评估 | 原incumbent | TCN均值 | GRU均值 | GRU−incumbent |
|---|---:|---:|---:|---:|
| I210_W | .934295 | .934220 | .935606 | +.001311 |
| I5_N | .954551 | .956094 | .956542 | +.001992 |

对应day-mean gain为+.001541/+.001956，名义CI为[.001024,.002103]/[.001318,.002830]。两方向pooled gain平均 **+.001651**，比原两fold小，不能把原gate或相关公开日期上的正区间解释为private稳健性。二月两方向missing180min目标仍0。A77c耗时300.61秒、GPU allocated峰值594.06MiB。

A84从公开raw train重建56份teacher及目标索引，独立复算560份保存role/day指标，最大误差1.11e−16；另外在每方向首末日期对incumbent/primary进行8次直接推断，预测最大差0。耗时21.26秒。**560是保存指标复算数，不是560次新预测**；这也不是独立重新训练。

```powershell
.venv-repro\Scripts\python.exe scripts\compare_recurrent_state.py --study reports\research\A72b_recurrent_I210_W_autumn_full_day.json --study reports\research\A73_recurrent_I5_N_winter.json --output reports\research\A76_recurrent_twofold_comparison.json
.venv-research\Scripts\python.exe scripts\verify_recurrent_forward.py --study reports\research\A77c_recurrent_forward_calendar_v3.json --output reports\research\A84_recurrent_forward_independent_verification.json
```

## 长失效反例与下一步

原后留及二月没有3h连续失效目标，因此另行生成公开数据3/6h压力测试，详见 [失效研究](PUBLIC_OUTAGE_STRESS_RESEARCH.md)。它重新构造全部被遮盖的测量和派生特征，不让原插值base泄漏移除真值。3h/30%有小收益、3h/70%区间跨零，6h/30%与6h/70%的GRU增益为−.002945/−.002284，名义区间均负。不能将普通模板收益表述为所有R3或长缺失已解决。

同一I210_W冻结策略在新增14日未额外失效control仍优于incumbent，合并原14日的28日次要pooled gain+.002033；control和合成失效目标人群不同，不用两者绝对分数算相对跌幅。独立验证168mask/teacher、672分数，并作16次stress直接推断与全部control重放。首次flag dtype错误保留；修复后当前101 research/85 CPU tests通过、5可选Torch模块跳过。

下一步在独立development日期/新mask seed研究无泄露失效增强或仅由输入决定的置信度门控，与相同预算/容量对照；本批已查看case只作回归。当前没有全273日生产拟合、没有新完整submission，正式最佳.82940保持冻结，0.95000+仍未达到。
