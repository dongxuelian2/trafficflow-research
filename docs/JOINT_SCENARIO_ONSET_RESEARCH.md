# 联合标签与可评分场景 C78–C81

正式最佳 **0.82940**，目标 **0.95000+** 未达到。当前经验联合场景模型没有显示新增稳定收益，四组primary继续保留incumbent。以下公开观测代理、已知小分布期望值与数学反例，均不是正式Queue/Total或真实latent交通成绩。没有新完整submission、上传或平台发布。

## 研究问题

C74–C77的calibrated非空IoU出现事后代理正区间，但未超过已有阈值的确定性证据。早期C40已用Monte Carlo展示过潜在正例总体与scored-positive子集的差异；本轮扩展到精确联合期望、六步evidence及可部署的fit-only场景决策。

重新核对本地冻结官方`score_task2.py`：先用`is_score_eligible`过滤，整个窗口无可评分单元为0，有证据而union为空为1。因此onset前五步标签已知为0时，其可评分单元也会影响“最后一步正例全部漏观测”的空预测得分。未来mask不可作为推断输入。

## C78：latent始终非空的精确分布

自主定义9个五节点分布，真值为均匀单节点拥堵或两个互斥拥堵簇，latent始终非空；枚举全部真值、最后一步mask和32个预测集合。额外指定前五步是否有可评分零标签单元，区分evidence与最后一步覆盖。预测依赖已知生成分布，不看某次已实现的未来标签/mask。

同时用完全相同总体计算：latent-positive人口期望，以及过滤到“至少一个可评分正例”的条件子集。先在子集选一个固定集合，再在原总体评价，不是按实际未来mask逐窗选点。

| 自主场景 | 未来有可评分正例概率 | 空预测在原总体的期望IoU | scored-positive子集选策略→原总体regret |
|---|---:|---:|---:|
| 单节点、覆盖率.95 | .950000 | .050000 | .002000 |
| 单节点、覆盖率.75 | .750000 | .250000 | .049219 |
| 单节点、覆盖率.25 | .250000 | .750000 | .360156 |
| 互斥簇、覆盖率1 | 1 | 0 | 0 |
| 互斥簇、覆盖率.75 | .956250 | .043750 | 0 |
| 互斥簇、覆盖率.25 | .493750 | .506250 | .091523 |
| 互斥簇、label-dependent覆盖 | .320850 | .679150 | .078536 |
| 同上、evidence只来自末步 | .320850 | .678482 | .078536 |
| 互斥簇、node-dependent覆盖 | .514000 | .486000 | .190200 |

regret相对该自主已知联合分布的全局期望最优诊断；它禁止部署，不是比赛隐藏答案。九种分布中的前缀搜索恰好能达到全局最优，不能据此推出任意联合分布都如此。C81另构造非空五节点反例，概率排序前缀最优.4360，全局集合最优.4425，gap **.0065**；仅提示搜索限制，不改变C79算法或选择。

这些例子证明代理选择条件可以改变策略排名，不能量化真实比赛里的偏差大小，也不能据此认定原非空策略正式会退步。C81用另一套逐truth/mask、标量集合运算重算9个分布与全部候选，最大差9.99e−16。

## C79：公开fit-only经验场景模型

联合建模思想参考 [Schefzik、Thorarinsdottir、Gneiting的ensemble copula coupling原始研究](https://arxiv.org/abs/1302.7149)。自主实现二值观测与缺失的随机区间构造，没有复制论文代码；不声称与标准ECC重排相同，或已校准真实交通联合分布。

冻结C71全部树、C74全部概率校准与三个参照策略：incumbent、raw threshold、calibrated非空IoU。仅使用各自C71 **fit** 的onset窗口标签及六步eligibility拟合新的经验场景库，不拟合树、校准器或分类器拟合内残差。

对每节点，按fit中eligible观测计算正例率`p_fit`；未知标签不当负例，不进分母。随机抽一个fit窗口：正例的潜在U放在`[0,p_fit]`，负例放在`[p_fit,1]`，未知位置用独立U。对当前合法校准概率p，以`U<p`生成假设标签。该构造在经验人口law、节点可评分条件下具有均匀U边际，有限抽样、时变关系与观测/latent偏差仍是限制。

每库固定两个MC seed20261013/20261029，各2048行，共4096行。独立与经验关联使用同一base jitter、同一训练窗口索引，分别保留关联或独立性；future可评分mask与六步evidence取同一fit模板。场景库是拟合期学得的假设分布，**不是当前预测窗口的实际future mask**。本版mask law不按当前历史质量条件化。

| case | fit onset窗口 | 完整域节点 | fit率介于0/1的节点 | bank末步eligible比例 |
|---|---:|---:|---:|---:|
| I5_S September | 72 | 100 | 17 | .767864 |
| I5_S October | 122 | 100 | 24 | .767439 |
| I10_W September | 37 | 107 | 13 | .764532 |
| I10_W October | 58 | 107 | 16 | .767859 |

四库合计289次fit onset窗口引用，部分fit日期/窗口重叠；4096行不是4096个独立训练事件。关联变换改变约9.43%…18.29%的U元素；未改变的节点多为fit中从未出现正例。两个MC成员只度量模拟噪声，不是独立重训。

四种同预算对照：independent/full、independent/empirical-mask、empirical-dependence/full、empirical-dependence/empirical-mask。每个假设场景计算完整域六步IoU：只有末步可能预测queue，前五步零标签的可评分性进入evidence。按校准p排序，检查k=0…N的所有前缀，优化4096场景均值。时间O(SN)，**一般联合分布下不保证全局最优集合**。

推断入口只收≤T历史、完整公开target、network、origin和condition；不收实际未来mask或label。ongoing逐格复制原强参照。stage顺序为bank seal→decision→decision seal→selection→selection seal→assessment。所有选择和门槛沿用先前原规则；MC单成员只作诊断，不能选更好seed。

## 公开观测代理结果

以下为已查看development日期上origin去重、onset/ongoing等权宏IoU。

| case | incumbent / primary | independent/full | independent/mask | dependence/full | dependence/mask |
|---|---:|---:|---:|---:|---:|
| I5_S September | .785466 | .810466 | .801057 | .810466 | .810466 |
| I5_S October | .779788 | .784098 | .784098 | .784098 | .784098 |
| I10_W September | .789289 | .789289 | .789289 | .789289 | .789289 |
| I10_W October | .672051 | .691281 | .691281 | .691281 | .691281 |

decision胜者仍为incumbent/incumbent/raw threshold/raw threshold；selection后均保留incumbent。新联合场景臂没有通过跨case晋级gate，不更新生产模型。

C81的13项事后同步日历比较未多重校正，也不参与选择：

| 比较 | 宏gain | 同步95% CI |
|---|---:|---|
| matched independent/full→dependence/full | 0 | [0,0] |
| independent/mask→dependence/mask | +.002352 | [0,.006250] |
| independent/full→independent/mask | −.002352 | [−.006250,0] |
| dependence/full→dependence/mask | 0 | [0,0] |
| raw threshold→dependence/mask | −.003710 | [−.031693,.023439] |
| calibrated非空IoU→dependence/mask | −.014174 | [−.031656,−.002083] |

四种经验场景方法相对强incumbent的区间均未形成跨case晋级证据；full独立MC与原精确IID解码的IoU也完全相同。不能将copula/mask联合作为已确认改进。

四case内去重后共83个onset origin，同日期的走廊不视作独立重复。full-law关联模型改动4个单元，**全部位于本地不可评分位置**。mask-law关联对照改动7个单元，其中3个可评分、4个不可评分；仅I5_S第一组有小量净收益。这个evaluator mask只用于事后误差定位，未用于预测或挑模型，诊断存于`joint_scenario_verification_record_v1.json`。

MC成员预测在多数case相同；最大差异为I5_S September independent/mask中3个预测单元的集合大小差。独立/full的4096场景曲线相对同概率模型精确IID期望，逐曲线最大误差均值.003353、最大.012884；这是条件模拟误差，非真实概率误差或Kaggle区间。增加抽样不会自动增加交通信息。

## 核验、失败记录及资源

C80首轮失败：核验器对float32标签直接求mean，生产模型按整数positive/eligible比率以float64计算，最大rate差2.62e−8。失败代码zip/注册/错误保留；C80b恢复独立整数比率，并修复核验器局部变量名称覆盖。逐位相等要求及模型/预测/选择/数值容差完全不变。

C80b四库逐位重建一致，包括全部289个fit窗口引用、两个seed的4096模板索引/jitter、eligible分母、unknown-label处理和六步evidence。3076条ensemble/member分数曲线用独立的“所有前缀指示矩阵×假设标签/eligible”计算交并集，**与cumsum实现最大差0**。决策、评分、封存、门控及所有成员probe一致。

实际8次decision onset Booster矩阵调用，另4次参照probability调用；24份evaluation概率记录沿用C75核验并重hash。新鲜109个便携历史窗口支持981次角色调用（含别名）、504次额外target乱序、36次未来历史拒绝，全部通过。明确复用C72b的2196次完整raw审核、C75的独立概率/校准证明，本轮没有新全量raw重建。

新增3单测1.61秒通过；当前全 **144research / 12.53秒**，CPU **123passed / 7.94秒、7可选Torch模块skip**。compileall通过。C79 CPU wall **103.51秒**、进程peak **462.54MiB**；C80b **207.44秒**、C81 **6.55秒**。时间含并行活动，不作隔离吞吐基准。

后继manifest `model_artifacts_checkpoint_joint_scenario_v1.json` 重校验全部6745个旧artifact不变，新增90、累计6835；5份完整CSV和依赖锁不变。报告Git字节证明为`joint_scenario_report_bytes_v1.json`。最终模拟seeds7907/8017/9109仍未生成。

```powershell
.venv-repro\Scripts\python.exe scripts/nonempty_mask_joint_stress.py --output reports/research/C78_nonempty_joint_mask_stress.json
.venv-repro\Scripts\python.exe scripts/joint_scenario_onset_experiment.py --output artifacts/research/C79_joint_scenario_onset
.venv-repro\Scripts\python.exe scripts/verify_joint_scenario_onset.py --study reports/research/C79_joint_scenario_onset.json --output reports/research/C80b_joint_scenario_verification.json
.venv-repro\Scripts\python.exe scripts/summarize_joint_scenario_onset.py --output reports/research/C81_joint_scenario_matched_summary.json
.venv-repro\Scripts\python.exe scripts/checkpoint_joint_scenarios.py
```

依赖先前冻结C71/C74模型、缓存及C72b/C75证明；已有输出写入一次，不能覆盖。下一步不扩大相同MC规模/经验关联超参数；可先用已知联合分布检验受控非前缀搜索是否值得预算，再以同一强参照做预登记公开验证。若不解决新增有效信息与观测/latent/eligibility契约，单纯增加集合解码复杂度不支持0.950+目标。
