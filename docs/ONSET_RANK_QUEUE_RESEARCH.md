# Queue onset 事件内排序 C65–C67

正式最佳仍 **0.82940**，0.95000+未达到。以下为公开TRAIN的观测标签代理；真实latent Queue、正式Queue/Total均未知。16个新模型未晋级，没有新完整submission或平台操作。

## 假设与对照

在C55–C58 balanced-v2的I5_S/I10_W两季事件上，预测onset最后T+30的相对位置，ongoing逐格沿用冻结incumbent。既有特征已有位置、TOD、交通状态；本轮不增加特征或标签。

每case两个seed20261013/20261029，各训练一个onset binary classifier及一个[LightGBM LambdaRank](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html)。每对使用相同eligible onset T+30行、unit weights、同plain/rich schema；query为完整窗口内已知监督行，只在训练时用future eligibility筛标签。每模型max600树/63叶、lr.035/min_child60/lambda8/feature_fraction.9/CPU4线程。

ranker固定truncation20、normtrue、binary label_gain0/1；该设置关注query内靠前结果，[参数说明](https://lightgbm.readthedocs.io/en/latest/Parameters.html#lambdarank_truncation_level)中的目标与IoU不同。ranker输出只是相对score，不是校准概率。相同数据/预算不代表内部梯度、query归一化或特征采样路径相同。

两seed固定均值；decision阶段各自从k1/2/4/8/16选一个，完整六步/全link预测域先排序、只在T+30取top-k；相同score按link ID打破平分，目标行乱序不改变集合。然后scorer才使用future mask。不能先删掉未知future cells再排序。classifier也用相同top-k，避免把decoder变化全算成ranking收益。

onset query/监督行数分别为I5_S秋107/8231、冬154/11830；I10_W秋52/4272、冬70/5766。feature维度56/46/56/56随冻结强参照一致。16模型全部实际600树。

k≥1是建模选择，不能保证future eligibility过滤后仍有预测正例，也不保证观测标签非空。沿用的proxy仍筛未来观测正例，与正式latent非空条件不同；此前公开TRAIN零eligible正例窗口的反例仍保留。

## 时间机制

沿用原fit/decision/selection/assessment日期；旧checkpoint期作未用buffer，固定树数不另选权重checkpoint。模型seal后读decision，decision seal后读selection，再整体seal后读assessment。候选需后15日origin去重宏gain≥.01且7日block CI下界正才能替换incumbent。原source JSON含此前结果，这些日期都是已查看开发资料，不称新盲测。

decision胜者四组均为incumbent，故后续gate比较为0；不能据此声称检验了非零challenger的gate效果。后留不反选赢家。

## 公开观测代理结果

| case | incumbent/最终保留 | classifier top-k | LambdaRank top-k | 两方法选k |
|---|---:|---:|---:|---:|
| I5_S秋 | .781151 | .770702 | .756328 | 4 |
| I5_S冬 | .797255 | .754866 | .753378 | 4 |
| I10_W秋 | .557659 | .622474 | .622474 | 1 |
| I10_W冬 | .814942 | .714942 | .731609 | 2 |

ranker相对incumbent gain−.024824/−.043878/+.064815/−.083333，day-block区间[−.091498,.047005]/[−.075894,−.013388]/[−.100000,.277778]/[−.171905,.020833]。I10_W秋次要正增益区间很宽且未在decision选中，不作晋级。

同top-k控制下ranker也没有一致优于classifier。与原参照的差异同时包含onset专用训练、feature subsampling、目标与decoder；相对classifier才是更匹配的目标对照。四fold共享fit日期/事件；两seed共享监督集，不是独立样本。

## 集合大小诊断：禁止作为预测器

C67只用公开assessment标签及eligibility，对已定义五个top-k集合逐窗挑最高IoU。没有新模型、二值预测导出、提交或任何选择变化；`deployment_allowed=false`。这是一组固定排序下的乐观评分诊断，不是合法未来decoder或可达成绩。

已见eligible fit正例数的median为3/3/1/1，min均1，max7/7/4/4。逐窗label-oracle下classifier宏代理为.805159/.817855/.779881/.881609，ranker为.789463/.817557/.761363/.881609。它提示集合大小值得单独检验，却不能证明因果cardinality模型能达到该值；尤其future质量与latent真值都未知。若继续，只能在fit中学习、在独立决策期选择，不能采用这个oracle。

## 核验与资源

C66重新读取公开raw历史及已发布masked T，重建全部32份缓存中的2345个窗口occurrences的特征、未来teacher/eligibility、query行/分组及unit weights；特征与缓存精确相同。独立top-k、全部decision网格、ongoing复制、窗口评分、origin去重及day-block gate一致。

128次Booster矩阵调用含16个fit调用；其余112次用于evaluation raw score/网格/存档重放。便携接口在每个evaluation缓存的每condition首末测试，共560角色窗口调用；另140乱序target调用、8未来history poison拒绝。不是所有2345窗口都独立调用了每个便携角色，也不是560独立事件或20个新模型。16新learner头部objective/参数另外核验。

C65 32.33秒CPU、进程peak726.04MiB；C66 302.65秒。3新单测通过，全套research131 passed/12.68秒，CPU110 passed/7.34秒、7可选Torch模块skip。部分测试/验证并行，不作隔离测速。

```powershell
.venv-repro\Scripts\python.exe scripts/onset_rank_queue_experiment.py --output artifacts/research/C65_onset_query_rank_public
.venv-repro\Scripts\python.exe scripts/verify_onset_rank_queue.py --study reports/research/C65_onset_query_rank_public.json --output reports/research/C66_onset_query_rank_public_verification.json
.venv-repro\Scripts\python.exe scripts/summarize_onset_rank_queue.py --output reports/research/C67_onset_query_rank_cardinality_diagnostic.json
.venv-repro\Scripts\python.exe scripts/checkpoint_onset_rank_queue.py
```

已有目录禁止覆盖；复现需改输出名及后继源码绑定。当前拒绝固定k排序方案，不扩大已看k/truncation网格。后续先用fit-only的集合大小/校准对照分离decoder与ranking信息，再作新rolling验证；原完整最佳、最终模拟seeds与平台暂停状态保留。
