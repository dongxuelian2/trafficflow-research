# 概率校准与期望 IoU 集合 C74–C77

正式最佳 **0.82940**，目标 **0.95000+** 未达到。以下是已公开、部分已查看 TRAIN 日期上的观测标签代理，以及自主五节点已知分布诊断；正式 Queue/Physics/OD/Total 都没有新测量。没有新完整 submission、上传或平台发布。

## 数学模型与自主实现

算法思想参考 Dai、Li 的 [RankSEG 原始论文，JMLR 2023，Lemma 13](https://www.jmlr.org/papers/volume24/22-0712/22-0712.pdf)。自主编写 suffix Bernoulli 卷积，没有使用该论文的代码或第三方参赛实现。条件独立只是一项建模假设，不是已验证的交通标签性质。

对完整域概率按降序排列，给定 top-k 集合，k>0 时计算

`E[IoU] = sum(p_inside) × E[1 / (k + number_of_positives_outside)]`。

外部正例数的分布用精确卷积逐步构造，对所有 k 计算期望；k=0 单独取全部标签为零的概率，匹配 empty/empty IoU=1。时间 O(N²)、内存 O(N)，相同分数按 link ID 排序，期望值相差≤1e−12时选较小集合。只改变 onset T+30，其他 onset 时刻为0，ongoing原样沿用强参照。

这里优化的是完整域、条件独立标签模型的期望值，**没有未来 score mask 输入**。真实 evaluator 会过滤未来 eligibility，真实交通标签还可能强相关；因此数学最优不等于本地代理最优，更不等于正式最优。非空 sensitivity arm 只在优化时禁止 k=0，不使用未来 mask，也不能保证评分过滤后有正例；没有将它输出为已校准的条件非空后验。

## 独立的概率校准期

冻结 C71 四个 case 的全部分类器、strong incumbent、feature arm 和原 raw threshold，不再拟合任何树/神经模型。每 case 此前未使用的15日 checkpoint buffer 改作概率校准：September case 为9/16–9/30，October case 为11/1–11/15，均早于其 decision。两个 feature arm 都先校准、封存，之后才进入新的 decision。

校准是 `sigmoid(a × logit(p) + b)`，仅使用该期 eligible onset T+30 标签。每个窗口等权，窗口内部按 eligible 行均分权重；平均 logloss 加 `.01/2 × ((a−1)²+b²)`，概率 clip1e−6，a∈[.05,5]、b∈[−8,8]。配置固定，没有按后段调正则或边界。少于3个窗口或缺类别时预登记回退identity，本次8个状态全部达到门槛并拟合。

| case | 新校准窗口总数中的onset | 选定arm校准行 / 正例 | 选定arm a / b |
|---|---:|---:|---|
| I5_S September | 10 | 763 / 28 | .680910 / .057239 |
| I5_S October | 9 | 676 / 24 | 1.373318 / .009090 |
| I10_W September | 5 | 405 / 8 | .884460 / .247628 |
| I10_W October | 3 | 249 / 5 | 1.348438 / .140290 |

新采样总62窗口，其中27个onset、35个ongoing；校准只监督上述27个onset。两个arm使用同一窗口，8个拟合不是8份独立数据。表内loss改善仅是拟合目标下降，不能当后段校准有效性证据。公开代理 sampler 仍筛未来观测正例，未知官方 latent selector 的偏差没有解决。

## 对照与时间选择

候选在 C74 运行前登记：incumbent、冻结 threshold_raw、重新在 decision 选阈值的 threshold_cal、iou_raw、iou_cal、iou_cal_nonempty。cal threshold使用31值 .05… .8 × force False/True，先不force再近.5；其余IoU臂没有决策阈值网格。所有角色ongoing相同，只有最后一步onset改变。

calibration seal→decision→decision seal→15日selection→selection seal→30日assessment。后两期原event seeds2909/3253/4001复用，物理origin去重；decision4133。新的算法/参数不会因source JSON已有历史assessment摘要而改变，但这些日期不能称盲测或独立最终测试。

decision胜者依次为 incumbent/incumbent/threshold_raw/threshold_raw；两个I10_W的selection gain为0，均退回incumbent。最后四组都保留原策略，primary gain为0。后段不能反过来选择表现较好的非空臂。

## 后段公开观测代理

以下为origin去重、onset/ongoing等权宏IoU，非正式Queue。

| case | incumbent / primary | threshold_raw | threshold_cal | iou_raw | iou_cal | iou_cal_nonempty |
|---|---:|---:|---:|---:|---:|---:|
| I5_S September | .785466 | .795681 | .800251 | .799713 | .810466 | .810466 |
| I5_S October | .779788 | .801545 | .801545 | .782374 | .784098 | .792719 |
| I10_W September | .789289 | .814289 | .789289 | .789289 | .789289 | .789289 |
| I10_W October | .672051 | .678461 | .697692 | .697692 | .691281 | .739358 |

I5_S September 的 iou_cal/nonempty 相对incumbent gain+.025000、95% CI [.001262,.055749]，但未被decision选中。其余单case区间跨零。原“至少两case gain≥.01且CI下界正，各case损失≤.01”的描述性gate，所有新策略均未通过，不晋级生产。

C77 另作11个事后配对比较：同日跨走廊同步抽样，两个不重叠assessment日历分别抽样，零事件日保留；每项4000次全部有效。以下摘录，全部未作多重比较校正，不参与选模型。

| 同步日历事后比较 | 等case/condition宏gain | 95% CI |
|---|---:|---|
| raw threshold→raw IoU | −.005227 | [−.033232,.022065] |
| raw IoU→calibrated IoU | +.001517 | [−.003041,.007199] |
| calibrated threshold→calibrated IoU | −.003410 | [−.026781,.016810] |
| calibrated IoU→calibrated非空IoU | +.014174 | [.002083,.031656] |
| incumbent→calibrated非空IoU | +.026310 | [.003884,.053022] |
| raw threshold→calibrated非空IoU | +.010464 | [−.010331,.029710] |

证据集中在代理中的非空约束，校准本身、精确期望IoU相对已调阈值都未显示稳定的独立增益。非空策略相对强参照的联合正区间是值得验证的信号，但仍未超过已有raw threshold的确定性证据，也未被primary选择。

尤其不能忽略：本 sampler **保证未来有可观测正例**；官方公布的潜在拥堵非空条件不等于future eligibility过滤后仍有可评分正例。这会放大非空策略的代理优势。不能把+.026310乘正式Queue权重来宣称Total增益。

## 已知分布假设压力 C76

自主定义7个五节点联合分布，精确枚举32个预测集合及所有真值/可评分mask状态。IID解码器获得正确的真边际概率；已知联合分布最优值只作期望风险诊断，`deployment_allowed=false`，不用某个已实现的未来标签或mask做决策。它不是交通模拟器或参赛label-oracle。

| 自主已知分布 | IID决策相对联合分布最优的期望IoU regret |
|---|---:|
| 三种独立、完全可评分场景 | 0 / 0 / 0 |
| 全空/全拥堵共同冲击 | .400000 |
| 始终非空、两个互斥拥堵簇 | .120000 |
| 独立标签、节点间覆盖率不同 | .029272 |
| 独立标签、标签相关的失效率 | .075318 |

这些反例说明即使边际概率正确，相关性/未知可评分机制也可能改变最优集合；不能据此断言真实比赛损失就是这些数。C77使用另一套逐truth/mask/预测集合的标量集合运算重算全部7场景，最大分数差6.66e−16。

## 独立核验、资源与复现

C75 显式复用 C72b 的旧 **2196次**raw输入审核，重新hash其源数据和未变构造源码，本次未重建全部旧窗口。新62个校准窗口的原始历史、公开masked T、两套特征和future观测teacher重新构建，共124特征臂。全部8个校准器用另一优化器SLSQP与KKT检查，参数最大差2.68e−7、投影梯度最大7.38e−10。

对582条新raw/calibrated IoU分数曲线（含重复抽样origin），独立计算积分 `∫[0,1] x^(k−1) × product(1−p_outside+p_outside×x) dx`；Gauss–Legendre阶数覆盖多项式次数，最大差 **1.90e−14**。它不复用suffix卷积。新threshold网格、全部决策、评分、门控与去重区间一致。

实际100次冻结Booster矩阵调用，872次便携角色调用（含策略别名）、448次额外乱序，32次未来历史拒绝通过；不是872个独立事件，也不是对全部窗口逐个运行所有便携角色。C77的11个联合区间另用逐次求和复核，最大差6.59e−17。

新增4项单测1.71秒通过；当前全套 **141research / 12.64秒**，纯CPU **120passed / 7.88秒、7可选Torch模块skip**。覆盖小分布穷举真值/预测全组合、空集合/边界/置换/完整域、校准窗口权重和梯度。compileall通过。C74 CPU wall **33.99秒**、进程peak **438.89MiB**；C75 **109.85秒**、C77 **5.28秒**。含并行测试，不作隔离吞吐基准。

后继manifest `model_artifacts_checkpoint_probabilistic_onset_v1.json` 重校验6649个旧artifact不变，新增96个、累计6745个；5份完整CSV和依赖锁不变。source zip、8校准状态、原始新校准缓存、全部便携策略、预测与注册报告保留；Git字节证明为`probabilistic_onset_report_bytes_v1.json`。最终模拟seeds7907/8017/9109继续未生成。

```powershell
.venv-repro\Scripts\python.exe scripts/probabilistic_onset_experiment.py --output artifacts/research/C74_probabilistic_onset_sets
.venv-repro\Scripts\python.exe scripts/verify_probabilistic_onset.py --study reports/research/C74_probabilistic_onset_sets.json --output reports/research/C75_probabilistic_onset_verification.json
.venv-repro\Scripts\python.exe scripts/expected_iou_assumption_stress.py --output reports/research/C76_expected_IoU_known_joint_stress.json
.venv-repro\Scripts\python.exe scripts/summarize_probabilistic_onset.py --output reports/research/C77_probabilistic_onset_matched_summary.json
.venv-repro\Scripts\python.exe scripts/checkpoint_probabilistic_onset.py
```

依赖C71冻结树与C72b审核；所有输出写入一次，已有目录不能覆盖。下一步先独立检验非空收益是否源于代理采样条件，再研究能表达相关拥堵与可评分不确定性的联合模型。继续保留原正式最佳，不按已查看后段反选校准器或强制非空规则。
