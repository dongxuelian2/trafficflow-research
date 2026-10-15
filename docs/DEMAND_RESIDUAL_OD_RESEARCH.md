# 共享需求中心与路径残差 OD 对照

正式最佳仍为 **0.82940**，目标 **0.95000+** 未达到。本轮只做本地模型与合法验证。原 Ridge、K05 entropy、K07 log-demand 与全部已验证完整文件不覆盖。

## 独立实现

`src/trafficflow/demand_residual_od.py` 使用两个阶段：先按 D17 已封存参数估计正值共享 origin/destination 需求 `g`，再构造几何中心

`r = max(max(b,20)^(1-alpha) * g^alpha, 2)`，

最小化 `0.5 * sum((Af-c)^2 / (tau*max(c,10))) + KL(f || r)`。后一步使用已验证的 measurement-space entropy solver，允许路径级偏离共享中心。alpha=0、tau=.05精确恢复固定强entropy，而且完全跳过需求拟合。测试用独立 primal finite-difference optimizer核验路径修正。

两个阶段重复使用同一计数，因此这是顺序正则化点估计，不能解释为独立观测的先验或完整Bayesian posterior。路径先验仍为正值，不能恢复精确真零support；新增路径自由度也没有增加观测rank。

仅在development比较四个 `(alpha,tau)`：`(.25,.05)`、`(.5,.05)`、`(1,.05)`、`(.5,.5)`。共享需求参数固定 `sd=.2/noise=.02/floor2`，不按本轮结果更换。选择依据为相对**固定强entropy**的九条件最差macro gain，再比较平均gain；要求全部case成功、均值≥.005、最差≥−.002，否则保留entropy。

## 数据与评估边界

开发使用九种既有自主需求/先验法律、十公开train网络、411/829/1201三个seed，共270个问题、1890条方法记录。架构受到既有研究结果启发，这些不是未见数据族；所有新参数选择只在development完成并封存。

封存后才生成本轮首用的研究seed **932003/932011/932023**。一组为九个同类法律的270问题；另一组为六个新登记压力法律的180问题，分别报告而不混成正式成绩：

| 新法律 | 自主生成的变化 |
|---|---|
| distance_gravity | origin/destination质量与路径link数量的摩擦项；link数只是模拟距离代理 |
| gamma_poisson_counts | Gamma路径需求、相对噪声prior、Poisson检测器计数 |
| correlated_count_bias | 四个按计数行取模的模拟传感器组，各有共同偏差；不冒充真实设备分组 |
| sparse_zone_shutdown | 关闭约四分之一origin的需求，weak prior仍给出正值 |
| heavy_tail_combined | 重尾prior、20%缺失prior路径与Student-t计数噪声共同出现 |
| dense_weak_prior_independent_truth | prior与truth只共享总尺度，路径级信号独立 |

全部truth来自独立代码生成，只利用公开train网络、元数据及weak-prior尺度；没有读取reference或隐藏答案。这些法律不是组织方生成器。最终seed **7907/8017/9109未生成**。共720个问题的truth/counts/prior和成功预测均保存NPZ及SHA，失败记录保留。deployment alias重复既有方法，不增加样本数。

## 复现与独立核验

```powershell
.venv-repro\Scripts\python.exe scripts\demand_residual_od_experiment.py --output reports\research\D21_demand_entropy_path_residual.json
.venv-repro\Scripts\python.exe scripts\verify_demand_residual_study.py --study reports\research\D21_demand_entropy_path_residual.json --output reports\research\D22_demand_residual_independent_replay.json
.venv-repro\Scripts\python.exe scripts\od_error_subspaces.py --study reports\research\D21_demand_entropy_path_residual.json --output reports\research\D23_demand_residual_error_subspaces.json
```

输出必须使用新目录，不能覆盖已有记录。D22重新生成全部case的输入并核对NPZ字节/hash，用独立公式复算所有成功分数；另外在两个方向、12个问题上绕过缓存，直接重跑完整model API。方法重复不作为独立性能样本。development leave-one-family-out只诊断残差超参数选择稳定性；上游共享模型参数已经看过全部族，不能称完整未见走廊验证。

D23对每个公开计数矩阵做SVD，使用正交rowspace basis，把相对平方L2路径误差拆成可观测rowspace与nullspace；逐case验证Pythagorean关系及 `A * null_error ≈ 0`。这是自主真值的几何诊断，**不是官方L1 S_od分解**，也不能按rank宣称某些具体路径可识别。原实验失败没有预测时不进入这个诊断，不能用成功子集包装整体模型成绩。

本轮新增四项kernel/generator测试；研究环境 **92 passed**，纯CPU **80 passed/4 optional-Torch skips**。所有模型均为局部CPU求解，无GPU与付费云资源。模型结果、资源和独立核验数字见本页后续实测结果及D21–D23报告。

## Development驱动的下限后续实验

D21已封存开发结果：alpha=.25/tau=.05平均gain **+.018222**，但missing-prior最差 **−.005876**，未过−.002门槛，primary继续固定entropy。alpha=.5/1及较弱计数权重也未通过；没有按assessment反选。

在D21 assessment结果尚未查看时，D24单独预注册“保留强entropy的20中心下限”：比较alpha=.25/.5与floor2/20，tau固定.05。选择继续只读取D21 development的输入/预测；原floor2的两模型和三个参考各270条共1350预测明确复用，不当作新求解器证据。只有新floor20两配置重新求解。D24使用另一组新研究seed **933003/933011/933023**，选定后才生成九同类/六新法律的评估；D21 assessment不参与其参数选择。

```powershell
.venv-repro\Scripts\python.exe scripts\demand_residual_floor_followup.py --output reports\research\D24_demand_residual_floor_followup.json
```

D24依赖先保存的 `D24_demand_residual_floor_followup.registered.json`，其中固定了完整development文件与选择seal的SHA，以及全部网格与fresh seeds。它是新的模型假设，不是放宽原数值证书、丢弃失败条件或修改D21成绩。

保留20下限的假设被development结果推翻：alpha=.25时missing-prior gain从−.005876减轻至 **−.000783**，但九条件平均gain变为 **−.000844**、最差 **−.010670**；alpha=.5平均 **−.002507**、最差 **−.021614**。原floor2/alpha=.25仍为开发最优且未过gate，primary继续entropy。这里约束的是prior中心，不是预测流量下界；测试明确验证预测仍可低于20。

## 第一轮独立实测

D21全部720个case、4140条含alias的方法记录均成功，wall time **634.65秒CPU**。未被gate接受的alpha=.25在同类新seed对entropy gain **+.019200**、CI **[.007648,.032899]**；六新压力条件 **+.008855**、CI **[−.000775,.020439]**。这些次要hypothesis结果不能反转已封存的开发拒绝，primary为entropy，所以primary gain为0；零比较不能证明选择gate有效。

D22全部720份truth/counts/prior从生成器逐值重建，4140份成功分数用独立公式复算最大差 **0**，12个问题的54次直接API调用预测SHA完全一致；耗时 **83.64秒CPU**，与其他实验存在并行。development条件leave-one-family-out均选原grid0，但gate只有排除D12_I405时通过，其余4次不通过；不能按其中一个子集选择晋级。

D23完成3690份不含deployment alias的L2分解，null分量计数残差最大相对值 **6.36e−14**。下面是相对平方L2误差，不是正式评分：

| 评估法律 / 模型 | 全路径 | rowspace | nullspace |
|---|---:|---:|---:|
| 九同类 / entropy | .369977 | .041899 | .328077 |
| 九同类 / 路径残差hypothesis | .389611 | .046501 | .343110 |
| 六新类 / entropy | .753916 | .130080 | .623836 |
| 六新类 / 路径残差hypothesis | .808060 | .135420 | .672640 |

同时，hypothesis平均相对路径L1在同类 **.551727→.501416**，新类 **.586731→.566634**。平均L1更低并不保证平方误差更低；误差分布与尾部成本需要一起考察。不能把不同指标排序差异解释为实现错误，也不能把rowspace/nullspace平方误差比例当官方L1 gain分解。组件加权增益另由D27核验。

## 第二批与六种子次要汇总

D24选择仍为原floor2/alpha=.25，而非新floor20。另一组三研究seed复核该冻结hypothesis：九同类gain **+.018551**、CI **[.007499,.031124]**；六新类 **+.009706**、CI **[.000547,.020791]**。两批参数相同，D27作六seed次要汇总：同类 **+.018876**、CI **[.007580,.031924]**；新类 **+.009280**、CI **[.000110,.020543]**。后者下界接近零，区间未作多重比较校正，只针对自主法律和固定五族，不是普适迁移或真实Kaggle结论。没有因此改变任何development gate。

D24全部720case记录亦无数值失败，实际438.02秒CPU、进程lifetime峰值300.27MiB；D21为634.65秒/290.70MiB。D24复用1350控制，且存在并行复核，二者不能作为模型推理速度对照。D26同类hypothesis平方L2 **.449764 vs entropy .410714**，新类 **.776173 vs .716385**，平方误差代价再次出现。

D25后续720份输入、4140分数亦全部精确重建，54次完整API调用exact；73.54秒CPU。D26有3690份分解，null count相对残差最大7.92e−14；13.04秒CPU。两批全部数值case成功，收益、缺失prior失败条件及平方误差代价一并保存。

两份研究共1440份case文件，其中270个development问题重复；**唯一输入组合1170个，900个fresh assessment问题**。54+54直接API调用也含相同开发问题，不能称108个独立样本。新API增加可选中心floor但默认仍2，D22验证原54次调用逐值不变；最终五项kernel/generator测试及全套 **93 research/81CPU passed、4 optional-Torch skips**。

```powershell
.venv-repro\Scripts\python.exe scripts\verify_demand_residual_study.py --study reports\research\D24_demand_residual_floor_followup.json --output reports\research\D25_demand_residual_floor_independent_replay.json
.venv-repro\Scripts\python.exe scripts\od_error_subspaces.py --study reports\research\D24_demand_residual_floor_followup.json --output reports\research\D26_demand_residual_floor_error_subspaces.json
.venv-repro\Scripts\python.exe scripts\od_metric_component_audit.py --study reports\research\D21_demand_entropy_path_residual.json --study reports\research\D24_demand_residual_floor_followup.json --output reports\research\D27_demand_residual_component_audit.json
```
