# 正值需求因子 OD 研究

正式最佳仍为 Kaggle **0.82940**，0.95000+ 目标尚未达到。本页所有恢复分数来自自主生成、真值已知的 OD 问题；公开检测器拟合、局部最优化证书与真实隐藏路径精度分别记录。原正式 Ridge、K01 的正式失败、K05 和既有完整候选均保留。

## 模型与可识别性

新增独立实现 `src/trafficflow/log_demand_od.py`，使用

`f_p = max(b_p,2) * exp(u_origin[p] + v_destination[p])`。

对数需求因子采用零均值 Gaussian 先验，检测器计数用相对 Gaussian 或 Huber 残差。它是正值、非线性的 MAP 点估计，与已有 additive Gaussian 路径协方差模型不同；没有独立路径残差，不支持真零路径，也没有声称提供校准后验。公开网络每个 OD 只有一条路径，当前实现不冒充备选路径选择模型。

四个 development 配置为 `(group_sd,noise_cv,floor,Huber)`：`(.2,.02,2,None)`、`(.6,.02,2,None)`、`(.6,.1,2,None)`、`(.6,.02,2,2.5)`。每个因子约束在 ±3，两种固定初始化。L-BFGS-B 后做 exact-Hessian 阻尼 Newton polish；仅允许目标下降或浮点舍入预算内的梯度残差减半，最终 projected-gradient 门槛仍为 **2e−4**。至少一个起点通过证书才能返回；多起点仍不证明全局最优。

局部测量 Jacobian 的 rank 审计在公开 weak-prior、零效应处进行：I5_N 为 **92 / 184** 因子，I210_W 为 **81 / 162** 因子。先验可以选出坐标，但没有增加观测信息。给全部 origin 加常数、全部 destination 减同一常数，路径流量不变，先验惩罚却变化；测试显式验证这一 gauge 方向。

## 合法实验与封存

只读取十个公开 train 网络与 weak prior，以自主生成器创建已知真值和计数。九种需求/先验条件包括独立路径、层次需求、总量偏差、加性噪声、对数噪声、缺失先验路径、分组偏差、稀疏活跃路径、检测器离群值。生成法律与此前 D12/D13 一致；这些不是组织方隐藏生成器。

开发种子 411/829/1201，十方向×九条件×三种子×六方法共 **1620 条**记录。参考为原 Ridge λ=.05 和已封存 entropy τ=.05/floor20。按九条件最差 gain、随后平均 gain 选择；所有 case 必须成功，均值 gain≥.005、最差 gain≥−.002。只按开发数据选出 `(.2,.02,2,None)` 并封存，开发 macro gain 对 Ridge **+.066286**，九条件全正。

随后首次生成研究评估种子 **931003/931013/931031**，270 个唯一问题。三种实际方法各270条，deployment alias再重复270条，总 **1080 条**；alias 不算额外求解器或独立统计证据。最终模拟种子 **7907/8017/9109 未生成**。同一网络的方向、同一 seed 的条件相关，区间只重采样五个固定走廊族与三个 seed，不能当作 Kaggle/private 置信区间。

## 数值失败与修复

D14 首次完整运行中，一个 development 配置案例以及选定参数的三个评估离群案例未通过原数值门槛。加上 deployment alias 共七条 failure 记录，实际四个唯一失败求解器案例。失败没有删除；原 report、配置、源快照和已生成的记录全部保留。失败时整体 gain CI 明确为空，不能用其余成功子集宣传收益。

D17 是同一目标、网格、seed、数据和 **2e−4 门槛**的数值重跑；仍保留原 D14 sealed choice，重新计算的开发选择只作回归检查。它复用已经查看的研究评估，**不是新的盲测**。D19逐条比较完整覆盖、失败恢复、已有成功记录的漂移，并要求 Ridge/entropy 参考预测 SHA 和分数逐值不变。

D15/D16/D18 使用独立有限差分 Jacobian 的 bounded trust-region least-squares 数值 oracle。选定参数六案例全部一致；D18全部配置 **16/18** 一致。弱正则 Gaussian 在 I5_N 离群案例中出现不同局部解，目标相差约 .015、两份路径流量相对 L1 差约15.5%，检测器计数相对 L1 差仅约 .064%；不能把低计数误差解释为路径识别。另一个 Huber 案例差异很小，但仍没有放宽原一致性门槛。

D19确认七条失败记录恢复、没有新失败，1080条参考预测/分数逐值一致；2693条原成功记录的最大评分漂移 **4.4781e−7**。D17记录的3240份多起点证书全部有效，最大projected gradient **9.92e−8**；这些记录含alias重复，不算独立实验。D14/D17分别456.71/498.35秒，进程lifetime峰值约295.86/293.99MiB，CPU四线程，无GPU。

## 与强参考的匹配结果

| 方法 | macro模拟S_ODME | 对Ridge gain / 配对CI |
|---|---:|---|
| Ridge λ=.05 | .664500 | 参考 |
| 固定entropy τ=.05/floor20 | .707992 | +.043492 / [.028484,.058255] |
| 开发封存的log-demand | .724308 | +.059808 / [.038371,.076213] |

新模型对entropy的配对gain **+.016316**、CI **[−.001239,.037405]**，不足以晋级。九种条件有五种比entropy差，missing-prior-paths差 **−.046655**；收益主要来自sparse-active与count-outliers（相对entropy分别+.134230/+.103242）。这提示先验表达与数据法律强相关，不能按这份已查看assessment挑混合权重。

## 完整本地控制

`artifacts/research/K07_log_demand_OD_LOCAL/submission.csv` 共 **6,980,503**行，409,645,782字节，SHA-256 **e797746aac25398f62d6b1a62392f59ecfdd763f9b2bc34628bca2c8f1e36c9d**。独立全行diff通过：仅替换validation/private的70,708个OD值，State/Queue/全部其他字段不变。另存原参数/研究报告SHA、源快照、全部输入哈希及每个公开split诊断，不覆盖正式最佳或K05。

30个公开train/validation/private OD子问题全部成功，count-only S_link范围 .983997–.997413，流量相对原weak prior的L1改变量 **41.11%–62.17%**。这些不是路径真值指标；大幅改变prior可能导致迁移风险。完整构建174.85秒CPU，进程lifetime峰值335.38MiB。正式ODME/Total仍未知，候选未上传。

```powershell
.venv-repro\Scripts\python.exe scripts\build_log_demand_od_candidate.py --study reports\research\D17_log_demand_OD_numeric_polish.json --numeric-verification reports\research\D18_polished_log_demand_numeric_verification.json --output artifacts\research\K07_log_demand_OD_LOCAL
.venv-repro\Scripts\python.exe scripts\verify_controlled_candidate.py --candidate artifacts\research\K07_log_demand_OD_LOCAL\submission.csv --allowed-task odme
```

## 复现

使用已锁定纯 CPU 环境；每项输出目录必须不存在，不覆盖任何旧失败或候选。

```powershell
.venv-repro\Scripts\python.exe scripts\log_demand_od_experiment.py --output reports\research\D14_log_demand_OD_broad_control.json
.venv-repro\Scripts\python.exe scripts\log_demand_od_experiment.py --reference-study reports\research\D14_log_demand_OD_broad_control.json --output reports\research\D17_log_demand_OD_numeric_polish.json
.venv-repro\Scripts\python.exe scripts\verify_log_demand_od.py --settings-index 0 1 3 --output reports\research\D18_polished_log_demand_numeric_verification.json
.venv-repro\Scripts\python.exe scripts\compare_log_demand_repair.py --before reports\research\D14_log_demand_OD_broad_control.json --after reports\research\D17_log_demand_OD_numeric_polish.json --output reports\research\D19_log_demand_numeric_repair_comparison.json
.venv-repro\Scripts\python.exe scripts\compare_log_demand_models.py --study reports\research\D17_log_demand_OD_numeric_polish.json --numeric reports\research\D18_polished_log_demand_numeric_verification.json --output reports\research\D20_log_demand_matched_model_comparison.json
```

以上 D14 命令在当前源码运行会使用修复后的 solver；逐字复现首次失败必须使用该实验 `source_snapshot.zip` 中的旧源码与记录环境。所有耗时为实际 wall time，期间与测试/独立数值复核存在并行，不是隔离吞吐比较。

研究参考：[Perrakis 等的 OD Bayesian mixture 工作](https://arxiv.org/abs/2011.06045) 与 [Chen 等的潜在时序需求工作](https://arxiv.org/abs/2403.04742) 提供概率建模背景。本实现没有复制其代码，也没有声称复现论文的 Poisson 混合、时序 GP 或 MCMC。

## 尚待解决

固定乘性 prior 的表达限制、零路径与局部多解尚未解决；需要独立预注册的需求/噪声分布、真正的后验校准和数据可识别子空间诊断。模拟正收益不能抵消此前 K01 的正式负结果，也不保证迁移到真实 OD。State、Queue、真实 Physics 及正式 Total 不由本轮模拟推算。
