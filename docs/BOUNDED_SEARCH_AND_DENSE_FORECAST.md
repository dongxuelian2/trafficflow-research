# 有限非前缀搜索与动态预测数据 C82–C86

正式最佳仍 **0.82940**，目标 **0.95000+** 未达到。本轮在公开决策期否决当前非前缀解码扩展，并完成后续动态预训练的数据清点；没有新后段模型成绩、完整submission、上传或平台操作。

## C82–C84：先检验受控非前缀搜索

C81的自主五节点分布曾给出前缀最优.4360、全局最优.4425。本轮自主实现有限单翻转搜索：从同一场景库的最佳前缀出发，每步比较全部节点的一次增删，最多8次严格改进，最低步增益1e−6；同分优先删除、再按link ID。只保证当前场景目标单调，不保证一般全局最优。

在C78的9个已知非空/mask分布及该非前缀反例上，另枚举全部32个预测集合。十例均达到各自全局期望最优，反例通过删除B把.4360提升到.4425。这个数学检查不是交通成绩，也不证明一般收敛。

公开模型复用C79的copula/empirical场景库、C74校准、C71分类器。另从同一fit数据构造独立模拟库（seeds20261107/20261119，各2048行），只在其中的paired模拟增益≥.001且近似正态95%区间下界>0时接受提议。该区间只衡量固定假设law下的模拟误差，不是独立交通测试、重训不确定性或正式置信区间；多个预测共享模拟库也不构成多重检验保证。

在读取新selection/assessment数组之前，先按预登记规则筛查decision：新模型必须同时超过incumbent和原prefix至少.005。未满足就停止此解码扩展，不追加后段实验。所有数据仍是已查看的公开development，源JSON含旧结果，不能称盲测。

| decision代理case | onset窗口 | incumbent | 原prefix | 新local proposal / validated | 提议改动 |
|---|---:|---:|---:|---:|---:|
| I5_S September | 19 | .810210 | .785210 | .785210 | 0 |
| I5_S October | 21 | .797619 | .751985 | .751985 | 0 |
| I10_W September | 6 | .872614 | .872614 | .872614 | 0 |
| I10_W October | 6 | .651519 | .651519 | .651519 | 0 |

共52个onset窗口、5284个单翻转邻居候选，全部停在原前缀。无提议、无模拟接受，四组均未shortlist；**没有新selection或assessment数组读取**，新后段成绩为未评估。不能以本次全零接受结果证明模拟接受门槛在真实非零提议下有效。

C84独立重放4个模拟库、以候选集合指示矩阵直接计算交并集复核所有邻居/停止/模拟区间/decision分数；另重核10个小分布的320集合全局枚举。8次decision onset Booster、4次参照probability调用；新鲜16个历史窗口支持48便携角色调用、24额外乱序及12未来输入拒绝，全部通过。旧raw/概率审核明确复用。

## C85–C86：新增信息应来自哪里

当前经验场景及局部搜索没有新增价值，转回动态预测的数据与表示。这里只清点C71各case **fit日期**，不读取calibration/decision/selection/assessment/private数据，不训练新网络。

每UTC日原点为第12至倒数第7个五分钟slot，仅按过去12步eligible覆盖≥.7筛采样；未来有无queue或可评分证据不用于选原点。模型可用输入仍为过去12步raw eligible观测加公开masked T；未来六步raw仅作训练teacher诊断。分别考察5分钟和固定30分钟步长。

340次fit-day引用涉及432份独特raw/masked源文件，滚动fit之间日期重叠，不是340个独立日。独特teacher cell与重复监督次数分别计数，不能把重叠时间窗当新独立标签。

| case | 原event-fit监督单元 | dense5独特teacher单元 | 倍数 | dense5窗口 | 同单元重复次数均值 |
|---|---:|---:|---:|---:|---:|
| I5_S September | 262518 | 1303404 | 4.965 | 16740 | 5.891 |
| I5_S October | 285865 | 2270580 | 7.943 | 29160 | 5.891 |
| I10_W September | 67385 | 1393856 | 20.685 | 16740 | 5.891 |
| I10_W October | 109498 | 2428487 | 22.178 | 29160 | 5.891 |

数量扩张不等于预测增益，且新增监督主要是背景/持续交通动态。checkpoint另从全部raw fit日独立重建合法观测代理的onset原点集合，证明**不是只有数量相等，而是集合逐项相等**：

| case | 原selected onset原点 / dense5全部同准则原点 | 固定30分钟网格保留此类原点 |
|---|---:|---:|
| I5_S September | 72 / 72 | 6 |
| I5_S October | 122 / 122 | 9 |
| I10_W September | 37 / 37 | 0 |
| I10_W October | 58 / 58 | 1 |

准则仅是“观测历史中无同节点两次queue、未来前五步无观测queue、T+30有观测正例、历史质量达标”。它不证明latent或官方事件已覆盖。已有fit已收全本准则的onset，扩大同一selector配额不会产生新onset监督；dense预训练的价值只能另行检验。固定30分钟相位会遗漏多数onset上下文，即使其teacher单元大多也出现在其他窗口中。

原event-fit的正例单元为46795/46090/3735/6145；dense5独特正例为102185/172892/6782/10654。新增正例也不构成新的同准则首次T+30事件。原始teacher的邻步降速≥20km/h单元仅1591/2754/388/591，不能把它们直接称为latent事故；目标严重不均衡需要保留。

masked T的eligible覆盖约.498… .513。dense5重复监督均值5.89说明应以日数组索引采样合法原点，避免把重复teacher存成大量独立样本。全部目标必须处于fit日期内、同日未来六步；未来raw/质量只能进入loss/诊断，不能成为输入。

C86以另一raw eligibility谓词、groupby数组、移位窗口和“未来时间集合并集”独立复算：680个daily/stride行、12240个整数统计、全部汇总比率与selected监督并集一致。所有432源文件重hash；checkpoint的exact-onset-support审核另存`local_dense_verification_record_v1.json`。没有把这些统计转换成State/Queue/Physics或Total。

## 验证、资源与复现

新增3项单测1.61秒通过，覆盖已知全局反例、预算/单翻转局部点/置换、独立假设场景接受与拒绝。当前全 **147research / 12.42秒**，CPU **126passed / 7.97秒、7可选Torch模块skip**；compileall通过。

C83 CPU wall8.05秒、进程peak442.32MiB，C84为10.46秒；C85清点19.79秒/peak450.39MiB，C86复算8.56秒。含并行测试与文件缓存，不是隔离吞吐基准。没有新分类器或概率校准器拟合，只有4个独立模拟库和可运行的research local-search策略。

后继manifest `model_artifacts_checkpoint_local_dense_v1.json` 重校验全部6835个旧artifact不变，新增42、累计6877；5份完整CSV及依赖锁不变。Git报告字节证明为`local_dense_report_bytes_v1.json`。最终模拟seeds7907/8017/9109仍未生成。

```powershell
.venv-repro\Scripts\python.exe scripts/local_search_known_laws.py --output reports/research/C82_bounded_search_known_laws.json
.venv-repro\Scripts\python.exe scripts/local_search_decision_screen.py --output artifacts/research/C83_bounded_search_decision_screen
.venv-repro\Scripts\python.exe scripts/verify_local_search_screen.py --output reports/research/C84_bounded_search_screen_verification.json
.venv-repro\Scripts\python.exe scripts/dense_forecast_population.py --output reports/research/C85_dense_forecast_fit_population.json
.venv-repro\Scripts\python.exe scripts/verify_dense_forecast_population.py --output reports/research/C86_dense_forecast_population_verification.json
.venv-repro\Scripts\python.exe scripts/checkpoint_local_search_dense.py
```

所有结果写入一次，已有目录不能覆盖。下一步转向通用动态预训练、随后同预算event fine-tuning的匹配实验：保持五分钟原点相位、未来teacher仅进loss、两optimizer seed及独立checkpoint/decision期。必须与scratch同结构网络、原强树共同比较，区分训练人口扩大与结构变化；当前数据清点尚未证明预训练收益。
