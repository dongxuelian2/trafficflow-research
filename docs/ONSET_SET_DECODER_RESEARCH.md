# Queue onset 集合大小与阈值 C68–C70

正式最佳仍为 **0.82940**，目标 **0.95000+** 未达到。本轮只使用公开 TRAIN 观测标签做代理实验；真实 Queue、Physics、OD path truth 和新 Total 都未知。没有新完整 submission、正式上传或平台发布。

## 研究问题与固定对照

C65 的固定 top-k 分类器没有稳定超过原强参照，C67 的逐窗 label-oracle 只提示集合大小可能影响评分。C68 不读取 C67，也不使用 oracle 的集合大小、预测或模型。为隔离解码器效应，本轮固定 C65 的四组双 seed 分类器、特征、排序分数和原 ongoing 预测，不再拟合树或神经网络，只改变 onset 最后 T+30 的集合生成方式。

四组分别为 I5_S 秋/冬与 I10_W 秋/冬，特征维数 56/46/56/56，沿用 107/154/52/70 个 fit onset query。候选如下：

| 候选 | 输入与拟合 | 决策方式 |
|---|---|---|
| incumbent | 原冻结 FullDomainQueueModel | 原强参照 |
| fixed_k | C65 原双 seed classifier | 原 decision 已选 k=4/4/1/2 |
| threshold | 同一 classifier 分数 | decision 选阈值与是否强制非空 |
| count_prior | fit 期校正计数的常数控制 | 同一 classifier 上取固定预测个数 |
| count_observed | fit 期观测正例个数，条件 Ridge | 同一 classifier 上取预测个数 |
| count_adjusted | fit 期校正计数，条件 Ridge | 同一 classifier 上取预测个数 |

两个条件计数臂每 case 各一个 Ridge，加一个常数控制，共 **8 个 Ridge、4 个标量状态**。条件模型使用 24 个合法因果摘要：当前速度阈值差、历史趋势/斜率、流量比例、缺失与覆盖/观测年龄、时间等。摘要不含未来标签、未来 eligibility、类别概率或拟合内分类器误差。标准化均值/方差仅用 fit 窗口，log1p 计数目标，固定 alpha=10；反变换后四舍五入并限制为 1…N。

校正目标为 `观测正例数 × 完整节点数 / future eligible 节点数`，只在 fit 标签构造中使用。它依赖未验证的 MCAR 假设；真实传感器失效可能与拥堵相关，不能将这个量称为 latent 真值或无偏真值。常数控制使用同一校正目标，避免把目标变化或合适的常数 k 误认成条件特征收益。

阈值网格为 31 个 .05… .8 的等距值，与强制非空开关组合，共 62 个/case，仅在 decision 选择。同分优先不强制，再选更接近 .5 的阈值。最终四组阈值为 .375/.175/.425/.15，均不强制非空。排序与解码始终先覆盖完整六步、全部 link 的目标域，按 link ID 稳定打破平分，然后 evaluator 才应用未来 eligibility。便携模型不接收未来 mask；count 最小 1 也不能保证 evaluator 过滤后有正例。

## 时间、选择与不确定性

沿用 C55–C58 balanced-v2 日期；旧 checkpoint 期作未用 buffer。秋季 fit 为 7/16–10/15，decision 10/31–11/29，selection 11/30–12/14，assessment 12/15–1/13；冬季 fit 为 7/16–11/30，decision 12/16–1/14，selection 1/15–1/29，assessment 1/30–2/28。均为已查看的公开开发日期，源 JSON 也包含历史结果，不能称新盲测或独立最终测试。

count 模型先封存，再读 decision；decision 策略封存后读 selection；selection 封存后才计算本轮 assessment。selection 沿用相对 incumbent 的去重 condition 宏 IoU gain≥.01 且 7 日 block bootstrap 95% CI 下界>0 的门槛。每个区间 4000 次重采样，保留无事件日期，缺少某 condition 的抽样排除。稀疏事件、两季重叠 fit、同步日历及多项未校正次要比较都限制推断，不能把四 case 当四个独立重复实验。

| case | decision 胜者 | selection gain / 95% CI | 最终研究策略 |
|---|---|---|---|
| I5_S 秋 | incumbent | 0 / [0,0] | incumbent |
| I5_S 冬 | threshold | +.072024 / [.039103,.107813] | threshold |
| I10_W 秋 | threshold | +.033333 / [0,.083333] | incumbent |
| I10_W 冬 | count_adjusted | −.025000 / [−.100000,.090000] | incumbent |

本轮有一个实际非零 challenger 通过研究 selection gate，但它在后段的增益大幅缩小且区间跨零。门槛不是泛化保证。I10_W 秋的后段阈值结果较好，也不能反过来改变已封存选择。

## 公开观测标签代理结果

以下为 assessment 的物理 origin 去重、onset/ongoing 等权宏 IoU；不是官方 Queue 或四任务加权 Total。

| case | incumbent | fixed_k | threshold | count_prior | count_observed | count_adjusted | 已选策略 |
|---|---:|---:|---:|---:|---:|---:|---:|
| I5_S 秋 | .781151 | .770702 | .786442 | .791028 | .694732 | .796319 | .781151 |
| I5_S 冬 | .797255 | .754866 | .801720 | .806184 | .717324 | .803208 | .801720 |
| I10_W 秋 | .557659 | .622474 | .631733 | .622474 | .705807 | .659511 | .557659 |
| I10_W 冬 | .814942 | .714942 | .864942 | .714942 | .764942 | .814942 | .814942 |

已选 I5_S 冬策略相对 incumbent 的后段 gain **+.004464，95% CI [−.044312,.052230]**。其余已选策略与 incumbent 相同。整体证据门槛要求至少两 case 的 gain≥.01 且 CI 下界正，并限制各 case 损失；已选策略符合正增益条件的 case 数为 **0**，统一 threshold 的次要诊断为 **1**，两者均未通过。不晋级生产模型。

C70 另作 16 个事后配对消融，独立复算，不用于重新选择：

| case | threshold 相对同 classifier fixed_k 的 gain / CI | threshold 相对强 incumbent 的 gain / CI |
|---|---|---|
| I5_S 秋 | +.015741 / [−.021826,.051344] | +.005291 / [−.022321,.038095] |
| I5_S 冬 | +.046854 / [.003662,.091628] | +.004464 / [−.044312,.052230] |
| I10_W 秋 | +.009259 / [−.222778,.200000] | +.074074 / [.027778,.125000] |
| I10_W 冬 | +.150000 / [.053482,.233333] | +.050000 / [0,.116667] |

阈值相对固定 k 在两个冬季 case 有正区间，但这是较弱匹配解码器对照，不能算相对原强流水线的独立收益。两个 count_observed 在 I5_S 秋/冬的 gain 为 **−.086420/−.079932**、区间全负；I10_W 秋的正值未获选择。失败保留。

count_adjusted 相对同目标 count_prior 的 gain 为 +.005291/−.002976/+.037037/+.100000，区间分别 [−.020283,.033716]/[−.015278,.010802]/[0,.063725]/[.055556,.159091]，没有一致的条件特征收益。常数控制预测 k=3/3/2/2；原固定网格不含 3。因此部分表面改善可由更合适的常数解释，不足以证明因果计数模型成功。完整域后段阈值在 I5_S 秋/I10_W 秋分别有 1/2 个空集合；保留这类决策，不用未来标签补点。

## 复现与独立核验

在项目根目录、CPU 环境运行（输出目录及 checkpoint 为写入一次；已有结果不要覆盖）：

```powershell
.venv-repro\Scripts\python.exe scripts/onset_set_decoder_experiment.py --output artifacts/research/C68_onset_threshold_cardinality
.venv-repro\Scripts\python.exe scripts/verify_onset_set_decoders.py --study reports/research/C68_onset_threshold_cardinality.json --output reports/research/C69_onset_threshold_cardinality_verification.json
.venv-repro\Scripts\python.exe scripts/summarize_onset_set_decoders.py --output reports/research/C70_onset_decoder_matched_summary.json
.venv-repro\Scripts\python.exe scripts/checkpoint_onset_set_decoders.py
```

依赖先前 C65 模型和 C66 输入证明。C68 显式绑定 C66 的报告/原始公开文件/旧特征构造源码 SHA，复用旧 **2345 次**原始窗口审核，本轮没有再次重建全部这些窗口。C69 独立重建 **665 次**新的 onset 摘要，最大差 0；12 个计数状态用增广最小二乘独立求解，最大参数差 **3.997e−15**。拟合目标、标准化、逆变换/取整、阈值网格、原 fixed_k、ongoing、gate、全部 assessment 分数及区间一致。

C69 实际执行 56 次冻结 Booster 矩阵调用；另从公开历史重建 112 个一小时加公开 masked T 的窗口，完成 **896 次**便携角色/窗口调用（含策略别名）、**448 次**额外目标乱序检查、**16 次**未来历史污染拒绝。它不是重新训练分类器，也不是对全部缓存窗口逐个调用所有便携角色。C70 的 16 个配对消融使用另一手算对照器复算一致。

C69 完成后仅删除核验脚本一行未执行的 `if False` 死代码；实际执行源码 zip 保留，数值与断言未改。C68 CPU wall **25.51 秒**、进程 peak **663.30 MiB**；C69 **135.62 秒**。部分测试并行，时间不作为隔离的吞吐基准。

新增 3 项单测 1.33 秒通过；全研究环境 **134 passed / 12.31 秒**，纯 CPU **113 passed / 7.37 秒、7 个可选 Torch 模块跳过**。覆盖摘要手算/完整域/置换、fit-only 计数及校正边界、阈值/force 与 future mask 分离。compileall 通过。

新 manifest 为 `reports/research/model_artifacts_checkpoint_onset_set_v1.json`：重新校验全部 **6374** 个旧 artifact 不变，累计 **6453** 个；5 份原完整 CSV 和依赖锁不变。新报告、注册配置、训练摘要、模型状态、阶段封存、预测、源码 zip 均保留；Git JSON 字节证据另存 `reports/research/onset_set_report_bytes_v1.json`。最终模拟 seeds 7907/8017/9109 未生成。

下一轮优先在预先固定的新 rolling 日历检验简单阈值相对强参照的稳定性，并研究预测起点之前的信息是否足以识别 onset 时间与范围。暂不扩大 count Ridge alpha/阈值网格；观测计数校正需要更明确的噪声与缺失机制证据。State/OD/Physics 的现有证据和限制继续保留。
