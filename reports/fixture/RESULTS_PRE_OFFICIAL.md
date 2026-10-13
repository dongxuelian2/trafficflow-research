# RESULTS — 2026-10-13 首轮交付

**没有取得正式公开榜分数，也没有测量到 0.940+。官方数据下载被 HTTP 401 阻断。**
当前已交付独立 Git 仓库、固定环境、四任务预测/验证系统、未改动的官方工具包，以及可复现的人工 fixture 消融和完整提交测试。
人工 fixture 不等于官方训练数据；以下人工结果不能用于估计排名或宣称达到研发目标。

## 官方参考与本项目当前状态

| 分项 | 官方 BASELINES 参考（validation） | 研发目标 | 本项目官方数据实测 | 当前候选 |
|---|---:|---:|---|---|
| State | 0.6929 | 0.960 | 未知，未取得数据 | 插值 + LightGBM 残差回归，保留 CatBoost 比较 |
| Queue | 0.3017 | 0.890 | 未知，所有 split 的真实标签未公开 | causal LightGBM + 按窗口 proxy IoU 选阈值 |
| Physics | 0.3549 | 0.940 | 未知，正式边界流未公开 | 状态输出上的 coverage-aware 诊断和可选约束 |
| ODME | 0.8359 | 0.980 | 未知，只有 S_link 可计算 | 稀疏 NNLS，λ=0.05，使用当前 split counts/prior |
| Total | 0.5534 | 0.940+ | **未知** | 四项必须齐全 |

参考值来自固定官方版本的文档，不是本项目复现成绩。官方 README 另写 Queue persistence=0.2518，与 BASELINES 不一致，未能独立确认原因。
RUN_LOCAL 的 D12_I5_N train State=0.6252 尚未在当前官方包复现。
真实综合分数按 `0.35 State + 0.30 Queue + 0.15 Physics + 0.20 ODME`；proxy IoU、物理 residual、S_link 都不会替代未知的正式分项。

## 实际完成的复现验证

- Python 3.13.1，60 个依赖版本固定；一个开发环境和一个不继承系统包的干净虚拟环境均运行完整流程。
- 10 项测试全部通过；干净环境 `pip check` 通过。
- 官方 33 个 vendored 文件逐个 Git blob 校验，与提交 `c88cddf533bbf0afa4ff1fc6c031d760a08e7a31` 一致。
- 原版历史均值、queue persistence、Task 3 state-anchored evaluator、ODME builder/link evaluator 在人工 fixture 上实际运行。
- 独立流程输出 55,664 个 state rows、9,216 个 queue rows、216 个 OD rows；主提交 **65,096 行**，无缺失值，严格覆盖 fixture master key。
- 开发环境和干净环境分别训练并预测；三份任务 CSV 和最终 CSV **字节一致**，所有预测最大绝对差为 0。

证据：[verification.json](reports/fixture/verification.json)、[环境](reports/environment.json)、[原版日志](reports/fixture/official_baselines/official_runs.json)。
测试提交在 `artifacts/clean_run_v2/submission.csv`，仅用于集成测试，不能上传比赛。

## 合法的人工数据验证方法

种子 2026；两个虚构走廊，42 天 train、各 3 天 validation/private；每走廊 8 links。
公开观测含缺测和质量标记；同一天只发布一个 mask regime，Task 2 history/horizon/buffer 不列入 Task 1 targets。
人工 generator 的 speed/flow 是测试信号，不是经过严格 LWR 校准的交通仿真。

日期固定为：profile 6 月 1–7 日，fit 6 月 8–27 日，calibration 6 月 28 日–7 月 4 日，test 7 月 5–12 日（均为 fixture 的 2030 年合成日期）。
训练特征只读 masked 当日；profile 只使用更早日期。calibration 选择模型和阈值，test 不参与选择。
树模型 150 iterations，4 CPU threads，单走廊最多 100,000 state training rows，未使用 GPU。

## State 消融：人工 test，按官方公式和模板计算

| 方法 | D12_I5_N | D7_I10_E | 两个 fixture family 等权均值 |
|---|---:|---:|---:|
| 历史 profile | 0.874819 | 0.878752 | 0.876786 |
| 时空插值 | 0.946548 | 0.939616 | 0.943082 |
| Kalman/RTS | 0.884855 | 0.881383 | 0.883119 |
| 插值 + LightGBM | **0.953113** | **0.946859** | **0.949986** |
| 插值 + CatBoost | 0.953016 | 0.945840 | 0.949428 |

两个走廊均由 calibration 选择 LightGBM。相对历史 profile，test 增益分别为 +0.078294 和 +0.068106；相对插值为 +0.006564 和 +0.007242。
R3 最难：所选模型 R3 分数为 0.950952 / 0.944575，speed RMSE 为 1.569 / 1.635 km/h，flow RMSE 为 19.760 / 26.226 vph/lane。
按 R1/R2/R3、congested/free observation proxy 的详细记录见 [State 消融](reports/fixture/state_ablation.csv) 和每个走廊的 experiment.json。

完整单走廊消融（包含 queue、OD 和物理候选）实测约 59.0 / 57.5 秒；进程 lifetime 峰值工作集约 226.6 / 229.7 MiB。计时受两个同时运行的复现实验影响，不能外推到 13 GB 官方包。

## Queue 消融：人工观测代理标签，非真实 S_queue

| 方法 | D12 test proxy IoU | D7 test proxy IoU |
|---|---:|---:|
| Persistence | 0.127663 | 0.102095 |
| 波传播 + 趋势概率 | 0.485098 | 0.441782 |
| LightGBM | **0.734338** | **0.758795** |

阈值仅从 calibration 搜索，分别为 0.30 / 0.25。
D12 LightGBM calibration=0.836929、test=0.734338，显示日期迁移和代理样本方差，不能把 calibration 当最终性能。
所有 feature histories 必须位于 `[T−60,T)`；真实预测仅使用 released window_history，不读取目标 horizon 或其后的 mainline observations。
代理标签来自 eligible train observation 的 `speed <= v_cut`，与正式 underlying-state queue 标签不同。
详见 [Queue 消融](reports/fixture/queue_ablation.csv)，正式 S_queue 保持未知。

## Physics 消融：只做诊断

| calibration 设置 | D12 State | D7 State |
|---|---:|---:|
| 原 LightGBM | 0.952896 | 0.948676 |
| speed/capacity bounds | 0.952896 | 0.948676 |
| 守恒强度 0.001 | 0.932325 | 0.931141 |
| 守恒强度 0.01 | 0.919263 | 0.913501 |

本轮 fixture 不启用守恒调整。它自身未严格守恒，因此此结果不足以判断约束在官方数据上的收益。
模块记录 FD residual、observable transition coverage 和 conservation RMS/MAE（车辆单位），缺失 ramp/latent boundary 不伪造为零。
原版 Task 3 fallback 已跑通，其 CSV 明确命名 `physics_diagnostic_NOT_FORMAL.csv`；打印值不作为成绩，不参与模型选择。
失败优化及修复记录见 [failed_experiments.json](reports/failed_experiments.json)。

## OD 消融：人工 train counts 的 S_link，非 S_ODME

| 方法 | D12 S_link | D7 S_link |
|---|---:|---:|
| Prior | 0.975994 | 0.952615 |
| NNLS/ridge λ=0.05 | 0.999632 | 0.999641 |
| NNLS/ridge λ=0.005 | 0.999963 | 0.999964 |
| 相对 prior 正则 | 0.999997 | 0.999997 |
| KL/概率 prior | 0.995474 | 0.995536 |

弱正则更贴 counts 不等于更准的隐藏 path flows。保守默认仍为 λ=0.05；S_od、S_dev、S_attr、S_ODME 均未知。
所有方法只使用 measured-link CSR 矩阵，不将未测 connector 当零流。
每次求解连同 sensor-holdout 诊断约 0.02–0.04 秒；规模仅 8×36、120 nonzeros，不能代表正式规模。
详见 [OD 消融](reports/fixture/odme_ablation.csv)。

## 官方实现核查

人工 fixture 的原版 Task 1 scorer 得到 0.066917，独立模板评分的同一历史 baseline 得到 0.945202（后者是全 train profile 的拟合分数，不是时间 holdout）。
该差异主要来自原版 scorer 把额外 eligible 双空 horizon cells 也列为 targets；文档要求模板。二者不能直接比较模型优劣，也不能把人工分数当官方分数。
真实包上的影响尚待验证，保留原版和独立 scorer 两条路径及原始日志。
Task 4 原版 baseline 实测人工 S_link=0.999637；builder 用 train counts，当前 split solve 写为 public_reference，不能拿它当隐藏真值。

## 剩余瓶颈及下一轮

1. 取得 Kaggle 官方当前包，核验版本/完整性/指纹，并实跑十走廊官方基线。下载请求已得到 401；本机凭据或用户浏览器下载是当前必需材料。
2. 在官方 train 做冻结的时间验证和多 rolling folds；核查不同方向、真实 mask regimes、ramp coverage 和拥堵场景。当前没有真实事故标签或跨走廊 holdout 结果。
3. 分别记录实际任务 runtime、memory、样本量，完善按观测急降速分组的 proxy 误差分析；不得把代理分组称作真实事故类型。
4. 针对真实数据消融联合状态估计/稀疏守恒，尤其检查 Task 1 准确度与物理一致性权衡；不以本地 fallback Task 3 做正式优化目标。
5. 在合法公开榜反馈中验证整体可迁移性；不依据隐藏标签、旧数据泄露或组织方私有资源调参。所有正式未知值继续记为 null。
6. 在最终提交前冻结数据、模型、命令、随机种子和依赖，准备奖项要求的复现代码与技术 PDF。该 PDF 尚未编写，因为正式数据实验未完成。

没有达到目标不会被当作停止研究的理由；当前限制是官方数据的账户访问条件。
