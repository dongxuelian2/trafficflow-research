# RESULTS — 0.950+ 高强度研究

**当前已核验的正式 Kaggle Total：0.82940。目标 0.95000+ 尚未达到。**

本人提交 ref `57138874`，2026-10-13 09:53:40 UTC，状态 COMPLETE。回下载文件与本地 `artifacts/submission_best/submission.csv` SHA-256 完全相同：`43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1`。四个正式分项仍未知，不能从 Total 唯一反推。

基准提交、模型、配置和代码 `15036c8` 已独立冻结，共 106 文件、712,127,414 bytes。详见 `reports/research/baseline_57138874_frozen.json`。本地最佳候选仍保留原文件；任何新候选另存目录。

新实验与失败结果见 `EXPERIMENTS.md`；候选对照见 `MODEL_COMPARISON.md`；优先路线见 `RESEARCH_ROADMAP.md`。
登记协议 SHA：`91e48a8f2f86775466c8c9b5ad6bdbe693ea25554ef9b18d9e12cd80d3256143`。

独立 CTM 的 54 个压力场景已运行，车辆守恒误差 ≤6.76e-13。其 latent oracle 对 noisy observation 的 State 为 0.952006，时空插值为 0.887547，RTS 为 0.848378。这些是自建模拟测量，不是比赛分数。
公开拓扑审计否定了“部分方向无可评分关系”的猜想：十方向均有可评分连接。原本地诊断错误地排除了外部连接，已修复；仍不能计算正式 Physics。

以下保留首轮交付报告作为历史实验说明。其“尚无正式分数”与 0.940 目标只描述首轮交付时状态，当前状态以本节为准。原报告也已进入冻结快照。

---
# RESULTS — 2026-10-13 官方数据首轮交付

**当前官方数据已完成下载和验证；尚无正式公开榜成绩，不能声称达到 0.940+ 或取得奖项。**
已完成十走廊四任务基线复现、独立改进、时间分组验证、模型冻结，以及 **6,980,503 行**完整提交。
公开 train 时间留出 State = **0.935606**；八走廊拥堵观测代理 IoU = **0.867664**。
真实 Queue、Physics、ODME 和正式 Total 均未知；这些代理/诊断不能替代正式分项。

## 正式指标与研发目标

| 分项 | 官方 BASELINES validation 参考 | 研发目标 | 本项目正式成绩 | 可验证的本地测量 |
|---|---:|---:|---|---|
| State | 0.6929 | 0.960 | 未知 | train 时间留出 0.935606 |
| Queue | 0.3017 | 0.890 | 未知 | 观测代理 IoU 0.867664 |
| Physics | 0.3549 | 0.940 | 未知 | 可观测守恒 RMS、FD residual、coverage |
| ODME | 0.8359 | 0.980 | 未知 | train S_link 0.999935；其他项未知 |
| Total | 0.5534 | 0.940+ | **未知** | 不将 proxy/diagnostics/S_link 拼成正式总分 |

正式规则为 `0.35 State + 0.30 Queue + 0.15 Physics + 0.20 ODME`。
参考文档固定在官方 commit `c88cddf533bbf0afa4ff1fc6c031d760a08e7a31`；参考 validation 成绩不是本项目复现成绩。
README 的 Queue persistence 0.2518 与 BASELINES 0.3017 不同，真实标签未公开，差异未能独立消除。

## 数据、环境与基线复现

- 仅使用 Kaggle 2026-09-11 重建的当前公开包，共 9,688 文件，归档 8,335,874,386 bytes，解压 12,387,782,323 bytes。
- ZIP 每项 CRC、官方列表大小和完整逐文件 SHA-256 均通过；归档 SHA-256 为 `ab2767c1c83f55d357ae16f4754d014b2bff8a6a1c28cbf9fc4876e522fb936f`。
- 数据是合成记录，使用真实走廊参数校准；合成日期不能匹配外部真实交通记录。
- Python 3.13.1，60 个锁定依赖，干净环境 `pip check` 通过，14 项测试通过，未使用 GPU。
- 官方 33 个 vendored Git blob 核验一致；原版 Task 1 builder/scorer、Task 2 persistence、Task 3 anchored evaluator、Task 4 builder/link scorer均在当前官方包实际运行。
- 原版 Task 1 十走廊 train 的 State = 0.732028，D12_I5_N = 0.625461，与 RUN_LOCAL 的 0.6252 接近。
- 原版 Task 2 生成 train+validation 174,000 rows 和 private 87,000 rows。真实标签未公开，无法复现其正式 IoU。
- 原版 Task 3 无 organizer boundary flux，只生成 `NOT_FORMAL` fallback 诊断；原版 Task 4 仅能测量 S_link。没有把 public_reference 当 path-flow 真值。

原版 train State 的日期/训练范围与下文 holdout 不同，不直接计算跨表增益。
当前包 train 的原版评分计数与模板一致，missing=0；人工 fixture 中额外 horizon blanks 导致的反例未在当前官方 train 观察到。
证据：[数据校验](reports/official/download_verification.json)、[全文件指纹](reports/official/release_manifest.json)、[原版 State](reports/official/official_state_baselines_train.csv)、[原版执行](reports/official/official_diagnostic_runs.json)。

## 时间验证和泛化

每走廊 273 个 train 日期：profile 2030-06-01–07-15（45 天），fit 07-16–11-24（132 天），calibration 11-25–2031-01-09（46 天），test 01-10–02-28（50 天）。
State 标签来自公开 train eligible observation；特征来自 masked 当日，profile 来自更早日期。每走廊最多 300,000 fitting rows，种子 2026。
模型/阈值/物理设置只按 calibration 选择，最终模型再使用后续公开 train 重拟合；其训练拟合值不当作 holdout。
D12_I5_N 是架构开发走廊，其 test 已多次查看；其余走廊使用预先固定候选与参数。Queue refinement 又在同一时间组做校准，因此这是一轮固定时间验证，不能称为完整 nested cross-validation。
已按方向、R1/R2/R3、拥堵/非拥堵观测和可用边界覆盖分组；更多 rolling folds、完全跨走廊 transfer holdout 及真实事故标签验证尚待完成。

## State：十走廊时间留出

| panel | 历史 profile test | 选定模型 calibration | 选定模型 test |
| --- | --- | --- | --- |
| D12_I405_N | 0.808033 | 0.952752 | 0.955335 |
| D12_I405_S | 0.752518 | 0.936967 | 0.934236 |
| D12_I5_N | 0.576658 | 0.939746 | 0.939808 |
| D12_I5_S | 0.633672 | 0.931664 | 0.932476 |
| D7_I10_E | 0.698937 | 0.933615 | 0.936318 |
| D7_I10_W | 0.798172 | 0.950247 | 0.947425 |
| D7_I210_E | 0.878762 | 0.936805 | 0.937946 |
| D7_I210_W | 0.753999 | 0.924223 | 0.924132 |
| D7_I405_N | 0.495627 | 0.920657 | 0.920987 |
| D7_I405_S | 0.610480 | 0.928402 | 0.927400 |

按五个走廊族等权，历史 profile test = 0.700686，当前模型 = 0.935606，增益 +0.234921。
所有方向选中 V2 LightGBM 与 geometry bounds；400 iterations、63 leaves，4 threads。V2 在 total-flow 空间插值后转换为目标每车道值，增加 density/走廊流量尺度/多跳上下游信息。
各 mask regime 的 RMSE、拥堵/非拥堵分组见 [state_strata.csv](reports/official/state_strata.csv)。

D12 开发消融（同一日期分组）：V1 插值 test 0.889830，Kalman 0.848687，LightGBM 150/31 0.930848，CatBoost 150 0.926643。
400/63 V1 calibration 0.935305；V2 calibration 0.939690。联合 density projection 的 blend 0.5/1.0 calibration 为 0.938578/0.934725，本轮采用 blend 0，未为复杂度保留无收益投影。
最终 bounds 后 D12 test 0.939808。完整原始记录在 `artifacts/official_experiments/D12_I5_N` 与 `artifacts/state_physics/D12_I5_N`。

急降速代理暴露瓶颈：D12 五分钟降速 ≥30 km/h 的 647 个目标 State=0.404768，恢复 ≥15 km/h 的 1,083 个目标=0.497832，稳定变化 <5 km/h 的 712,397 个目标=0.950816。
这些是公开观测分组，不能命名为真实事故类型；详见 [scenario_D12_I5_N.json](reports/official/scenario_D12_I5_N.json)。后续优先研究拥堵前缘和突变，而非仅增加树数量。

## Queue：合法观测代理，真实 S_queue 未知

| panel | Persistence test proxy | 400/23 test proxy | 最终 test proxy |
| --- | --- | --- | --- |
| D12_I5_N | 0.225440 | 0.822997 | 0.824194 |
| D12_I5_S | 0.238417 | 0.747631 | 0.781689 |
| D7_I10_E | 0.325386 | 0.918080 | 0.932298 |
| D7_I10_W | 0.277531 | 0.769697 | 0.774242 |
| D7_I210_E | 0.382534 | 0.983416 | 0.981557 |
| D7_I210_W | 0.316930 | 0.895659 | 0.913964 |
| D7_I405_N | 0.308792 | 0.866200 | 0.880950 |
| D7_I405_S | 0.297483 | 0.817518 | 0.852420 |

按四个被评分走廊族等权，Persistence proxy=0.296564，最终 proxy=0.867664。
特征只取发布的 `[T−60,T)` history 和公开 topology/FD 几何；不读取 horizon、未来 ramp 或 buffer 后观测。
代理 horizon 标签来自 train eligible speed≤released cutoff，与 underlying-state 真标签不同。代理窗要求 horizon 非空，condition 按当前 contract v2 的同 link 至少两次 history queue 分类，跨日保留 360 min spacing。
比较 persistence、物理波、400/23、600/63 LightGBM；增加最近拥堵前缘距离、波到达时间、5/15/30 min 下游范围速度、走廊拥堵比例。
31 点阈值网格比较 global/condition threshold 和空预测时最大概率单元的公开非空先验；calibration 决定设置，test 仅报告。参数、阈值、Brier、onset/ongoing 和每窗 IoU 见 [完整摘要](reports/official/campaign_summary.json)。
各走廊 calibration/test 落差仍明显，且正式每 split 仅十窗；proxy 方差和噪声阈值差异是主要过拟合风险。

## Physics：约束权衡，不是正式成绩

初轮 V1 D12 calibration：未调整 State=0.930799、平均每日可观测 RMS=15.188 vehicles；强度 0.001 为 0.928000/10.261，强度 0.01 为 0.901778/9.854。
强约束的 State 损失明显。本轮另保留弱约束候选：在 calibration State 损失≤0.004 的范围内选最小可观测 RMS，比较 0、1e-5、1e-4、1e-3，再报告 test。它是 Pareto 研发启发式，不能称为综合最优。
缺失 ramp、latent/external boundary 不伪造为零；只修改 target cells。没有 50 vph floor，没有将 fallback 数值作为正式 Physics。

| panel | strength | State test | State change | RMS baseline veh | RMS candidate veh | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| D12_I405_N | 0.001000 | 0.954048 | -0.001287 | 11.092905 | 7.836726 | 0.028025 |
| D12_I405_S | 0.000000 | 0.934236 | 0.000000 | 未知 | 未知 | 0.000000 |
| D12_I5_N | 0.001000 | 0.936338 | -0.003470 | 14.312918 | 10.862186 | 0.073844 |
| D12_I5_S | 0.000000 | 0.932476 | 0.000000 | 未知 | 未知 | 0.000000 |
| D7_I10_E | 0.001000 | 0.935699 | -0.000620 | 10.653004 | 8.483930 | 0.033630 |
| D7_I10_W | 0.000000 | 0.947425 | 0.000000 | 未知 | 未知 | 0.000000 |
| D7_I210_E | 0.001000 | 0.935038 | -0.002907 | 10.538233 | 8.276224 | 0.138891 |
| D7_I210_W | 0.000000 | 0.924132 | 0.000000 | 未知 | 未知 | 0.000000 |
| D7_I405_N | 0.001000 | 0.917475 | -0.003512 | 15.623260 | 12.522234 | 0.130428 |
| D7_I405_S | 0.000000 | 0.927400 | 0.000000 | 未知 | 未知 | 0.000000 |

五个南向/西向走廊在严格边界条件下没有可观测转换，保留 strength=0；其余方向 test coverage 约 3%–14%。RMS 下降无法推出真实 S_physics 提升。正式边界流未公开，所有 Physics/Total 字段继续为 null。

## OD：独立稀疏反演和收敛核验

比较 prior、ridge λ=0.05/0.005、relative-prior regularization、count-weighted residual+KL prior；逐走廊 S_link、sensor holdout、prior L1 change、失败和耗时见 [odme_ablation.csv](reports/official/odme_ablation.csv)。
默认 λ=0.05；较弱正则更贴 counts，不能证明更准的隐藏 path flows。S_od、S_dev、S_attr、S_ODME 全部未知。
matrix 仅包含 counted links，未测 connector 不当作零。正 λ 的独立 block-pivot NNLS 使用 Woodbury，把自由系统缩至 measured-links 维度，并检查非负与 KKT。
16 个随机问题与独立 dense SciPy NNLS 最大 path-flow 差<1e-6。正式十走廊与原版 dense NNLS 的最大差为 3.1575036e-05，相对最大 reference flow 为 1.61e-08；详见 [parity](reports/official/od_all_panel_parity.json)。
D12 validation 同一 ridge 目标从旧独立稀疏迭代的 461.38 秒降至 0.86 秒，S_link 几乎相同；这是非隔离运行测量，不是与原版 dense NNLS 的测速比较。
例如 D12_I5_N 的 257 counts、4,278 paths，nullity 至少 4,021；高 S_link 不能识别全部 OD。

## 完整性、复现与资源

完整主文件 `artifacts/submission_best/submission.csv`：State 6,735,795、Queue 174,000、OD 70,708，总 6,980,503 rows。
严格 SQLite 合并检查 master key 全覆盖、自然键和 ID 的跨 chunk 重复、finite/nonnegative、binary queue 以及额外 rows；全部通过。
主文件 SHA-256：`43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1`。Task 3 自动使用同一 State，无额外行。
弱约束候选 `artifacts/submission_physics/submission.csv` 也包含相同完整行集；其独立合并验证和 SHA-256 见 `reports/official/submission_physics_verification.json`。Queue 与 OD 输出保持同一候选，State 的弱约束单独比较。
另一个环境重新推理 D12_I405_S private 的 107,416 rows，与原结果所有字段完全相同，最大 speed/flow 差为 0；不据此声称整个七百万行已独立重新训练验证。
Kalman/邻域向量化在三个官方日期的全部 features/base 上数值完全一致，单日约 0.8→0.37 秒；证据 [repeat](reports/official/repeat_inference.json)、[向量化](reports/official/state_vectorization_parity.json)。

本机 Windows 11、16 logical CPU，无 GPU；主 campaign 两个进程，每模型四线程，Queue refinement 一个进程。以下是并发环境下的实际 wall time，峰值为进程 lifetime，不是隔离单模型显存/内存：

| panel | 实验秒 | 最终拟合秒 | 预测秒 | 实验进程峰值 MiB |
| --- | --- | --- | --- | --- |
| D12_I405_N | 231.126631 | 92.734506 | 48.707085 | 955.617188 |
| D12_I405_S | 133.550475 | 56.351370 | 28.640754 | 688.000000 |
| D12_I5_N | 700.513921 | 335.447589 | 125.887040 | 1528.636719 |
| D12_I5_S | 318.454820 | 141.117297 | 52.194141 | 899.128906 |
| D7_I10_E | 494.086567 | 206.480912 | 77.959379 | 1025.539062 |
| D7_I10_W | 348.351104 | 126.027731 | 59.482775 | 946.445312 |
| D7_I210_E | 884.210458 | 373.671629 | 122.817722 | 1867.675781 |
| D7_I210_W | 327.749926 | 145.549618 | 48.432643 | 837.722656 |
| D7_I405_N | 754.294966 | 301.940250 | 93.178795 | 2101.527344 |
| D7_I405_S | 283.190253 | 115.610826 | 49.979213 | 943.757812 |

完整合并耗时 194.0 秒、进程峰值 266.9 MiB。依赖、随机种子、数据/模型 SHA、命令和测量协议均保存。
复现入口：`scripts/reproduce.ps1`；加 `-PhysicsCandidate` 同时生成弱约束候选。源码与报告在 Git，大数据和模型在本机 artifacts，未纳入仓库。

## 剩余瓶颈与下一轮

1. 取得正式 Kaggle 反馈后验证四任务的综合迁移性；当前未上传，真实 Total 未知。
2. State 重点为急降速/恢复及 R3 缺失区间；研究合法 occupancy/density 辅助观测、变化点和拥堵前缘，先消融再扩展。
3. Queue 重点为低分方向、噪声阈值与少窗口方差；额外 timestamp≤T 历史/ramp 仍未消融。禁止任何 timestamp>T 输入。
4. Physics 检查弱约束对正式分数的影响；局部 RMS 优化可能无法迁移到未观察边界，不能以本地不完整评分代替正式验证。
5. OD 继续检验 prior/regularization 稳定性；只有 path truth 的正式评测才能判断其准确率。
6. 更多 rolling folds、跨走廊 transfer 和独立全量复现；准备奖项注册、最终代码和技术 PDF。登记与最终包分别截至北京时间 10 月 26 日/11 月 11 日 19:59；当前 Kaggle API deadline 为 11 月 7 日 14:55，每日最多五次。团队信息及注册尚未完成，未发送邮件。

人工 fixture 的旧轮结果单独保存在 `reports/fixture/RESULTS_PRE_OFFICIAL.md`，不能作为比赛性能。
第三方仅阅读公开架构，无代码/数据/预测导入；公开资产和许可证见 `docs/EXTERNAL_ASSETS.md`。没有使用隐藏标签、旧泄露包或组织方私有资源。
目标尚未达到不会被当作技术结论；本轮交付建立了可以继续做合法实验和正式验证的完整系统。

