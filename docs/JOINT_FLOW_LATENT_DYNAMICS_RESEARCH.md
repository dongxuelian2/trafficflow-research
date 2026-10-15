# 连续流量监督与潜状态传播 C91–C94

正式成绩仍为 **0.82940**，0.95000+ 未达到。本轮实际训练 **12 个神经网络、4 个因果 State LightGBM**；两个 Queue 晋级策略仍选原强参照。以下均为公开 TRAIN 的本地证据，没有新正式提交或 Physics 成绩。

## 固定假设与比较

沿用 C87 的两个走廊 I5S_September / D12_I5_S、I10W_September / D7_I10_W。fit 为 2030-07-16…09-15，checkpoint 09-16…09-30，decision 10-01…10-30，selection 10-31…11-14，assessment 11-15…12-14。日期已用于此前开发，不称盲测、最终测试或 private split。

三臂共享旧图编码器初态、全部新增模块初态、每 seed 的 400×8 dense 预训练批次和 800×4 event 微调批次；optimizer seed 20261013/20261029，微调前重置 AdamW。lr .001、weight decay 1e−4、gradient clip 1。C87 的批次文件另复制并核对 RNG；本轮均为新增训练。

| 臂 | 共同输出 | 流量监督 | 未来潜状态 |
|---|---|---|---|
| direct_no_flow | 六步 speed/queue、六步 flow、当前 speed/flow 分析头 | 0；flow 分支保留外推 | 无 |
| direct_flow | 同上 | .1×（future flow L1 + current masked flow L1） | 无 |
| latent_flow | 同上 | 同 direct_flow | 六步 graph message + GRU，反馈预测 v/q，修正共同直接预测 |

全部保留原 warm/event speed+queue 损失，另加 .1×当前 masked speed L1。speed/25、flow/lanes/600 归一化。direct_no_flow 是本轮匹配控制，新增 current-speed 监督使它不再是 C87 的逐权重复现。三臂分配同样的参数，但 direct 不使用 GRU；相同更新数不等于相同 FLOPs，实测潜状态训练更慢。

这是学习的残差动态模型，没有 CTM、密度守恒损失、隐藏边界流估计或新物理约束。当前 State 只调用分析头，GRU 只作用于未来 forecast；它对当前 State 的影响来自共享权重的训练。

## 两套合法输入

Queue 使用发布契约允许的过去 12 槽 raw 历史及 masked T。State 分析使用**过去及当前全部 13 槽 masked 观测**，条件位也仅由 masked 历史计算。两次编码共享参数，未来/raw 当前标签只进入 loss/evaluator。不能将 Queue 的 raw 历史直接用于缺测 State 验证。

除事件窗口，selection seal 之后另评估后段 30 天、每 UTC 日全部 T=01:00…23:55 的 **276 个当前原点**。两走廊共 16,560 上下文；原始真值 eligible、当前两字段均缺失的单元进入重建统计，按 R1/R2/R3 计算 .54×speed RMSE 截断项 + .46×flow-per-lane RMSE 截断项。首小时未覆盖，模型只能看过去；这不是完整 Task 1 模板或正式 State。

强树只使用同样 masked 13 槽输入，129 个特征包括时序值/指示、最近值/观测年龄、公开几何及邻域最近值。每字段固定 400 棵、63 leaves、min child 60、lr .03、L2、lambda 1、四线程，不按后段选参数。它覆盖全部 62 日×276 fit State 原点：I5S **429658**、I10W **487992** 个特征行，独特当前标签多于 NN 抽到的标签。树与 NN 的差距不能全部解释为架构差距。

NN 在 200/400/600/800 微调步保存权重、Adam 状态与 checkpoint 预测，仍按 checkpoint 条件/窗口 Brier 选权重，**没有按 State 分数选权重**。每臂先平均两 seed，再固定 .75 原强树 + .25 NN，只在 decision 选 Queue 完整域阈值及非空策略。selection 要求 gain≥.01、7 日块 CI 下界>0，然后才读 assessment；没有按后段反选候选。

## 实际结果

Queue 为观测代理 IoU，origin 去重、两 condition 等权；后段独特事件原点 I5S 105、I10W 28。

| Queue 后段 | 原强参照 / primary | direct_no_flow | direct_flow | latent_flow |
|---|---:|---:|---:|---:|
| I5S | .785466 | .787215 | .751338 | .763234 |
| I10W | .789289 | .738131 | .802072 | .734693 |

I5S decision 保留 incumbent。I10W decision 选择 latent_flow（.905494 vs incumbent .872614），但 selection gain **−.048572**、95% CI **[−.119367,−.022222]**，拒绝晋级。I10W direct_flow 的后段正点值不能用于重新选择；跨走廊方向并不一致。

两走廊 Queue 同步日历比较中，direct_flow−direct_no_flow **+.014032**、CI **[−.029730,.055723]**，不支持稳定流量监督增益；latent_flow−direct_no_flow **−.013710**、CI **[−.026576,−.003032]**。所有区间事后未校正、条件于这两组已拟合模型，不作为新选择规则。

| 合法因果 State 子集 / regime 宏公式 | 历史外推 | 强因果树 | direct_no_flow | direct_flow | latent_flow |
|---|---:|---:|---:|---:|---:|
| I5S | .830759 | .873903 | .831437 | .831914 | .830411 |
| I10W | .904693 | .933223 | .901654 | .902971 | .904331 |

两个走廊同日历同步 7 日块重采样、4000 次，全部 replicate 另用标量公式重算。State 子集 direct_flow−direct_no_flow **+.000897**、CI **[.000802,.001042]**；direct_flow−强树 **−.036121**、CI **[−.038283,−.033444]**。潜状态相对 direct_flow **−.000072**、CI 跨零。区间是事后、未多重校正、条件于这些已拟合网络的描述，不含 optimizer seed 不确定性，不是晋级或正式指标。

R3 仍是弱点。I5S 的 126568 个后段 R3 单元上，强树 speed RMSE **4.9170 km/h**、flow **56.4751 veh/h/lane**，direct_flow 为 **6.3453 km/h / 83.4122 veh/h/lane**。I10W R3 共 101268 单元，强树 **1.4599 / 45.3293**，direct_flow **2.0054 / 81.9047**，单位同前。数量是相关的时间/空间单元，不能当作同样数量的独立事件。

额外按 eligible raw T 到未来 speed 变化≥20km/h 的急降速/恢复、teacher queue 分层评估未来 speed/flow。分层只在 evaluator 使用未来 teacher；不是已知事故标签，也不是 State 或 Physics。C94 保存 origin 去重后的 cell-pooled RMSE；与 Queue 的窗口宏 IoU 不能混加为 Total。

direct_flow 的 I5S 全体 future speed RMSE 为 **14.2069 km/h**，526 个急降速/902 个恢复单元分别 **59.4954/68.1064**；I10W 全体 **7.3819**，70 个急降速/140 个恢复单元 **44.5652/45.4650**。强烈变化仍是明确失败场景，且单元来自少量相关事件，未把模型微小平均误差变化当作解决拥堵波。

C94 实际曝光并集显示，NN 每 seed 的独特 current-speed teacher：I5S **85838/85004**，I10W **85997/86415**；对应强树为 **429658/487992**。direct_no_flow 未优化 current flow，其余臂 flow teacher 数同 speed。下一轮需要匹配这些真实标签集合和选择目标，才能分离人口与共享梯度效应。

## 核验、资源与限制

C91 worker **615.44 秒**，进程 working set 峰值 **2716.49 MiB**，CUDA allocated 峰值 **146.61 MiB**；不是系统 GPU 总显存。两走廊强树训练分别 49.04/65.08 秒。I5S 三臂每 seed 神经训练约 22–45 秒，潜状态更贵。

- C92 重建 124 fit 日包、34224 masked State 上下文、33480 合法 Queue 原点和 742 fit/checkpoint 事件；744 次 rawpast/future 污染检查通过。State 条件不依赖 raw history/teacher。原 raw-array 独立参考实现明确复用，新增 State/flow/current 标签另外核对。
- C93/C93b 核验 12 初态/批次/曝光、48 个权重/optimizer/checkpoint 预测快照、792 个 checkpoint 与 1896 个事件 member forward。316 新 evaluation 事件及全部 60 个 State 日期的 16560 上下文独立重构，99360 次 State member forward 全量重放。
- 强树全部 **917650** 个 fit 特征行和标签/坐标 SHA 重算；精确 SHA 使用原 canonical feature generator，并用独立 double 空间求和实现另查语义，最大差约 **6.32e−7**，小于预定 2e−6。没有独立重新优化四个树或十二个 NN。
- 完整决策网格 1128 次评分、门控/seals、56 新便携历史上的 336 Queue / 168 State 调用、168 次乱序、18 future 拒绝通过。旧 Queue 强树 feature 的 raw 证明明确复用，不扩大核验范围。
- 原 C93 sampled replay 通过后，补充 67.86 km/h 阈值回归发现 portable State 条件把 float32 speed 乘回 25 再与 float64 cut 比较，会与训练的 normalized cut 比较不一致。只修入口比较；C93b 对同一权重/数据全量再重放，未改模型、阈值或选择。原成功审计、失败回归日志、首次未命中边界 probe 及 CPU 可选依赖放置错误均保留。
- 当前全研究环境 **158 passed / 14.92s**，CPU **132 passed / 9 optional-Torch module skips / 11.41s**，compileall 通过。最初六个新 kernel 测试通过，边界是第七个新增测试。

公开 teacher 为带噪观测，float32 归一化后参与这些诊断，不能当作 latent truth。State 和 Queue 的输入域、采样人口、损失和输出用途不同。当前 State 弱结果还受 NN 独特 current 标签较少、权重按 Queue Brier 选择的影响；不能据此断言更充分训练的联合模型必然无效。

归档重新核验全部旧 **7247** artifact、5 份完整 CSV 和依赖锁；新增 **432** 个产物后累计 **7679**，公开源文件 **544** 个再核 SHA。C93b 核验耗时 **187.95 秒**，C94 总结 **3.89 秒**。无新完整提交、正式晋级或平台操作。

## 复现入口

需要相同官方数据与 C71/C74/C87 冻结元数据；所有 output 不可覆盖。

```powershell
.venv-research\Scripts\python.exe scripts/joint_dynamics_queue_experiment.py --output artifacts/research/C91_joint_flow_latent_dynamics
.venv-repro\Scripts\python.exe scripts/verify_joint_dynamics_data.py --study-folder artifacts/research/C91_joint_flow_latent_dynamics --output reports/research/C92_joint_dynamics_data_verification.json
.venv-research\Scripts\python.exe scripts/verify_joint_dynamics_models.py --output reports/research/C93b_joint_dynamics_model_verification.json
.venv-repro\Scripts\python.exe scripts/summarize_joint_dynamics.py --audit reports/research/C93b_joint_dynamics_model_verification.json --output reports/research/C94_joint_dynamics_matched_summary.json
.venv-repro\Scripts\python.exe scripts/checkpoint_joint_dynamics.py --audit reports/research/C93b_joint_dynamics_model_verification.json
```

审计当前绑定固定 C91/C92；新 output 复现训练要显式改相应输入绑定，旧 audit 不证明新训练。当前修正入口与训练输入一致，原 C91 预测数组不依赖旧 portable 条件实现。

下一步先分离**当前监督覆盖不足、checkpoint 选择目标、任务共享梯度**三个因素，使用相同已曝光标签的 State 专项控制和完整标签强树分别比较。不要扩大同一 GRU/flow-weight 网格，也不把小幅辅助损失改善当作综合成绩突破。原最佳、未开启最终模拟 seeds 7907/8017/9109 及仅本地研发策略保持。
