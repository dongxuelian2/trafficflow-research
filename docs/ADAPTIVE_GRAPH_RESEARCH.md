# 自适应图模型：独立研究与复现

神经训练的 grid 标签对同 link 的多个 station 取均值，评估保留全部 station truth 和各 station lane 数。它与 LGB 的逐 station fitting rows 并非相同损失权重；比较控制相同日期与可用真值范围，不宣称相同重复采样量。

读取 [STAEformer 原论文](https://arxiv.org/html/2308.10425v2)关于时空输入表示的分析，以及 [PyTorch 2.8 MultiheadAttention 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html)后，独立实现 FD 几何条件化、检测器可用标记和拓扑关系偏置的 State 网络。未读取第三方参赛源代码、模型权重或预测。论文的交通预测成绩不作为本项目证据。

12 个输入通道来自 masked State frame：速度、每车道流量、各自可用标记、早期 profile、时空插值、occupancy 及可用标记、日内 sin/cos。FD free speed / per-lane capacity / jam density 条件化隐藏层。四层时间卷积与有向图扩散采用 dilation 1/2/4/8，两个空间注意力层使用上游、下游和自身关系的可学习 float bias。没有固定 node-ID embedding 表。

这是离线 Task 1 重建，使用已发布的未来可见观测合法。Queue 预测继续使用独立的因果数据接口；不能将此离线网络的未来信息直接输入 Queue。训练真值只来自公开 train 的重建目标，输入不含 target truth。

图几何对照与注意力候选固定相同宽度、优化器、patch RNG、训练预算与输入。第三个候选加入 .01 observation-derived density-increment loss，只约束有可见锚点或 target 的相邻五分钟对；没有官方边界流。拟合均匀抽取 R1/R2/R3，使用按 patch 的速度/流量 RMSE 近似任务目标。它不等于正式 pooled scorer。

`scripts/adaptive_graph_experiment.py` 使用原图缓存和冻结的全标签 LGB 权重。checkpoint、架构与 ensemble alpha 都由 calibration 决定；选择封存后才载入 assessment。缓存保持原值，强 LGB comparator 在内存中重建。评估期已经用于其他架构开发，不能冒充未看过的最终测试。

`scripts/export_adaptive_state_model.py` 将封存选择导出为 `AdaptiveStateModel`，在完整 calibration 日期上独立构造输入并复跑。逐通道数组必须与缓存一致，State 和 density-increment 诊断必须与封存值一致。使用 `.venv-research` 环境；依赖锁、源代码快照和 SHA 保存于每个实验目录。模型可加载到 CPU 或现有 GPU。

晋级仍需跨季节、跨方向、多个拟合 seed 的强基线对照，以及受控正式 Kaggle 证据。calibration State、assessment State 与 density proxy 均不代表正式 Total 或正式 Physics。

A41 完成第三个 I5_N fit seed，A43 完成 I405_N 冬季；六组 attention-vs-geometry 的日均 CI 为正。A44 对 A33/A35/A41 三个各自原 cal 封存成员固定等权，比较同三个 full-label LGB 的等权预测，assessment .951168 vs .942742。所有成员的逐日指标重建；组合 wrapper/save/reload 另外只检查一个完整 assessment 日。种子共享 14 个日期，平均种子日分数后再 bootstrap，不把它们算成 42 个独立日。源文件 `state_ensemble.py` 要求同 profile/version、相同完整 key 顺序、有限非负预测，保存时移除 GPU runtime cache。

```powershell
.\.venv-research\Scripts\python.exe scripts/adaptive_seed_ensemble_experiment.py --study artifacts/research/A33_adaptive_attention_strong_autumn --study artifacts/research/A35_adaptive_attention_autumn_seed20261029 --study artifacts/research/A41_adaptive_attention_autumn_seed20261107 --output artifacts/research/A44_independent_repeat
```

当前组合只有一个季节的开发证据，且推断需要三个成员；它保留为独立开发 artifact，不覆盖单 seed 的生产 K03/K04 或正式最佳。

当前已完成 A33–A35 的两个季节、两个 NN/LGB fit seed 对照。原 cal winner 的三个部署文件各在 14 日 calibration 独立复跑通过。`assess_adaptive_ablation.py` 使用各架构原已封存的 checkpoint/alpha 评估模块贡献，不改原 winner。几何图本身有收益，注意力进一步改善；密度损失的独立 State 收益未稳定，冬季退化。A37 无目标历史标签的跨 family 迁移失败，不能据此部署全网共享模型。

从项目根目录使用锁定的研究环境复跑，每次指定新输出目录：

```powershell
.\.venv-research\Scripts\python.exe scripts/adaptive_graph_experiment.py --cache artifacts/research/A01_D12_I5_N_autumn_seed20261013 --baseline-file artifacts/research/A14_full_label_LGB_control/lightgbm_large_budget.pkl --output artifacts/research/A33_independent_repeat
.\.venv-research\Scripts\python.exe scripts/export_adaptive_state_model.py --directory artifacts/research/A33_independent_repeat
.\.venv-research\Scripts\python.exe scripts/assess_adaptive_ablation.py --directory artifacts/research/A33_independent_repeat
```

缓存目录名和 baseline 文件应以原报告 `registered.protocol` 为准；脚本会核验原 profile、目标顺序、输入数组和 checkpoint SHA。既有目录拒绝覆盖。原 checkpoint SHA 与部署文件 SHA 是不同层次的记录；CPU/GPU 数值容差不代表跨设备模型字节相同。

P01 生产准备保留首轮 frozen profile，对其后 228 日的全部 4,409,727 个 required targets 拟合全标签 LGB，构造同合法输入的 NN 缓存。P02 固定 attention / attention+density 两个开发支持的假设、24 epochs、alpha=.75；没有生产 calibration 或 checkpoint 选择。它使用全部公开 train，包括此前开发 assessment，故不报告生产 holdout 分数。

```powershell
.\.venv-research\Scripts\python.exe scripts/train_production_adaptive_state.py --prepared artifacts/research/P01_D12_I5_N_production_preparation --output artifacts/research/P02_independent_repeat
.\.venv-research\Scripts\python.exe scripts/build_adaptive_state_candidate.py --production artifacts/research/P02_independent_repeat --architecture adaptive_attention --output artifacts/research/K03_independent_repeat
.\.venv-research\Scripts\python.exe scripts/verify_controlled_candidate.py --candidate artifacts/research/K03_independent_repeat/submission.csv --allowed-task state --allowed-panel D12_I5_N
```

完整候选只替换 I5_N 的 State 数值；所有其他方向、Queue、OD 保持首轮基准。Task 3 由 State 派生，因此候选是 State–Physics 联合控制，不能宣称 Physics 不变。严格 merge 检查全任务覆盖，独立 streaming diff 额外限制任务、字段、方向及 master-key 身份。构建脚本不上传、不发布、不覆盖正式最佳文件。
