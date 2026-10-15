# Queue onset 阈值的滚动重训 C71–C73

正式最佳仍为 **0.82940**，目标 **0.95000+** 未达到。本轮为公开 TRAIN 观测标签代理，不是 latent Queue 或正式总分；没有新完整 submission、上传或平台发布。

## 为什么重训

C68–C70 的阈值相对固定 top-k 有局部改善，但尚未稳定超过原强参照。C71 改用两个更早的滚动切点，在 I5_S、I10_W 分别重新拟合强参照与 onset 分类器，检查此前结论对时间选择的敏感性。不能拿用较晚标签训练的旧模型比较较早时段，所以原强参照的训练配方也在各自 fit 日期内重训。

公开 train 覆盖 2030-06-01…2031-02-28，本轮没有获得更晚的新数据。下列日期在其他研究中部分出现过；这是预登记的新权重/滚动划分实验，仍属已公开、部分已查看的 development，不能称盲测、最终测试或 private。

| 切点 | fit（起始均为7/16） | 未用 checkpoint buffer | decision | selection | assessment |
|---|---|---|---|---|---|
| September | 至2030-09-15 | 9/16–9/30 | 10/1–10/30 | 10/31–11/14 | 11/15–12/14 |
| October | 至2030-10-31 | 11/1–11/15 | 11/16–12/15 | 12/16–12/30 | 12/31–2031-1/29 |

四组配置在任一新 assessment 前统一登记，没有根据前一组结果改变后续配置。fit 数据全部只在指定日期内；checkpoint 期不读值。模型封存→decision→决策封存→selection→选择封存→assessment，输出写入一次。

## 容量、输入与选择

每 case 拟合 plain/rich 两个强参照树（seed20261013、600树/63叶、lr.035、minchild60、lambda8），监督全部 eligible 六步标签。rich 增加既有 occupancy 和公开 condition，其他原合法因果特征不变。decision 按完整预测域校准并选择更强 arm，同分选 plain。

每个 arm 在 decision 之前另拟合两个 onset 分类器（seeds20261013/20261029、同600树/63叶，colsample.9、unit weights），只用 fit onset T+30 eligible 行。总计 **8 个强参照树、16 个 onset 树**；不重训 ranker、计数模型或神经网络。

| case | fit 窗口 | fit onset query | baseline 每 arm 监督行 | onset 每模型监督行 | decision 选 arm |
|---|---:|---:|---:|---:|---|
| I5_S September | 572 | 72 | 262518 | 5523 | plain |
| I5_S October | 622 | 122 | 285865 | 9367 | plain |
| I10_W September | 137 | 37 | 67385 | 3036 | plain |
| I10_W October | 223 | 58 | 109498 | 4768 | rich |

fit sampler 沿用 balanced-v2：每 condition 最多500个、全局30分钟间距、seed20261013。配额不会创造稀缺事件；I10_W 的 ongoing 只有100/165个。evaluation 每 condition 最多30个、全局360分钟间距；decision seed4133，selection/assessment seeds2909/3253/4001。跨 seed 的同物理 origin 去重，不能把重复抽中同一事件当新增独立样本。

选定 arm 的 onset 分数为两 seed 等权均值。threshold 在 decision 的31个 .05… .8 阈值×force开关中选择，优先不 force，再选更近 .5 的阈值；fixed-k 仍用1/2/4/8/16网格，仅作次要匹配解码器控制。所有角色的 ongoing 逐格复制本 case 的强参照，仅 onset 最后 T+30 可变。

完整六步全部 link 先预测，再由 evaluator 使用 future eligibility。输入只有预测起点以前的一小时公开历史、公开 masked T、静态网络及发布 condition；不读取未来 mask 来排名、强制非空或选位置。与 fixed-k 比较隔离了解码器效应；与 incumbent 比较还包含 onset 专用训练、训练人口及双 seed 集成差异，不能把它全归给阈值本身。

primary 仅在 incumbent/threshold 中由 decision 选择，同分保留 incumbent。selection 要求去重 condition 宏 gain≥.01 且7日 block 4000次 bootstrap 的95% CI下界>0，否则退回 incumbent。fixed-k 的后段较好也不反选。

| case | decision incumbent | decision threshold | 阈值 / force | fixed-k | decision→selection |
|---|---:|---:|---|---:|---|
| I5_S September | .810210 | .783017 | .30 / false | 4 | incumbent→incumbent |
| I5_S October | .797619 | .792064 | .10 / false | 4 | incumbent→incumbent |
| I10_W September | .872614 | .900392 | .50 / false | 1 | threshold→incumbent |
| I10_W October | .651519 | .693186 | .25 / false | 1 | threshold→incumbent |

两个 I10_W challenger 的 selection gain 和区间均为0，未通过门槛。稀疏 selection 事件不能提供有力的非零增益检验；不因后段出现较好点值改变选择。

## 后段观测代理证据

以下为物理 origin 去重、onset/ongoing 等权宏 IoU，不是正式 Queue，也不能按四任务权重外推 Total。

| case | incumbent / 已选策略 | fixed-k | threshold（次要） | threshold gain / 95% CI |
|---|---:|---:|---:|---|
| I5_S September | .785466 | .757240 | .795681 | +.010215 / [−.010753,.033341] |
| I5_S October | .779788 | .764845 | .801545 | +.021757 / [−.018702,.054445] |
| I10_W September | .789289 | .747622 | .814289 | +.025000 / [0,.093750] |
| I10_W October | .672051 | .700897 | .678461 | +.006410 / [−.057692,.066667] |

后段独立 onset origin 仅31/29/10/13个，数据量有限。所有已选策略仍是 incumbent，primary gain为0；统一 threshold 的四个点值均正，但没有一组达到预定 gain≥.01 且 CI 下界严格正。两种策略都未通过“至少两 case 有正区间、各 case 损失受控”的描述性门槛，**不晋级生产模型**。后段点值不用于重选。

C73 另作等 case/等 condition 的事后统计；同一 assessment 日期的两个走廊同步抽样，两个不重叠日历分别抽样，保留无事件日期。每项4000次抽样全部有效，区间由第二套逐次求和实现复核，最大差8.07e−17。它仅度量固定这些模型时的日历变异，不包括重训 seed 的不确定性，也不是正式赛事聚合；不改变 C71 primary 或晋级规则。

| C73 事后比较 | 等case宏IoU（左→右） | gain | 同步日历95% CI |
|---|---|---:|---|
| incumbent→已选策略 | .756648→.756648 | 0 | [0,0] |
| incumbent→统一threshold | .756648→.772494 | +.015846 | [−.000587,.037369] |
| incumbent→fixed-k | .756648→.742651 | −.013997 | [−.044371,.016446] |
| fixed-k→threshold | .742651→.772494 | +.029843 | [.003025,.058984] |

阈值相对匹配 fixed-k 的联合区间正，但相对强参照仍跨零。完整域预测大小也不是固定数：I5_S October 的阈值有一个10节点集合；两个 I10_W case 的阈值各有一个空集合。这些是冻结模型的诊断，不按后段标签修正。

## 资源、核验与复现

C71 全24模型实际均600树，CPU wall **488.97秒**，进程 lifetime peak **2520.88 MiB**；没有 GPU 训练。包含采样/特征/预测/评分，不作为隔离树吞吐基准。

C72 首次核验因检查器假定 `DatetimeIndex.asi8` 为纳秒而失败；本环境是 `datetime64[us, UTC]`，整数最小间隔1800000000实际为30分钟。失败注册、执行源码zip、错误与探针结果保留。C72b 改为直接与 `pd.Timedelta(minutes=spacing)` 比较，仍严格执行30/360分钟标准；未改变模型、输入、预测、选择、评分或数值容差。

新增三项科学测试 **3 passed / 2.54秒**：同日跨走廊相反误差在每次共享抽样中抵消、重复 event seed 去重；不同但重叠日历拒绝错误独立抽样；缺失及完全无可评分窗口的188次校准分数独立复算。另一个独立 smoke 的188次校准与阈值解码也通过。本轮未改生产 src；上一阶段134research/113CPU与7可选Torch skips是继承证据，不重复声称新运行。

C72b 完成 **32份缓存、2196次**新原始窗口重建，逐窗重建 plain/rich 共 **4392次**特征，与公开观测标签、future eligibility、选中事件的历史覆盖/condition/onset timing/persistence/spacing约束一致。检查全部24模型的容量、schema及实际监督行，**1504次**baseline校准分数、全部新阈值与k网格、选择封存、去重区间均复算通过；88次直接Booster调用。没有独立重训模型，也没有重新实现整个候选宇宙与采样RNG。

便携接口在每个非fit缓存的各condition首末窗口抽查，共 **436次**角色调用、**224次**额外目标乱序、**16次**未来输入拒绝，全部通过。I10_W October selection每seed只有一个ongoing窗口，所以首末去重后总调用数小于448；按实际样本数计数。旧builder保存但本轮模型不用的神经输入`x`没有独立重建，不把它算作已验证模型输入。所有采样期raw文件、选中masked T来源与静态网络CSV保存SHA；来源hash不等于独立复现整个采样宇宙。C72b wall **351.51秒**，C73 **3.05秒**。

新代码compileall通过。独立报告分别为`C72b_rolling_onset_threshold_verification.json`与`C73_rolling_onset_matched_summary.json`，失败C72保留。新manifest为`model_artifacts_checkpoint_rolling_onset_v1.json`：重新校验全部6453个旧artifact不变，新增196个、累计6649个，5份完整CSV及依赖锁不变；报告Git字节证据为`rolling_onset_report_bytes_v1.json`。最终模拟 seeds7907/8017/9109保持未生成，原完整提交保持冻结。

```powershell
.venv-research\Scripts\python.exe scripts/rolling_onset_decoder_experiment.py --output artifacts/research/C71_rolling_onset_threshold_refits
.venv-repro\Scripts\python.exe scripts/verify_rolling_onset_decoders.py --study reports/research/C71_rolling_onset_threshold_refits.json --output reports/research/C72b_rolling_onset_threshold_verification.json
.venv-repro\Scripts\python.exe scripts/summarize_rolling_onset_decoders.py --study reports/research/C71_rolling_onset_threshold_refits.json --audit reports/research/C72b_rolling_onset_threshold_verification.json --output reports/research/C73_rolling_onset_matched_summary.json
.venv-repro\Scripts\python.exe scripts/checkpoint_rolling_onset_decoders.py
```

原输出目录均写入一次。C71 的旧 Data builder 为复用历史缓存格式导入 Torch，实际只做 CPU 树训练；便携树策略及独立核验不需要 Torch。
