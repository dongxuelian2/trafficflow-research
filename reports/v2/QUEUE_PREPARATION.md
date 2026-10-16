# Queue 条件事件后验：准备与预注册

日期：2026-10-16。本文件保存启动前准备与假设记录。第一阶段 State 放行后，真实公开代理实验已完成并否决本模型，详见 [QUEUE_RESULTS.md](QUEUE_RESULTS.md)。没有新 Kaggle 或正式总分测量。

## 已审阅的负证据与范围

- `docs/V2_RESEARCH_DECISION.md`：H3 要求新增条件位置/规模信息，不能将同概率上的阈值收益称作新信息。
- `docs/PROBABILISTIC_ONSET_RESEARCH.md`、C74–C77：概率校准与精确独立 Bernoulli IoU 没有稳定超过已调阈值；非空的事后信号受到未来观测正例代理采样偏差影响。
- C55–C61：严格 fit/checkpoint/decision/selection/assessment 五阶段、两方向两季的 graph 主模型没有晋级；I10_W checkpoint onset 少，不能把重复 seed 当独立事件。
- C78–C84：固定树边际上的经验联合关联没有稳定增益，受控非前缀搜索在 decision 筛查停止。此次不扩大旧 copula、MC 样本或搜索预算。

I5_N 现有 C30/C33 是旧三阶段缓存。首轮改用现成 C55–C58 balanced-v2 的 I5_S/I10_W 两季四 case，保留真正独立的五阶段，避免把旧模型已选择过的 calibration 切片伪作新选择期。所有日期此前已用于开发，不是盲测。

## 数学对象与可证伪假设

拟合期事件轨迹记为 `j`，其观测位置摘要为 `B_j`。合法历史 `H` 是缓存已审计的 13 个时刻，公开条件 `E` 为 onset/ongoing。固定 8 个节点组上的速度、早晚变化、flow、occupancy、覆盖率和时刻摘要构成 58 维描述；节点组仅用于降维，节点顺序不解释为物理传播方向。

同条件库内，标准化历史距离的 k 近邻权重为

`w_j(H,E) ∝ 1[j in kNN(H,E)] exp(-(d_j-min d)/(2 max(median d,.05)))`。

由此 `p(B=b|H,E)=sum_{j:B_j=b}w_j`，`p(Z|H,E)=sum_j w_j p(Z|j)`。这是一个小型条件经验混合模型，不声称发明 kNN，也不声称识别容量下降的真实因果位置。与 C79 的重叠在于经验联合场景；区别在于本模型直接由 H,E 预测新的联合轨迹权重与边际概率，C79 保留原树边际。

模板概率为 `Phi((v_cut-v_observed)/sigma)`，`sigma=clip(MAD(delta_5min v)/(.67448975 sqrt(2)),1.5,8)` km/h，仅由拟合历史估计。该尺度混合真实交通变化和观测噪声，不能解释为已识别的纯传感器噪声。拟合缺失未来目标用合法历史速度回退，不将缺失标签当负类。共享模板加固定 .5 的时间 Gaussian copula 产生六步联合观测轨迹，边际保持该平滑概率。模型并未获得未知 latent 真值。

瓶颈标签取拟合期可观测软拥堵的持续质量最大节点。前沿范围只使用可观测拥堵和 sensor 图中可达关系；缺失中间路段不虚构连边。方向报告 upstream/stationary/downstream/unknown；onset 仅有末步事件，无法据此识别传播方向，保留 unknown。输出位置概率、方向概率、前沿均值及 q10/q50/q90；这是图不完整下的观测摘要。

## 四路对照与信息屏障

1. 冻结旧强树概率 + 原完整域阈值解码。
2. 完全相同旧概率 + 新联合场景期望 IoU 解码。
3. 新条件事件边际 + 相同原解码算法和 calibration 网格。
4. 新条件事件联合轨迹 + 新期望 IoU 解码。

新解码的固定候选预算相同：空集合、5 个阈值集合、7 个期望基数附近前缀、16 条抽样轨迹，最多 29 个候选，256 场景。旧概率使用 IID 场景，新模型使用条件联合场景。该四路设计区分条件边际和决策变化，但不能单独识别联合依赖中每个假设的收益。没有全局集合最优的声明。

所有动作覆盖完整公开 node×6 域。onset 前五步由已公布选窗约束置空。未来 eligibility、quality、speed 不进入预测接口；它们只作拟合目标缺失标记或动作完成后的评分域。缓存原代理 sampler 仍筛选未来可观测正例，其选窗人口不等同于未知官方 latent selector，本次没有解决该偏差。

## 冻结、统计门槛和复现

`QUEUE_CONDITIONAL_RUN.registered.json` 已写入。先 checkpoint 从 k={8,24} 选择 Brier 最小者；随后 decision 仅用原网格校准新事件的旧解码器；封存 calibration 后读 selection，selection 封存后读 assessment。原强树和原阈值全部冻结。两 MC seeds=20261016/20261029 检查决策稳定性，不是两份训练样本；无随机拟合优化。

primary 为事件+新解码。研究门槛是 condition/case 等权观测代理 IoU 增益≥.02、同步日期 7 日 block bootstrap 95% 下界>0、四个方向季节 case 均正、至少 30 个去重 onset。首轮不通过即停止扩大。结果不乘正式权重推算官方总分。所有失败写入新目录的 failure.json，禁止覆盖。

执行时保存代码 HEAD、完整源 ZIP 哈希、数据缓存与模型 SHA-256、seeds、wall time 和进程 peak RSS。State 第一阶段报告存在且显式通过 `--stage1-report` 后才可 `--execute`。输出只在独立 `reports/v2/QUEUE*` 目录。

验证：`python -m pytest tests/test_v2_event.py -q`，9 passed / 1.66s。包括小型联合分布逐集合标量穷举、empty/empty、公共 onset 约束、条件分库、缺失目标 poison、未来参数拒绝、14 槽未来输入拒绝、图方向置换、固定 origin seed、场景边际检查、四路 evaluator poison 不变、日期 origin 去重。首次测试有一处浮点概率和严格等号比较导致断言失败；改为数值容差后通过，未改变模型以迎合数据。

运行命令：

```powershell
.venv-repro/Scripts/python.exe scripts/v2/run_queue_experiments.py --execute --stage1-report reports/v2/<第一阶段实际报告>.json
```

当前判断：经验混合的概率构造与有限候选 IoU 风险计算数学成立；真实公开代理效果与跨季节一致性证据不足，须实际执行后才能接受或否决。latent Queue 真值和正式收益均不可由本轮公开代理确认。
