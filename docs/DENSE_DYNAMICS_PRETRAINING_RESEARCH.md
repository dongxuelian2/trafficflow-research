# 全时段动态预训练的匹配对照 C87–C90

正式 Kaggle 最佳仍为 **0.82940**，0.95000+ 未达到。本阶段新增 12 个网络训练，两个走廊的已选策略均保留原强参照，没有生成或上传新的完整 submission。

## 数据和设计

采用 C71 的 I5S_September / D12_I5_S 和 I10W_September / D7_I10_W。两者日期相同：公开 TRAIN 2030-07-16…09-15 拟合，09-16…09-30 选择权重，10-01…10-30 选择候选及阈值，10-31…11-14 进行晋级检验，11-15…12-14 做后段描述。各阶段依次封存，读取时记录前一 seal SHA。这些公开日期此前已经查看，不能称为盲测、最终测试或 private split。

每走廊 62 个拟合日、16,740 个五分钟原点；仅过去 12 步 quality 覆盖决定资格。输入为 13 槽、9 通道：速度/流量/occupancy 及观测指示、时间正余弦和条件。过去 12 槽来自合法公开历史，T 槽仅用公开 masked 观测。未来 speed、queue proxy、quality 只进监督或评分。通道与旧神经输入逐窗一致，future/raw-T 扰动不能改变当前输入。

保持 `CausalGraphQueue(hidden=32,use_graph=True)`，输出六步速度与 queue logits。**流量是输入，没有 flow 预测头或新物理损失**。两个 optimizer seed 为 20261013、20261029；每 seed 三臂共享相同初始权重：

| 训练臂 | 预训练 | 共同事件微调 |
|---|---|---|
| scratch | 无 | 800 步、batch 4 |
| event_warm | fit 事件按 onset/ongoing 平衡，400 步、batch 8 | 同 seed 使用完全相同的微调批次 |
| dense_warm | 全部合法五分钟 fit 原点均匀采样，400 步、batch 8 | 同上 |

两 warm 臂使用相同全六步 weighted BCE（positive weight 4）+.1 归一化速度 L1。微调均恢复原事件损失：onset 前五步不计分类，速度覆盖六步。每臂微调前重置 AdamW，lr .001、weight decay 1e−4、gradient clip 1。event_warm 控制额外步数及目标；dense_warm 同时改变人口与采样分布，不称为纯架构收益。

在 200/400/600/800 微调步保存权重、optimizer 和 checkpoint 预测，按独立 checkpoint 条件/窗口等权 Brier 选择。仅做 `sigmoid(logit-log(4))` 解析修正，没有拟合 Platt；不保证是校准后验。checkpoint 仅 I5S 24、I10W 9 窗口，选择不确定性不可忽略。

每臂先平均两个冻结网络，再固定 .75 强树 + .25 NN；只在 decision 选择完整域阈值/非空策略，不搜索混合权重。incumbent 沿用 C71 强参照及设置。单 seed 套用 arm-mean 冻结阈值，仅诊断，未单独调参或参与 primary 选择。

## 结果和负面证据

下表是 origin 去重、onset/ongoing 等权的**公开观测代理 IoU**，不是正式 Queue。后段独特原点数为 I5S 105、I10W 28；采样 seed 重复原点不当成独立事件。

| 走廊 / 后段 | 强参照 | scratch blend | event_warm blend | dense_warm blend | 已选策略 |
|---|---:|---:|---:|---:|---|
| I5S | .785466 | .775668 | .770797 | .792565 | 强参照 |
| I10W | .789289 | .735999 | .739262 | .735791 | 强参照 |

I5S decision 均未超过强参照。I10W decision 选 scratch（.905097 vs .872614），但随后 selection gain **−.089910**、7 日块 95% CI **[−.233987,−.034722]**，拒绝晋级。未按已看后段反选 I5S dense_warm。

| 后段匹配比较 | gain | 7 日块 95% CI |
|---|---:|---|
| I5S dense − scratch | +.016897 | [.003667,.031985] |
| I5S dense − event_warm | +.021768 | [.007221,.039003] |
| I10W dense − scratch | −.000208 | [−.004164,.002413] |
| I10W dense − event_warm | −.003472 | [−.010211,.002364] |
| 两走廊 dense − event_warm | +.009148 | [.000189,.018831] |
| 两走廊 dense − 强参照 | −.023199 | [−.070007,.027711] |

联合区间同步抽同一日历的日期块，case/condition 等权、保留无事件日期，4000 次；所有保留 replicate 用标量实现再算。这是事后、多重比较未校正、条件于已拟合模型的描述，不含训练 seed 不确定性，不是独立走廊复制或新晋级标准。第二 seed 的 dense−event 联合区间跨零（−.001193…+.021074）。

| 纯 NN 双 seed 均值 / 后段 | scratch | event_warm | dense_warm |
|---|---:|---:|---:|
| I5S corrected-score Brier | .021808 | .018849 | .019211 |
| I10W corrected-score Brier | .010982 | .010245 | .009394 |
| I5S 速度 forecast RMSE，km/h | 10.926572 | 10.842307 | 10.832744 |
| I10W 速度 forecast RMSE，km/h | 5.200442 | 5.025862 | 5.167036 |

Brier 域按事件分类损失，速度误差按六步 eligible 观测 teacher；逐窗再条件等权。它们不是 latent traffic、Task 1 State 或 Task 3 Physics。I10W Brier 更低而 IoU 更低，不能推断多任务增益。

## 曝光量、资源和核验

每个 warm 臂每 seed 抽 3200 上下文。dense 独特上下文为 2911/2928；event 在 I5S 为 547/544、I10W 137/137。实际独特速度 teacher 单元 dense/event 比为 I5S **3.47–3.57**、I10W **13.84–14.14**，低于完整人口理论上限。总监督 occurrence 接近，但 I10W dense 正例仅 7961/8607 次、event 为 62668/58768 次。重叠 teacher 和相邻原点不当独立标签或新增稀有事件。

C87 整个 worker **188.04 秒**，进程 working set 峰值 **1875.50 MiB**，RTX 4060 Laptop CUDA allocated 峰值 **102.80 MiB**；后者不是 GPU 系统总显存。各臂实际耗时、曝光、批次和选择权重在原报告及 C90。

- C88：124 个 fit 日包、33,480 原点、742 个 fit/checkpoint 神经事件逐项重建，372 次实际 future/raw-T poison。另一实现重算 quality/数组/窗口；图几何仍按既有公开网络配方重建，不称独立几何算法。
- C89b：共同初始化、全部 RNG 批次和曝光重算；48 个权重/AdamW/checkpoint 快照核对，792 次 checkpoint 与 1896 次 evaluation 神经 forward，概率/归一化速度最大差均 **0**。316 个新增 evaluation raw 窗口重建，1128 次决策网格评分及全部 gate/seals 重放。
- 便携入口：56 个新历史窗口、336 角色调用、168 额外 target 乱序、12 future 输入拒绝。含角色别名及重复事件，不当独立事件数。
- 原强树 feature 证明明确复用并重核 SHA；没有独立重训树或重新优化全部 12 个网络。保存初态/批次/状态/源码支持复现，但 forward 核验不等于重训复现。
- 初次 C89 在便携入口漏 `released_origin_rows` 导入，源码 ZIP 和 partial 保留。终端原 traceback 在工具截断后不可得，未伪造；补 import 后另建 C89b 完成，模型/阈值/选择/容差未改。
- 新增 4 个 kernel 测试通过；全研究环境 **151 passed / 13.39s**，CPU **129 passed / 8 optional-Torch module skips / 10.31s**，compileall 通过。

清单重新核验旧 **6877** artifact、5 份完整 CSV 和依赖锁；加入 **370** 个新文件后累计 **7247**，另复核 510 个公开源文件。未开启最终模拟 seeds 7907/8017/9109，无平台操作。

## 复现及后续

仓库根目录、相同官方数据及 C71/C74/C85 元数据准备好后：

```powershell
.venv-research\Scripts\python.exe scripts/dense_pretrain_queue_experiment.py --output artifacts/research/C87_dense_dynamics_pretraining
.venv-repro\Scripts\python.exe scripts/verify_dense_pretrain_data.py --study-folder artifacts/research/C87_dense_dynamics_pretraining --output reports/research/C88_dense_pretraining_data_verification.json
.venv-research\Scripts\python.exe scripts/verify_dense_pretrain_models.py --output reports/research/C89b_dense_pretraining_model_verification.json
.venv-repro\Scripts\python.exe scripts/summarize_dense_pretraining.py --audit reports/research/C89b_dense_pretraining_model_verification.json --output reports/research/C90_dense_pretraining_matched_summary.json
```

产物和报告不可覆盖。审计/summary 当前绑定固定 C87/C88；新 output 的复现训练必须显式更新相应输入绑定，旧 audit 不证明新训练。checkpoint 脚本只归档此次证据，不训练或选模型。

继续保留强参照，不扩大已查看日期的阈值/混合网格。下一步在相同 dense 数据和匹配计算预算下，研究连续速度/流量联合预测及缺失感知潜状态传播，以合法 State 重建和事件 forecast 双重检验表征价值；独立收益确认后再扩滚动时间/走廊。物理诊断仍不能冒充正式 Physics。
