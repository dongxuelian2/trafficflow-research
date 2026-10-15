# State 专项、权重选择与实际标签人口 C95–C97

正式 Kaggle 最佳仍为 **0.82940**，0.95000+ 未达到。本轮新增 **4 个 State-only 网络、24 个匹配树字段模型**，另重选 4 个已有 direct_flow 网络的快照；重选不计新增训练。所有新候选在 decision 筛查均未超过全标签因果强树，**没有为新候选读取新的 selection、assessment 或 late-transfer 标签/数组**，没有新完整提交或正式成绩。

## 固定比较

沿用 I5S_September / D12_I5_S 与 I10W_September / D7_I10_W：fit 2030-07-16…09-15，checkpoint 09-16…09-30，decision 10-01…10-30。之后的 selection、assessment 和固定 2031-01-15…02-13 late-transfer 仅在门槛通过时读取，本轮未触发。日期属于已查看过的公开开发数据，不称为盲测或 private split。

State 只能看过去及当前全部 masked 的 13 槽观测；条件位也仅由 masked 历史计算。当前 raw teacher 仅监督/评分，未来 Queue/speed/flow 不进入 State-only loss 或预测接口。保留原 C91 初态、每 seed 400×8 warm / 800×4 event 更新、AdamW .001、weight decay 1e−4、clip 1、微调前 optimizer reset。只有 Queue/future 辅助损失移除，当前损失仍为 .1×speed L1 + .1×flow-per-lane L1。

选择因素分开比较：原 joint Queue-Brier 快照；同一小事件 CP 的当前 State-MSE 快照；完整 15 日 State CP 的快照。共同 State 选择准则为 .54×归一化 speed MSE + .46×归一化 flow MSE；它是 pooled-cell 准则，并非官方逐 regime RMSE 宏平均。新 State-only 的主版本使用完整 CP，事件 CP 版本只作诊断。所有预测 speed/25、flow/lanes/600，输出当前两字段。

匹配树使用 NN 实际抽到的相同独特 current teacher：I5S 两 seed **85838/85004**，I10W **85997/86415**；全标签强树则为 **429658/487992**。每个 NN 均有 6400 次带当前标签的上下文抽取，没有零监督批次解释。343254 是四组种子内行数之和，跨 seed 的重复标签不能当独立样本。

| 固定树比较 | 唯一标签 | 权重 | 目标 |
|---|---|---|---|
| full_tree，原强参照 | 全 fit 当前缺测标签 | 每行等权 | L2 |
| matched_l2 | NN 实际独特标签并集 | 每行等权 | L2 |
| matched_l2_weighted | 同上 | NN nominal update 权重 | L2 |
| matched_l1_weighted | 同上 | 同上 | L1 |

129 个因果特征和 400 树、63 leaves、min child 60、lr .03、lambda 1 等设置不变。每 seed 各训练两字段三版本，共 24 个新字段模型；主版本平均两 seed，单 seed 比较另报，以区分覆盖与 ensemble 效应。

每标签 nominal 权重为所有实际抽取的 `1/(batch_size×当窗该字段缺测标签数)` 之和，按字段归一化到平均 1。它复现静态累计损失测度，不等价于 NN 的 optimizer 顺序、reset、clipping 或 LGB 正则化。事件重采样造成很不均匀的风险：I10W 权重中位约 .26，99% 分位约 20，最大约 78–94；不能仅按标签数称为数据匹配。

## 决策期结果及停止规则

以下为两字段均缺测、raw eligible、每日 01:00…23:55 共 276 个当前原点的**公开因果 State 子集公式**；首小时排除，不是完整 Task 1 或正式 State。

| Decision 子集 | I5S | I10W | 等 case 宏平均 |
|---|---:|---:|---:|
| 全标签 L2 强树 | .865029 | .930189 | .897609 |
| joint，原 Queue CP | .823286 | .900926 | .862106 |
| joint，完整 State CP | .824272 | .901663 | .862967 |
| State-only L1，完整 State CP | .823259 | .902676 | .862967 |
| 匹配标签 L2 等权树 | .858401 | .928151 | .893276 |
| 匹配标签 L2 nominal-weight 树 | .852740 | .918797 | .885769 |
| 匹配标签 L1 nominal-weight 树 | .831052 | .908723 | .869888 |

唯一全局候选族必须比 full_tree gain≥.002、两个 case gain≥0、各 R3 下降≤.002、两个 15 日半段 gain 均>0，才进入后续独立 selection。最好的新族 matched_l2 仍 **−.004333**，两个 case **−.006628/−.002038**、两个半段也都负，按预登记规则停止。原因果参照不等于正式最佳整任务 State 模型；没有覆盖原冻结提交。

| 事后诊断差异 | gain | 同步 7 日块 95% CI |
|---|---:|---|
| joint：完整 State CP − 原 Queue CP | +.000861 | [.000811,.000931] |
| State-only − joint，均完整 State CP | +.00000013 | [−.000086,.000190] |
| 匹配 L2 等权 − 全标签 L2 | −.004333 | [−.005191,−.003362] |
| 匹配 L2 nominal − 匹配 L2 等权 | −.007507 | [−.007875,−.007127] |
| 匹配 L1 nominal − 匹配 L2 nominal | −.015881 | [−.017425,−.013430] |
| State-only L1 − 匹配 L1 nominal 树 | −.006920 | [−.007675,−.006156] |

C97 共 12 个均值、8 个单 seed 比较，每个保留的 4000 次日期块 replicate 均另用标量公式检查。全部区间为**用过的 decision 数据上的事后描述、未多重校正、条件于拟合模型**，不能当新的晋级检验或训练 seed 不确定性。覆盖对照的双 seed 均值还包含 ensemble 差异，单 seed 结果单列；末项包括特征工程和优化器差异，不是纯架构结论。

这条诊断链提示，应优先直接测试 NN 的平方误差及风险权重，而不是假定更多数据或移除辅助任务即可解决问题。匹配树中的 L1/L2 效应不能外推成 NN 已取得的增益。32 个固定 FIT 梯度对中 11 个 cosine<0、均值+.1400，但 State-only 整体近乎无增益，也不能仅凭局部梯度冲突认定因果机制。

## 独立复现与精度修正

C96b 从不同的上下文/权重组装及解析损失代码独立重训**预先固定的 I5S 第一 seed 网络**，4 个快照逻辑 tensor 最大差 **0**；同一 seed 的 weighted-L2 两字段树独立重训，booster 文本完全一致。其余 3 个 NN / 22 个字段树没有独立重新优化，但全部权重、标签、特征/权重 SHA、选择和预测均复核。首次失败的 C96 也有部分复训产物，均保留，不冒充整体成功。

完整 CP **8280** 上下文、事件 CP **33**、新的 decision **16560** 上下文及 60 日 teacher 逐项重建。16 个新增 NN 快照与 16 个旧 joint CP 记录分别以原模式重放 **133008** 个成员上下文，再以完整 FP32 重放同样数量。343254 个匹配标签行、32 个梯度对、全部 gate/seals 复核。

首次 C96 在单窗便携预测的 1e−5 绝对检查上失败。输入逐元素相同，诊断定位 cuDNN 默认 TF32 的 batch-size 舍入：失败窗口归一化单窗/批量差 speed **1.38e−5**、flow **1.78e−5**；重复同一输入填充 48 个 batch 与原 batch 首行差 0，没有发现该窗口跨样本数值依赖。完整 FP32 后对应差 **0 / 5.96e−8**。

修正只让便携接口局部关闭 TF32，并恢复外部 backend 设置。**没有放宽原 1e−5 检查、重训候选、改旧预测数组或改选择**。C96b 分别核验原 TF32 研究 profile（最大重放差 0）和完整 FP32 profile（便携单窗最大差 **4.77e−7**），后者的新日期数组另存。全部 checkpoint 选择及全局拒绝结论一致。

所有已检查 decision 模型/上下文的两 profile 最大归一化变化为 speed **4.3869e−5**、flow **9.6321e−5**，由 RMSE 三角不等式与截断项给出的该子集 State 点值变化上界 **6.7997e−5**。实际等 case State-only 平均只变约 **−1.07e−8**。此界不适用于未知隐藏数据或正式 Total，不把修正称为模型性能增益。

便携核验覆盖 12 个历史窗口、108 个角色调用、36 个乱序、36 个时区等价调用和 18 次 future 拒绝。纯 CPU、未安装 Torch 的环境另对一个真实 matched_l2 导出窗口运行，输出与原保存值差 0。164 research / 16.45s、136 CPU / 13.02s、10 optional-Torch module skips、compileall 通过。

## 资源、归档与下一步

C95 worker **578.29 秒**，进程 working-set 峰值 **2943.68 MiB**，CUDA allocated 峰值 **167.44 MiB**，不是系统总显存。C96b 含独立复训、双 profile 推断 **386.42 秒**，C97 **10.30 秒**。新候选拟合与验证复训分开计数。

冻结清单重核旧 **7679** 产物、5 份完整 CSV 和依赖锁不变；新增 **277** 后累计 **7956**，442 个公开源文件重核。新的 selection / assessment / late-transfer 分数均为**未知**，最终模拟 seeds 7907/8017/9109 仍未开启，平台操作保持暂停。

```powershell
.venv-research\Scripts\python.exe scripts/state_specialization_controls.py --output artifacts/research/C95_state_specialization_controls
.venv-research\Scripts\python.exe scripts/diagnose_state_control_precision.py --output reports/research/C96_state_precision_diagnostic.json
.venv-research\Scripts\python.exe scripts/verify_state_specialization_controls.py --output reports/research/C96b_state_controls_verification.json
.venv-repro\Scripts\python.exe scripts/summarize_state_specialization.py --audit reports/research/C96b_state_controls_verification.json --output reports/research/C97_state_controls_factor_summary.json
```

需相同官方数据及 C91/C93b 冻结元数据；output 不覆盖。精确原 TF32 研究数字应使用 C95 start-source ZIP；当前 portable 实现是已核验的完整 FP32 profile。审计脚本固定绑定 C95，新的复现 output 需显式更新输入绑定，不能把旧证明用于新模型。原训练权重与两种 profile 数组都保留。

下一轮预先比较 **L1/window、L2/window、L2/独特标签风险校正**，使用同初态、原批次与精确标签并集。风险校正应抵消 context 重复频率及每窗逆缺测数，同时保持累计权重总量；继续与已有 weighted/unweighted L2 树对照。需实际验证 NN，而不是根据树的负面结果承诺收益；正式 State/Physics/Total 和 0.950+ 都仍未验证。
