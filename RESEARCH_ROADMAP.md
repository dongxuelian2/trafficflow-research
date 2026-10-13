# 0.950+ 研究路线

目标为可复现的正式 Kaggle **0.95000+**；当前正式 **0.82940**。不将阶段性较低分数作为终点。

1. **A：强 State 比较基线与独立残差收益。** A14 的全标签 LGB 已超过原图组合。先在冬季和较大方向核对标签预算，再用强基线评估图网络的独立增益，区分速度收益与流量退化。A15 增益 0.001077 低于晋级门槛；不能依照旧的 300k 比较组继续宣称架构优势。GPU 限制 8GB，使用小批次。
2. **B：交通数字孪生。** 独立 CTM 已通过守恒与密度界检查。先评估模型在未知事故位置、失效比例、需求扰动下的鲁棒性；辨识 public train 动力学时只估计一般参数。官方 flux 是观测投影，模拟 CTM flux 与 observation-projected flux 要并列报告。修正外部连接后的本地残差依旧不等于正式 Physics。
3. **C：合法 T 观测与稳定事件决策。** 旧代理最早优先、未严格模拟 persistence safeguard；新 sampler 改用随机候选和公开筛选约束，但 latent truth 与真正 selector 仍不可见。C04–C08 发现合法 T 观测能改善部分候选，也会退化；小 selection 集的模型选择不稳定。扩大可得 calibration 窗口、增加拟合 seed、按唯一 origin 去重并按日估计不确定性，然后比较固定概率决策、长历史和传播特征。不要按 assessment 选模型。
4. **D：概率 OD。** D02 在独立模拟中有增益，但 K01 正式 OD-only 控制结果下降 0.06466，拒绝此候选。研究 broad prior distribution 失配、矩阵 rank/nullspace 和 interval coverage；增加反向 noisy-prior 生成与整体/局部 bias。S_link 接近 1 不构成路径恢复成功，不从这一次正式反馈细调 Bayes 参数。
5. **组合。** 每次正式上传前登记假设、基准 SHA、改变的任务与离线证据；每天最多两个新研究提交，低于官方五次限制。负增益候选保留，最佳版本不可覆盖。private split 最终成绩依然未知。
6. **封存与奖项。** 最终独立模拟 seed 在选择封存后启用；全量独立复跑、锁定依赖、模型 checksum、代码和 PDF 是奖项准备的一部分。奖项注册邮件及团队资料尚未核验；不自动发送邮件。

A01/A02 相对 300k LGB 在两个季节、三个 NN seed 改善，但 A14/A15 表明收益主要由标签预算解释。A05 跨 family 迁移失败，A13 有标签的逐方向图模型也无增益。A10 的 public-train density-increment 约束仅改善代理诊断，A11 强权重失败。下一轮先完成强基线控制与合法 Queue 输入实验，避免继续增加没有独立收益的结构。

A16–A20 进一步确认强 LGB 对照下的图增益小于 .002。C12–C14 的 occupancy 特征在 I5_N 两个季节存在正向事件 block 证据，但 I10_W 明显退化；下一步优先 45 天 calibration / 30 天 assessment、事件均衡损失和分方向专家的稳定选择，而不是统一增加该特征并上传。A19 以全标签重检 direct-density 回归；第一轮 joint-LightGBM 失败仍保留。

更长 Queue 评估 C15–C17 削弱了 occupancy 的独立增益，I10_W 仍退化；C18/C19 改用更多合法训练事件，但保留原 assessment spacing。A23b 的 State occupancy 增益仅 .000281，暂不追加复杂度。A24 与正式模型同类的 500k 预算为 .938184，full 为 .942340；A25 第二个 full seed 为 .942295。优先完成预算控制的多 seed / 多方向证据，再考虑一份保留原 Queue 与 OD 的 State–Physics 正式控制候选。

D05 未发现重复计数列或完全不被计数的路径。可识别性不足源于低秩线性组合，不能用列重复来解释；所有单独路径的线性 leverage 都远小于 1。K01 在未计数网络路段上的负载变化约 8.2%，仍无其真实负载可供评分。

当前优先级：500k/full 三种子增益已在 I5_N autumn 达到登记门槛，继续核查其他方向和冬季，再准备保留原 Queue/OD 的 State–Physics 控制候选。Queue 的训练事件预算比复杂事件损失更有收益，但 I10_W 的 at-T 模型持续退化，不能按一个方向全局替换。B02 学习密度增量并做硬锚同化，首先用强 State 基线与 target-pair 诊断筛选。官方 Physics 仍需受控正式测量。

源代码公告按用户最新指示由用户手动发布论坛；Notebook 仅为未发布草稿，不再自动发布。GitHub 推送授权继续有效。

B01 揭示 flux contract 对诊断影响极大：同一个观测 oracle 在实际 CTM flux 下模拟 Physics 0.340882，在我们按观测真值投影的 flux 下 0.995759。纯 CTM 守恒不能直接视为比赛评分信号；物理损失需结合状态误差和受控正式评测。

矩阵数值 rank 比测量行数更小：例如 D12_I5_N 92 / 4,278 paths，nullity 4,186。rank/n 是行空间维数比例，不是“某些具体路径已经可识别”的证明。区间是未做非负截断的高斯近似，不能冒充可靠 posterior coverage。

规则：当前公开 Kaggle 截止 2026-11-07 06:55 UTC，每日 5 次、团队最多 4 人。奖项最终评价还包含报告。源规则记录在 `data/published_rules_response.json`；需要持续核对 Evaluation 页面，不能用旧榜成绩比较当前重建数据。
