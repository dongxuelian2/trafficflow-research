# 0.950+ 研究路线

目标为可复现的正式 Kaggle **0.95000+**；当前正式 **0.82940**。不将阶段性较低分数作为终点。

1. **A：图时空联合状态估计。** 在注册的三个 rolling folds 重训相同的 LightGBM；实现缺失感知图时空残差网络，分别比较原始观测、occupancy、扩展时间上下文和残差 ensemble。优先分析 R3、骤降与恢复，再做跨 family transfer 和多 seed。GPU 限制 8GB，使用小批次。
2. **B：交通数字孪生。** 独立 CTM 已通过守恒与密度界检查。先评估模型在未知事故位置、失效比例、需求扰动下的鲁棒性；辨识 public train 动力学时只估计一般参数。官方 flux 是观测投影，模拟 CTM flux 与 observation-projected flux 要并列报告。修正外部连接后的本地残差依旧不等于正式 Physics。
3. **C：事件 Queue。** 当前代理按时间最早优先、未严格模拟 persistence safeguard，并忽略未知未来标签；这会改变抽样分布。新 sampler 采用随机候选、条件配额、间隔、覆盖和 persistence 筛选，但公开资料未提供真正的选择器或 latent truth。先比较旧采样/新采样/独立事件模型，再探索合法 ≤T 的长历史、ramp、概率校准和事件传播。
4. **D：概率 OD。** D02 在已知真值独立模拟中测试分层 prior、路径噪声、整体偏差。研究矩阵 rank/nullspace 和近似 posterior intervals，检查失配时的覆盖。S_link 接近 1 不构成路径恢复成功。优先一次 OD-only 正式控制实验。
5. **组合。** 每次正式上传前登记假设、基准 SHA、改变的任务与离线证据；每天最多两个新研究提交，低于官方五次限制。负增益候选保留，最佳版本不可覆盖。private split 最终成绩依然未知。
6. **封存与奖项。** 最终独立模拟 seed 在选择封存后启用；全量独立复跑、锁定依赖、模型 checksum、代码和 PDF 是奖项准备的一部分。奖项注册邮件及团队资料尚未核验；不自动发送邮件。

规则：当前公开 Kaggle 截止 2026-11-07 06:55 UTC，每日 5 次、团队最多 4 人。奖项最终评价还包含报告。源规则记录在 `data/published_rules_response.json`；需要持续核对 Evaluation 页面，不能用旧榜成绩比较当前重建数据。
