# 资料核查 — 2026-10-13

已阅读官方 BASELINES、SCORING_SPEC、RUN_LOCAL、SUBMISSION_SCHEMAS、DATA、MASK_SPEC、Task 3 modes 和实现。
最初通过搜索索引阅读赛事规则及完整 Foundational Rules。2026-10-13 又通过官方竞争者 API `/pages?pageName=rules` 取得并阅读当前完整赛事规则；只访问公开页面端点，没有访问 host settings。

## 来源

- [官方版本](https://github.com/jacky850/trafficflowbench-public/tree/c88cddf533bbf0afa4ff1fc6c031d760a08e7a31)
- [评分](https://github.com/jacky850/trafficflowbench-public/blob/c88cddf533bbf0afa4ff1fc6c031d760a08e7a31/docs/SCORING_SPEC.md)
- [赛事规则](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/rules)
- [奖项及时间](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview/evaluation)
- [公开榜](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/leaderboard)
- [第三方架构](https://github.com/MAYANKSHARMA01010/2026-ieee-big-data-traffic-flow-bench)

## 比赛要点

每人单账号、单队，最多四人；禁止队外私下共享。公开外部工具须合法并披露依赖。只允许通过已发布数据建模预测，不能获取隐藏资产。数据于 2026-09-11 重建，奖项只认可重开提交后的版本。
奖项需要邮件注册、最终代码/PDF 和复现审核；登记截至 10 月 25 日 23:59 AoE，最终包截至 11 月 10 日 23:59 AoE，分别对应北京时间 10 月 26 日/11 月 11 日 19:59。
当前公开 API 核验的 Kaggle deadline 为 `2026-11-07T06:55:00Z`，即北京时间 11 月 7 日 14:55；daily cap=5，team size=4，submissionsDisabled=false。Overview 日期快照写 11 月 6 日；精确设置以 [官方公开 API](https://www.kaggle.com/api/v1/competitions/get/2026-ieee-big-data-traffic-flow-bench) 为准。merger/new entrant deadline 未返回，final selections 数仍未知，原始核验保存在 `reports/official/competition_metadata.json`。
官方奖项流程从 private 前十合资格队中审核，报告也影响最终排序。索引中的榜首为 0.92406；这是参考快照，不保证当前即时榜首。

## 不能混淆的指标

真实 queue labels 所有 split 均未公开；Task 3 正式边界流未公开；OD 真值未公开，只有 S_link 可验证。
官方公开 reference path-flow 文件来自本地求解，不是 path truth。
官方 BASELINES 的 queue 参考为 0.3017，而 README 写 persistence 0.2518，无法仅凭当前资料消除差异。

## 代码与规范的不一致

固定版本 Task 1 scorer 从所有 eligible 双空值选 truth targets，而 MASK_SPEC 指定模板；这会包含 Task 2 额外隔离空值。
fixture 专门制造额外 horizon blanks，原版 scorer 的计数与模板不同。当前官方包十走廊 train 的原版 baseline 缺失预测数为零，计数与模板一致，未观察到该人工反例在官方 train 上发生。
Task 4 builder 的 `baseline_submission` 用 train counts，`public_reference` 用当前 split counts；两者都正则到当前 split prior。不能拿 reference 给自己评出 S_od=1。

## 第三方架构审查

仅阅读公开 README。它声称使用空间/时间插值、LightGBM、波传播、FD/LWR 和 bounded OD solver，分项成绩未经独立证实。未获取其源代码、数据、submission 或 prediction。
独立实现加入严格日期隔离、观测年龄、拓扑 latent-link traversal、Kalman/RTS 消融、稀疏 measured-link OD operator、多先验对比、coverage-aware 守恒诊断和提交完整性拒绝机制。
不采用它提出的 flow floor 来改善评分，不将其声称成绩作为本项目结果。

## 当前包和网页文字的差异

当前数据包 contract v2.0.0 用“同 link 至少两次 history queue”区分 ongoing，搜索索引中的 overview 仍写“onset 全 history 无 queue、ongoing 在 T 有 queue”。代理窗口遵循当前公开包契约，并记录 condition，不把噪声观测代理当真实标签。
当前规则允许 Task 2 使用 timestamp <= T 的其他已发布历史（包括过去的 masked mainline/ramp 和完整 train）；禁止任何 timestamp > T 的输入。当前模型只使用发布的 60 分钟 window_history，尚未测量额外历史/ramp 对预测的增益。
