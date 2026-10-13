# 资料核查 — 2026-10-13

已阅读官方 BASELINES、SCORING_SPEC、RUN_LOCAL、SUBMISSION_SCHEMAS、DATA、MASK_SPEC、Task 3 modes 和实现。
Kaggle 动态 open 无正文，浏览器也未成功；通过搜索索引取得并阅读赛事规则及完整 Foundational Rules。索引可能滞后，提交前仍须核对在线 settings。

## 来源

- [官方版本](https://github.com/jacky850/trafficflowbench-public/tree/c88cddf533bbf0afa4ff1fc6c031d760a08e7a31)
- [评分](https://github.com/jacky850/trafficflowbench-public/blob/c88cddf533bbf0afa4ff1fc6c031d760a08e7a31/docs/SCORING_SPEC.md)
- [赛事规则](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/rules)
- [奖项及时间](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview/evaluation)
- [公开榜](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/leaderboard)
- [第三方架构](https://github.com/MAYANKSHARMA01010/2026-ieee-big-data-traffic-flow-bench)

## 比赛要点

每人单账号、单队，最多四人；禁止队外私下共享。公开外部工具须合法并披露依赖。只允许通过已发布数据建模预测，不能获取隐藏资产。数据于 2026-09-11 重建，奖项只认可重开提交后的版本。
奖项需要邮件注册、最终代码/PDF 和复现审核；登记截至 10 月 25 日 23:59 AoE，最终包截至 11 月 10 日 23:59 AoE，分别对应北京时间 10 月 26 日/11 月 11 日 19:59。Kaggle 最终提交标为 11 月 6 日，精确 cutoff/daily cap/final selections 未取得。
官方奖项流程从 private 前十合资格队中审核，报告也影响最终排序。索引中的榜首为 0.92406；这是参考快照，不保证当前即时榜首。

## 不能混淆的指标

真实 queue labels 所有 split 均未公开；Task 3 正式边界流未公开；OD 真值未公开，只有 S_link 可验证。
官方公开 reference path-flow 文件来自本地求解，不是 path truth。
官方 BASELINES 的 queue 参考为 0.3017，而 README 写 persistence 0.2518，无法仅凭当前资料消除差异。

## 代码与规范的不一致

固定版本 Task 1 scorer 从所有 eligible 双空值选 truth targets，而 MASK_SPEC 指定模板；这会包含 Task 2 额外隔离空值。
fixture 专门制造额外 horizon blanks，原版 scorer 的计数与模板不同。保留原版作为对照，并以模板实现独立合法本地验证；真实数据影响仍需核查。
Task 4 builder 的 `baseline_submission` 用 train counts，`public_reference` 用当前 split counts；两者都正则到当前 split prior。不能拿 reference 给自己评出 S_od=1。

## 第三方架构审查

仅阅读公开 README。它声称使用空间/时间插值、LightGBM、波传播、FD/LWR 和 bounded OD solver，分项成绩未经独立证实。未获取其源代码、数据、submission 或 prediction。
独立实现加入严格日期隔离、观测年龄、拓扑 latent-link traversal、Kalman/RTS 消融、稀疏 measured-link OD operator、多先验对比、coverage-aware 守恒诊断和提交完整性拒绝机制。
不采用它提出的 flow floor 来改善评分，不将其声称成绩作为本项目结果。
