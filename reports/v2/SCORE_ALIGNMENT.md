# 官方评分对齐实测

官方版本：`jacky850/trafficflowbench-public` commit `c88cddf533bbf0afa4ff1fc6c031d760a08e7a31`，本地 `vendor/official/UPSTREAM.json`。独立实现见 `src/trafficflow/v2/score_aligned.py`。原始不可覆盖证据为 [v2_score_audit_v2.json](v2_score_audit_v2.json) 与同名预注册。

最终通过GitHub API核验当前上游main为[`f9e3fbc`](https://github.com/jacky850/trafficflowbench-public/commit/f9e3fbc35c066419726987e470448f33d4c72fd9)，比vendor仅多1个commit：README、BASELINES、SCORING_SPEC及Task2 persistence builder的T时刻说明/基线修正。四个scorer与overall源码均无变化，本轮评分契约仍有效。[上游差异证据](OFFICIAL_UPSTREAM_CHECK.json)。

## 实际覆盖与结果

按每方向每regime最早两个公开train日预先选择：**10方向×3regime×2日=60日期分区**；不是完整273日历。包含 **662,141 个完整模板目标、1,449,841 个官方LWR时间对**。实际运行403.682秒，进程峰值RSS2083.64MiB，CPU，无GPU。

| 契约 | 实际比较 | 结果 |
|---|---|---|
| State目标/eligibility | eligible raw 与 masked两通道空白，逐key对比published template | 全30组差异0 |
| State速度/每车道flow | 独立SSE、missing计数与官方Task1 | 所有组匹配，分数容差1e−12 |
| State宏平均 | regime→方向→family→overall | 与官方S_state相同 |
| FD参数 | link几何、station FD均值、重算kcrit、lanes/jam fallback | 最大绝对差0 |
| Ramp | split=train、attachment、>=75质量、eligible/有限流量、on/off聚合 | 最大绝对差0 |
| Anchored frame | 本regime日期、eligible站点、只替换模板目标、站点均值 | 最大绝对差0，missing0 |
| Physics frame | q/max(v,1)、库存Lk、拓扑fallback、有效性标志 | 最大绝对差0 |
| FD/LWR/Physics | 独立reducer对官方Task3每panel/regime | 所有分项和n_rows匹配，容差1e−12 |
| Task2 | 源码审计，空eligible契约；额外构造数学与因果测试 | 没有公开latent标签，不能本地验证正式IoU |
| Task4 | 源码审计公开计数S_link及family归约 | 仅占ODME的25%；其余需隐藏路径真值 |

## 必须保留的实现细节

- Task1代码从空白读目标，Task3从模板读目标，不能未经检查认为总相同。本轮60日相同，未据hash自行补目标。
- 站点速度/流量分别平均后才取比值。FD参数车道总量与每车道量转换不能混用；kcrit=mean(cap)/mean(vf)。
- FD使用全部清洗后的anchored行；empty-road guard对`is_target>0.5`的聚合行检查，低于50vph占比**严格大于**.20才归零。
- 缺失内部邻接按`topology IDs ∩邻接 IDs`是否包含于本regime frame link集合筛选。外部connector不当作缺失内部证据；这也不意味着它的真实边界流已知。fallback对存在邻接列但该时刻缺值采用pandas sum语义。
- LWR先按link/time排序、向前shift，再筛0<dt<=5.1分钟。Mode A/B只检查**区间起点**对应on/off ramp有效性；无该类结构匝道时零为有效。
- 同一regime的相邻日期可能有跨午夜时间对，最终候选重放必须跨日期拼接后评分。训练脚本初始逐日代理只作日内训练/校准诊断；独立`compare_candidates.py`会对封存选择补齐跨午夜对，State SSE无需更改。不会按评估结果反选系数或强度。
- Task3 standalone对panel直接等权；Task1/2/4包括family层。当前完整10方向每族2方向不能成为忽略该差别的理由。

## 可知与不可知

展开总分为 .35 State+.30 Queue+.05 FD+.10 LWR+.20 ODME。公开代码有topology fallback，组织方还可提供隐藏boundary flux。源码文档声称后者投影到发布观测；**没有读取隐藏flux，不能数值证实b*=D N***。本轮分别报告官方可执行fallback和观测增量代理，不报告新正式Physics或Total。

四任务scorer SHA保存在原始JSON。第一次审计因Windows拒绝hardlink、尚未产生成绩即失败；改为工作区复制后在新目录v2重跑，原登记与失败记录保留。构造测试另覆盖多站点比值、缺失内部邻接/合法外部边界、非连续时间、ramp失效、Mode A/B/C、空流量保护以及宏平均。
