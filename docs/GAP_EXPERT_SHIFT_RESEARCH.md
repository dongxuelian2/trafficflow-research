# 缺失模式专家与跨分布失效验证

正式最佳仍为0.82940，0.95000+未达到。本研究固定原权重，先区分混合权重与训练增强的影响，再验证只用观测可用性的门控。不使用隐藏真值、不上传、不计算正式Physics或Total。

## 官方任务分布与研究边界

已复查本机官方 `vendor/official/docs/MASK_SPEC.md`，SHA-256为`8425656f94f9d82c75ed8fc0bad35a79499996ab211fc1e27aeab2981e0ee749`。官方R1/R2/R3是按公开稳定哈希移除20%/30%/50% eligible单元，每日仅一个regime；Queue窗口的未来30min及后续60min还会产生额外空缺，template才定义Task1目标。R3不等同于多小时连续检测器断测。

因此长失效专家的合成收益只证明特定额外缺失下的鲁棒性，不能当正式State提升。若普通模板没有达到门槛的长缺失，门控可能完全不改变正式目标预测；下一步研究优先级必须依据这一事实，而不是只看很大的合成增益。

## 相同alpha的独立对照

A92首次试图从保存的混合预测线性反推alpha.5，第一日原alpha.75模型的直接预测差为0.00016276 total flow units，超出1e−8复现门槛。原alpha的四个模型和baseline均精确重放；误差来自neural float32乘法舍入，混合输出已丢失部分精度。失败注册/源快照保留，没有放宽旧复现容差。

A92b改为全部14普通日期及168失效case×4个perseed模型，直接从checkpoint设置alpha.5推断，共728次predict调用。该批A88日历已经查看，是次要解释性诊断，没有重训、重新选epoch或选择新的最佳alpha。两臂每个seed的选中epoch仍20/24，初始化、原监督曝光及参数预算已匹配。

同alpha.5的普通模板State：control均值 **.932724**、增强均值 **.930013**，gain **−.002711**。四种3h/30%、3h/70%、6h/30%、6h/70%合成case均值gain分别 **+.099034/+.133283/+.138672/+.172670**。性能取舍并非仅由原cal混合权重不同造成。实耗95.36秒；舍弃的线性反推在全部案例最大差0.00020345 total units，该字段不是新直接预测的复现误差。

## 冻结的输入门控

A93使用两个已独立部署核验的两seed均值：原control与增强expert，保留各自旧cal alpha。只在目标所属link的公开speed或flow连续缺失至少36步（180分钟）时选择expert，否则使用control。门控按完整公开State输入网格计算，不读teacher、regime、事故标签、未来评分mask或人为失效geometry。

State是离线重建，允许同日已公开的前后观测；该门控不能用作Queue在T时刻的因果预测输入。两日跨午夜case仍各自单日重建，不把相邻日观测借入边界；截断缺口只按可见日内长度计数。速度和流量作为一个状态对共同切换，没有按真值逐通道挑模型。

序列门控测试覆盖35/36步、首尾截断、不跨越观测单元、有限测量值变化不改变门控、跳过未启用专家、保存重载与键顺序。额外失效生成器只读timestamp/link/milepost及RNG，所有8条件poison测量值后mask不变。

## 新日历和失效法律

A93在读取新标签前固定普通日期2030-12-11…12-25、14个anchor date12/11…12/24及三个seed20261121/20261123/20261127。日期与本组profile/fit/cal/原assessment及A88均不重叠，但其他历史研究可能看过这些公开日期，不是未开启final/private。

八种预注册条件为2h/15%、4h/50%、8h/90%、日初90min/50%、日末4h/50%、全走廊4h、30%独立散点失效及跨午夜4h/70%。它们不是原3/6h30/70%尺寸的再次调参。跨午夜从23:00开始，第一日1h、第二日3h，两个segment的SSE和样本数合并后才计算一个case分数，不把两个日分数直接平均。

普通指标只使用官方Task1模板；合成指标只用自定义region的公开raw-train eligible station真值，分开报告，不能混算总分。336个case geometry共享14个anchor日，42个跨午夜case还共享相邻日；按7日block的区间仅为未校正的条件诊断，不当独立事故或Kaggle CI。全部条件保留，不根据新分数调整阈值。

## 新日历实测与反例

普通15日模板State为LGB **.930402**、control **.936243**、未门控expert **.934366**、gated **.936243**。全部普通目标中gate触发数为 **0**，因此普通指标精确相同。该结果证明当前样本预测没有被门控改动，没有提供新增Task1收益证据，不能宣传为“无损增益”。

| 新自主失效条件 | control合成分数 | gated合成分数 | gain / 配对anchor-day CI | 目标门控比例 |
|---|---:|---:|---|---:|
| 2h / 15% | .837425 | .837753 | +.000329 / [−.000914,.001571] | 2.934% |
| 4h / 50% | .667706 | .736424 | +.068718 / [.045964,.093461] | 100% |
| 8h / 90% | .537600 | .599210 | +.061610 / [.048537,.074683] | 100% |
| 日初90min / 50% | .944177 | .944179 | +.000002 / [.000001,.000003] | .157% |
| 日末4h / 50% | .910954 | .910720 | −.000234 / [−.003841,.003374] | 100% |
| 全走廊4h | .707999 | .649217 | **−.058782 / [−.065502,−.052063]** | 100% |
| 随机散点30% | .928020 | .928020 | +.0000004 / [0,.0000008] | .0067% |
| 跨午夜4h / 70% | .933244 | .938142 | +.004897 / [.004110,.005685] | 74.889% |

某些原短时缺失与额外缺失拼接，令短case有少量目标超过180min；跨午夜日内截断使第一日大部分目标不触发。全走廊反例说明只按时间缺失长度选专家不能覆盖空间上下文完全丢失的分布变化。小case/散点的极小数值不作实质增益；8种区间未校正多重比较。

本研究不晋级该门控，也不根据这批新结果立即加一条全走廊例外后仍称其为盲测。A94耗时251.31秒、GPU allocated峰值230.45MiB，部分时间与测试并行；包含所有对照的完整运行时间不是门控生产延迟。新模型、全部反例和旧完整提交均保留，正式Physics与Total未知。

下一步不继续围绕严重合成断测扩大阈值网格。优先回到官方20/30/50%稳定随机mask与Queue90min空缺对原Task1模板的影响，以公开fit标签构造合法训练变体并保持时间验证；另外保留空间信息完全缺失的压力测试作为回归。

## 独立核验与资源

A95不调用模型的run-length函数，而是从modified frame按eligible观测groupby重建可见性，再逐列寻找缺失段边界；全部393份普通/segment门控数组逐值一致。重新核对15普通日、336case的378个day segment及其raw-train teacher/lanes/indices；60普通指标、1512segment分数、1344合并case分数的独立计算最大差2.22e−16。

另在首末普通日和首anchor/首seed的8条件全部segment，对4个角色（含真正序列化的GapGatedState）作44次直接predict调用，预测最大差0。不是每一个保存case均重新推断。独立核验31.14秒；当前114 research tests通过/11.79秒，CPU98通过、5可选Torch模块跳过/9.15秒。

## 复现入口

输出必须使用新路径，原失败目录不覆盖。模型、预测和源快照保存于同名artifacts目录。

```powershell
.venv-research\Scripts\python.exe scripts\equal_blend_augmentation_diagnostic.py --output reports\research\A92b_equal_blend_direct_diagnostic.json
.venv-research\Scripts\python.exe scripts\freeze_gap_gated_state.py --output reports\research\A93_I210_W_gap_expert_frozen_policy.json
.venv-research\Scripts\python.exe scripts\evaluate_gap_expert_shift.py --policy reports\research\A93_I210_W_gap_expert_frozen_policy.json --output reports\research\A94_I210_W_gap_expert_shifted_calendar.json
.venv-research\Scripts\python.exe scripts\verify_gap_expert_shift.py --study reports\research\A94_I210_W_gap_expert_shifted_calendar.json --output reports\research\A95_I210_W_gap_expert_shift_independent_verification.json
```
