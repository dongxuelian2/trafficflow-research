# 非负 MAP 与 OD 专家组合

D12 比较 Gaussian count loss 与 Huber count loss，并加入可保留零值的 MAP。设路径尺度 `sd`、计数尺度 `.02*max(c,10)`，目标为 count loss + `.5 Σ ((f-b)/sd)^2 + η Σ f/sd`，约束 `f>=0`。令 `z=f/sd`，二次先验中心变为 `b/sd-η`；复用独立实现的 measurement-space bound pivot，Huber 用 IRLS 并核验原目标的 projected KKT。绝对尺度为 `cv*max(mean(b),2)`，相对尺度为 `cv*max(b,2)`。

这是连续 Gaussian/exponential 先验的 MAP，边界可以是零；不是带离散 spike 的 hurdle posterior，也不提供路径激活概率或可靠 posterior coverage。小矩阵对独立 SLSQP，完整 903-path 算子对独立 augmented path-space TRF/LSMR，通过数值对照，后者最大差 1.23e−8。

D12 六配置在九种独立真值生成分布上比较：原七种、sparse active paths、count outliers。D11 已排除误称 route-choice 的 singleton OD demand 同类复核。选择一个全局配置及 prior top15% mass 阈值；只允许开发 seeds 411/829/1201 选择，最低平均增益 .005、最差 regime 不低于 −.002。额外 shrinkage 与 Huber 没有被保守 Ridge-fallback 选择器选中。其旧 research seed gain .004085、CI [.000466,.007706]，但 30 个公开 weak prior 全不触发；这不是当前赛事输入的改进。

D13 扩大 fallback 选择至已冻结 D08 entropy τ=.05/floor20，仍只读取 D12 development scores 选择。选中的专家为：prior top15% mass <.75 用 entropy；否则用 absolute Gaussian/exponential MAP `cv=.75,η=1,noise_cv=.02`。开发最差 regime gain .023582；配置先封存，随后使用新的研究 seeds 5003/6203/7001。

| 生成分布 | 新 seed Ridge ODME | 新 seed 专家 ODME |
|---|---:|---:|
| independent paths | .831022 | .864914 |
| hierarchical demand | .837995 | .866395 |
| prior total bias | .845886 | .860765 |
| additive prior noise | .657992 | .727728 |
| lognormal prior noise | .512276 | .594056 |
| missing prior paths | .619187 | .686268 |
| group-biased prior | .535718 | .631851 |
| sparse active paths | .653270 | .765436 |
| count outliers | .478680 | .596152 |

九类 macro .663559→.732618，gain .069060、family/seed CI [.052272,.086461]。仅反映这些自行选择的生成分布；新研究 seeds 也不是最终测试集。Final seeds 7907/8017/9109 及其 offsets 仍未使用。540 method/case 完整、0 failures；两个当前代码分支的点预测 SHA 精确复跑，开发选择从 development-only cache 独立重算一致。

14/270配对案例退化，最差 D7_I405_N / seed7001 / count_outliers 为 −.028287，entropy分支 .555405→.527118。九类均值正不构成逐例鲁棒性保证；这批新seed没有用于重新选参，最差十例单独保留。

完整 LOCAL K05 为 `artifacts/research/K05_OD_expert_LOCAL/submission.csv`。当前 validation/private 的20份 prior 都低于阈值，实际全部 entropy；不能声称它在赛事上已表达稀疏分支收益。合并全部 6,980,503 行后，与正式基准逐行对照只有70,708个 OD path_flow 字段改变，State/Queue 和其他字段完全一致。SHA `eef9ec04375aa33a21d0824419f4f92a84be324fe4869e1911ac574534a1825e`，未上传，真实 ODME/Total 未知。

可从根目录复跑，新输出不可覆盖旧记录：

```powershell
.\.venv-research\Scripts\python.exe scripts/zero_aware_od_experiment.py --output reports/research/D12_new_repeat.json
.\.venv-research\Scripts\python.exe scripts/od_expert_experiment.py --cache-study reports/research/D12_new_repeat.json --output reports/research/D13_new_repeat.json
.\.venv-research\Scripts\python.exe scripts/build_od_expert_candidate.py --study reports/research/D13_new_repeat.json --output artifacts/research/K05_new_repeat_LOCAL
.\.venv-research\Scripts\python.exe scripts/verify_controlled_candidate.py --candidate artifacts/research/K05_new_repeat_LOCAL/submission.csv --allowed-task odme
```

相同 seeds 的重复是复现，不是新的独立评估。旧 Bayes 的 K01 正式失败不因这些模拟而消失；不得根据 public_reference 解、不可见 path truth 或单次正式反馈调整参数。下一步扩大 prior/noise 分布变化和需求结构，保留正式基准。
