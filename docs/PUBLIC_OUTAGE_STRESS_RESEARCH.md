# 固定State模型的公开数据长失效测试

本实验使用公开train真值与自主生成的额外观测失效，不是比赛隐藏数据、正式State或Physics评测。它针对原后留集没有3小时连续缺失目标这一证据缺口；不根据结果修改任何权重、alpha或模型选择。

## 注册、输入与标签边界

固定I210_W的四个已导出策略：full-LGB、原incumbent、新TCN两seed均值、新TCN+GRU两seed均值。日期2030-11-13…11-26与这些权重的profile/fit/cal/原assessment均不重叠，其他历史研究可能看过这些公开日期，因此不称最终盲测。

三个mask seed为20261031/20261103/20261107。每个date×seed只按公开timestamp、link ID、milepost与RNG选中心；中心时间09–15UTC，按milepost排序选连续30%/70%检测器，移除3/6小时。四个尺寸共用中心并嵌套，42个中心对应168个相关case，不能当168个独立事故。

移除speed、flow、occupancy、occupancy-derived density和pct_observed，同时一致更新observed/imputed/eligible/missing flags。每次从扰动后的frame完整重建StateFeatures，不能只把缓存的measurement通道清零，因为旧base/插值特征可能仍含刚被遮住的真值。Geometry不读取速度/流量或模型预测，单元测试验证改动所有测量值不改变mask。

受影响评估人群为region内全部raw-public-train eligible station行；与原Task1模板目标分别记录。原模板control使用未额外扰动的官方masked输入，两个人群的绝对分数不能直接比较。额外失效分数采用同样25/600通道尺度，但字段明确为synthetic reconstruction score，不计入正式Total。没有eligible目标的case保留未知、不重新抽样；本轮168case均可评分。

## 结果

| 自主失效条件 | 原incumbent | TCN均值 | GRU均值 | GRU−incumbent / 14日区间 |
|---|---:|---:|---:|---|
| 3h / 30% | .735689 | .736554 | .737391 | +.001701 / [.000241,.003162] |
| 3h / 70% | .697971 | .696575 | .698035 | +.000064 / [−.003224,.003353] |
| 6h / 30% | .638638 | .639134 | .635694 | −.002945 / [−.005720,−.000170] |
| 6h / 70% | .621071 | .620212 | .618787 | −.002284 / [−.004740,−.000142] |

3小时小范围有小幅收益；6小时的两个名义区间为负，说明原14日State收益不能推广为普遍长失效鲁棒性。6h/70%下full-LGB为 **.633125**，也高于GRU的.618787。区间重采样date均值，三个mask seed和嵌套case共享数据，未作多重比较校正，只针对这些自主设定。

相同14个新日期、未额外失效的原Task1模板control State：full-LGB **.928702**、incumbent **.933584**、TCN均值 **.934605**、GRU均值 **.935716**。A83将其与原14日后留合成同一冻结策略的28日次要汇总，incumbent **.933915**、GRU **.935949**，pooled gain **+.002033**；day-mean gain+.002148、CI[.001292,.002968]。这不改变原模型选择，也不能替代其他方向、2月或正式Kaggle证据。

## 失败、独立核验与资源

首次A81在赋值observation flags时退出：真实Parquet为int64，当前Pandas拒绝把bool False写入；尚无stress case预测。原注册、源快照与异常保留。按原flag dtype写入0/1，并增加bool/int两种fixture后，在新A81b目录复跑；没有改变geometry、日期或模型。

A81b全部168case、672份方法分数完成，wall time130.49秒、进程GPU allocated峰值230.58MiB。A82逐个重建mask与teacher数组，672分数独立重算最大差0；另重算14日全部control，并在4个stress case对4种策略作16次直接预测复核。独立核验29.12秒。测试与winter训练并行，耗时不是隔离吞吐。

```powershell
.venv-research\Scripts\python.exe scripts\public_outage_stress.py --output reports\research\A81b_I210_W_public_outage_stress.json
.venv-research\Scripts\python.exe scripts\verify_public_outage.py --study reports\research\A81b_I210_W_public_outage_stress.json --output reports\research\A82_I210_W_outage_independent_replay.json
.venv-repro\Scripts\python.exe scripts\combine_recurrent_calendar.py --output reports\research\A83_I210_W_28day_frozen_calendar.json
```

所有输出必须用新目录。正式最佳仍.82940，最终模拟seeds未开启，没有平台操作。若研究置信度门控或额外mask训练，必须仅从独立development case选择，并使用新日期/seed复核；不能在这些已查看case上挑阈值后宣称增益。
