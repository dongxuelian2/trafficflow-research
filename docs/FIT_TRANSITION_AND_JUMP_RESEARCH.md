# 拟合期突变统计与速度差分损失：A122–A127

正式最佳仍Kaggle **0.82940**，0.95000+未达到。A122/A123是公开fit诊断。A124–A127双seed、后留/新日历及独立计数/部署核验现已完成，固定.05目标未取得收益。

## A122：先检查监督人口

只打开I210_W原92个fit日期、I5_N冬季153个fit日期的raw观测、原缓存、四份point/Queue-shaped视图及实际NN曝光计数。完整源JSON含过去研究摘要，但统计计算不读cal/assessment/forward的teacher或预测数组。数据均为公开train，不是latent state。

大幅速度变化定义为连续5分钟有效link-grid观测差至少20km/h。单步返回指下一步回到事件前±5km/h；持续指后两步都与事件前相差至少10km/h且同方向。流量阈值为每车道150vph、返回50vph、持续75vph。相邻测量链接按公开拓扑遍历，时间±10分钟内同方向变化≥10km/h才计支持。日界不循环、不填未观测标签。

| 全部raw fit人口 | I210_W | I5_N |
|---|---:|---:|
| 有效相邻格点对 | 1,295,413 | 6,855,779 |
| 降速≥20 / 恢复≥20 | 1495 / 1800 | 6576 / 7822 |
| 两类合计占有效相邻格点 | .2544% | .2100% |
| 有效未来两步子集 / 全部大变化 | 2009/3295（61.0%） | 9088/14398（63.1%） |
| 持续10分钟 / 上述子集 | 1944/2009（96.76%） | 8984/9088（98.86%） |
| 单步返回 / 有效下一步子集 | 37/2577（1.44%） | 66/11434（.58%） |
| 同方向邻居支持 / 邻居可用子集 | 553/955（57.91%） | 12095/14349（84.29%） |

不能把这些条件比例推广到缺少未来标签或邻居的案例；I210的邻居可用子集仅955个。持续变化也可能含观测误差/共同扰动，**没有认定真实拥堵波、事故或无噪声状态**。

原target mask上，大幅降速/恢复分别贡献输入base速度SSE的3.60%/2.01%（I210）和5.25%/5.32%（I5），合计5.61%/10.57%。base来自输入strict12的通道6/7，不是训练NN预测；这些不是正式State误差归因。选择I5作为首个损失对照，是基于fit人口和base误差，不按后留挑走廊。

四份视图复用同一raw观测，不是独立事件。NN实际曝光计数也单独统计：例如I5两个Queue seed的总格点曝光为45754069/45844015。逐格累计曝光不能还原共同出现在同一个patch中的相邻监督对，故A124另存实际pair计数，不以这些累计数代替。

## 独立复核与测试

A122耗时48.93秒；独立raw teacher与缓存最大差2.38e−7归一化单位，在原1e−6容差内。A123用独立shift/window实现重建全部245日、3185诊断数组，复核3430个人口/实际曝光记录及全部base SSE汇总；最大绝对统计差3.73e−9来自浮点求和，35.60秒。它沿用相同raw数值teacher定义，未声称所有算法完全独立。

新增四项核验覆盖持续阶跃、孤立返回、缺失/日界、手算endpoint梯度和无相邻监督时严格退回原目标。全套research **122 passed / 10.03秒**；CPU **104 passed、6可选Torch模块跳过 / 7.99秒**。

## A124：固定对照与负面结果

保留原97307参数GRU、2seed、24×120/batch4、context128/core64、Queue-shaped输入及所有存储采样计划。旧A104 Queue模型直接引用A107核验过的导出，不重训控制。初始状态hash须逐seed相同；每个新模型完成后，全部逐格曝光必须与同seed旧控制精确一致。

新增目标为原patch State RMSE加 **.05 × speed jump pair RMSE**。只对core内连续两格都已经被监督、float32标准化速度标签差绝对值≥.8的pair启用；.8对应20/25，float32边界可能与raw float64诊断略有不同。只用这两个已有标签，不增加监督格点，不添加流量pair项。额外目标改变优化预算，不能称完全等损失。

仍按普通cal State每4epoch选checkpoint，alpha沿用0/.25/.5/.75/1。新双seed均值只有cal gain≥.0005且各regime损失不超过.001才替换NN；两模型及选择全部seal后才读旧后留。新日历在训练前固定为2031-02-15…02-28，仍属公开研究日期，其他历史研究可能查看过。

两新模型均按普通cal选epoch24，alpha分别.75/.5；旧控制两seed均为.75。因此这是配对优化及cal策略比较，完整差异不能完全归因单独梯度项。cal均值从.955885降至.951560，gate保留旧NN。全部训练1089.42秒，两fit/cal角色517.88/519.50秒，GPU allocated峰值2100.56MiB；未重新拟合旧控制。

| 公开TRAIN日期域 | 原NN均值 | 新pair-loss均值 | pooled State gain | day-mean 95% CI |
|---|---:|---:|---:|---|
| 原12/30–1/12后留 | .957512 | .952669 | −.004843 | [−.005022,−.003977] |
| 预登记2/15–2/28 | .959782 | .956831 | −.002950 | [−.003752,−.002703] |

两seed相对各自冻结控制的pooled gain：旧后留−.004930/−.004862，新日历−.002643/−.003328。各seed共享日期和标签，不能算独立数据集重复；CI只是14日、7日moving-block的日均诊断，不是pooled State或Kaggle区间。新日历R1/R2/R3为7/1/6日，R2只有一日。

目标子集未改善：旧后留急降速/恢复速度RMSE从9.983/10.016升至12.733/11.980；新日历15.084/11.531升至18.316/14.051。新日历60min缺失速度1.368→1.327略好，但flow27.680→29.540更差；原后留该子集速度/flow均更差。旧增量L1两日历.650629→.686754、.662477→.699876也变差，正式Physics未知。没有根据某个较好子集反选模型。

两个seed逐格监督分别45,754,069/45,844,015次，与原控制精确一致；新增pair endpoint曝光30,192/30,858次。A126独立重建四个计数archive的全部日数组和初始化/patch序列，验证196次保存角色predict及168指标，最大预测差4.55e−13、评分差2.22e−16，213.94秒。A125所有7导出×14日实际predict98次，112.73秒；A127重建14日teacher/diagnostics、98指标，另14次直接predict差0，30.00秒。后续评估部分并行，不当隔离测速。

阶段一manifest仅冻结当时不可变start archive，原清单不覆盖；完成后继清单才加入所有权重/计数/预测。这个单一系数和预算失败，不证明所有时间差分目标都无效，但没有理由扩大当前系数网格。

## 运行入口

```powershell
.venv-repro\Scripts\python.exe scripts/audit_fit_transition_population.py --output reports/research/A122_fit_transition_population_audit.json
.venv-repro\Scripts\python.exe scripts/verify_fit_transition_audit.py --study reports/research/A122_fit_transition_population_audit.json --output reports/research/A123_fit_transition_independent_verification.json
.venv-research\Scripts\python.exe scripts/train_speed_jump_ablation.py --output artifacts/research/A124_I5_N_speed_jump_pair_loss
.venv-research\Scripts\python.exe scripts/evaluate_exported_state_forward.py --study reports/research/A124_I5_N_speed_jump_pair_loss.json --output reports/research/A125_I5_N_speed_jump_forward_calendar.json
.venv-research\Scripts\python.exe scripts/verify_speed_jump_training.py --study reports/research/A124_I5_N_speed_jump_pair_loss.json --output reports/research/A126_I5_N_speed_jump_training_verification.json
.venv-research\Scripts\python.exe scripts/verify_native_remask_forward.py --study reports/research/A125_I5_N_speed_jump_forward_calendar.json --output reports/research/A127_I5_N_speed_jump_forward_independent_verification.json
```

A124现有目录禁止覆盖。后续命令已完成：`evaluate_exported_state_forward.py`生成A125、`verify_speed_jump_training.py`生成A126、`verify_native_remask_forward.py`核验A125生成A127。原完整提交和最终模拟seeds均保持冻结。
