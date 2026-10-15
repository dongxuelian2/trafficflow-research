# 拟合期突变统计与速度差分损失：阶段一

正式最佳仍Kaggle **0.82940**，0.95000+未达到。A122/A123是公开fit诊断，没有新模型验证分数。A124双seed训练正在执行；后留和新日历成绩未知。

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

## A124：一个固定对照，尚未完成

保留原97307参数GRU、2seed、24×120/batch4、context128/core64、Queue-shaped输入及所有存储采样计划。旧A104 Queue模型直接引用A107核验过的导出，不重训控制。初始状态hash须逐seed相同；每个新模型完成后，全部逐格曝光必须与同seed旧控制精确一致。

新增目标为原patch State RMSE加 **.05 × speed jump pair RMSE**。只对core内连续两格都已经被监督、float32标准化速度标签差绝对值≥.8的pair启用；.8对应20/25，float32边界可能与raw float64诊断略有不同。只用这两个已有标签，不增加监督格点，不添加流量pair项。额外目标改变优化预算，不能称完全等损失。

仍按普通cal State每4epoch选checkpoint，alpha沿用0/.25/.5/.75/1。新双seed均值只有cal gain≥.0005且各regime损失不超过.001才替换NN；两模型及选择全部seal后才读旧后留。新日历在训练前固定为2031-02-15…02-28，仍属公开研究日期，其他历史研究可能查看过。

中途cal不作已完成模型、assessment或正式成绩。阶段manifest只冻结运行任务的不可变start archive，排除所有仍可变化的checkpoint/progress/export/counter文件。

首seed已完成fit/cal，517.88秒，45,754,069次逐格监督与原控制精确一致（训练worker断言），实际jump-pair曝光30,192次，alpha=.75。这是单角色进度；两seed的独立A126核验和完整模型选择尚未完成，旧后留/新日历仍未知。第二seed沿用同一worker继续运行，不重启。

## 运行入口

```powershell
.venv-repro\Scripts\python.exe scripts/audit_fit_transition_population.py --output reports/research/A122_fit_transition_population_audit.json
.venv-repro\Scripts\python.exe scripts/verify_fit_transition_audit.py --study reports/research/A122_fit_transition_population_audit.json --output reports/research/A123_fit_transition_independent_verification.json
.venv-research\Scripts\python.exe scripts/train_speed_jump_ablation.py --output artifacts/research/A124_I5_N_speed_jump_pair_loss
```

A124现有目录禁止重启覆盖；先poll同一任务。全部完成后，依次运行`evaluate_exported_state_forward.py`生成A125、`verify_speed_jump_training.py`生成A126、`verify_native_remask_forward.py`核验A125生成A127。新模型实际成绩确定后另作后继归档。原完整提交和最终模拟seeds均保持冻结。
