# 固定通道集成：A112–A115b

正式最佳仍为 **Kaggle 0.82940**。下表均为公开 TRAIN 的本地 Task 1 State，不能加权推算正式 Total；正式 Physics 未知。

## 假设与选择

A103 的同视图 LightGBM 在开发 cal 上流量 RMSE 优于 A97 GRU、速度 RMSE 更差，因此探索速度与流量使用不同的固定模型权重。这个假设来自已查看 cal，并非盲假设。没有新增拟合、标签或目标误差驱动的门控。

固定候选为 point/Queue-shaped 两种树均值，各速度和流量权重取 0/.25/.5/.75/1，共50组。只按 cal 选取：State 增益至少 .0005、每个 regime 损失不超过 .001，否则保留 NN。结果选择 **速度全用 A97 NN、流量全用 point 树双 seed 均值**。cal 增益 .001138，最小 regime 增益 .000809。全部权重封存后才读取旧 assessment；2031-02-06…02-19 的14日历在本次选择前登记。

## 实测

| 公开 TRAIN 日期域 | A97 NN | 固定通道集成 | pooled State 增益 | 配对 day-mean 增益 / 95% CI |
|---|---:|---:|---:|---|
| 原14日 cal（选择用） | .937716 | .938855 | +.001138 | 不作独立泛化证据 |
| 原14日 assessment（已查看回归） | .938919 | .940049 | +.001130 | +.000998 / [.000879,.001117] |
| 2031-02-06…02-19 | .940073 | .941077 | +.001004 | +.001059 / [.000948,.001170] |

CI 以日为单位、7日 block、4000次重采样，只是 day-mean 诊断；不是非线性 pooled State 或 Kaggle CI。公开日历可能被其他历史策略查看，不称私有/最终盲测。原 assessment 的 R3 只有1日；新日历 R1/R2/R3 的目标数为7081/26319/62059，相关格点不计作独立实验。

新日历三个 regime 的流量 RMSE 从31.030649/31.033393/32.295119下降到30.003204/29.735803/30.691105；速度输出逐格保持 NN。急降速、恢复和拥堵50km/h子集的流量 RMSE 分别从97.441/138.992/45.282下降到81.464/122.293/41.951。180min断测目标为0，没有该子集收益证据。

旧增量诊断 L1：原 assessment .693890→.683813，新日历 .725316→.717154。它缺少官方边界流等信息，**不是正式 Physics**；分通道组合仍可能改变 q/v 动态，不能凭这个数晋级正式 State–Physics 联合候选。

## 复核、资源与失败

- A112：25.92秒，原28个 cal/assessment 日期逐日直接调用保存策略，与通道公式最大差0。
- A113：29.35秒，4角色×14日共56次直接调用，全部日期保存模型与固定通道公式差0。
- A114：10.24秒，重建14日 raw teacher/lanes/target order/诊断mask，复算56指标，另8次直接 predictor 调用差0。
- A115b：7.25秒，另行从 raw TRAIN 重建28日 teacher/单位，复算112保存指标、50个cal候选及选择规则，核对封存顺序和序列化权重；最大评分差1.11e−16。没有重训，也没有新增直接推断。
- A115 首次在无Torch的 CPU 环境中完成数值检查后，嵌套GRU模型反序列化缺依赖失败；原注册、源码及 failure.json保留，A115b改用research环境通过。
- 全套测试：research **118 passed / 9.95秒**；CPU **102 passed、5可选Torch模块跳过 / 7.79秒**。首次未指定工作区 basetemp 的运行产生13个临时目录 setup error，修正运行路径后通过；没有更改数值断言。

模型为 `artifacts/research/A112_I210_W_channel_state_ensemble/State_primary.pkl`，SHA-256 `67b37cefe2b245fe3544fbf502351cd76e3e5ca6b0a0e4ab36a807ff6f06b7c6`。其递归清除临时网络缓存、保存后恢复当前进程缓存，并检查模型特征/profile兼容、输出键/顺序/非负有限值。

## 复现

```powershell
.venv-research\Scripts\python.exe scripts/channel_state_experiment.py --output reports/research/A112_I210_W_channel_state_ensemble.json
.venv-research\Scripts\python.exe scripts/evaluate_channel_forward.py --study reports/research/A112_I210_W_channel_state_ensemble.json --output reports/research/A113_I210_W_channel_state_forward_calendar.json
.venv-research\Scripts\python.exe scripts/verify_native_remask_forward.py --study reports/research/A113_I210_W_channel_state_forward_calendar.json --output reports/research/A114_I210_W_channel_state_forward_independent_verification.json
.venv-research\Scripts\python.exe scripts/verify_channel_state.py --study reports/research/A112_I210_W_channel_state_ensemble.json --output reports/research/A115b_I210_W_channel_selection_independent_verification.json
```

已有目录禁止覆盖；重新实验须使用新输出名。模型组件和研究输入由源报告/阶段manifest关联。没有全273日生产训练、新完整 submission 或正式成绩；下一步先完成冬季重新mask的跨方向验证，再决定是否值得投入同样通道树对照。
