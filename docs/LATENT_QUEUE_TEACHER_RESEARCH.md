# Queue：已知模拟真值下的标签对照 C62–C64

正式最佳仍 **0.82940**，0.95000+未达到。本轮只训练自主CTM模型；没有组织方latent truth、公开比赛模型迁移或新完整submission。以下IoU不能与正式Queue或旧公开观测代理直接比较。

## 固定协议

32cell/2day的自身Godunov/CTM law。fit生成seeds947101/947107/947119/947129，cal947137/947149，assessment947161/947167/947177/947183；三阶段互斥。assessment/stress共用四seed，不算额外独立样本。final seeds7907/8017/9109未生成。

六个原场景，fit/cal σ1/3/8；assessment σ0/3/8/16。四个预登记stress改变事故位置、FD、需求、失效长度/比例和flow噪声。268scene的不同noise保持相同latent/quality，不能称268独立重复实验。

三个teacher：硬观测阈值、`Phi((cut-observed)/known_sigma)`软标签、已知自主latent标签。soft假设平坦先验且使用已知模拟σ，**不是真实后验**；实际比赛σ未知。latent模型是标签控制，不是数学上界；原注册“upper reference”只指诊断用途，实际不保证领先。

每teacher两个optimizer seeds20261013/20261029，同272907行/权重/58特征，300棵31叶cross-entropy树。lr.035/min_child40/lambda10、feature/bagging.9、CPU4线程。condition/window等权，窗口内合法监督cell等权；onset早五步不监督，预测固定0。sigma仅用于生成fit teacher，不进入推断。

58特征仅接受13个past/current slots：归一化v/q、availability、局部统计/趋势、5km位置邻域、公开condition/cutoff/lead。不输入latent、noise、scene/cell/time身份、事故位置或未来quality。过去12步使用eligible raw noisy observations，T仍masked；与C51旧诊断不同，分数不能直接比较。

用自己已知latent选事件：history quality≥.7、过去12步同cell两次queue判ongoing，否则onset；horizon有queue，onset首次T+30，latent persistence IoU≤.9。所有重叠origins保留，不要求未来观测有正例；这不是组织方selector。

先seal六模型再生成cal。50阈值/force配置分别按观测/latent cal选择；后者只作诊断。soft仅在observed-cal gain≥.005且各condition损失≤.01时替换hard。实测gain **−.000664686**，primary保留hard。全部模型/两种cal都选onset.1、ongoing.4、不force，本次没有阈值差异的额外变化。选择seal后才生成heldout。

## 实测与限制

当前窗口集合的condition宏平均、已知latent标签IoU（observed-cal政策）：

| 模型 | 同类新seed留出 | 未见事故/FD/失效stress |
|---|---:|---:|
| 固定因果persistence | .272934 | .252671 |
| 硬观测teacher | .461195 | .393638 |
| Gaussian软teacher | .460679 | .399256 |
| 已知latent标签控制 | .461134 | .396266 |

soft按此统计量变化 **−.000516/+.005618**。另一预登记统计量先等scene/可用condition、再按seed平均，gain+.001306/+.005305，四seed bootstrap区间[−.007221,.009834]/[.002964,.007553]。权重不同，不只展示有利的一组；四独立seed区间很粗，无多重比较校正，不是正式置信区间。

同类四seed gain+.015238/−.007132/+.004429/−.007310；stress+.006018/+.008479/+.004776/+.001945。latent标签控制相对hard的seed均值gain仅+.000039/+.002115，两区间均跨零，尚不能把阈值标签噪声认作主要瓶颈。

效果不随噪声单调变化：等scene latent IoU里，assessment soft在σ3/8下降.005039/.006273，在σ16改善.016221；stress σ8改善.011417，σ16反而下降.000773。不根据已看noise/场景重新选参数。

hard的latent onset/ongoing为同类 **.090053/.832336**、stress **.061107/.726168**；soft分别.089728/.831630、.068778/.729734。当前因果特征/树模型的onset很弱，latent标签控制也没解决；不能推断比赛onset不可预测或分布相同。

assessment2632窗口（onset176/ongoing2456），stress1488（128/1360）；eligible cell377392/212424，latent/观测不一致2690/1631，未来零观测正例窗口28/12。窗口重叠，预测先在完整域作出，不用未来eligibility改变force位置。

## 核验与复现

C63重新模拟268scene、独立逐行核验7090窗口的特征/selector/teacher/mask/fit权重：feature最大差1.49e−8、soft差0。18portable booster预测差0，300cal配置和57680角色窗口评分差0。C64独立核验8个四seed区间、保存64行noise摘要；没有独立重训六模型。

C62整体93.78秒、CPU进程peak491.11MiB，六fit各4.42–4.53秒；C63 79.66秒。三个新单测通过；全research128 passed/13.63秒，CPU107 passed/8.79秒、7可选Torch模块skip。部分测试/验证并行，不当隔离测速。

```powershell
.venv-repro\Scripts\python.exe scripts/latent_queue_noise_experiment.py --output artifacts/research/C62_CTM_Queue_hard_soft_teachers
.venv-repro\Scripts\python.exe scripts/verify_latent_queue_noise.py --study reports/research/C62_CTM_Queue_hard_soft_teachers.json --output reports/research/C63_CTM_Queue_hard_soft_independent_verification.json
.venv-repro\Scripts\python.exe scripts/summarize_latent_queue_noise.py --output reports/research/C64_CTM_Queue_hard_soft_summary.json
.venv-repro\Scripts\python.exe scripts/checkpoint_latent_queue_noise.py
```

原目录禁止覆盖，复现需新输出路径并相应修改后继源码绑定。不向真实Queue晋级softteacher，不扩大已看噪声网格。下一步检验公开≤T事件信息、预测不确定性及可辨识的观测模型；公开数据迁移和正式效果仍未知。
