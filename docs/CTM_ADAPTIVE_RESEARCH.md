# 固定 State 网络的独立 CTM 压力验证

B07 用 A33 已封存的 attention / attention-density 两个 checkpoint，未重新训练或按模拟结果选择混合权重。输入仅为质量过滤后的 masked speed/flow、公开 FD 聚合参数、方向邻接及时间。Profile 使用 vf / .4 capacity，occupancy 完全缺失；不从 latent density 生成 occupancy。两个日网格分别为 288 / 289 个时刻。

六个固定条件为 mild、onset、上下游严重事故、检测器失效、额外观测噪声；三个 assessment seed 2909/3253/4001 × 三个缺失比例 .2/.3/.5，共 54 个组合。它们是自主 Godunov/CTM 生成，非组织方生成器。Final seed 7907/8017/9109 尚未使用。B01 曾使用质量不合格但有限的观测；B07 统一过滤，因此不能将两次绝对分数之差视为算法进步。

| 固定方法 | 模拟 observation State 均值 | CTM-flux Physics 诊断均值 | observation-projected 诊断均值 |
|---|---:|---:|---:|
| temporal/spatial interpolation | .853404 | .358359 | .675411 |
| attention | .848350 | .327080 | .521114 |
| attention-density | .855957 | .327728 | .619295 |
| .5 interpolation + .5 attention | .863126 | .328410 | .646200 |
| .5 interpolation + .5 attention-density | .864132 | .332950 | .672995 |
| observation oracle | 1.000000 | .354164 | .995759 |
| latent oracle | .952324 | .439453 | .806580 |

固定 density 混合相对 interpolation 平均 +.010728，但 mild 场景 .952471 → .941526，退化 .010945。三个 seed 的平均增益均正，只有三个独立 seed；缺失比例共享真值，不能把 54 个组合当 54 个独立样本。原始指标及按场景/缺失比例的对照见 B07 报告和 verified 汇总，不选择模拟 assessment 上的最优方法。

速度/流量 State 改善未转为已知 CTM-flux 诊断改善。Simulator 的 snapshot FD flow 与实际五分钟平均边界 flux 不同；physics scorer 只改 target，其他位置锚定带噪 observation，故 latent oracle 也不等于完整无噪守恒场。两种 flux contract 均为自主模拟，没有比赛 hidden boundary flux。上述 Physics 都不是正式成绩。

检测器失败位置、五分钟降速与恢复的 latent RMSE 单独保存。adapter 的接口排除了 latent truth/target values，测试核验质量不合格值任意变动不会影响输入或 interpolation。独立进程复跑全部 54 组，registered protocol、input SHA、每个候选数值和汇总完全相同。

复跑使用新路径，已有证据不可覆盖：

```powershell
.\.venv-research\Scripts\python.exe scripts/ctm_adaptive_stress.py --source artifacts/research/A33_adaptive_attention_strong_autumn --output reports/research/B07_new_repeat.json
.\.venv-research\Scripts\python.exe scripts/summarize_ctm_adaptive.py --source reports/research/B07_frozen_adaptive_CTM_stress.json --repeat reports/research/B07_new_repeat.json --output reports/research/B07_new_verification.json
```

下一步应把 public-observation 与 latent/noise 目标分开，评估 learned-profile 缺失导致的偏差，并在公共 train 的强 LGB 对照下验证新的同化/动力学结构。B07 的固定混合不替换生产模型。

## B08/B09：连续缺失和适配器因素

B08 另设 uniform half、中央半走廊整段缺测、事故附近120min矩形缺测、全网45min缺测，共72个配置。几何 profile / temporal-first interpolation 的模拟 State 为 .832553 / .711935 / .118663 / .576647。独立进程全部 protocol、input SHA 和候选数值一致。这些是自主压力模式，不复刻官方 R3。

B09 将 geometry vs development-history prior 与 temporal-first vs deployed-weighted interpolation 作2×2对照，并增加事故时钟+1h；144种配置，每种4个adapter。history从独立 dev seeds 的 eligible noisy masked-observation 字段按288 slot/cell聚合，不能使用评估真值或造 occupancy。原72个geometry/temporal-first control精确复现。六个history元素没有输入，按几何回退；同seed不同场景共用随机噪声，36合成日不是36独立样本。

改用 StateFeatures 本来就有的 age weight `1/(1+age/10)`、space weight .35 后，120min矩形缺失的 interpolation .118663→.227091。此前 B07/B08 的 temporal-first 比较基线较弱；修复评测适配器不是生产算法新收益。加权基线下，history-conditioned density network 为 .229198，仍未解决长缺失。

history-only 在 network45min缺失下 .229790，经+1h事件时钟变为 .011636；冻结 density network with history 在同加权输入下 .657787→.624722。不同 clock/pattern 的目标人口不同，只能报告压力场景表现，不能把绝对差解释为同标签算法增益。无模式重新选择或正式 Physics 声称。

A50 对三个公开train缓存的availability/target masks审计，R3 input连续段p99=60min、最大150–185min，target段p99=25min、最大70–75min。质量不合格输入与人工 target 缺失不同；这些日内 run 在午夜截断，不推断private mask或组织方生成器。

```powershell
.\.venv-research\Scripts\python.exe scripts/ctm_contiguous_stress.py --source artifacts/research/A33_adaptive_attention_strong_autumn --base-study reports/research/B07_frozen_adaptive_CTM_stress.json --output reports/research/B08_new_repeat.json
.\.venv-research\Scripts\python.exe scripts/ctm_profile_ablation.py --source artifacts/research/A33_adaptive_attention_strong_autumn --base-study reports/research/B07_frozen_adaptive_CTM_stress.json --output reports/research/B09_new_repeat.json
```
