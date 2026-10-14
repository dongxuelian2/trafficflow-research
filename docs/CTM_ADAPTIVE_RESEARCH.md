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
