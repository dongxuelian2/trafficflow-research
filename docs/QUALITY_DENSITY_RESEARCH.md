# 质量观测与密度坐标研究：A51–A57

本轮没有新的正式成绩：最佳 Total 为 .82940，.95000+ 尚未达到。模型只用公开数据，没有平台发布、推送或正式提交。

## 输入与匹配控制

在已有 D7_I210_W 秋季缓存中追加 any-quality speed/25、flow-per-lane/600、两项 finite availability、pct_observed/100（限于0…1）和质量字段 availability。原 strict12 数据完整保留。数据源仍是官方 masked public train；目标处 v/q 为空，未从 raw 标签填输入。新增输入权重零初始化，其余所有参数从同一 strict12 初始化复制。四个 v/q 模型及两个 density-coordinate 模型均采用24 epochs ×120 steps、batch4、64-slot patches、hidden48、2 heads，AdamW .001、weight decay 1e−4、clip1，seed20261013。patch 序列完全相同。

fit 为2030-07-16…10-15，cal10-16…29，原 assessment10-30…11-12；历史 profile 来自06-01…07-15。神经模型采样会重复目标，与全标签 LGB 的标签暴露量并非相同。A52 两个 strict12 控制复现 A49 全部最终权重和训练历史；这是精确控制，不是独立 seed。

每架构先按纯 cal State 选择 checkpoint，再在α∈{0,.25,.5,.75,1}上选择组合，架构也只用 cal 选择。joint 另外最小化 anchored density L1，约束 State≥full-LGB cal，且距当前 primary / 可选 strict12 control 中较高者不超过 .001。本轮 primary 与 joint 选同一 checkpoint，所有 assessment 标签在封存选择后加载。

## 密度坐标

输入的插值基准为 v₀、q₀，归一化输出残差为 rᵥ、rρ：

`v = v₀ + 25 rᵥ`

`q = max(q₀ + 600 rρ, 0) × max(v,1) / max(v₀,1)`

因此网络原始输出的 `q/max(v,1)` 由 density head 决定，初始流量 head 的导数尺度与原模型一致。参数化没有强加 jam density 真值、隐含边界流或完整 CTM。部署对 v/q 加公开 FD 上界并与 LGB 混合后，这个解析解耦关系不再严格成立。密度损失λ=.01仍是公开带噪观测的相邻增量监督，不是正式 Physics。

## 实测与失败

| 模型（仅 cal 选择） | 原14日 State | 追加30日 State | A55 v2 projected increment L1 |
|---|---:|---:|---:|
| full LGB | .927102 | .927004 | .419699 |
| 原 strict12 attention | .933577 | .932585 | .404858 |
| quality18 attention | .932965 | .932977 | .398451 |
| strict12 density coordinates + λ=.01 | .932951 | .932583 | .389999 |

追加日期2030-11-13…12-12含R1/R2/R3各10/13/7日，冻结已有 checkpoint、α、profile，无调参或模型选择。质量与密度的30日日均 State gain CI 分别为[−.000510,.000797]与[−.001017,.000476]。这些置信区间是7日块、4000次 paired bootstrap 的日均统计，不是 nonlinear pooled State 或 Kaggle Total 的区间。30日仍是公开 train 开发日期，不是最终盲测。

质量 cal State .932729，density cal .932896，都高于原 .932591；这些很小的 cal 变化没有转为稳定独立 State 增益。原14日只有一个R3日，不能以该子组的小幅差异晋级。两新模型都未通过 State .002 gate，保留为研究候选。

## 物理代理的口径

公开 scorer 用非目标 eligible raw observations 构建 state_for_regime，只替换 Task1 required targets；这些 raw observations 中有部分在模型输入中因 Queue horizon/buffer 不可见。A55仅在公开 train 以同样方式设置 evaluator anchors，不向模型提供它们。过滤 observable internal topology、Mode A公开 ramp valid（区间开始时），对全部有效相邻5分钟对计算 N=q/max(v,1)×length 的增量误差。

诊断假定 RHS 等于公开观测的真实ΔN，指标为 `sum(abs(ΔN_pred−ΔN_observed))/sum(abs(ΔN_observed))`，再对R1/R2/R3等权。没有读取组织方 boundary_flux，因此不是 S_LWR 或正式 Physics。原 attention .404858→density .389999，仅支持 observation-increment 代理改善；日均误差 reduction CI [.006049,.009574]。此前 target-touching-only 代理的 anchors、覆盖及分母不同，数值不能直接相减。Task3官方 RHS、方向聚合与 hidden evaluation 的真实增益仍未知。

A55 v1 未对 LGB 加统一上界，源归档为 postrun/preedit；v2 以 start snapshot 与共同上界重新测量，只有4个 LGB 预测被上界改变。两份报告均保留。A55/A57 原 evidence() GPU 标志误用了默认false；更正说明根据执行源码与本机CUDA环境推断使用GPU，原运行没有单独记录device或GPU峰值。原报告不修改，更正见 quality_density_metadata_corrections_v1.json，后续代码按实际network device记录。A54原选择文字未完整描述 optional control reference，但运行代码和封存reference可审计；本轮其primary高于control，文字补充不改变结果。

## 复现

在仓库根目录使用 `.venv-research/Scripts/python.exe`。所有新输出目录必须不存在，禁止覆盖历史证据。下面用 fresh 后缀示例；父缓存、公开数据和强基线需存在。

```powershell
.\.venv-research\Scripts\python.exe scripts/prepare_quality_state.py --help
.\.venv-research\Scripts\python.exe scripts/quality_state_experiment.py --cache artifacts/research/A51_quality_D7_I210_W_autumn_cache --output artifacts/research/A52_quality_replay_fresh
.\.venv-research\Scripts\python.exe scripts/quality_state_experiment.py --cache artifacts/research/A51_quality_D7_I210_W_autumn_cache --output artifacts/research/A54_density_replay_fresh --inputs strict12 --coordinates density --control-study artifacts/research/A52_quality_density_D7_I210_W
.\.venv-research\Scripts\python.exe scripts/verify_quality_state.py --help
.\.venv-research\Scripts\python.exe scripts/audit_state_increment_coverage.py --help
.\.venv-research\Scripts\python.exe scripts/state_temporal_extension.py --help
```

原 registered JSON保留实际参数与输入 SHA；每次训练有 start source archive、初始化权重、各checkpoint、cal选择seal、部署pickle与报告。A51缓存120个日文件，包含92 fit/14 cal/14 assessment。A52四模型410.51秒、GPU峰值288.59MiB；A54两模型222.82秒、288.42MiB。新checkpoint manifest保存哈希，不在Git分发比赛数据或模型大文件。

53项研究测试通过；CPU46项通过、三个Torch模块跳过。测试包含新增权重嵌套初始化可学习性、独立密度解析梯度、质量字段缺失/inf/clip与旧模型fallback。新 primary/joint 各14日cal独立重建，额外日期没有修改模型。下一步验证显式置信度门控与噪声状态同化、跨季joint目标，保留当前纯v/q对照和失败证据。
