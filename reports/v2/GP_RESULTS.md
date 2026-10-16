# 双向传播残差 GP：公开 train 首轮结果

结论：该固定的 192 诱导点、每日最多 512 个 counterfactual anchor 的近似未产生独立预测收益。I5_N、I210_W 秋季的普通 GP、门控普通 GP和双向 GP均在 calibration 选择强度 0；匹配后的 assessment 收益为 0，配对区间为 [0, 0]。不部署、不扩大到其他方向，也不启动冬季 GP。此结果否定本次受限近似的晋级资格，未否定所有可能的传播 GP。

基模型是冻结的强 LightGBM/Attention ensemble。另将其内部原始纯 LightGBM 作为固定诊断，在相同两方向、全部 14 个 calibration 日期和 14 个 assessment 日期独立重放；没有重训、没有修改 GP 选择。ensemble 已优于纯 LGB，不能把这个既有差异归功于新增 GP。

这些是公开开发集上的两方向 State 指标，采用官方 eligibility、regime pooled RMSE 和 family 宏平均公式；不是十方向正式 State，更不是 Kaggle Total。正式 State、Physics、Total 均为 null。

## 数据角色与数学契约

- 基模型训练和 checkpoint 选择截止 2030-10-29；2030-10-30 至 11-12 的前推留出残差仅拟合两通道残差幅度；11-13 至 11-26 calibration；11-27 至 12-10 assessment。这是单次时间前推 cross-fit，不是多折训练误差。
- 每日从严格合法观测随机采样最多 512 个 anchor，分两组共同遮挡本组 anchor 的速度、流量、occupancy 和 density 四通道，再用冻结模型预测已遮挡 anchor。原始 target 始终保持四通道缺失；target 的强基线预测直接复用缓存。
- 三种 GP共享同 anchor、合法状态特征、种子、残差幅度、192 诱导点、noise=.10 和 length_x=2km、length_t=.5h；传播速度固定为 +80/-15 km/h，普通核为 c=0。门控普通核隔离门控与传播的影响。
- 只沿互证拓扑、节点匹配、非分支链建立累计长度坐标，再用里程方向和距离比校验；里程不产生任何新边。可信组件以外严格回退强基线。平方根门控、分量外积、Gaussian 拉回及 Schur 乘积的 PSD 证明见 GP_MATHEMATICS.md。
- 每个核独立在 calibration 的 {0,.25,.5,1} 中选强度，保存 selection_sealed.json 及 SHA 后才读 assessment。另报告每个固定强度 1 的失败，未按 assessment 反选超参。

## Assessment 固定对照

库存列为各 regime 的公开库存增量相对 L1 再等权平均；越低越好。它不假定隐藏 boundary b* 等于公开真值增量。

### I5_N_autumn

| 模型 | State | R3 State | 库存增量相对 L1 |
|---|---:|---:|---:|
| 冻结 LGB/Attention | 0.956234145 | 0.952945305 | 0.359389584 |
| 普通 GP，固定强度 1 | 0.953627460 | 0.950424369 | 0.378033044 |
| 门控普通 GP，固定强度 1 | 0.953630869 | 0.950426231 | 0.377907772 |
| 双向传播 GP，固定强度 1 | 0.954964442 | 0.951667217 | 0.373094798 |
| 既有图求解器，同 anchor，强度 1 | 0.956231501 | 0.952942183 | 0.359359817 |
| 原始纯强 LightGBM，固定诊断 | 0.947706409 | 0.944378330 | 0.402182533 |

Calibration 选择：ordinary=baseline, gated_ordinary=baseline, directional=baseline, graph_residual=graph_residual@1.0。

### I210_W_autumn

| 模型 | State | R3 State | 库存增量相对 L1 |
|---|---:|---:|---:|
| 冻结 LGB/Attention | 0.931141064 | 0.927414352 | 0.411396328 |
| 普通 GP，固定强度 1 | 0.930012989 | 0.926319427 | 0.417843709 |
| 门控普通 GP，固定强度 1 | 0.930021295 | 0.926344031 | 0.417904760 |
| 双向传播 GP，固定强度 1 | 0.930996399 | 0.927278134 | 0.413130701 |
| 既有图求解器，同 anchor，强度 1 | 0.931131144 | 0.927428146 | 0.411356726 |
| 原始纯强 LightGBM，固定诊断 | 0.924732277 | 0.922252043 | 0.427097625 |

Calibration 选择：ordinary=baseline, gated_ordinary=baseline, directional=baseline, graph_residual=baseline。

固定强度 1 的方向核虽然比固定普通核少退化，但两者都未胜过冻结强基线，且 calibration 都选择 0；不能将“少退化”宣称为独立收益。I5 的 calibration 所选 graph@1 在 assessment 反而下降 0.000002644；I210 的 graph 也回退基线。

## 急降与恢复

事件由预测完成后的原始真值相邻五分钟速度差定义：急降 <=-20km/h、恢复 >=20km/h。下表均为 target 上的速度 RMSE，完整逐事件车道流量 RMSE 和计数在 JSON 中。

| 方向 | 急降 n | 恢复 n | 冻结 ensemble 急降/恢复 | 方向 GP@1 急降/恢复 | 纯 LGB 急降/恢复 |
|---|---:|---:|---:|---:|---:|
| I5_N_autumn | 217 | 251 | 9.553377/8.312079 | 9.561577/8.327395 | 17.538103/15.590764 |
| I210_W_autumn | 76 | 119 | 26.364790/13.327459 | 26.357379/13.324862 | 29.058029/16.707049 |

方向 GP 在 I210 急降/恢复速度上有极小的局部变化，但没有转化成独立的 State 或库存代理收益。

## 覆盖、午夜对与核查

| 方向 | 可信 links | assessment target 覆盖 | 库存有效 pairs | 其中跨午夜 pairs |
|---|---:|---:|---:|---:|
| I5_N_autumn | 257/258 | 99.604% | 578952 | 730 |
| I210_W_autumn | 15/81 | 18.851% | 145143 | 118 |

I210 的公开拓扑只能支持 7 条很短的可信链，因此其全方向阴性证据受覆盖限制。JSON 另报告可信组件自身 State，未按里程补边扩大覆盖。I5 的 2 条链覆盖 257/258 links。

每个方向首个 calibration 和 assessment 日期的全部候选均与原 scorer 逐字段核对，最大绝对差 3.45607986674e-11。纯 LGB replay 另对两个模型（纯 LGB、冻结 ensemble）、两个 role、三个 regime、两个方向，共 24 组完整日历运行原 increment_proxy；有效 pair 数完全相同，L1 最大绝对差 7.27595761418e-11。

午夜对仅连接同 panel、同 regime、相隔 <=5.1 分钟的相邻日期端点，使用起点匝道有效性和完整 link-set guard。时间戳先显式转 nanoseconds 再取 epoch 秒，避免历史微秒量纲错误。若多站点/同 link、非有限有效 anchor、缺失 target endpoint 或 regime 内各日期完整 eligible link 集合不同，则快捷 scorer 直接拒绝。

## 复现与失败记录

- GP 主运行 201.543s；进程 CPU RSS 峰值 2911.66MiB；GPU allocated/reserved 峰值 146.64/238.00MiB，低于 8GB 预算。BLAS=1，冻结 Torch 推理请求封顶为 2 CPU threads。
- 纯 LGB 固定重放 150.498s，CPU RSS 峰值 2640.63MiB，未使用 GPU。
- 11 项针对性测试通过，包括三核 PSD、传播方向、断连/潜在 link、里程拒绝、稀疏 GP 与稠密有限秩后验一致、四通道遮挡、微秒与午夜/regime 契约。
- v1/v2 均只在 calibration 中因重复 scorer 和全表 ramp 日期 strftime 性能中止，assessment 未读取。部分模型、日志、注册、源快照均保留。首日期三核 pickle 与最终 v3 逐字节一致。已记录运行时间下界约 421.11/162.00s；未收集的中止峰值明确为 null。
- 新增午夜测试最初因 Pandas 只读 to_numpy view 的测试构造失败 2 项；test-only copy=True 修复后 11 项通过。只读进程查询的 CIM 不可用、可选 psutil 未安装也记录在中止证据中，未安装新依赖。
- 未部署 GP，未向 Kaggle 上传，未新增任何正式分数。跨季节 GP 尚无证据：首轮未通过晋级，故停止扩大。

主运行命令：

```powershell
.venv-research/Scripts/python.exe scripts/v2/run_directional_experiments.py --cache-root artifacts/research/v2_state_v2 --output artifacts/research/GP_v2_same_day_autumn_v3 --case I5_N_autumn --case I210_W_autumn
.venv-research/Scripts/python.exe scripts/v2/run_directional_experiments.py --replay-lightgbm reports/v2/GP_v2_same_day_autumn_v3.json --cache-root artifacts/research/v2_state_v2 --output artifacts/research/GP_pure_lightgbm_replay_v1
```

运行时基准 commit：`9f65c96857eb0be8373eb30cf0b18d1ecf39130d`，包含当时未提交研发修改的精确源快照为权威复现对象；最终提交由主进程另行记录。

| 文件/模型 | SHA-256 |
|---|---|
| GP 主报告 | `f5d0b45df9679903ee28501cae1b4c3f42cbb8edcd9fd75df73b30fc673cb9ca` |
| GP 主源快照 | `9f2edbdf0fb651bf7852a4208349d5760f1f6732bf0c612925fe9cdf6cee92f1` |
| 纯 LGB 重放报告 | `539dfcd445119b018a3414ad8edbccbcf28fea7166666a645eaa48b068f4d2aa` |
| 纯 LGB 重放源快照 | `2dfa91ffea3e7a37c1344c4a48e0068e79366d27889813f7c11fa7a949eb302d` |
| I5_N_autumn 冻结 ensemble | `d5067934dec3facdf66ca1f4d22089aa69e646c9329a63ddbaeabfabc0d83cef` |
| I5_N_autumn 纯 LGB 嵌套原始 pickle | `0252263f10bb6aaa57eaf329eb1aadbe672009ed0008e6b7eba22f63a38aa0b4` |
| I5_N_autumn calibration seal | `7b1edd3cd30e2e06a9abc557c3acd983b25701dd4ba84a6cf42538059afbfe14` |
| I210_W_autumn 冻结 ensemble | `d850f1f7ec9409b4bb186e36424331fcdb98ad263004ca35289306662e9ddc25` |
| I210_W_autumn 纯 LGB 嵌套原始 pickle | `bf33afdbfd566cdc41a2f4087a6d191940209c2eb2123524524de72414136bb5` |
| I210_W_autumn calibration seal | `d2d9c4845a82cd4ed2f563b0a21de03d498ab3cbafe43caa5c902e2f812be6e9` |

168 个 GP 后验模型 SHA、56 份合法 anchor cache SHA、graph 输出 SHA、全部 GP cache SHA 与 calibration/assessment 的 raw/masked SHA，以及源文件清单在主 JSON 和源快照中；correction_fit 原始数据来源另见第一阶段缓存 manifest。LGB 重放保留 56 个固定预测缓存的 SHA 和整批 pair 复核明细。
