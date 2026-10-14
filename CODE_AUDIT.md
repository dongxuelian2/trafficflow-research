# 当前实现与评分接口审计

本轮新增核验：六个 adaptive State 的独立 14 日 calibration 重建/预测均通过；A44 各成员在所有 cal/assessment 日复现，组合 wrapper/save/reload 只额外核验一个完整 assessment 日，不能把它写成组合的完整独立训练。Queue 原树在 C31–C34 的相同事件上逐窗复现；所有 NN 输入均来自 T 及此前一小时，波速图只用公开几何。

Entropy 双空间 Newton 与独立 path-space 优化器及大数消减 scalar oracle 通过对照。D06 的 69 个失败来自目标值浮点分辨率附近线搜索停滞；D07 出现一个新失败，D08 改用与误差预算匹配的 Newton decrement 后全部闭合。tol 2e−6 未放宽；生成器/种子/全局选择 gate 不变，Ridge 原分数完全相同，既有成功 entropy 分数漂移最多 2.32e−9。数值回归不等于独立泛化或正式 OD 成绩；约 99% 解变化在计数零空间。

当前研究环境 **43 passed（11.07 秒）**，CPU **38 passed / 3 optional-Torch module skips（6.51 秒）**。Causal Queue 的 class-weight 修正是显式 flag，默认保留 C30 原行为。旧源代码和所有失败实验均在各自 start snapshot/报告中保留；后续源文件修复不改旧模型或提交。

基准源码固定在 `15036c8`；正式提交 ref 57138874 回下载 SHA 与本地文件一致。官方代码为 `c88cddf533bbf0afa4ff1fc6c031d760a08e7a31`，2026-10-13 再次 ls-remote 核查仍为此版本。完整公开 Rules、Evaluation、Foundational Rules 与 Timeline 已读取并保存于 `data/published_competition_pages.json`。

| 路径 / 接口 | 审计结论 | 证据 / 限制 |
|---|---|---|
| `data.py` / Release | raw observations 仅允许 train；target 模板、ID、quality 和 ramp missing/structural-zero 分开 | 原全数据 fingerprint 与 contract tests；不读取隐藏 split raw |
| `submission.py` | SQLite 全局唯一键、ID、完整覆盖、有限非负、Queue binary、跨 chunk 重复检查 | 两次完整 6,980,503 行校验；K01 再做逐行单任务 diff |
| `metrics.py` / official evaluator | 本地 State / proxy IoU / S_link 有定义；真实其他三分项不可本地计算 | Evaluation 的完整任务缺行记零比 vendor 局部 missing-zero 更严格；上传前完整校验使这一区别不会进入候选 |
| `state.py` | mask-only 特征；离线插值与 RTS 可以使用目标之后的公开观测 | 合法 offline；未把 unmasked targets 放回输入 |
| `experiments.py` | 首轮单时间划分、calibration 选择 | 首轮 test 均已查看；新 rolling 不冒充未知最终 test |
| `queue.py` / `queue_calibration.py` / `causal_queue.py` | 旧模型保持严格 [T−60,T)；新模型显式允许已发布的 T，拒绝任何 >T 输入；nonempty prior 来自公开合约 | T 行用 Parquet 谓词读取；未来值扰动不改变输入的测试通过；旧代理采样分布有偏差 |
| `physics.py` | 外部连接以前被误当作 unavailable，导致五方向诊断覆盖为零 | 已修复并新增 external-vs-internal 回归测试；仍是 incomplete topology fallback，不能称正式 Physics |
| `odme.py` | measurement-space 非负 Ridge，KKT 与官方数值 parity 已核查 | counts 可识别维数远少于路径维数；S_link 不代表 path truth |
| `bayesian_od.py` | 显式分层 Gaussian prior，nonnegative MAP、未截断 posterior SD | D02 自建模拟增益没有迁移；K01 正式总分 0.76474；拒绝晋级 |
| `digital_twin.py` | 独立 Godunov/CTM、CFL 子步、显式 ramp supply、车辆数守恒 | B01 54 场景最大守恒误差 6.76e-13；两种模拟 flux contract 并列 |
| `graph_state.py` / graph experiment | missing-aware temporal conv + directed graph diffusion；occupancy 仅来自 masked 输入；joint speed/flow residual | 日前 profile、相同 fit/selection/assessment 日期、selection seal 后 assessment；GPU 344 MiB；多 seed/跨走廊继续研究 |
| `queue_events.py` | 均匀打乱候选、条件配额、六小时间隔、quality 和 persistence 筛选；C24 起可选 onset 首次观测 queue at T+30 | 组织方公告 742350 公开时序属性；其余 selector 未公开，仍为带噪代理。C01–C22 不匹配时序，仅压力测试；不同 seed 可重复 origin |
| `runtime.py` / evidence | CPU 实验 source fingerprints、进程峰值、训练前源码 ZIP | A29/A30 新实验在训练前保存源代码、依赖锁与逐文件 SHA；过去实验仅有 Git/运行指纹与模型校验 |
| 下载 / API / 发布 | 请求官方 public participant APIs 与本队 submissions；秘密和签名 URL 不写日志 | 用户最新指示先做模型；本阶段只做本地 Git 提交，不处理论坛/Notebook、GitHub 推送或新正式上传 |

上一阶段独立 CUDA 环境 **35 tests passed**（14.32 秒），CPU 环境 **30 passed / 3 optional-Torch module skips**（8.73 秒），源码 compileall 已通过。测试核验事件权重、density projection、target 隔离、锚点同化、daily pooling、onset 时序；新增独立枚举 expected-IoU 最优集合、density 缓存 key 对齐/不变、因果 Queue 十三步历史拒绝未来值/quality 过滤/无观测 fallback/onset-loss domain 检查。测试临时目录位于工作区 .cache。A01 与 A09 的共同模型 tensor/SHA 完全一致；C24–C29 三组共享模型独立重建/拟合后树文本、阈值、逐窗 IoU 一致。均不等于正式七百万行提交的全量独立复现。

A14 审计发现重要比较混杂：原 graph 与 LGB 的日期相同，但 LGB 只用 300k 标签。全标签 LGB 1,804,908 行得到更高 State，A15 重做强基线对照后 graph 独立收益仅 0.001077。所有旧实验保留，解释同步修正。

开放瓶颈：缺乏正式分项；独立模拟分布与赛事分布失配；Queue latent labels 未公开；物理 flux 投影使单纯 CTM surrogate 不可靠；OD 大零空间；Graph 单走廊模型的跨 family 泛化尚不确定。

A33–A36：可变节点的 FD 几何条件与拓扑偏置图注意力通过形状/有限梯度测试。四个 standalone 各在完整 14 天 calibration 独立构造输入并预测，数组逐通道一致，State/密度诊断匹配封存结果。配对架构消融复现原赢家 assessment，原选择保持。A38/A39 场景差分按 station AND link 分组，且严格相隔五分钟；原历史 scenario 脚本也已修正，当前 I5_N 原 50 日复跑的全部指标/计数仍相同，新文件另存。训练前源码快照是主记录；运行结束的 on-disk fingerprint 可能包含期间新增的部署代码。当前正式成绩仍为 .82940，未上传新候选。

K03/K04 各 6,980,503 行严格自然 key 覆盖通过，独立 streaming diff 同时验证 task、数值字段、方向与 master-key 身份。只有 I5_N 的 1,131,560 State rows 改变；原 Queue/OD 文件 SHA 相同，其他 State 方向字段逐行相同。不是全量独立训练复現或正式成绩验证。A38 初版 runtime 的 `gpu_used=false` 是调用 evidence() 时漏传参数，不能用它判断设备或 GPU 峰值；数值与部署验证不受影响。脚本现在记录实际 CUDA availability，A39/生产候选记录为 true；A38 原记录与训练源码快照保留。

`public_onset_timing()` 只约束标记为 onset 的 T+5…T+25，保留 ongoing 与 T+30。模型默认不开启，新公告对齐实验显式设置标记，旧冻结模型行为不变；针对条件/lead 的测试通过。K02 使用本队冻结预测与公开 metadata，未读任何隐藏标签或 val/private 未来数值；完整 6,980,503 行独立 diff 仅 5 个 Queue 值。计分 eligibility 未公开，因此更紧的 .001369 Total 界必须注明条件，安全界为 .0075。
