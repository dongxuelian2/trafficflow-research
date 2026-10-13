# 当前实现与评分接口审计

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
| `queue_events.py` | 均匀打乱候选、条件配额、六小时间隔、quality 和 persistence 筛选 | 官方 selector 未公开，仍为带噪观测代理；不同 seed 可能抽到相同窗口 |
| `runtime.py` / evidence | CPU 实验 source fingerprints 与进程峰值 | GPU 训练另存 device / torch / actual GPU peak；进程 lifetime peak 不是隔离模型峰值 |
| 下载 / API / 发布 | 请求官方 public participant APIs 与本队 submissions；秘密和签名 URL 不写日志 | Kaggle API token 不能替代网页登录；论坛公告发布授权已取得，网页会话尚待恢复 |

当前独立 CUDA 环境 **21 tests passed**；CPU 环境 **20 passed / 1 optional-Torch skipped**，源码 compileall 通过。A01 与 A09 中相同缓存、相同 seed 的三个共同模型独立训练后，tensor 和模型文件 SHA 完全相同；这不等于全量独立重训首轮七百万预测的复现。

A14 审计发现重要比较混杂：原 graph 与 LGB 的日期相同，但 LGB 只用 300k 标签。全标签 LGB 1,804,908 行得到更高 State，A15 重做强基线对照后 graph 独立收益仅 0.001077。所有旧实验保留，解释同步修正。

开放瓶颈：缺乏正式分项；独立模拟分布与赛事分布失配；Queue latent labels 未公开；物理 flux 投影使单纯 CTM surrogate 不可靠；OD 大零空间；Graph 单走廊模型的跨 family 泛化尚不确定。
