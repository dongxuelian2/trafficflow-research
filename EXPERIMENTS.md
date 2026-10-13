# 实验台账 — 0.950+ 攻关

所有模拟指标、观测代理与正式 Kaggle 指标分别记录。失败实验不删除，既有模型与提交不覆盖。

| 编号 | 假设 / 操作 | 数据与划分 | 结果 | 资源 / 决策 |
|---|---|---|---|---|
| F00 | 冻结首轮正式提交 | 本人 Kaggle 回下载，与本地文件 SHA 比对 | 正式 Total **0.82940**，ref 57138874；四项正式分数未知 | 106 个冻结文件，712,127,414 bytes；保留基准 |
| P00 | 公共拓扑是否导致部分方向无法评分 | 公开拓扑、原版 scorer、人工理想两时刻状态 | 十方向全部有可评分连接，否定结构性 0.950 上限猜想 | 本地诊断漏掉外部连接；修复并测试 |
| B01 | CTM、缺失、事故和噪声压力实验 | 独立 54 个场景，3 seed × 6 条件 × R1/R2/R3 | 守恒误差 ≤6.76e-13 vehicles；State observation oracle 1.0，latent oracle 0.952006，时空插值 0.887547，RTS 0.848378 | 7.60 秒 CPU；模拟数据不等于正式生成器 |
| D01 | 分层贝叶斯 OD 对比 Ridge | 两走廊、独立已知真值模拟 | 初步增益，但 zone set 的迭代顺序依赖 Python hash seed | **不作为可复现结论**；保留记录，改为 sorted 后重跑 D02 |
| D02 | 固定组顺序的分层贝叶斯 OD | 十走廊，每方向 3 生成分布 × 3 assessment seed；selection seed 独立 | 已完成，详见 JSON；尚无正式 OD 增益证据 | public_reference 从未读取；正式候选须保持其他任务相同 |
| C01 | 随机事件采样、onset/ongoing 分别建模 | rolling_autumn；按公开限制构造带噪观测代理 | 运行中 | 同 14 天抽样种子会重叠，不视为独立样本 |
| ENV01 | 独立 CUDA 环境 | Python 3.13.1 / RTX 4060 8GB | ensurepip 在当前系统失败；uv 已启动官方 torch 2.8.0+cu128 安装 | 原 .venv-repro 不修改 |

复现入口：`scripts/register_research_protocol.py`、`audit_physics_topology.py`、`simulate_stress.py`、`od_probability_experiment.py`、`queue_event_experiment.py`。
正式反馈只读本队提交。禁止隐藏标签、组织方内部资产和其他队伍预测。

现有首轮 January–February test 已经查看；新 rolling assessment 仍是公开 train 开发证据。独立模拟 final seed 7907/8017/9109 尚未生成，需在架构和参数封存后一次性评估。
