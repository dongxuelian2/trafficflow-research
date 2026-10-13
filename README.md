# TrafficFlow Research — IEEE Big Data Cup 2026

独立、可复现的四任务交通流研究系统。0.940+ 是研发目标，正式公开榜成绩仍未知。
当前 Kaggle 官方包已下载、CRC/大小/SHA-256 校验完成，数据位于 `data/kaggle_public`。测量结果见 [RESULTS.md](RESULTS.md)。

## 环境和数据

Python 3.13.1；60 个依赖固定在 `requirements.lock.txt`。正式数据实验使用不继承系统包的 `.venv-repro`，每模型 4 CPU threads，没有使用 GPU。

```powershell
.\scripts\setup.ps1 -Python 'C:\path\to\python.exe'
.\.venv\Scripts\python.exe scripts\verify_vendor.py
.\.venv\Scripts\python.exe -m pytest -q --basetemp artifacts\pytest_readme
.\.venv\Scripts\python.exe scripts\download_data.py --output data --workers 8
```

下载器使用本机 Kaggle API 凭据，支持标准 Range 断点续传，不记录密钥或签名 URL。
当前官方包创建于 2026-09-11，共 9,688 文件，归档 8,335,874,386 bytes，解压 12,387,782,323 bytes。归档和逐文件指纹保存在 `reports/official`。
数据是由真实走廊参数校准的合成交通记录，不能匹配外部真实日期。

## 本轮最佳方案的复现入口

在仓库根目录运行，选择新的 RunRoot。入口重新完成数据审计、十走廊时间验证、最终拟合、八走廊拥堵校准、模型冻结和完整提交合并。

```powershell
.\scripts\reproduce.ps1 -ReleaseRoot E:\ieee\data\kaggle_public -RunRoot artifacts\reproduction_01 -Python .\.venv\Scripts\python.exe
```

Task 1 比较历史 profile、插值和 V2 时空/流量/密度上下文 LightGBM，最多 300,000 training rows/走廊，400 iterations、63 leaves；bounds 仅由 calibration 选择。
Task 2 比较 400/23 与 600/63 两组 LightGBM、全局/condition 阈值和公开的非空 horizon 先验，使用 train 观测代理标签校准。
Task 3 使用 Task 1 输出，只记录公开可观测诊断，不产生额外提交。Task 4 使用当前 split counts/prior 的 measurement-space 非负 ridge λ=0.05。

本轮 artifact：`artifacts/campaign_v1`、`artifacts/queue_refinement`、`artifacts/best_models`。
完整提交：`artifacts/submission_best/submission.csv`，涵盖 validation/private 全部主 key 行。数据和大文件未纳入 Git；测量摘要、消融、运行命令和校验在 `reports/official`。
弱守恒候选：`artifacts/submission_physics/submission.csv`。复现时增加 `-PhysicsCandidate`，在 calibration State 损失≤0.004 的范围内选择可观测残差较低的设置；未知边界不补零，不能据此宣称正式 Physics 提高。

## 官方基线和研究入口

```powershell
.\.venv\Scripts\python.exe scripts\run_official_state_baselines.py --workers 2
.\.venv\Scripts\python.exe scripts\run_official_diagnostics.py
.\.venv\Scripts\python.exe vendor\official\src\task2\build_task2_persistence_submission.py --release-root data/kaggle_public --split all --output artifacts\official_queue.csv
.\.venv\Scripts\python.exe -m trafficflow experiment --release-root data/kaggle_public --models-dir artifacts\exploration --panel D12_I5_N
.\.venv\Scripts\python.exe -m trafficflow predict --release-root data/kaggle_public --models-dir artifacts\best_models --panel D12_I5_N --output artifacts\repeat_panel
```

`experiment` 比较历史、插值、Kalman、LightGBM、CatBoost、物理约束和五种 OD 方法。`scripts/state_research.py` 比较 V1/V2 与联合密度投影，`scripts/queue_research.py` 比较 causal 多跳上下文，`scripts/scenario_audit.py` 检查急降速/恢复/稳定的观测代理场景。
Task 1 targets 来自模板；Task 2 只读 `[T−60,T)`，隐藏 split 的 raw reader 显式拒绝访问。严格合并检查键、覆盖、有限非负值、binary queue 和跨 chunk 重复，不会静默补零。pickle 只加载本流程自己生成的可信文件。

## 指标边界

正式总分为 `0.35 State + 0.30 Queue + 0.15 Physics + 0.20 ODME`。公开 train State 可按官方公式验证；queue proxy IoU、physics residual 和 OD S_link 不能替代正式隐藏分项。未知项保留 null。
D12_I5_N 是开发走廊，test 已多次查看。时间组与选择协议见 [方法](docs/METHODS.md)。人工 fixture 的旧轮结果保存在 `reports/fixture/RESULTS_PRE_OFFICIAL.md`，只证明集成流程，不能上传其 submission。
官方 MIT 工具包固定到 `c88cddf533bbf0afa4ff1fc6c031d760a08e7a31`，33 个 Git blob 已核验。第三方只阅读公开架构说明，未导入代码、训练资产或预测。规则差异和许可证见 `docs/RESEARCH_NOTES.md`、`docs/EXTERNAL_ASSETS.md`。
GitHub 仓库：[dongxuelian2/trafficflow-research](https://github.com/dongxuelian2/trafficflow-research)。数据和模型产物保留在本机；没有发送注册邮件或上传 Kaggle。
