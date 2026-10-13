# TrafficFlow Research — IEEE Big Data Cup 2026

独立四任务研究系统。0.940+ 是研发目标，当前没有测得的公开榜成绩。
官方数据下载返回 HTTP 401；已完成的测量来自人工 integration fixture，不能代表比赛性能。

## 环境和数据

使用 Python 3.13。依赖固定在 `requirements.lock.txt`，实际环境记录在 `reports/environment.json`。

```powershell
.\scripts\setup.ps1 -Python 'C:\path\to\python.exe'
.\.venv\Scripts\python.exe scripts\verify_vendor.py
```

先在自己的 Kaggle 账户完成必要登录/规则接受，在本机配置 Kaggle API 凭据；不要提交密钥到仓库或聊天。
下载只访问官方 competition endpoint，HTTP 状态保存为 `data/download_status.json`。

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py --output data
```

如解压结果多一层目录，`--release-root` 应指向同时包含 `corridors/`、`task1/`、`task2/`、`task4/` 的 `kaggle_public/`。
必须使用 2026-09-11 重建之后的当前官方包；13 GB 是官方文档的近似解压大小。

## 单入口完整流程

```powershell
.\scripts\run.ps1 -ReleaseRoot E:\ieee\data\kaggle_public
```

该命令依次验证和记录数据 SHA-256，按日期划分 profile/fit/calibration/test，比较五种状态模型、三个队列模型及 OD 优化方法，按 calibration 选择状态和物理设置，再拟合最终模型、预测三个 split、严格合并三类提交。
Task 3 使用 Task 1 文件，不产生额外提交。任务输出缺失、键重复、非法值或主 key 未覆盖会使本流程失败，避免静默填零。
完整官方包尚未实跑，可能需要依据实际 schema 做兼容修正。

## 分步运行

```powershell
.\.venv\Scripts\python.exe -m trafficflow validate --release-root data/kaggle_public --output artifacts/audit --full-hash
.\.venv\Scripts\python.exe -m trafficflow official-baselines --release-root data/kaggle_public --split train --output artifacts/official
.\.venv\Scripts\python.exe -m trafficflow experiment --release-root data/kaggle_public --models-dir artifacts/experiments --panel D12_I5_N
.\.venv\Scripts\python.exe -m trafficflow fit-final --release-root data/kaggle_public --models-dir artifacts/experiments --panel D12_I5_N
.\.venv\Scripts\python.exe -m trafficflow predict --release-root data/kaggle_public --models-dir artifacts/experiments --panel D12_I5_N --output artifacts/panel_predictions
```

单走廊用于开发；完整合并要求覆盖主 key 的所有走廊和 split。`--split all` 默认预测 train/validation/private。
`baseline` 是独立的历史均值、persistence、当前 counts 正则 NNLS；`official-baselines` 执行未改动的官方脚本，两者 OD baseline 的 count 来源不同。

```powershell
.\.venv\Scripts\python.exe -m trafficflow baseline --release-root data/kaggle_public --output artifacts/independent_baseline
.\.venv\Scripts\python.exe -m trafficflow score --release-root data/kaggle_public --split train --output artifacts/independent_baseline
.\.venv\Scripts\python.exe -m trafficflow merge --release-root data/kaggle_public --output artifacts/independent_baseline
```

## 无官方数据的功能复现

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m trafficflow fixture --release-root artifacts/demo_data --days 42
.\.venv\Scripts\python.exe -m trafficflow run --release-root artifacts/demo_data --models-dir artifacts/demo_models --output artifacts/demo_run --iterations 150 --max-rows 100000
```

人工数据带 `FIXTURE_ONLY.json`，不会发布真实 queue labels 或隐藏 OD reference。不能把生成的 fixture submission 上传比赛。
模型 pickle 仅加载本流程自己产生的可信文件。预测输出目录需使用新目录，确保旧文件不会混入。

## 方法与指标边界

- State：历史 profile、按物理位置插值、Kalman/RTS 残差平滑、LightGBM/CatBoost 残差回归。特征只来自 masked 当日，profile 来自更早日期。
- Queue：只读窗口 `[T−60,T)`，训练代理标签由公开 train 观测速度和官方 cutoff 构造。波到达不确定性、趋势、上下游特征和按窗口 IoU 的阈值搜索；真实 S_queue 未知。
- Physics：缺失 ramp 不置零；仅在边界可观测时计算守恒残差。可选稀疏非负 flow 调整，只修改 target cells；真实 S_physics 未知。
- OD：稀疏 measured-link operator、正则 NNLS、相对先验和 KL/概率先验。只报告 S_link、prior change、sensor holdout；真实 S_ODME 未知。

综合分数固定为 `0.35 State + 0.30 Queue + 0.15 Physics + 0.20 ODME`。未知项保留 null，不以 proxy、physics diagnostic 或 S_link 代替。
详见 `RESULTS.md`、`docs/METHODS.md` 和 `docs/RESEARCH_NOTES.md`。

官方 MIT 工具包置于 `vendor/official`，固定到 `c88cddf533bbf0afa4ff1fc6c031d760a08e7a31`；33 个文件通过 Git blob 校验。
第三方方案仅阅读公开架构说明，未导入其源代码、数据或预测。仓库创建在本机，未发布到 GitHub。
