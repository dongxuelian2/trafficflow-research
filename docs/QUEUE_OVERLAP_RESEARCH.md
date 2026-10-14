# 窗口级 Queue 目标与观测标签审查

正式最佳 .82940，.95000+ 尚未达到。C42–C48 为本地研究；不发布、不推送、不上传。

## 设计与方法

使用 C38 已独立重建的九通道因果序列和完整六步概率缓存。两季、optimizer seeds 20261013/20261029；事件 sampler 固定20261013。三个 arm 共用 30min graph、hidden32、同初始化/每个 batch，20×80、batch4、AdamW .001/wd1e-4/clip1。原 equal-window pos_weight4 BCE+.1 speed MAE 不变，另外分别加 .25 soft-IoU 或 Lovász binary hinge。Lovász 按 [Berman 等论文的集合扩展定义](https://arxiv.org/html/1705.08790) 自主实现，没有复制第三方参赛代码或预测。

每个窗口只在 TRAIN eligible 标签上计算 overlap；onset 只用 T+30 分类域。soft-IoU 对非空 truth 为 1−Σpy/(Σy+Σp(1−y))，空 truth 显式采用 mean(p)。Lovász 按 margin errors 降序排列，计算每个前缀 Jaccard loss 的差分并加权正部 error。64 个二值顶点以独立集合枚举核对；测试还验证被屏蔽格和 onset 早期零梯度、窗口等权。

checkpoint 统一选每五 epoch 的最低 cal Brier。只在完整预测域校准 α/threshold/force，先施加公开 onset 时序，future mask 只作 scorer。全部选择先封存再读 assessment。新 loss 下 logit−log4 不再保证校准概率，故只把 sigmoid 输出称 score。辅助 speed loss、cal checkpoint 及 threshold 都可能改变实际训练策略，不把指标变化全归因单个 gradient。

## 结果：公开观测 proxy

| period/seed | BCE | soft-IoU | Lovász | primary |
|---|---:|---:|---:|---|
| 秋/20261013 C47 | .879028 | .842850 | .872422 | BCE |
| 秋/20261029 C42 | .872422 | .872422 | .872422 | tree |
| 冬/20261013 C48 | .879943 | .878782 | .857370 | BCE |
| 冬/20261029 C43 | .881507 | .878744 | .878723 | Lovász |

数值为去重 origin 后 condition 等权的 IoU。秋新 seed 全部 α0，因此 loss comparisons 为零并不表示 raw networks 相同。冬新 seed 的 cal 获选 Lovász 比同 seed BCE 差 .002783，CI[-.006516,.001192]；没有反选 BCE。soft-IoU 冬新 seed −.002763，CI[-.004825,−.000714]；原 seed 冬 Lovász −.022573，CI[-.045600,−.005924]。两季各109/103个独立origin仍是已查看开发日期；两个 optimizer seed 共用数据，不是新的事件样本。

C46 的16个等 α 次要对照保存全部正负结果。原 seed 冬 Lovász α.25 为 .883696 vs .879943，gain .003753、名义CI[.000310,.007953]；cal 原选 α.5，另一 seed 不复现，故不晋级。soft/Lovász 纯网络相对纯 BCE 有正区间，但最高 .830712，仍低于强 tree .868240。没有多重比较校正，也没有把次要最优当 primary。

## 公开窗口审查

C44 读取八个 scored panel 的80个公开 TRAIN 窗口。history condition 与观测均一致；40 ongoing 的首个正例均 T+5，39 onset 为T+30。I5_S `Q2_D12_I5_S_train_005` 没有任何 eligible 阈值正例，当前 `sample_events` 会排除。C45 在质量过滤之前的全部有限速度上也无正例，不能把差异归因质量过滤；潜在 queue_true 未发布，原因未知。

这些窗口的 index future_coverage=1，而实际 eligible 观测覆盖 .721713–.798354，字段不能被当作未来评分域。7100个 finite threshold positives 中6457个满足评分质量，只是观测诊断。没有读取 validation/private truth、寻找隐藏组织方资源或将该诊断用于某个隐藏实例推断。

## 复现与资源

训练命令：`.venv-research\Scripts\python.exe scripts/queue_overlap_experiment.py --source-study C32_D12_I5_N_wave_graph_Queue --seed 20261029 --output artifacts/research/C42_I5_N_autumn_window_overlap_seed2`。冬替换 source 为C34，原 seed 改20261013；输出目录必须新建，禁止覆盖。

四组 wall time199.11/216.23/203.29/216.44秒，约110.33MiB GPU allocated峰值。末组与部署部分并行。四组全部 cal histories 独立重建，raw probabilities 和导出完整模板 decisions 各660次精确；两个原 seed BCE checkpoint 全张量复现 C32/C34。首次 verifier 因 pandas 不接受 numpy.str_ 失败，旧source zip及失败记录保留，转换str后新目录复跑。成功部署73.53/29.37秒。

71 research tests passed；CPU59 passed/4 optional-Torch skips。正式分项未知；原正式 submission SHA及K06都保留。下一步优先跨种子概率组合/选择不确定性和独立 latent/observation 事件诊断，随后跨方向；不继续盲扫当前 overlap λ。
