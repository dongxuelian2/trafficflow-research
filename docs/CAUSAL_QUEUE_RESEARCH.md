# 因果 Queue：概率修正与公开波速邻域

输入严格为 T−60 分钟至 T 的十三个五分钟 slot：速度/车道流量、可用标记、occupancy/标记、日内相位及已提供的 condition。T 行使用已发布 masked 数据；不读取 >T 数值，也不将离线 State 的未来重建值传给 Queue。六个未来速度与拥堵只作为公开 train 的监督标签。标签来自 noisy observations，不冒充 underlying latent Queue。

C30 同预算训练 temporal 与 directed graph（各 20,076 参数），相对固定共享 LGB 均由 cal 选择 α=0。正类权重 w=4 的理想 BCE 最优 odds 满足 `logit(r)=logit(p)+log(w)`；C31 固定 checkpoint 对比 `sigmoid(logit-log(4))`，graph Brier .015419→.012059，但 IoU 混合仍选 0。独立期望损失最小化测试验证公式；有限网络、辅助 speed loss 与事件抽样仍可能不校准。

`--correct-probability` 显式开启修正，默认保留 C30 原行为。C33 冬季使用修正后的 Brier 选择 checkpoint，普通图的去重代理 gain .008485，CI 跨零。

C32/C34 的三种图范围使用相同 cache、loss、优化器、patch RNG、宽度和预算。公共 wave speed 为 capacity/(jam−critical)，根据公开有向路网最短 travel time 构造 15/30 分钟指数衰减邻域；direct graph 是对照。重复层和 GroupNorm 能传递更远信息，因此它是表示而非硬传播锥或正式 CTM 约束。

两季 cal 均选 30min/α=.25，去重代理增益秋 .006606、冬 .011703，day-block CI 均跨零。秋季未选的 15min 更好，保留原 cal 选择，不反选。源树逐窗 IoU、采样 metadata、未来 target order 与缓存 SHA 均独立核对。NN 使用更丰富的历史/condition，树对照不能单独归因架构；三图范围之间控制相同输入。

复跑必须使用新目录：

```powershell
.\.venv-research\Scripts\python.exe scripts/causal_graph_queue_experiment.py --matched-study reports/research/C29_D12_I5_N_winter_onset_specialist.json --correct-probability --output artifacts/research/C33_independent_repeat
.\.venv-research\Scripts\python.exe scripts/queue_wave_graph_experiment.py --directory artifacts/research/C33_independent_repeat --output artifacts/research/C34_independent_repeat
```

公开 onset 时序只在最后一步出现 Queue，来自组织方公告 742350。其他正式 selector 与 latent labels 未公开。模型范围/ensemble/阈值全部在 calibration 封存后再加载 assessment 数组；所有当前 assessment 都是已查看的开发日期。晋级仍需跨方向、多 fit seed 的可靠独立收益，真实 Kaggle Queue 未知。
