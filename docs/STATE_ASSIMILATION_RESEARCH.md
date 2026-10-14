# 冻结先验的观测置信度同化：A58–A65

正式最佳仍为 .82940；这些实验不是 Kaggle 测量。没有推送、发布或正式上传。原最佳、全部训练好的先验及完整候选不覆盖。

## 模型与数据边界

冻结已有神经网络/LGB 组合，将每个 link×5min 的速度和每车道流量预测分别以25、600归一化为 m。由 supplied masked public frame 提取合法测量 z：严格 eligible 且有限的观测权重为1；其他有限观测需有已知质量，权重为 .25×min(pct_observed/.75,1)²。缺失或质量未知的附加观测权重为0。推断不读取 raw train 标签。

求解归一化残差 δ 的严格凸二次型：

`λ₀ ||δ||² + Σ wᵢ (δᵢ − (zᵢ − m_anchor,ᵢ))² + λₜ Σtime (δᵢ−δⱼ)² + λₛ Σgraph (δᵢ−δⱼ)²`

图边取公开拓扑中压缩的相邻检测器，无向残差平滑不能解释成真实拥堵波动力学。λ₀>0确保无观测格点也可唯一求解。Jacobi预条件CG的rtol为1e−8，最多400步，实际相对残差必须≤2e−8；非收敛即失败，不返回伪分数。最终输出 m+αδ并施加公开FD上界。

五个因素组依次为 strict temporal、加入质量值、固定Huber式创新权重、加入图平滑、按先验急变抑制时间边。归一化创新cutoff=.4/.2，对应10km/h和120vph/lane；时间边可用相同cutoff的指数权重。每组λ₀∈{.25,1}、λₜ∈{1,4}，图组λₛ=.5，共20个配置；α∈{.25,.5,1}，另有α=0的冻结先验，共61个cal候选。权重是校准选择的代理，并非已辨识测量方差或可靠Bayesian posterior。

只在原calibration选最高pooled State；另选最小旧target-touching增量代理、cal State不低于先验−.001的joint候选。两者均先封存，再读取原assessment缓存。每个因素组也与α=0对照选择；未胜出方案不强制进入assessment。所有日期仍是已查看的公开train开发集，没有启用final simulator seeds。

## 遮观测残差

A60/A61/A63仅改变创新的中心：每个有有限v或q的公开可见cell，按固定seed、panel/date与grid顺序随机分成5组。每组临时遮住自身v、q、occupancy与density_occ四字段，再重新构建StateFeatures并用同一个冻结先验预测一次。每个观测仅被遮一次，其他组保持可见。原始Task1目标先验m、观测值与solver网格均保持不变；只有m_anchor来自遮观测预测。

这是额外缺失条件下的推断，不是重新交叉拟合训练标签，也不保证无偏残差。原始条件先验可能已经读取同一个观测；遮观测会提高anchor innovation RMS，这不能当目标恢复变好或潜在真值精度提高。方法需要额外5次完整先验推断，收益应与计算消耗比较。

## 实测入口与可复现性

三个冻结先验分别为A49 I210_W秋季、A34 I5_N冬季、A43 I405_N冬季；每组14日cal/14日assessment，各保留普通和遮观测两条路线。具体封存选择、分项、区间和资源见A58–A64 JSON与MODEL_COMPARISON.md。没有按assessment选择更高分模型，新的State增益仍需达到原.002门槛及跨seed/方向证据。

A58初次启动在第一份cal输入前因pandas `datetime64[us]`整数单位被误当ns而退出；无成绩或模型产生。原注册与源归档保留。修复为显式转ns后校验5分钟，新增测试同时拒绝真正的10分钟间隔。

独立验证从原masked Parquet重建全部cal日期的输入，加载保存pickle并比对封存预测；若primary/joint文件SHA相同，明确共用该独立复跑。CG另与SuperLU直接求解比较真实24-slot子集；不把该子集冒充全日直接求解。五项新测试覆盖独立手工normal equations、无观测/未知质量、鲁棒权重与SPD、时间单位/真实间隙，以及四通道确实被遮且原帧不变。

```powershell
.\.venv-research\Scripts\python.exe scripts/state_assimilation_experiment.py --cache artifacts/research/A51_quality_D7_I210_W_autumn_cache --prior-file artifacts/research/A49_adaptive_attention_D7_I210_W/State_selected.pkl --output artifacts/research/A58_replay_fresh
.\.venv-research\Scripts\python.exe scripts/state_assimilation_experiment.py --cache artifacts/research/A02_D12_I5_N_winter_seed20261013 --prior-file artifacts/research/A34_adaptive_attention_strong_winter/State_selected.pkl --output artifacts/research/A61_replay_fresh --anchor-folds 5
.\.venv-research\Scripts\python.exe scripts/verify_state_assimilation.py --study artifacts/research/A61_replay_fresh --output reports/research/A61_replay_fresh.deployment.json
```

新目录/报告必须不存在，禁止覆盖原版本。依赖沿用现有NumPy/SciPy/pandas和已冻结GPU先验，不需要付费云服务。State可使用已公开的前后观测；这些双向同化结果禁止作为Queue起点T的未来输入。

## A65 对损失覆盖假设的更正

A65独立读取公开TRAIN eligible标签，保留原120日缓存的所有数组，另加y_full、eligible_full及Mode A lwr_pair_valid。raw标签只能用于训练/评估loss，不能进入x；目标标签与原y逐值一致。没有读取boundary_flux或计算正式Physics。

**I210_W本fold的fit/cal/assessment全部非目标eligible input blanks为0。** 因此这个fold没有补充被buffer遮住的锚点信息，不能宣传额外训练信息收益。真正的差异是旧密度loss不筛选公开Mode A ramp/topology coverage，且分母只取target-touching pairs：92日fit旧loss中有177,808对不属于新的公开coverage；完整coverage共有952,300对，其中494,725对触及target。cal/assessment相应旧loss额外对为27,636/23,546。新label cache尚未训练，未取得任何新成绩。

下一步应固定strict12/vq初始化、patch序列及预算，仅改变loss的coverage与分母进行对照；不继续扩大微小增益同化网格。公开完整观测增量仍是代理，真实官方RHS未知。随后优先检查Queue观测标签噪声及事件采样一致性，继续OD失配研究；不会以当前局部成绩结束.95000+目标。

`public_density_change_loss()`现已实现上述teacher anchors和完整pair分母。独立手算用N真值[6,12,36,48]、target预测[6,18,36,48]验证loss12/42；排除中间pair后为1/3，目标speed/flow归一化gradient为[-.25,1/3]，非目标输出梯度为0。这个kernel尚未训练。新增测试中误放旧断言的失败源快照单独保留；恢复原测试边界后59项研究测试通过，CPU51项通过、三个可选Torch模块跳过。
