# 公开密度覆盖损失与Queue评分域：A66–A71、C35–C37

正式最佳.82940保持冻结；.95000+未达到。本轮所有数据为公开TRAIN与公开网络/ramp/模式资料，无平台发布、推送或正式上传。

## 匹配density loss

四个hidden48、heads2、strict12/vq模型使用同一初始化及seed20261013，每组24 epochs×120 batches，batch4、64 slots；AdamW .001、weight decay1e−4、clip1。它们的唯一区别为State-only、旧visible/target loss、公开Mode A target-touching coverage loss、公开Mode A全部valid pair分母loss。非零λ都为.01，full原始TRAIN真值只作loss/evaluator anchors，从不进入x。

完整pair包含不触及target的对：其预测与teacher相同，分子0、分母可能非0。因此full loss严格等于public-target loss乘相应clamped denominator比率（忽略浮点误差）。A70在实际计划2,880 batch中，两个case平均比率.542741/.538899，p05…p95约.40….69。它改变了batch-specific有效强度，不是新的观测信息；coverage还改变误差对象，不能视为纯标量变化。

纯cal State选择checkpoint，然后选α∈{0,.25,.5,.75,1}；primary比较各模型cal State，joint在不低于强LGB及距最佳primary≤.001的cal候选中，最小化三regime等权public increment L1。全部选择在读取assessment之前封存。public metric是ΔN观测误差，不是实际组织方boundary flux或S_Physics。

| case | State-only | legacy | coverage | full denominator |
|---|---:|---:|---:|---:|
| I210_W秋14日 | .933577 | .932989 | .931394 | .932287 |
| I5_N冬14日 | .954329 | .953338 | .954403 | .954407 |

秋cal选coverage α=.5但assessment退化.002184。额外2030-11-13…12-12冻结复核含10/13/7个R1/R2/R3日，State .932585→.932105，日均CI[−.001044,−.000118]；不是final。冬cal选择State-only，其他两新loss稍高但CI跨零，不能反选。两case四模型的patch序列一致；State-only/legacy分别逐张量复现A52/A34。两部署角色全部14日cal独立masked输入、raw TRAIN标签和预测复现。

复现用新目录，禁止覆盖原证据：

```powershell
.\.venv-research\Scripts\python.exe scripts/prepare_full_density_labels.py --parent artifacts/research/A02_D12_I5_N_winter_seed20261013 --output artifacts/research/A67_labels_replay_fresh
.\.venv-research\Scripts\python.exe scripts/full_density_experiment.py --cache artifacts/research/A65_full_public_density_labels_I210_W_autumn --baseline-file artifacts/research/A30_State_budget_directions_b/D7_I210_W/full.pkl --initialization artifacts/research/A52_quality_density_D7_I210_W/shared_initialization.pt --output artifacts/research/A66_replay_fresh
.\.venv-research\Scripts\python.exe scripts/verify_full_density.py --study artifacts/research/A66_replay_fresh --old-control artifacts/research/A52_quality_density_D7_I210_W --control-format quality --output reports/research/A66_replay_fresh.deployment.json
```

## Queue未来评分域修正

正式forecast template公开完整六步link域，future is_score_eligible属于evaluator。旧代理先按future观测available筛x，然后对剩余cell进行expected-IoU prefix或force nonempty；集合决策因此受到未知未来可用性的影响。训练只保留有标签的cell是允许的，但推断/集合决策必须先覆盖完整公共forecast域，评分才筛mask。

固定旧models/thresholds/mixture/checkpoints/origins，C35在三组specialist研究的48个model/phase复现旧metric，C37在两季wave研究的32个model/phase复现旧metric。历史JSON、模型和SHA不覆盖。新API predict_full_forecast不接收future mask；score_full_forecast在预测完成后单独筛选，全部无eligible cell的window按官方规则为0。

反例概率[.9,.4]、未来mask=[False,True]：完整prefix预测[1,0]，第二cell真值1时IoU0；旧域先丢掉first cell，强制prefix转而预测second，虚假IoU1。这个反例证明接口问题，不是任何实际比赛标签。

实际回放C35有41次评分cell决策变化，C37有8次，含跨model/seed重复，不是49个独立事件。三组原cal选expert-threshold均force=false，其输出未变。C27相对force=true基准的去重gain从−.007850改为+.017150，CI[−.044440,.080005]，仍不晋级。C28/C29主比较不变；C32秋绝对baseline/wave30从.897422/.904028修正为.872422/.879028，增益.006606和CI不变；C34冬force=false，主比较不变。

上述是固定旧规则的域回放，不是重新优化后的新模型。旧校准/神经runner只适合复现历史结果。下一步在完整域重做calibration、独立检验公共onset时序应在force之前限制候选的规则，再探索事件级相关概率。latent horizon非空并不保证eligible horizon有正例；future QC未知，强制非空不是安全分数保证。旧观测采样要求future已有观测positive，还存在latent/噪声与事件选择偏差。

```powershell
.\.venv-research\Scripts\python.exe scripts/audit_queue_decision_domain.py --output reports/research/C35_replay_fresh.json
.\.venv-research\Scripts\python.exe scripts/audit_wave_queue_domain.py --output reports/research/C37_replay_fresh.json
```

C35逐窗从仅≤T的Parquet历史重建特征，先预测再读取future TRAIN观测评分；C37还从历史重建九channel NN输入并与冻结cache逐值对齐。旧预测只为刻意复现受影响的历史验证，不用于晋级或正式上传。现有正式提交使用全公共template，没有访问隐藏future eligibility；它及所有先前完整候选SHA保持不变。

63研究测试通过，CPU55通过、三个Torch模块跳过。新增测试覆盖public density metric的独立手算与macro aggregation、未来mask不改变完整set预测及无eligible evidence的0分。真实Physics、Queue、Total增益仍未知。

另核验正式基准的Git源代码15036c8：predict_windows对子窗口完整template推断，未读取future eligibility/queue_true；八个冻结queue_final.pkl的SHA都不变。D7_I10_W与D7_I405_N为force=true，仍应在后续校准完整域中复核这些超参数。这是本地校准可信度问题，不是正式预测读取隐藏数据；原正式.82940记录继续保留。
