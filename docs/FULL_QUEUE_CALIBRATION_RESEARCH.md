# 完整预测域Queue校准与选择偏差压力测试

正式最佳仍.82940，.95000+未达到。C38–C41均是旧公开TRAIN时期的观测代理或自主模拟，没有正式Queue/Total新成绩。没有平台发布、推送、上传。

## 校准与部署

C38从仅≤T的Parquet历史重新构建全部9-channel输入，匹配冻结cache；固定tree及direct/15min/30min graph，完整六步link域保存tree和raw corrected-BCE neural概率。future score mask与label只作evaluator数组。C39每种时间顺序在cal比较tree和三graph的α=.25/.5/.75/1（α0与tree等价），31个threshold .05….80、global/condition与可选nonempty。全部选择先封存，再读取assessment。

两个时序控制为旧“force之后按onset清零”和新“先限制onset候选到T+30再force”。后者是合法逻辑修正，但两季最优cal均force=false，故实际selected predictions和before/after gain都相同。不能把接口修正声称性能突破。

| case | cal选择 | tree assessment proxy | mixture assessment proxy | gain CI |
|---|---|---:|---:|---|
| I5_N秋 | wave30 α=.25、ongoingτ=.275/onsetτ=.5、force=false | .872422 | .879028 | [−.004692,.018938] |
| I5_N冬 | wave30 α=.25、ongoingτ=.4/onsetτ=.375、force=false | .868240 | .879943 | [−.001776,.032738] |

这是两种旧fit权重、一种fit seed及已查看时期；checkpoint早先按Brier选定，当前不重训。当前tree输入不含NN的完整occupancy序列，因此混合对照不是单一architecture归因。逐窗条件等权、origin去重day-block bootstrap与真正formal latent Queue不同。

FullDomainQueueModel的predict_window只收公共完整template、≤T history、网络/topology、origin与condition；不接收未来mask或labels。它将模板映射到固定公开node顺序，验证完整性并标准化UTC，按原权限追加公开T观测。保存只包含CPU weights、tree与public geometry，清除GPU cache。四个artifact各从Parquet重建52/58个cal窗口，总220次完整二值预测精确复现；当前I5_N validation/private各15480行推断成功。

```powershell
.\.venv-research\Scripts\python.exe scripts/audit_wave_queue_domain.py --output reports/research/C38_replay_fresh.json --save-probabilities
.\.venv-research\Scripts\python.exe scripts/full_queue_calibration_experiment.py --probability-report reports/research/C38_full_queue_probability_cache.json --output artifacts/research/C39_replay_fresh
.\.venv-research\Scripts\python.exe scripts/verify_full_domain_queue.py --study artifacts/research/C39_full_domain_Queue_calibration --output reports/research/C39_replay_fresh.deployment.json
.\.venv-research\Scripts\python.exe scripts/predict_full_domain_queue.py --model artifacts/research/C39_full_domain_Queue_calibration/C34_before.pkl --panel D12_I5_N --split validation --output artifacts/research/C39_replay_fresh_validation.csv
```

新输出目录/报告必须不存在。当前全概率cache58.77秒（含GPU推断与历史重建），calibration395.24秒CPU，220-call独立部署45.06秒。67研究测试通过，CPU59通过、三个Torch模块跳过。

## 采样偏差模拟

C40是自定义的已知真值模拟，不能当官方生成器。32个位置上均匀选择长度1/4/16的contiguous queue block，只有T+30有queue，其他五步为空；future eligibility为独立Bernoulli .5/.75/.95。三个research seeds，每case5000事件，共27case/135000事件。forecast只获知模拟law的marginal，不获知实际block位置或mask。

分别评价完整latent-positive事件总体，以及**同样事件中仅scored-positive子集**，三个policy用相同事件。宽度1、eligibility=.75时，空预测在latent总体模拟IoU .252533，合法nonempty .095533；过滤到scored-positive后为0/.032356。未来mask oracle为.031800/.042521，只用于展示不合法域选择，不导出或部署。

比较规则时必须在同一population内；两个population绝对差不是模型gain。该例说明nonempty与观测正例采样可能偏置，**不证明组织方真实窗口会以这些概率或规则抽样**。隐藏latent labels和额外采样细节仍未知，不能把模拟结论写成真实Queue成绩。

## 完整本地控制K06

`artifacts/research/K06_I5_N_full_domain_Queue_LOCAL/submission.csv`覆盖6980503行，SHA-256 `d280c84352bff14b46c05ca64c6c8c9da2b7988e9c450a78f897088957280459`。相对冻结ref57138874，仅I5_N415个queue_pred改变（132validation、283private）；State、OD、其他Queue与其他字段完整逐行一致，Physics输入不变。独立master-key验证通过。

这是Queue-only研究控制，不是新最佳或已晋级方案，**未上传、正式分数未知**。它同时更换T输入、graph mixture、calibration域及onset时序，不能孤立解释单项algorithm影响。原正式.82940文件及其他完整候选保存不变。下一步应检验合法train窗口的观测/latent偏差、不同方向和新的fit seeds，而不是继续调同一cal阈值。
