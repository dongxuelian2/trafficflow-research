# Queue 跨种子集成与标签契约诊断

正式最佳仍 .82940，目标 .95000+ 未达到。本轮仅本地模型/验证/归档，没有平台动作或最佳替换。

## 冻结集成与校准敏感性

C49从C42/C43/C47/C48的已冻结两optimizer seed权重和完整域score构造21个候选/季：tree，old/new BCE各四个α，以及跨seed同loss的mean-BCE/soft-IoU/Lovász各四个α。没有新增训练或事件样本。所有阈值仅在完整模板calibrate；旧single-seed cal结果精确重放，所有政策在assessment读取前封存。

| policy | 秋去重condition宏观测IoU | 冬 |
|---|---:|---:|
| 旧BCE α.25 | .879028 | .879943 |
| cal-selected mean-BCE | .880013 (α.25) | .858806 (α.5) |
| 全候选cal argmax | 旧BCE | mean-BCE α.5 |
| within1 paired-gap SE最简单 | tree .872422 | tree .868240 |

mean-BCE秋gain .000985/CI[−.001310,.003718]；冬gain−.021137/CI[−.040511,−.004430]。冬cal只有.865309 vs旧.865149，多.000160；没有根据assessment改α。平均、混合强度和cal决策共同变化，不把下降全归因平均运算。新的简单政策均退回tree，分别相对原模型−.006606/−.011703、CI跨零，拒绝替换旧参考。

敏感性用45个完整cal日期、4000次7日circular blocks，计算最佳与各候选gap的bootstrap标准差；gap≤该SE的候选中按active networks/α/条件阈值/force/BCE/旧seed优先级简化。checkpoint和threshold都已使用这些cal标签，因此这里只是in-sample sensitivity，不能提供独立置信保证。秋tree gap.002087/SE.003998；冬.011712/SE.012719，故均退回更简单但稍弱的tree。没有将此称为提升或正式成绩。

## 自主CTM：latent、带噪观测和评分域

C51使用自身Godunov/CTM law，三个research seeds943003/943013/943031、六个需求/事故位置/检测器失效配置、σ0/1/3/8，共72个counterfactual场景。它不是组织方生成器。每个seed/case的latent状态和quality数组完全相同，事件selector只用自己已知truth构造合法模拟集合；预测函数只收到13个past/current noisy speed与history quality。两个固定基线为persistence和近30min线性趋势，公开onset时间约束只允许T+30。

每种噪声都有503个重叠latent-selected origins；从这同一集合加观测代理guard后剩495/495/492/490，zero-positive horizons5/5/8/9。零噪声也有future quality遗漏正例。各噪声 eligible cell计数71870（含overlap重复），threshold标签与latent标签的disagreement share为0/.000473/.002894/.006734。这些比例仅描述自主模拟，不识别任何隐藏赛事标签或selector。

17个有事件scene/condition设置的等权描述均值里，σ8 onset persistence的known-latent IoU为0、noisy-observation为.029412；linear trend为.009804/.039216。相同预测和相同评分域，标签契约不同即可变化，不是新算法增益。六case共用三个seed，origin重叠，不能把72场景或503个origin当独立样本；没有逐origin CI或正式四任务加权分。未将公开训练网络移植到CTM或声称其泛化通过。

## 推断与完整性

概率接口与binary共用完整canonical域，输出最后恢复输入行顺序。ensemble保存CPU权重，不序列化live GPU cache。C50对全部旧模型660次cal-window输出精确复现，94.75秒GPU；C53对10个导出政策角色，score/binary各550次精确，134.66秒GPU。角色存在重复，没有10个新训练模型或550个独立事件。

C49首次严格manifest查找失败：旧manifest遗漏geometry NPZ，已封存秋cal但尚未读assessment。失败zip/seal/JSON保留；v2先和SHA保留的旧portable model中的全部geometry数组核对，再登记当前bytes，258.92秒CPU完成。C54独立从公共FD/拓扑/节点顺序重建六个旧NPZ全部数组，再纳入当前manifest；这些图文件当前字节首次登记，不追溯冒充原先完整hash覆盖。

C51首次CPU import复用旧GPU脚本CASES导致可选Torch缺包，发生在注册/建输出之前；保存失败source snapshot后，将不变六case配置拆到无GPU依赖模块，CPU复跑9.29秒。78 research tests passed；CPU66 passed/4 optional Torch skips。所有旧模型、完整submission和依赖锁保持冻结；K06不变，没有新正式分数。

复现：`scripts/queue_seed_ensemble_experiment.py --output artifacts/research/C49_two_seed_Queue_ensemble_uncertainty_v2`，`scripts/ctm_queue_observation_diagnostics.py --output reports/research/C51_CTM_Queue_latent_observation_diagnostics.json`；使用相应Python环境和新的输出名，禁止覆盖原记录。下一步优先新方向下真正分开的checkpoint/decision/selection日期和latent/observation噪声建模，保留原强参考。最后模拟seeds7907/8017/9109未生成。
