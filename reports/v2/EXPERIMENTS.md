# V2 可复现实验登记与证据

本轮开始HEAD为`9f65c96857eb0be8373eb30cf0b18d1ecf39130d`。所有新运行在拟合/校准前保存协议及源码ZIP，HEAD与实际未提交源码的区别由ZIP逐文件SHA链覆盖。最终提交只包含代码、数学、JSON/Markdown报告；模型、预测NPZ、source ZIP、训练数据与完整submission保留在本地。公开日历以前已被开发查看，不能称全新盲测。

## 运行入口

在`E:\ieee`使用`.venv-research\Scripts\python.exe`。所有输出不可覆盖；复跑时换新目录名，并同步下游`--cache-root`/`--state-root`路径。

```powershell
.venv-research/Scripts/python.exe scripts/v2/audit_score_alignment.py --output artifacts/research/v2_score_audit_v2
.venv-research/Scripts/python.exe scripts/v2/run_state_experiments.py --output artifacts/research/v2_state_v2
.venv-research/Scripts/python.exe scripts/v2/run_ltm_experiments.py --output artifacts/research/v2_boundary_precheck_v2
.venv-research/Scripts/python.exe scripts/v2/run_queue_experiments.py --execute --stage1-report artifacts/research/v2_state_v2/I5_N_autumn/experiment.json --output reports/v2/QUEUE_CONDITIONAL_RUN_V2
.venv-research/Scripts/python.exe scripts/v2/run_directional_experiments.py --cache-root artifacts/research/v2_state_v2 --output artifacts/research/GP_v2_same_day_autumn_v3 --case I5_N_autumn --case I210_W_autumn
.venv-research/Scripts/python.exe scripts/v2/compare_candidates.py --state-root artifacts/research/v2_state_v2 --output artifacts/research/v2_exact_replay_v1
.venv-research/Scripts/python.exe scripts/v2/verify_gated_candidate.py --output artifacts/research/v2_gated_fallback_v1
```

完整源ZIP包括`src/**/*.py`、`scripts/**/*.py`、官方scorer、配置及锁文件。此次修正了旧runtime快照漏掉`scripts/v2/`与源码指纹漏掉包子目录的问题；旧已运行记录以其启动ZIP为主，不用运行中磁盘指纹代替启动代码。

## 数据角色与匹配控制

State四case：I5_N/I210_W×秋冬。秋：base拟合/选模≤10-29，correction_fit10-30—11-12、cal11-13—11-26、assessment11-27—12-10。冬：I5_N base选模≤12-29，correction_fit12-30—01-12、cal01-13—01-26、assessment01-27—02-09；I210_W冬沿用冻结秋模型，明确是时间转移。数据年份2030/2031为发布的合成日历。每case各角色14日。

每组冻结主模型；比较同12维U/24系数的线性State、稀疏State、稀疏State+库存权重.02/.10；强度0/.25/.5/1只用cal选择。固定seed20261016、L-BFGS-B、系数界±.25、ridge .001。没有训练小残差网络：只有低维模型先达到.002晋级门槛才允许扩大容量。目标标签、库存误差只在fit损失/独立scorer使用，绝不进入U。

GP：同日512个合法可见anchor、2折四通道共同遮挡后的冻结强模型残差；192诱导点，普通/门控普通/双向核和已有图二次求解器使用同anchor。独立前推fit残差仅用于尺度。c_f=80、c_b=−15km/h，空间尺度2km、时间尺度.5h，参数预先固定。可信拓扑链外回退基线；cal封存后读assessment。

Queue：C55–C58严格fit/checkpoint/decision/selection/assessment五角色，I5_S/I10_W秋冬。I5_N缺少同样五角色缓存，因此预先记录方向偏离。仅checkpoint从k=8/24选择，四路概率×解码对照；256场景、MC seeds20261016/20261029，旧概率与新事件相同候选预算。未来availability与truth仅用于训练target或完成预测后的scorer。

边界：使用评分审计固定公开日期，仅检查观测通量映射、24个12对非重叠片段、Q正定、低维边界先验SD .1/1/10的固定/联合/边缘代数对照。固定协方差时联合剖面与边缘二次型应该相同，不能将它们当作新增预测信息。

## 失败、数值修正与评估边界

| 记录 | 发生阶段 | 处理 |
|---|---|---|
| `v2_score_audit_v1.failed.json` | 尚无成绩，Windows拒绝hardlink | 工作区复制，v2新目录；原记录保留 |
| `v2_state_pilot_v1.incomplete.json` | correction_fit准备中，尚未拟合/读评估 | 全模板日期格式化约26秒/日；改UTC区间筛选，v2重跑，不改日期或参数 |
| `v2_boundary_precheck_v1.failed.json` | 尚无比较结果 | Series改用`.dt.as_unit`；v2新目录 |
| `QUEUE_CONDITIONAL_RUN/failure.json` | decision，未读selection/assessment | 概率凸组合超1一ulp；仅clip数值到[0,1]，新目录同协议重跑 |
| GP v1/v2性能中止 | 均在cal阶段，无assessment成绩 | 固定pair向量评分；再去除4,469,379行ramp日期重复格式化（21.98秒/次）；逐候选核对后v3同协议重跑，旧目录保留 |
| 初始评分单测 | 不可能的同panel同日多regime测试输入 | 明确拒绝重复panel/date，测试改为实际日历契约；未影响实际唯一日期数值 |
| State早期缓存列名metadata | `feature_columns`是object dtype | 只按需读安全数值数组，不开启pickle；后续导出改Unicode，不改冻结缓存 |
| State固定预算优化 | I210秋/冬`sparse_increment_010` | 两次达到120迭代上限（其余14次达到容差）；有限迭代点与原因均报告，不按assessment追加迭代 |

训练与初始cal库存目标使用日内局部线性化。最终精确重放对封存系数/强度跨日期拼接同regime，补齐跨午夜≤5.1min时间对；独立重算State SSE并断言一致。该重放不根据assessment重新选择模型。原始日内代理与最终完整pair代理均保存，不能混为同一指标。

全部实验JSON记录参数、数据角色、seed、模型hash、代码快照、时间与内存；GPU峰值是各进程观测，RSS为进程生命周期峰值，不冒称隔离单模型。预测/模拟指标与正式成绩严格区分。

代码复核另指出初始State JSON的`gate_pass`只能表示单case筛查，不能代表两方向/两季晋级。当前各case均低于.002，结论不受影响；最终重放另计算跨两季分层日期bootstrap的campaign gate，新源码将字段明确改为`case_gate_pass`。没有为这项报告语义修正重新训练或反选模型。
