# 熵正则 OD 反演：已知真值模拟与数值核验

仅使用公开 path set、path-link incidence、link counts 和 weak prior；不读取 public_reference、隐藏 path truth 或未发布的 complete counts。所有模拟真值均独立生成，真实 ODME 未知，正式最佳仍为原 Ridge .05 / Total .82940。

目标为 `0.5 Σ (Af-c)^2/variance + KL(f || max(b, prior_floor))`，其中 `variance=τ*max(c,10)`。τ 的含义与 Ridge .05 不同。令 `f=max(b,floor)*exp(-A.T*u)`，双空间 Hessian 为 `A diag(f) A.T + diag(variance)`，只求解 counted-link 维度的线性系统。Newton/Armijo 与原容差 2e−6 的 primal stationarity 核验共同决定成功；这只证明优化，不证明路径真值恢复。

D06 的 69 个失败已保留。精度附近线搜索停滞的真实案例 trace 显示，步长缩到约 2e−7 或 1e−10 后参数不再变化。D07 以残差下降和舍入预算接受近最优步骤，恢复原 69 例但新增 1 例；D08 用 Newton decrement 与误差预算匹配替代绝对 near-gradient gate，全量 0 failures。Ridge 分数逐项相同，原成功 entropy 模拟分数最多漂移 2.32e−9；原数据、网格、选择门槛和容差未修改。小矩阵独立 path-space L-BFGS-B、单计数大数消减 scalar oracle 及真实失败病例共同核验。

七种生成分布包括 public-prior forward perturbation 与独立 gamma truth 的 reverse-generated weak prior。统一设置在 development seeds 上按最差 regime gain、再按平均 gain 选择；要求平均至少 +.005、任何 regime 不低于 −.002、无优化失败，否则选 Ridge。D08 选 τ=.05/floor20，assessment 模拟 ODME .747273 vs Ridge .691063，family/seed gain CI [.043844,.068733]。assessment seeds 已在先前模拟研究中使用，final seeds 仍未生成；这不是新盲测。旧 Bayes 在同一模拟里 .728094，却已有正式失败 K01，因此不能据模拟保证赛事增益。

公开 train 的 entropy S_link-only .987973–.998652、总需求约为 Ridge .920–1.084 倍；约 99% 改变量处于计数零空间。没有把这些稳定性数字当隐藏 S_od/S_dev/S_attr。当前 entropy 是点估计，不提供未经校准的 posterior coverage。

从项目根目录使用锁定环境，新结果必须另存：

```powershell
.\.venv-research\Scripts\python.exe scripts/entropy_od_experiment.py --reference-study reports/research/D07_entropy_OD_numeric_repair.json --output reports/research/D08_independent_repeat.json
.\.venv-research\Scripts\python.exe scripts/compare_entropy_numeric_repair.py --before reports/research/D06_entropy_OD_broad_prior.json --after reports/research/D08_independent_repeat.json --output reports/research/D08_independent_verification.json
```

该脚本只生成本地研究报告，不产生正式参赛记录。下一步应扩大新的先验/路径选择失配分布，验证不确定性和未计数负载稳定性，再考虑完整独立候选。

## D09/D10：区间与新失配证据

**D11 更正：下面的 route-choice 历史标签实际是 OD demand 扰动。** 十方向全部公开 OD 对只有一条 path，preference 归一化必然抵消；该条件与 D08 independent_paths 是同类生成分布，不能当替代路径选择证据。全部 path-set SHA/分组计数见 D11 审计。原数字和源快照保留，两种真正新失配的 macro gain .016966、CI [−.016780,.051299]，仍不晋级。

固定 D08 τ=.05/floor20，不按新区间或新真值重调点预测。局部 Hessian 为 `diag(1/f)+A.T diag(1/variance) A`。令 `D=diag(f)`，其逆为 `D-D A.T (diag(variance)+A D A.T)^-1 A D`；按路径 block 求对角避免 paths-square 内存。数值测试与独立完整 Hessian 求逆匹配；这只是优化曲率，KL 权重和 count variance 不保证反映真实生成分布。

七种旧分布、三个 development seed、十方向的 210 个案例，校准一个 >=1 倍数，使每种开发分布的 equal-case 平均 path coverage 达到 .95。倍数为 **6.402476365**，在生成 assessment 之前封存；不是 conformal coverage 或正式置信区间。最终 simulation seeds 尚未使用。

| 新失配 | 模拟 Ridge ODME | 模拟 entropy ODME | raw interval coverage | calibrated coverage | width / mean true flow |
|---|---:|---:|---:|---:|---:|
| 历史 route_choice_shift（OD demand） | .831746 | .861299 | .747646 | .999109 | 2.542238 |
| 15% active sparse paths + spurious prior | .648198 | .572888 | .039943 | .976592 | 3.051320 |
| 5% count outliers | .474949 | .584191 | .324027 | .966035 | 3.446678 |

原三个登记条件的 macro gain .021162，family/seed CI [−.003262,.045799]；D11 只汇总两种真实新增失配的 macro gain .016966、CI [−.016780,.051299]。稀疏场景显著退化。覆盖率提升依赖很宽的区间，后两种分布没有任何全部路径同时覆盖的案例；不能视为 OD 可识别或准确恢复的证据。真实隐藏路径和 attraction truth 仍未知。

D09 的 flat metadata `method` 覆盖模型身份导致汇总失败、assessment 数值未写出，失败快照/封存区间/数组保留。D10 只修复记录隔离和阶段保存。两个运行 210 个 calibration arrays 完全相同、同一倍数；旧 D08 的 420 个 Ridge/entropy point scores 逐项相同。全部 600 个 method/case 0 solver failures，验证报告另存。

```powershell
.\.venv-research\Scripts\python.exe scripts/entropy_od_uncertainty.py --frozen-study reports/research/D08_entropy_OD_decrement_repair.json --output reports/research/D10_new_repeat.json
.\.venv-research\Scripts\python.exe scripts/verify_entropy_uncertainty.py --source reports/research/D10_new_repeat.json --previous-study reports/research/D08_entropy_OD_decrement_repair.json --failed-directory artifacts/research/D09_entropy_OD_uncertainty_mismatch --output reports/research/D10_new_verification.json
```

下一步由 development 数据选择能保留稀疏路径的先验与计数噪声模型，再作完整十种生成分布对照；不可针对已查看的某个 assessment 单独调 floor。原正式 Ridge 提交保持不变。
