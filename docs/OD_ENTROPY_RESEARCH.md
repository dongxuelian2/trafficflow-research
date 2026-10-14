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
