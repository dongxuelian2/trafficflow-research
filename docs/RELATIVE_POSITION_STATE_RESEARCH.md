# 相对里程注意力：A130–A133

正式最佳仍Kaggle **0.82940**，0.95000+未达到。以下全部为公开TRAIN的本地State；没有正式Physics或Total测量，也没有替换原完整提交。

## 预登记结构与控制

既有注意力已经全连接。新增关系为`bias = 原relation bias − gamma * D`，`D=min(|milepost_i−milepost_j|/median_positive_gap,20)`。两个attention block各新增一个有符号可学习gamma，初值0；网络97307→97309参数。共同参数逐字初始化相同，普通patch State RMSE、原4view Queue-shaped输入、所有存储采样计划及监督格点不变。没有增加速度差分项。

里程只从首个fit日masked文件的link_id/milepost列读取。距离是统计先验，不修改官方道路拓扑，不是推断的物理边或隐藏流量。模型固定link顺序并保存完整距离buffer。输入base/profile原本可能间接含空间信息，不能称原模型没有空间输入。

控制为A97两seed Queue模型，强性能参照为A112的“旧NN速度＋point树流量”。额外预定组合仅为“新NN均值速度＋原强参照流量”，不再搜通道权重。只有cal相对强参照gain≥.0005且各regime损失不超过.001才可替换；纯新NN和固定组合均合格时按cal取最高。两seed/checkpoint/alpha/primary整体seal后才读旧后留。不是只比较较弱NN就晋级。

I210_W沿用profile45日、fit92日（7/16–10/15）、cal10/16–10/29、旧后留10/30–11/12。新日期2/15–2/28在拟合前固定；其他历史研究可能查看过这些公开日期，不称最终/私有盲测。最终模拟seeds不生成。

## 实测State与不确定性

| 策略 | cal | 原14日后留 | 预登记2/15–2/28 |
|---|---:|---:|---:|
| 冻结A97 NN均值 | .937716 | .938919 | .939762 |
| A112强通道参照 | .938855 | .940049 | .940685 |
| 新位置NN双seed均值 | .939456 | .941372 | .941081 |
| 新速度＋原树流量（cal primary） | **.940300** | **.942187** | **.941826** |

primary相对强参照的pooled gain为 **+.002138 / +.001141**，day-mean gain为+.001821/+.001260，名义95% CI **[.000865,.002862] / [.000743,.001822]**。纯新NN相对强参照+.001324/+.000396，新日期区间下界仅.000032，不能把组合收益全部写成纯NN收益。

相对各自配对NN控制，两个seed旧后留gain+.002211/+.002908，日均区间均正；新日期第一seed仅+.000059、CI[−.000648,.000574]，第二seed+.002311、CI[.001797,.003704]。不能宣称每个seed都稳定复现。所有seed共享数据，14日/7日moving-block/4000重采样只是日均诊断，未作多重比较校正，不是pooled State或Kaggle区间。旧后留R3只有1日，新日期R1/R2/R3为3/5/6日；尚只有一个方向和fit。

两新模型与原配对控制的checkpoint均为epoch24，alpha均.75。所选gamma分别[.325412,.100394]、[.422598,.048287]；正值表示本次拟合偏向近里程节点，不证明因果传播或跨方向增益。

## 保留误差代价

组合flow逐格沿用强参照。旧后留速度RMSE：降速15.304→14.864、恢复10.240→8.305、60min缺失3.426→2.708。新日期降速却 **16.646→18.774（66目标）**，恢复15.840→15.429、60min缺失2.395→2.286、拥堵3.415→3.380。不能把总体增益当急降速鲁棒性改进；180min缺失目标为0。

旧增量L1旧后留.683813→.678066，未来.723815→.723928略差。缺少官方边界流等信息，这些都不是正式Physics收益；需要保留对事故分布变化的风险。

## 验证与资源

两fit/cal角色225.93/223.62秒、全A130468.31秒，GPU allocated峰值725.07MiB。监督曝光14,368,985/14,336,327次与原控制精确一致。只新增两个拟合模型，其余角色是冻结参照或固定组合。

A132独立重建公开位置/归一化距离/两个共同初始化和完整逐格曝光，核验选中pt、距离buffer/gamma/alpha、强参照cal gate及固定组合；28日×9导出252次predict最大lane-unit差2.27e−13，224保存指标评分差1.11e−16，126.60秒。A131九导出×14日实际126次predict，76.52秒；A133全部14日teacher/诊断mask、126指标另算，另18次实际predict差0，18.25秒。部分任务并行，不当隔离测速。

新增三项测试：距离单位/平移缩放/置换；非零output head下初始函数与控制完全相同且距离能获梯度；模型键/总流量单位/缓存清除与保存重载。全套research125 passed/10.41秒；CPU104 passed、7可选Torch模块跳过/8.04秒。

研究primary模型：`artifacts/research/A130_I210_W_relative_position_attention/State_primary_export.pkl`，SHA-256 `00b42f664b1d2a2c38dc27bfe91323550808935f4f1ce1e2b372ce0a02bdc086`。它尚未作全273日生产训练或全网完整submission；原正式最佳和完整候选不覆盖。

## 复现

```powershell
.venv-research\Scripts\python.exe scripts/train_relative_position_ablation.py --output artifacts/research/A130_I210_W_relative_position_attention
.venv-research\Scripts\python.exe scripts/evaluate_exported_state_forward.py --study reports/research/A130_I210_W_relative_position_attention.json --output reports/research/A131_I210_W_relative_position_forward_calendar.json
.venv-research\Scripts\python.exe scripts/verify_relative_position_training.py --study reports/research/A130_I210_W_relative_position_attention.json --output reports/research/A132_I210_W_relative_position_training_verification.json
.venv-research\Scripts\python.exe scripts/verify_native_remask_forward.py --study reports/research/A131_I210_W_relative_position_forward_calendar.json --output reports/research/A133_I210_W_relative_position_forward_independent_verification.json
```

已有目录禁止覆盖，重跑用新输出名。下一步保持此结构/普通损失，在其他方向独立训练并与各自最强cal参照比较；I5_N原树流量迁移失败，不可直接复用I210的流量选择。
