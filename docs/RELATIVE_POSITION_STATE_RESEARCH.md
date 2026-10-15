# 相对里程注意力：A130–A137

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

已有目录禁止覆盖，重跑用新输出名。I5_N原树流量迁移失败，不可直接复用I210的流量选择。下面记录同一结构的跨方向检验。

## 冬季 I5_N：固定协议

A134沿用A104两seed Queue-shaped控制、A107已核验导出和全部存储采样计划。258测量节点，profile45日，fit153日（7/16–12/15），cal12/16–12/29，原后留12/30–1/12；预登记新日历仍为2/15–2/28。只新增两个位置参数，训练预算、共同初始化、普通损失及监督格点全部匹配原控制。

性能参照为冻结NN，固定候选为纯位置NN均值和“新NN速度＋原NN流量”。不拟合或导入冬季流量树；仍按cal gain≥.0005、各regime损失不超过.001选择。原后留只在两seed/checkpoint/alpha/primary整体封存后读取。

两方向非对角距离达到cap20的比例为I210_W 63.06%、I5_N 89.80%。同一距离规则在不同节点密度下不表示相同的物理作用范围；本轮保持规则不变，不按评估结果调整cap。跨方向日均诊断把共同14日同步重采样，不能算成28独立日或正式走廊权重。

```powershell
.venv-research\Scripts\python.exe scripts/train_relative_position_ablation.py --winter --output artifacts/research/A134_I5_N_relative_position_attention
.venv-research\Scripts\python.exe scripts/evaluate_exported_state_forward.py --study reports/research/A134_I5_N_relative_position_attention.json --output reports/research/A135_I5_N_relative_position_forward_calendar.json
.venv-research\Scripts\python.exe scripts/verify_relative_position_training.py --study reports/research/A134_I5_N_relative_position_attention.json --output reports/research/A136_I5_N_relative_position_training_verification.json
.venv-research\Scripts\python.exe scripts/verify_native_remask_forward.py --study reports/research/A135_I5_N_relative_position_forward_calendar.json --output reports/research/A137_I5_N_relative_position_forward_independent_verification.json
.venv-research\Scripts\python.exe scripts/checkpoint_relative_position.py --winter
```

本阶段仅修改研究runner/verifier/checkpoint脚本，生产src不变。上一阶段125 research、104 CPU通过及7 Torch模块跳过为历史内核证据，不重复记作本轮新测试。

## 冬季结果与限制

| 策略 | cal | 原14日后留 | 2/15–2/28 |
|---|---:|---:|---:|
| 原NN均值（封存primary） | .955885 | .957512 | .959782 |
| 新位置NN均值 | .956055 | .957498 | .960319 |
| 新速度＋原NN流量 | .955930 | .957438 | .960207 |

纯位置cal仅+.000169850，R1损失.000833375，未过总gain+.0005门槛。固定组合cal仅+.000045589，R1损失.001047952，还触发regime保护。旧后留纯模型gain−.000014246、日均CI[−.000362412,.000247461]；新日历+.000536968、日均CI[.000251633,.001053473]。固定组合旧/新gain−.000074244/+.000424800，区间分别跨零/为正。不得在看到二月后把新NN改为primary。

两个新模型均epoch24/alpha.75，原配对控制也相同。gamma为[.343653,.091409]和[.345850,.095314]；正的近邻偏好并不保证State晋级。原后留R1/R2/R3为5/5/4日，新日历7/1/6日；名义日均区间不作pooled State/Kaggle区间，无多重比较校正。

纯新NN误差分层：旧降速速度9.983→8.186、恢复10.016→11.144、60min缺失1.301→1.353、拥堵2.012→1.686。新日历降速15.084→12.653、恢复11.531→11.858、60min缺失1.368→1.216、拥堵3.384→2.686。急降速flow新日历95.743→101.651更差；不能只报速度收益。180min缺失两时段均无目标。

旧增量L1旧后留.650629→.633118、新日历.662477→.652704；原后留State几乎持平，因此代理改善不作为正式Physics晋级证据。两fit/cal角色517.67/513.91秒，A134整体1084.44秒，GPU allocated峰值2099.63MiB。实际监督曝光45,754,069/45,844,015次与控制一致。没有新全网生产训练或完整submission。

相对各自配对控制，seed20261013/20261029旧后留gain+.000616910/−.000723851，日均CI分别[.000060812,.001173398]/[−.001122753,−.000456724]。新日历+.000914910/+.000164071，区间[.000482675,.001642075]/[−.000145278,.000562918]。不能只保留第一seed或把第二seed的未来小正值当稳定证据。

A136完整独立几何/共同初始化/逐格曝光、选中pt/gamma/alpha、固定组合及cal gate核验通过。28日×8角色224次实际predict最大lane-unit差4.55e−13，196保存指标score差2.22e−16，283.78秒。A135全部8角色×14日112次predict、156.89秒；A137全部14日teacher/diagnostics和112指标另算，首末16次predict差0、score差1.11e−16、35.63秒。评估曾并行，不能当隔离速度基准。

跨方向共同二月的纯新NN相对各自NN，等方向day-mean gain+.001046100、同步7日block区间[.000772262,.001319939]；cal-selected策略相对各自强参照+.000630046、区间[.000371730,.000911031]，其中冬季策略完全没变。它们只有共同14日期、两个拟合方向，未按正式走廊权重，不能取代冬季失败的cal gate或解释成全网State/Total收益。
