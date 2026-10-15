# 冬季固定通道迁移：A116–A120

正式最佳仍为 Kaggle **0.82940**，目标0.95000+未达到。这里研究公开TRAIN本地State；没有正式Physics或Total测量。

## 预定假设

I210_W的A112仅按cal选择了“NN速度＋point树均值流量”，在旧后留和后续14日取得约+.001本地State增益。A116在I5_N直接迁移这个组合：不重新搜索通道权重，不重训NN速度，只从头拟合两个seed的point流量树。原LGB速度树仅作为流量模型内可部署的占位速度分支；所有主对比都使用冻结NN速度。

profile为原45日；fit为2030-07-16…12-15的153日，cal为12-16…12-29，旧后留为12-30…2031-01-12。新forward在拟合前固定为2031-02-06…02-19；公开日期可能被其他历史研究使用，不能称最终/私有盲测。NN采用A104校准选中的Queue-shaped双seed均值，A107已作336次独立部署核验。

固定transfer始终报告；另有预定cal gate决定primary：State gain至少.0005，且各regime损失不超过.001，否则保留NN。没有按后留选择seed、通道权重或gate。两seed均值权重各.5，400树/63叶/4CPU线程及原60维v2特征保持固定。

## 数据与资源

原目标3,042,465行，四份point视图增加12,175,783行，共 **15,218,248行、765个date/view组**。前者有效权重占.5，后者四份各.125；每组按regime/date/row均匀赋权，总有效权重3,042,465（float64求和误差约2.3e−9）。标签人口与NN不同：完整视图池覆盖7,294,027个独立link/time位置，NN两个point seed实际覆盖7,002,824/6,995,008个；总行数不当独立观测数。

按视图逐块写入float32特征、float64流量teacher/base/weight和int32行键的磁盘映射文件，共4,200,237,088字节，准备243.03秒。原speed tree只复用，流量树独立训练；拟合仍会分配LightGBM内存，磁盘映射不代表整个算法在磁盘上运行。耗时和峰值工作集均为实际进程记录，部分验证与拟合并行，不能当隔离吞吐比较。

## 审计边界

A118只审计已经完成的不可变训练表，在模型仍拟合时并行执行。独立实现坐标随机数、history/guard区域和所有测量删除，然后重建全部60特征、mask、标量teacher/base、权重和行键；raw teacher采用固定1e−6归一化容差。首末日四份view共8次实际目标poison检查覆盖完整60维特征。此阶段不读取cal/assessment，不报告模型分数。

后续完整部署审计以A118报告SHA、原注册和dataset SHA绑定，重新校验训练文件及源数据字节；不会把复用审计写成再次重建765组或再次8次poison。全部保存模型再与原cal/assessment预测逐日对比，并独立计算评分和cal gate。

旧增量L1仅是公开数据上不完整的动态诊断。NN速度与树流量的组合可能改变q/v关系，不能由State增益或这个诊断推断正式Physics增益。

## 拟合与原后留结果

两seed新流量树耗时290.99/304.11秒，A116整体888.84秒；进程生命周期峰值工作集6917.55MiB。速度树没有重训，整个拟合/原后留流程使用保存NN预测，不作GPU推断。

| 固定模型 | 原14日cal State | 原14日后留State |
|---|---:|---:|
| 原NN primary | .955885 | .957512 |
| NN速度＋seed13树流量 | .955071 | .956612 |
| NN速度＋seed29树流量 | .955048 | .956592 |
| NN速度＋树流量均值（固定transfer） | .955084 | .956624 |

cal gain−.000801，按预案primary保留NN。固定transfer原后留gain−.000888，day-mean−.000886、名义CI[−.000995,−.000777]；两seed单独区间也负。旧增量L1从.650629恶化为.672539。这是失败的跨方向迁移，不能由I210_W结果推为统一通道规则。

A121在新的flow forward预测打开之前，只用14日cal数组拟合一个有界流量混合系数。最优树权重.132571，in-sample State gain仅.00001953，远低于原.0005门槛；不导出新模型、不扩大权重网格。源JSON元数据包含已查看后留的摘要，但优化器数值输入仅为cal预测和teacher文件；此为开发诊断，不是独立收益。源码/注册/14份输入SHA保留。

A118独立765组/15,218,248行全部通过、完整60维8次poison通过；raw流量teacher最大归一化差1.19e−7，低于原1e−6容差，权重差0。耗时344.87秒，与拟合部分并行；不把两项时间相加作为整体elapsed。

## 固定新日历检验

A117的2/6–2/19 State：NN **.955189**、固定transfer **.954343**，gain **−.000846**、day-mean CI **[−.000920,−.000766]**，两seed单独也都更差；primary继续等于NN。旧增量L1 **.656166→.680281** 恶化。急降速/恢复/60min缺失/拥堵的流量RMSE分别从123.247/137.323/32.561/46.298恶化为155.013/157.296/34.039/47.238，速度输出保持NN。

直接推断四个基本角色＋固定transfer，每日5次，共70次，91.83秒；其余角色按固定数组推导。A119另行重建14日全部raw teacher/lanes/indices/诊断mask，126指标复算，首末日9个保存角色另作18次predict，最大差4.55e−13，46.37秒。公开研究14日、7日moving-block的名义CI不代表非线性pooled或Kaggle分数区间。I210_W局部成功没有迁移；不晋级统一通道策略，也不扩大同样树/混合网格。

A120复用绑定注册/数据manifest/报告SHA的A118，重新hash全部训练文件及raw源，而非重复765组重建或8次poison。28日×9角色252次实际predict最大差4.55e−13，238保存指标最大score差2.22e−16，cal gate/封存顺序/原speed booster一致和每q booster400树均通过，305.82秒。新阶段生产src未修改；原118research/102CPU测试记录不重复计作新实验。

## 复现入口

```powershell
.venv-research\Scripts\python.exe scripts/winter_flow_transfer.py --output artifacts/research/A116_I5_N_point_flow_transfer
.venv-research\Scripts\python.exe scripts/verify_flow_transfer.py --study reports/research/A116_I5_N_point_flow_transfer.json --table-only --output reports/research/A118_I5_N_flow_table_independent_verification.json
.venv-research\Scripts\python.exe scripts/evaluate_flow_transfer_forward.py --study reports/research/A116_I5_N_point_flow_transfer.json --output reports/research/A117_I5_N_flow_transfer_forward_calendar.json
.venv-research\Scripts\python.exe scripts/verify_flow_transfer.py --study reports/research/A116_I5_N_point_flow_transfer.json --table-proof reports/research/A118_I5_N_flow_table_independent_verification.json --output reports/research/A120_I5_N_flow_transfer_deployment_verification.json
.venv-research\Scripts\python.exe scripts/verify_native_remask_forward.py --study reports/research/A117_I5_N_flow_transfer_forward_calendar.json --output reports/research/A119_I5_N_flow_transfer_forward_independent_verification.json
```

训练表审计须在dataset.json完成后执行，后续命令须等主报告生成。已有目录禁止覆盖；重跑使用新输出名。最终模拟seeds仍封存，所有原完整提交保留。
