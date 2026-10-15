# 第三个走廊族：I405_S 配对 State 实验

正式 Kaggle 最佳仍 **0.82940**，0.95000+未达到。以下是公开TRAIN研究；没有正式Physics/Total、新全网模型或完整submission。

## 固定协议

A138使用D12_I405_S的44个测量节点，复用A46的原始公开mask缓存、A31全标签LightGBM和A47封存注意力模型。A47含density auxiliary loss，作为既有强模型参照，不作为本轮严格架构控制。所有模型共享原profile45日，fit92日（7/16–10/15），cal10/16–10/29，原后留10/30–11/12。

新训两seed（20261013、20261029）各一普通GRU和一位置GRU，共四模型。配对网络所有共同参数初始化字节相同，位置模型仅多两有符号gamma，初值0；仍为abs里程差/正间距中位数、cap20。使用普通patch State RMSE、24epoch×120step、batch4、context128/core64、相同AdamW及完整存储采样计划。每对实际监督格点计数精确相同。

这里仅使用原始公开fit target mask，**没有native-remask或mock Queue额外标签位置**。前两方向位置实验使用增强标签池，因此不能把三个方向当成完全同分布的架构重复实验。强LGB可用标签日期相同，但遍历标签与NN重复patch的实际曝光数不同。位置是统计关系，不修改道路物理拓扑。

每4epoch取纯神经cal State首次最大，随后独立从0/.25/.5/.75/1选cal混合权重，精确同分保留较小alpha。固定普通双seed均值、新位置双seed均值，以及新位置速度＋原强参照流量的组合；不搜通道权重。原强参照从冻结LGB/A47中按cal取最高，精确平分保留A47。三个新候选只有cal gain≥.0005且每regime损失≤.001才可替换；合格时取最高cal，所有选择整体seal后再读旧后留。

二月2/15–2/28在拟合前固定，只部署封存模型、全部角色每天直接调用。这些公开日期此前可能被其他研究查看，不是最终/私有盲测。最终模拟seeds7907/8017/9109保持未生成。

## 复现命令

```powershell
.venv-research\Scripts\python.exe scripts/third_family_relative_state.py --output artifacts/research/A138_I405_S_matched_relative_position
.venv-research\Scripts\python.exe scripts/evaluate_exported_state_forward.py --study reports/research/A138_I405_S_matched_relative_position.json --output reports/research/A139_I405_S_matched_relative_forward.json
.venv-research\Scripts\python.exe scripts/verify_third_family_relative.py --study reports/research/A138_I405_S_matched_relative_position.json --output reports/research/A140b_I405_S_matched_relative_verification.json
.venv-research\Scripts\python.exe scripts/verify_native_remask_forward.py --study reports/research/A139_I405_S_matched_relative_forward.json --output reports/research/A141_I405_S_matched_relative_forward_verification.json
.venv-repro\Scripts\python.exe scripts/checkpoint_third_family_relative.py
```

已有目录不覆盖，重跑更换输出名并相应改后继路径。生产src未改；旧125research/104CPU与7可选Torch模块skip不重复计作本轮新测试。

一次单日preflight误用无Torch的`.venv-repro`失败，未修改训练或产物；改用`.venv-research`后2030-07-16的原始输入、归一化teacher和1899个target格点全部精确一致。完整92日重建另由A140b执行。

首轮A140的完整网格复算失败：核验器把纯神经预测恢复成float64后乘alpha，训练/部署原路径却先在float32乘alpha再加float64 baseline。独立56次纯cal调用定位到预测最大差6.10e−5 lane units、State最大差5.69e−10；恢复native float32后20个网格评分及选中预测差均为0。模型、阈值、选择、保存预测都未修改，没有放宽1e−12评分容差。失败源码ZIP/注册和`A140_I405_S_alpha_precision_diagnosis.json`保留，完整重放用新A140b目录。

## 本地结果与误差

| 策略 | cal | 原后留 | 固定二月 |
|---|---:|---:|---:|
| 全标签LGB | .938017 | .941500 | .940198 |
| 冻结A47注意力（强参照） | .943913 | .941872 | .946264 |
| 新普通GRU双seed | .945306 | .944237 | .947864 |
| 新位置GRU双seed（cal primary） | .946282 | .944199 | .949435 |
| 新位置速度＋强参照flow | .944968 | .943074 | .948193 |

primary相对冻结A47 pooled gain+.002327297/+.003170351，day-mean CI[.001843144,.003428928]/[.000524099,.003934329]。普通新GRU对冻结A47也改善+.002365715/+.001599932，区间均正。原A47单seed、不同目标与推断窗口，不能称两参数或GRU分支本身带来整个增益。

位置相对同预算普通GRU pooled gain却为−.000038417/+.001570419；day-mean gain+.000620624/+.000191112，区间[−.000192521,.001488933]/[−.000879695,.001249492]均跨零。State为各regime先pool RMSE再等权平均，day-mean诊断不是同一统计量，不能用日均正值改写pooled结果。旧R1/R2/R3为4/4/6日，新3/2/9日；区间只是14日/7日block/4000抽样的名义日均诊断，非pooled/Kaggle区间，没有多重比较校正。

位置相对普通GRU的旧regime gain为R1+.001487、R2−.003536、R3+.001934；新为R1+.002891、R2+.003169、R3−.001349。新R3速度RMSE1.163→1.235更差，flow38.809→38.541改善。不能把整体State收益当R3鲁棒性提升。

配对seed13/29旧后留gain−.001115316/+.001078856，日均区间分别跨零/为正；二月+.000960548/+.002448189，两个日均区间均跨零。普通模型选epoch24，位置模型都选epoch20，四个alpha均.75；既是位置结构也是cal checkpoint政策的完整比较。选中gamma为[.306242,.042891]、[.257193,.044321]。

相对普通GRU，位置模型旧/新降速速度RMSE7.419→7.198 / 9.151→7.842，但恢复39.413→41.336 / 11.801→12.213更差，60min缺失.899→.926 / 1.632→1.660也更差。新降速flow72.483→79.813、拥堵速度6.476→6.748均退步。旧/新降速仅6/11个target、恢复7/12个，180min缺失无target；不宣称事故场景或极端失效泛化。

旧增量L1相对普通GRU.731653→.724604 / .754929→.753362，小幅改善不等于正式Physics。当前primary SHA-256为`06fc0321beb05cfe9dc9d051397db5a0ac1c677797e0c1fb8c2f649d143dff9d`，只保存研究模型，未全网生产拟合或生成完整submission。

## 核验与资源

四fit/cal角色139.78/100.36/100.02/100.45秒，总464.45秒，GPU allocated峰值439.57MiB。两seed实际监督曝光8,254,636/8,270,435次，每对精确相同；启动预热与并行工作影响耗时，不当独立模型速度比较。

A140b独立重建全部92fit日的原始输入、teacher、target mask、公开图/距离、独立RNG计划及四份实际曝光；核对共同初始化与选中pt/gamma/alpha。28日×11导出308次predict最大lane-unit差2.27e−13，280保存指标score差2.22e−16；另56纯神经cal接口调用验证20个所选checkpoint的alpha网格评分，83.38秒。未重放优化器更新或未选择的epoch。

A139全部11角色×14日154次实际predict，45.25秒；A141全部14日teacher/lanes/diagnostic masks和154指标独立复算，首末另22predict差0、score差2.22e−16，9.96秒。11角色包含冻结模型、固定组合及别名，只有四个新fit；部分验证并行，不当隔离测速。原完整最佳和其他完整候选全部保留。
