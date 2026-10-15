# 实验台账 — 0.950+ 攻关

所有模拟指标、观测代理与正式 Kaggle 指标分别记录。失败实验不删除，既有模型与提交不覆盖。

| 编号 | 假设 / 核验 | 实测结果 / 资源 |
|---|---|---|
| C65 | 同公开onset行/特征/容量，binary vs LambdaRank+固定top-k，ongoing冻结 | 4case/16fit，decision全部原模型；ranker后留gain−.024824/−.043878/+.064815/−.083333；32.33秒CPU/peak726.04MiB |
| C66 | 全原始输入/teacher/query/网格/评分/去重CI与便携接口复核 | 2345窗口、128Booster调用；560便携+140乱序精确，8future poison拒绝；302.65秒 |
| C67 | 公开assessment标签下逐窗最优已定k的诊断，禁止用于选择 | k有评分空间，但需未来标签/eligibility；无新模型/选择/候选，不能当可达成绩 |

I10_W秋次要正值区间很宽且未被decision选中；其余排序退步保留。3新单测通过，全131research/12.68秒、CPU110/7.34秒+7Torch skips。两季训练事件重叠，model seeds共数据；不将4fold/560角色调用视作独立事件。原完整最佳保留，无正式上传。

| 编号 | 假设 / 核验 | 实测结果 / 资源 |
|---|---|---|
| C62 | 同X/曝光/权重，hard/known-sigma-soft/known-latent各2seed | soft observed-cal−.000665拒绝；latent窗口宏同类−.000516/stress+.005618；93.78秒CPU/peak491.11MiB |
| C63 | 全源重模拟、独立特征/teacher/selector/权重/部署/评分 | 268scene/7090窗口，18booster预测差0、300grid/57680score差0、feature差1.49e−8；79.66秒 |
| C64 | 独立4seed区间及noise/scene摘要，无新选择 | 8区间精确重现、64摘要行；同类σ3/8及stressσ16退步均保留 |

六fit均272907行/300树，model seeds20261013/20261029；固定两seed均值。模型seal后读cal、选择seal后才生成heldout。10个生成seed中仅4个heldout独立seed，noise/case counterfactual及重叠origins不算新独立样本。3新单测通过，全128research/13.63秒、CPU107/8.79秒+7Torch skips。无public迁移/正式提交。

| 编号 | 假设 / 核验 | 实测结果 / 资源 |
|---|---|---|
| A138 | I405_S原始公开targets、2seed各配对普通/位置GRU，共四fit | cal选位置均值；原后留.944199 vs A47 .941872、普通GRU .944237；464.45秒、GPU allocated439.57MiB |
| A139 | 全11导出在冻结二月日期逐日直接调用 | 位置.949435 vs A47 .946264、普通GRU .947864；154predict、45.25秒 |
| A140 | 全92fit inputs/teacher/exposure及全部选中checkpoint网格复算 | 首轮在alpha-grid报错；核验器float64混合与训练float32不同，源码ZIP/注册保留 |
| A140 precision | 56次纯cal调用定位float精度顺序，20网格复算 | 原最大预测差6.10e−5、score差5.69e−10；恢复原生float32后均0，无模型/容差/选择修改；8.45秒 |
| A140b | 独立全92fit、4archive、选中pt、28日部署及alpha-grid | 308predict最大差2.27e−13，另56pure cal、280保存指标/20grid通过；83.38秒 |
| A141 | 新14日全部teacher/diagnostics/154指标与首末调用 | 另22predict差0、score差2.22e−16；9.96秒 |

位置相对普通GRU的旧/新pooled gain−.000038/+.001570，两day-mean区间均跨零，新R3−.001349；未按较好后留反选普通模型或seed。两对曝光8,254,636/8,270,435次精确一致，普通checkpoint24、位置20，alpha均.75。单日preflight最初误用无Torch的repro解释器，research解释器复核成功；生产src未改，历史单元测试不算新测试。没有新完整提交/正式测量。

| 编号 | 假设 / 核验 | 实测结果 / 资源 |
|---|---|---|
| A134 | I5_N同普通loss/共同初始化/完整曝光的两参数位置偏置，2seed | cal新均值+.000170未晋级；旧后留−.000014/CI跨零；1084.44秒、GPU allocated2099.63MiB |
| A135 | 不改选策略，固定2/15–2/28全部8导出直接调用 | 原NN .959782、新均值.960319；day-mean CI[.000252,.001053]；primary仍NN；112predict、156.89秒 |
| A136 | 独立位置/初始化/完整曝光/选中pt及28日部署和评分 | 224predict最大差4.55e−13、196指标最大差2.22e−16；cal gate与固定组合通过，283.78秒 |
| A137 | 新14日全部teacher/diagnostics/112指标及首末直接调用 | 另16predict差0、score差1.11e−16；35.63秒 |

新两seed旧后留配对gain分别+.000617/−.000724，区间一正一负；二月+.000915/+.000164，第二seed区间跨零。没有按后留挑seed。所有新/旧控制均epoch24/alpha.75，实际曝光45,754,069/45,844,015次精确匹配；恢复速度两日历退步。仅研究脚本修改，125research/104CPU及7Torch模块skip是前阶段内核记录，没有重复计新测试。两个评估曾并行，耗时不是隔离测速。

| 编号 | 假设 / 核验 | 实测结果 / 资源 |
|---|---|---|
| A130 | 两个初值0的相对里程attention系数，同普通loss/初始化/曝光，I210双seed | cal选新速度+原树流量；旧后留.942187 vs强参照.940049；468.31秒、GPU allocated725.07MiB |
| A131 | 固定2/15–2/28，用强通道模型作reference | .941826 vs .940685，gain+.001141/day-mean CI[.000743,.001822]；9角色126次predict、76.52秒 |
| A132 | 距离/共同参数初始化/曝光/选中pt与全部cal/ass部署核验 | 252predict最大差2.27e−13，224指标score差1.11e−16，固定组合/cal gate通过；126.60秒 |
| A133 | 新14日teacher/diagnostics/126指标及首末部署 | 另18predict差0，score差1.11e−16；18.25秒 |

第一seed未来gain接近0/区间跨零，新日期急降速速度退步，旧增量诊断略差，均保留。三项新kernel/roundtrip测试通过；全套125research/10.41秒，CPU104/7Torch模块跳过/8.04秒。详见 `docs/RELATIVE_POSITION_STATE_RESEARCH.md`。尚无跨方向增益或正式成绩。

| 编号 | 假设 / 核验 | 实测结果 / 状态 |
|---|---|---|
| A122 | 只用两方向fit检查突变、持续/返回、邻居支持及actual exposure | 245日，速度大变化约.21–.25%；有效未来两步子集中持续96.76–98.86%，子集覆盖约61–63%；48.93秒，无模型验证 |
| A123 | 独立shift/window重建事件mask及全部人口/SSE汇总 | 245日3185数组exact、3430记录复核，最大绝对SSE差3.73e−9；35.60秒 |
| A124 | 仅已有连续监督pair的速度差分RMSE，系数.05，2seed同GRU/输入/采样 | 完成；cal−.004325，后留−.004843/负区间，保留NN；1089.42秒、allocated峰值2100.56MiB |
| A125 | 固定2/15–2/28所有7导出直接预测 | NN .959782 vs新均值.956831，gain−.002950/负区间；98predict，112.73秒 |
| A126 | 独立初始化/原cell与新pair计数/全部导出及score | 4archive全部日数组exact，196predict最大差4.55e−13，168指标差2.22e−16；213.94秒 |
| A127 | 新日历全teacher/diagnostics/score与首末导出 | 14日98指标差1.11e−16，另14predict差0；30.00秒 |
| A128 | 10方向公开静态测量图覆盖；source/CPU探针核对注意力范围 | I210_W53/81、I5_N0/258孤立；注意力已有全局通信，不能称reader缺陷；.92秒 |
| A129 | 不用neighbors helper，直接CSV ID/union-find核验 | 10图全部边/孤立/分量数一致；.36秒，无预测指标 |

两seed逐格监督45,754,069/45,844,015次与控制精确一致，pair曝光30,192/30,858次，alpha=.75/.5；独立A126现已完成。原控制alpha均.75，不能把完整策略差异全归因额外梯度。当前无运行中任务，后继manifest保留模型、预测与全部负面分层。

新四项诊断/梯度测试通过，全套122research/10.03秒，CPU104/6Torch模块跳过/7.99秒。A124旧控制明确复用；最终模拟seeds仍封存。详见 `docs/FIT_TRANSITION_AND_JUMP_RESEARCH.md`。

最新冬季固定通道迁移失败；cal拒绝替换，未扩大树/权重网格。

| 编号 | 假设 / 核验 | 实测结果 / 资源 |
|---|---|---|
| A116 | 迁移I210固定NN速度+point树均值流量，2seed仅拟合流量 | cal−.000801；旧后留−.000888/负区间，保留NN；1522万行、765组、2×400树；888.84秒、peak6917.55MiB |
| A117 | 固定策略2/6–2/19 | NN .955189 vs transfer .954343；day-mean CI[−.000920,−.000766]；70次直接predict，91.83秒 |
| A118 | 拟合并行时，独立坐标mask/全部60特征/teacher/base/weights/keys | 765组15218248行exact，8实际poison通过；raw teacher差1.19e−7归一化单位、weight差0；344.87秒 |
| A119 | 新日历raw teacher/diagnostics/126指标与采样导出 | 14日全重建；另18次predict最大差4.55e−13；46.37秒 |
| A120 | 复用已绑定SHA的A118表审计，28日×9导出部署/238指标/cal gate | 252次predict最大差4.55e−13，score差2.22e−16；speed booster原样、两q树各400；305.82秒，不重复计整表重建 |
| A121 | forward打开前，cal-only误差协方差与单标量最优混合 | alpha.132571、in-sample gain.00001953未过.0005门槛，不导出新模型，.20秒；非独立验证 |

新阶段没有修改生产src；原118research/102CPU及5Torch skips记录保留，不重复计作新实验。新增训练/整表/模型部署审计分别记录，最终模拟seeds不生成。详见 `docs/WINTER_FLOW_TRANSFER_RESEARCH.md`。

最新完成固定通道集成及冬季跨方向重mask训练。所有分数为本地公开TRAIN State；没有按新日历改选模型。

| 编号 | 假设 / 核验 | 结果 / 资源 |
|---|---|---|
| A112 | NN/point树/Queue树按速度和流量分开，50组cal常数权重 | 选NN速度+point树流量；旧后留.940049 vsNN .938919；28次直接部署差0，25.92秒 |
| A113 | 固定通道策略2/6–2/19 | .941077 vsNN .940073；day-mean CI[.000948,.001170]；4role×14日56次直接推断，29.35秒 |
| A114 | 新日历全teacher/诊断mask/保存指标及首末部署 | 56指标，另8次predict差0；10.24秒 |
| A115 / A115b | 旧28日raw teacher、112指标、50cal候选与封存权重独立重建 | 初次CPU环境缺Torch无法读嵌套GRU pickle，保留失败；research重跑通过，score差1.11e−16、7.25秒，无重训/新直接推断 |
| A106 | 冬季四模型完成后固定1/23–2/5 | 原GRU .954500，point .955275，cal primary Queue-shaped .954872；primary CI跨零，140.97秒 |
| A107 | 全部实际标签曝光/28日×12角色独立部署 | 4份exposure数组exact；336次predict最大差4.55e−13 lane units；277.12秒，不是重训 |
| A108 | 冬季新日历全teacher/诊断/168指标及采样部署 | 14日全部重建，score差2.22e−16；另24次predict差0；33.47秒 |

当前全套research118 passed/9.95秒；CPU102 passed/5 Torch skips/7.79秒。初次默认沙箱temp不可写导致各13个setup错误，指定工作区新basetemp后通过。没有更改数值断言。详细误差分层、失败和复现见 `docs/CHANNEL_STATE_RESEARCH.md`。

此前同视图树对照与冬季训练的数据准备：

| 编号 | 假设 / 核验 | 结果 / 资源 |
|---|---|---|
| A102 | I5_N153fit日同策略4view×point/Queue-shaped | 1224view，591.11秒；profile/fit/cal/assessment分离 |
| A103 | I210_W2seed×2variant同标签池强LGB，固定400树/63叶 | 旧后留point/queue .931238/.930716 vsNN .938919；cal仍选NN，500.86秒CPU；标签重加权同时变化 |
| A104 | 冬季2seed×2variant原GRU同初始化/预算，复用已核验控制 | 四模型完成，cal选Queue-shaped；原后留.957512 vs原GRU .956818，CI跨零；2154.28秒、allocated峰值2099.12MiB |
| A105 | 冬季全部输入/label masks与poison检查 | 1224view exact、16poison通过；归一化teacher差2.38e−7，498.93秒 |
| A109 | 树/NN策略固定后的1/23–2/5日历 | NN .929704，point/queue树 .924704/.924905；.25混合略正但CI跨零，primary不变；32.96秒、含GPU推断 |
| A110 | 全树训练表/权重及cal/ass角色核验 | 920组exact；168树role/day+4选中策略调用，392保存指标复算；182.26秒 |
| A111 | forward全teacher/diagnostics/score及采样部署 | 14日/210指标差≤1.11e−16，另18predict差0；9.92秒 |

该树阶段117 research passed/13.92秒、CPU101 passed/5 Torch skips/11.42秒为历史检查；最新测试见本文开头。表格数据策略、特征编码、损失及优化预算存在差别，不把结果归因单一组件。原0.82940及所有完整候选保持冻结，冬季结果已在上文更新。

最新A96–A101回到官方随机mask目标，记录实际数据增强预算，不以严重断测合成分数代替State。

| 编号 | 假设 / 验证 | 实测 / 资源 |
|---|---|---|
| A96 | 原92fit日输入审计+4seed×point/Queue-shaped新mask | 86.81%原目标gap≤25min，≥180min为0；736views、131.93秒CPU；新teacher只来自raw public fit |
| A97 | 2optimizer seed×2新数据策略，原GRU control显式复用 | Cal选Queue-shaped均值；原assessment .935491→.938919（+.003428）；point .938770；887.62秒、725.01MiB |
| A98 | 模型/alpha/primary固定后评估预登记一月14日 | primary .931933→.934816（+.002883），6个R3日；第二seed CI仍跨零；51.86秒、234.58MiB |
| A99 | 独立raw teacher、mask、history/guard、全输入重建 | 736view exact，16次真实loss-target poison通过；归一化float32 teacher差≤2.38e−7/固定1e−6容差；113.30秒 |
| A100 | 独立实际loss-mask曝光与全保存角色部署 | 4个完整曝光数组/标签计数一致；28日×12role共336次predict最大差2.27e−13 lane units；84.89秒 |
| A101 | 固定forward全部teacher/诊断/指标与采样直接推断 | 14日、168保存指标差≤2.22e−16；另24次predict差0；11.16秒 |

新数据策略与原控制标签位置/曝光数不同；两个新模型均值跨时间正面，但仅1个方向/同2seed，未生产晋级。原assessment R3仅1日，一月6日；bootstrap重采样日期而非目标cell。Queue-shaped对point差异尚弱，模拟origin不是官方条件事件。恢复速度/拥堵flow负面结果保留，真实Physics/Total未知。116 research tests通过/9.63秒，CPU100通过/5 Torch skips/7.04秒；新增跨午夜guard两kernel测试1.00秒，当前fit数据不变。

最新A92–A95进一步分离alpha影响，并检验输入门控的跨缺失分布泛化，保持负面结果。

| 编号 | 假设 / 核验 | 实测 / 决策 |
|---|---|---|
| A92首次 | 从已保存混合结果反推alpha.5 | 原alpha.75 neural float32乘法舍入，直接差.00016276 total units；原权重输出exact，未放宽复现门槛，旧源快照保留 |
| A92b | 全部14普通日+168case×4模型直接alpha.5推断 | 728次，普通gain−.002711；四长失效gain+.099034/+.133283/+.138672/+.172670；95.36秒；已查看数据上的解释性诊断 |
| A93 | 固定180min输入缺失门控，冻结新日期/8法律/3seed | 只选control或增强expert，权重及旧cal alpha不变；新标签前保存模型和协议 |
| A94 | 新15日普通模板+14anchor×3seed×8失效 | 普通State .936243与control相同，gate0；全走廊4h−.058782/负CI；4h50、8h90、跨午夜合成增益；不晋级。251.31秒、230.45MiB |
| A95 | 独立RLE/teacher/score以及真实保存门控部署 | 393gate数组exact，60普通/1512segment/1344case分数核验；44次直接predict差0；31.14秒 |

当前114 research passed/11.79秒；CPU98 passed/5 optional-Torch skips/9.15秒。官方mask为20/30/50%单元随机移除，加Queue90min缓冲空缺；严重额外断测成绩不能当正式State。原0.82940和全部完整候选不变，本轮没有新全网训练或完整submission。

最新A86–A91为同架构、同标签、同原采样预算的输入增强消融；负面结果保留，不因更多视图而晋级。

| 编号 | 假设 / 验证 | 实测 / 资源 |
|---|---|---|
| A86 | 92个fit日×4种metadata-only失效，完全重建strict12/base，原y/mask不变 | 368视图，63.72秒CPU；不打开raw teacher |
| A87 | 两seed×control/augmented，固定.5概率、24×120、97307参数相同 | 原公开assessment均值.935491→.931590，−.003901；两seed−.004898/−.002411，cal primary保留原GRU；892.30秒、725.01MiB |
| A88 | 训练前固定11/27–12/10、3新mask seed，所有选择不变 | 普通control .933808→增强.929945（−.003863）；四自主失效gain+.075265/+.113193/+.102134/+.140683；156.20秒、231.51MiB |
| A89 | 不调用增强移除helper的完整缓存复核 | 368输入/region逐值一致，8次实际测量poison检查；59.10秒，与训练并行 |
| A90 | 独立重建全部训练RNG/exposure和保存策略部署 | 两control权重/alpha与原策略exact；280次role/day推断最大差2.27e−13 lane units；74.52秒 |
| A91 | 新日期teacher/score复核与固定样本直接预测 | 126普通指标、168mask/teacher、1512合成分数重建，最大score差1.11e−16；另54次predict差0；19.99秒 |

两个control checkpoint所有参数与原A72b逐值一致；本轮按原JSON更正旧文档seed29 alpha（TCN .5 / GRU .75），没有修改旧模型/预测/分数。103 research tests通过、CPU87通过/5可选Torch跳过。新日期/mask在训练前登记，普通重建退步、合成长失效改善；后者目标人群不同，42中心/168嵌套case不能当独立事件，也不当正式增益。保持原cal primary，不按新结果改方案。

最新State两方向训练、固定模型二月评估与长失效压力测试全部完成；局部State增益和长失效退步同时保留。

| 编号 | 假设 / 对照 | 实测结果 | 资源 / 决策 |
|---|---|---|---|
| A72原试跑 | context128只监督中心64 | 22.25% fit目标无法被抽到；TCN完成、GRU epoch20后主动停止 | exit1，两个checkpoint保留；未读assessment，不作架构结论 |
| A78 | 精确fit-mask训练覆盖审计 | I210_W115049/517171、I5_N674637/3042465原不可监督；core-first理论全日可采样 | 只读fit；实际曝光另保存，不以理论代替实际 |
| A72b | 两seed、共享初始化/采样、TCN vs增加双向GRU分支 | cal primary GRU均值；后留State .933577→.935491，gain+.001914；两个匹配gain+.001321/+.002452 | 755.46秒GPU、725.33MiB；四模型实际覆盖全部517171格点 |
| A74初次 | 完整保存策略与独立部署 | JSON str传入仅接受Path的loader，预测前退出 | 原注册/源快照/异常保留；修复后新A74b目录 |
| A74b | 28日×10role从masked Parquet重建 | 280推断，input/graph exact，最大预测差2.28e−13 lane units | 74.78秒；均值单位换算浮点容差明确 |
| A79 | 第一方向冻结策略比较与曝光审计 | R3匹配gain+.002491/+.006832；3h缺失0目标；急变/flow收益不一致 | 不重选ass seed29较高的单模型；保留cal均值primary |
| A73 | I5_N冬相同四模型/两seed | cal primary GRU均值；后留.954329→.956818，+.002488；匹配TCN gain+.001062/+.000810 | 1970.18秒GPU、2100.55MiB；四模型全部3042465目标实际覆盖 |
| A75 | 冬季独立策略部署 | 28日×10role共280次推断，input/graph exact，最大差4.55e−13 lane units | 219.53秒；保存全部导出与预测 |
| A76 | 两fold共享初始化/采样/实际曝光对照 | 原两fold mean gain+.002201，本地State gate通过；四匹配seed均正 | 不当正式Physics/Total增益，不因assessment改选 |
| A77/A77b/A77c | 冻结模型的2月日历评估登记链 | 两方向各28日；GRU gain+.001311/+.001992，名义日均CI均正 | 300.61秒GPU、594.06MiB；旧注册/修订原因保留，日期/模型不变 |
| A81首次 | 独立生成3/6h、30%/70%额外失效 | flag bool写入int64被Pandas拒绝；尚无stress case预测 | 源快照/注册/异常保留；按dtype修复，在新A81b目录复跑 |
| A81b | 冻结四策略、全部特征从扰动后输入重建 | 168嵌套case；GRU对inc gain+.001701/+.000064/−.002945/−.002284；两个6h名义区间负 | 130.49秒GPU、230.58MiB；42个共享中心，不当168独立事故 |
| A82 | 独立失效验证 | 168mask/teacher重建、672分数差0；16stress策略调用及全部control重放 | 29.12秒；不以合成指标代替正式State |
| A83 | 同一冻结策略连续28日次要汇总 | I210_W incumbent .933915→GRU .935949，gain+.002033 | 合并原14日和新增14日未额外失效control；不覆盖6h负面结论 |
| A84 | 二月公开teacher及预测复核 | 56日/560保存指标独立重算差≤1.11e−16，另8次直接策略调用差0 | 21.26秒；不是560次新预测 |
| A85 | 普通模板与合成失效的证据图 | 分开坐标/目标人群，明确day-mean CI与pooled gain不同 | 已视觉核验、图与四来源报告SHA记录 |

当前101 research tests通过（9.97秒），CPU85通过/5可选Torch模块跳过（7.38秒）；包括batch独立、边界全覆盖和bool/int失效输入。原两fold本地State gate达到，但二月平均收益缩小、六小时失效退步，暂不扩成全网新最佳。新分支参数97307 vs匹配TCN75994，非等容量对照。训练/验证有并行，wall time不是隔离吞吐。原正式.82940与所有完整候选保持冻结，无新完整submission或正式成绩。

最新D21–D27研究共享需求中心+路径残差；两个development gate均失败，primary保留强entropy。

| 编号 | 假设 / 对照 | 实测结果 | 资源 / 决策 |
|---|---|---|---|
| D21 | 四个几何中心/计数权重配置，270开发问题后封存，再450新seed问题 | alpha.25开发mean+.018222、missing-prior−.005876；同类gain+.019200/CI正，新六类+.008855/CI跨零 | 634.65秒CPU；零solver failure，未晋级 |
| D22 | 全数据生成/score/直接API独立重跑 | 720份输入exact，4140分数差0，54次API预测exact | 83.64秒CPU；方法重复不算独立样本 |
| D23 | 计数rowspace/nullspace平方误差 | 3690份分解；null count残差6.36e−14；hypothesis的平均L1改善但平方L2更差 | 13.37秒CPU；不是正式L1分解 |
| D24 | development预注册保留floor20，alpha.25/.5；1350控制预测显式复用 | floor20整体mean−.000844/−.002507；仍选floor2且gate失败；另组三seed同类/新类gain+.018551/+.009706 | 438.02秒CPU；300.27MiB；不按D21assessment选择 |
| D25 | 后续全数据/分数/API独立重放 | 720输入exact，4140分数差0，54次API exact（包含floor20开发模型） | 73.54秒CPU；无覆盖失败或更改选择 |
| D26 | 第二批计数子空间误差 | 3690分解，null count相对残差7.92e−14；平方误差代价仍在 | 13.04秒CPU；公开十个算子复用，不当新网络 |
| D27 | weighted组件及相同hypothesis六seed次要汇总 | 同类gain+.018876/CI[.007580,.031924]；新类+.009280/CI[.000110,.020543] | 组件加和核验通过；未校正多重比较，不晋级 |

第一轮leave-one-family-out始终选grid0，但仅1/5个开发子集过gate；上游共享参数已见过全部族，不能称完整unseen-corridor结果。两轮模型均是count-informed center再作path correction，未校准posterior，不能恢复真零support。新增floor测试证明prior中心下限不等于预测下界；最终93 research passed、CPU81 passed/4 optional-Torch skips。正式最佳、完整候选与final seeds保持冻结。

最新D14–D20完成独立正值需求因子模型与数值修复；正式Total仍.82940，无新正式最佳。

| 编号 | 假设 / 对照 | 实测结果 | 资源 / 决策 |
|---|---|---|---|
| D14 | 乘性origin/destination log-demand MAP，4配置、9条件、10方向、3开发seed | development选sd=.2/noise=.02/floor2；初评3个选中case未过证书；含alias共7failure记录 | 456.71秒CPU；原失败、源快照、空整体CI保留 |
| D15 | 两方向×3条件×2配置，独立有限差分trust-region oracle及rank | 10/12一致；弱正则离群案例存在显著不同流量解 | 47.16秒CPU；不放宽门槛、不声称global MAP |
| D16 | 原sealed选中参数的独立数值复核 | 6/6一致 | 11.23秒CPU；仅development数值证据 |
| D17 | same-case exact-Hessian polish，保持目标/2e−4/选择 | 2700记录全部case成功；研究macro .724308 vs Ridge .664500 | 498.35秒CPU；已查看种子回归，不是新盲测 |
| D18 | polish后3配置独立oracle | 16/18一致，选中6/6；两旧差异仍记录 | 57.17秒CPU；局部多解风险保留 |
| D19 | 完整数值修复回归比较 | 7记录恢复、无新失败；1080参考记录exact；旧成功分数最大漂移4.48e−7 | 全部coverage与原选择一致；不删失败案例 |
| D20 | 同270个问题直接比较强entropy | 新模型对entropy+.016316，CI[−.001239,.037405]；5/9条件更差 | 不晋级；稀疏/离群条件收益不能冒充普适收益 |
| K07 LOCAL | log-demand OD-only完整控制，State/Queue原组件 | 6980503行；30个公开子问题全成功；prior L1变化41.11%–62.17% | 174.85秒CPU、335.38MiB；真实路径效果未知、未上传 |

新kernel测试覆盖Gaussian/Huber解析梯度、独立trust-region小系统、gauge方向及无效输入。最终88 research passed，纯CPU76 passed/4 optional-Torch skips。计数Jacobian rank I5_N92/184、I210_W81/162；局部KKT与高计数拟合不能证明路径真值。正值因子只作MAP点估计，尚无校准后验或真零support。最终模拟7907/8017/9109未生成，原正式.82940及已有完整候选不变；平台任务继续暂缓。

最新C38–C41与完整LOCAL K06完成，正式Total仍.82940；新时序规则gain为0，没有新模型晋级。

| 编号 | 假设 / 对照 | 实测结果 | 资源 / 决策 |
|---|---|---|---|
| C38 | 完整forecast概率、raw graph logits校正概率cache | 两季8个cal/assessment cache；原C37旧指标及新全域回放均复现 | 58.77秒GPU推断，无权重训练 |
| C39 | 固定checkpoints，全域混合/threshold校准；onset时间约束before/after force | 两季都wave30 α=.25 force=false；两时序输出相同 | 395.24秒CPU；不宣称新性能收益 |
| C39 deploy | 四模型全部校准历史重建与保存/加载推断 | 52/58窗口×两个时序，220次全部binary predictions精确一致 | 45.06秒；validation/private各15480行完整 |
| C40 | 合法latent-positive与scored-positive过滤的已知真值压力测试 | 单格/r.75 latent总体force .095533 vs空.252533；过滤后force .032356 vs空0 | 27case×5000事件，81rule/case共享样本；非官方生成器 |
| C41 | 选中混合与同样全域校准tree公平对照 | 秋gain.006606、冬.011703，两CI均跨零 | 旧观测proxy/一个fit seed，未晋级 |
| K06 | I5_N Queue-only完整本地控制 | 6980503行；只415个Queue值变，132val/283private | master-key全行diff通过；正式分数未知、未上传 |

C39 cal同时比较两种规则，先封存全部选择，再读取assessment。graph checkpoint早先按逐点Brier选择，当前不重训；source probabilities及geometry保持冻结。推断没有future score mask参数；scorer包含无eligible证据window=0。cal非空假设在当前观测正例采样中仍可能偏置，C40不是实际组织方政策证据。67 research /59 CPU tests、3Torch skips，完整本地候选不覆盖正式最佳。

最新A66–A71与C35–C37完成；没有新模型晋级，正式Total仍.82940。

| 编号 | 假设 / 对照 | 实测结果 | 资源 / 决策 |
|---|---|---|---|
| A66 | I210_W秋：State / legacy / coverage / full-denominator，λ=.01 | cal选coverage；State相对State-only−.002184，proxy L1改善.008442 | 689.64秒GPU；不反选、不晋级 |
| A67 | I5_N冬full标签缓存，153fit/14cal/14assessment | 原inputs/targets不变，额外eligible buffer labels=0；旧fit对304034不计公开coverage | 117.13秒CPU；无新模型信息增益 |
| A68 | I5_N冬，同四损失 | cal仍State-only；其他coverage/full State gain.000074/.000077，CI跨零 | 1122.99秒GPU；不按assessment反选 |
| A69 | 两组匹配对照及日级不确定性 | I210coverage State CI全负；冬proxy error-reduction CI正但State收益小 | 只有一个fit seed、两个panel/fold，不晋级 |
| A70 | 固定fit采样的分母强度审计 | 两组full/target系数均值.542741/.538899；batch有变化 | λ变化不是新增物理信息；无cal/assessment调参 |
| A71 | A66 cal模型冻结，追加30日 | State .932585→.932105，日均CI[−.001044,−.000118] | 33.51秒；public-train开发复核，非final |
| C35 | 旧C27–29模型同origins，full-domain vs future-eligible domain | 41次评分单元决策变化，选中expert模型全部不变；48组旧指标精确复现 | 127.79秒CPU；不重调权重或threshold |
| C36 | 使用修正域去重重算CI，保留旧cal选择 | C27 gain改为+.017150但CI跨零；C28/C29不变 | 不构成新模型收益；旧负/正记录保留 |
| C37 | C32/C34固定波速checkpoint/alpha/threshold全域复跑 | 秋absolute IoU整体−.025，gain不变；冬不变；32组旧指标精确复现 | 因nonempty/domain而非输入未来值；不重新选模型 |

A66/A68 State-only与legacy参数张量复现A52/A34，新的两种loss只改监督条件与分母。原seed不是新replication。cal保存角色各14日从masked数据和raw TRAIN evaluator labels独立重建。63 research /55 CPU tests、3 Torch skips；全源码compileall通过。C35/C37未来mask只可在scorer使用；旧nonempty/expected-prefix的filtered-domain数字标记为历史诊断。新的全域校准和timing-before-force尚待独立实验，不能把当前fixed-rule重放包装成已经完成的重新优化。

最新A58–A65完成；没有新State晋级或正式成绩。同化不重新训练先验，仅选择cal precision/strength；含CPU求解及GPU冻结先验推断。

| 编号 | 假设 / 对照 | 实测结果 | 资源 / 决策 |
|---|---|---|---|
| A58初次 | 原条件先验残差同化 | 第一份cal输入前时间单位验证失败，无分数/模型 | 原注册与start snapshot保留；显式转ns修复 |
| A58修复 | I210_W秋，20配置×3strength+原对照 | 所有因素组都cal选原先验；assessment .933577不变 | 39.94秒；不测未选非零assessment方案 |
| A59 | I5_N冬，相同配置网格 | primary .954329→.954406，gain .000076；joint gain .000003 | 118.90秒；增益过小，不晋级 |
| A60 | I210_W秋，5组遮观测anchor center | 原观测/目标先验逐值不变；cal仍选α=0 | 146.01秒；额外推断未取得收益 |
| A61 | I5_N冬，相同遮观测中心规则 | .954329→.954453，gain .000124、日均CI [.000059,.000151] | 472.95秒；proxy比A59 primary更差，不晋级 |
| A62 | I405_N冬，普通中心跨方向 | .933986→.933992，gain .000005、日均CI跨零 | 223.48秒；不晋级 |
| A63 | I405_N冬，5组遮观测跨方向 | .933986→.934099，gain .000112、日均CI [.000075,.000156] | 769.57秒；远低于gate，不晋级 |
| A64 | 三paired策略的输入/数值/误差对照 | 三组各28日prior/strict/quality/edges完全一致；3836次channel CG成功 | cal选择参数不同，非单一模块归因；旧proxy口径保留 |
| A65 | public TRAIN full-label/Mode A coverage缓存 | 120日全部input/target不变；额外eligible buffer anchors=0；fit旧损失177808对不计入公开coverage | 75.68秒CPU；未训练，没有模型增益 |

时间为完整实验的实测wall time，多任务存在并行，不能作为隔离吞吐比；遮观测机制本身需额外五次全帧先验推断。每项保留注册、源快照、selection seal和部署模型。各不同文件SHA的模型都在14日cal独立推断；相同角色SHA共享复跑明确记录。直接求解对照仅为真实24-slot子集，全grid另有CG残差证书；不扩大验证范围。59 research /51 CPU tests、3 Torch skips；正式Total仍 .82940。full-coverage density loss已实现并手算验证，但未训练；测试编辑边界错误导致一次58 passed/1 failed，原snapshot保留，复跑59 passed，没有修改实验分数。

最新完成 A51–A57；formal Total 仍 .82940，未晋级新 State 或声称 Physics 提升。

| 编号 | 假设 / 对照 | 实测结果 | 资源 / 决策 |
|---|---|---|---|
| A51 | 在原 strict12 缓存追加六个合法质量通道 | 120 日原输入/target/labels 不变；fit 151,109 行低质量且 v/q 有限 | 43.23 秒 CPU；训练目标的 occupancy 全缺失 |
| A52 | 12/18 通道 × λ=0/.01，完全共享初始参数与 patch RNG | cal 选 18 通道、epoch24/α=.5；State .932965 vs strict12 .933577 | 410.51 秒 GPU、288.59 MiB；不晋级；两控制复现 A49 全部张量/历史 |
| A53 | 质量可见率与旧模型兼容审计 | R3 .381506→.451711；旧模型 14 日 cal .932591 重建一致 | 不把可见率当准确率；旧 pickle 未重新保存 |
| A54 | 同初始化 strict12，联合 v/ρ 坐标代替独立 v/q | cal 选密度 loss；State .932951 vs 原 .933577；旧 proxy L1 .701016 vs .716127 | 222.82 秒 GPU、288.42 MiB；State 不晋级，保留 joint challenger |
| A55 v1/v2 | 对齐公开 Task3 非目标真值 anchors、Mode A 与全部有效相邻对 | v2 projected L1 .404858→.389999；统一上界仅改4个 LGB 预测 | 真边界流未访问，真实 Physics 未知；v1 无统一上界原样保留 |
| A56 | 匹配 State / coverage-proxy 的日块不确定性 | 两新模型 pooled State 为负；density proxy 日均 reduction CI [.006049,.009574] | 不用 proxy 替代正式分项，不根据 assessment 改选 |
| A57 | 冻结模型追加30日时间复核，无训练/调参 | 原 .932585；质量 .932977；密度 .932583，两 gain CI 跨零 | 37.38 秒；7个 R3 日；public-train development，不是 final |

primary/joint 在 A52、A54 各选同一 checkpoint；每个角色全部14日 cal 部署复跑一致。53 research /46 CPU tests +3 Torch skips；全脚本 compileall 通过。A55/A57 GPU 标志默认 false 的记录错误单独披露于 `quality_density_metadata_corrections_v1.json`，不覆盖原 SHA 链；后续 code 已记录实际 network device。复现命令、参数化公式和覆盖差异见 `docs/QUALITY_DENSITY_RESEARCH.md`。

本轮完成 A48–A50、B08/B09、D12/D13，以及完整 LOCAL K05。正式 Total 仍 .82940，没有上传或替换正式最佳。

| 编号 | 假设 / 对照 | 实测结果 | 资源 / 决策 |
|---|---|---|---|
| A48 | D7_I210_W 同秋季图缓存 | 92 fit / 14 cal / 14 assessment、81 nodes | 75.55 秒 CPU，输入与强基线 profile 匹配 |
| A49 | 三架构与 full-label LGB 匹配 | cal 选 attention α=.5；State .927102→.933577；日均 CI [.004342,.006415] | 311.71 秒 GPU、288.65 MiB；14 日 cal standalone 重建通过 |
| A50 | 公开 train 缺失段长度 | 三 cache R3 input p99=60min、target p99=25min | 仅 availability/mask；跨 panel 共 360 个文件不是独立 360 日 |
| B08 | 四种自主连续缺失 × 六条件 × 三 seed | 插值 State：uniform .832553、半走廊 .711935、事故 120min .118663、全网45min .576647 | 4.86 秒；独立复跑 5.99 秒，全 72 case 精确相同 |
| B09 | prior × interpolation 2×2，并检验 +1h 时钟 | 加权插值使 120min 缺失 .118663→.227091；prior-only 时钟偏移严重退化 | 26.59 秒 GPU；144 case、576 adapter 配置；无模型选择 |
| D12 | 六 MAP/Huber 配置与可见 prior 规则 | 选相对 Gaussian cv=.25、η=0、threshold .75；旧 seed gain .004085，CI 为正 | 749.94 秒 CPU；额外 sparse penalty/Huber 未选；30 个赛事输入全部退回 Ridge |
| D13 | entropy fallback + MAP 专家，开发缓存选定后开三个新 research seed | 九类模拟 .663559→.732618；gain CI [.052272,.086461]；最差类均值也正 | 89.52 秒 CPU；540 method cases 无失败；非 final / 正式测量 |
| K05 | 完整 OD-only 本地控制 | 6,980,503 行；仅 70,708 OD 值改变；所有 State/Queue 字段不变 | 146.60 秒构建；全行独立 diff 通过；未上传，真实成绩未知 |

D12 的 standalone 相对 Gaussian 还会在 missing-prior 与 group-biased 场景退化，不能按平均分直接替换所有方向。D13 只用开发案例选择了 absolute cv=.75/η=1 的稀疏分支；新 seed 平均 sparse ODME .653270→.765436，但当前 validation/private prior 浓度都不触发，K05 实际全为 entropy。旧 K01 正式失败仍是反证，新的模拟并不消除该风险。

D13 新种子中14/270案例为负，最差 count_outliers .555405→.527118（−.028287），属于D7_I405_N、seed7001、entropy分支。误差审计另存，模型不根据已看案例改选；下一步应从开发端检验可见计数不一致特征及稳健 likelihood，而不是只优化均值。

B08/B09 比较严格分离 target population：不同 missing pattern/时钟有不同 targets，不把它们的绝对分数差当同标签算法增益。B09 同场景 weighted vs temporal-first 是配对适配器对照；该加权方式本来就在生产 StateFeatures 中。历史 profile 仅使用 dev seeds 411/829/1201 的 eligible masked-observation 字段，全局按 288 slots/cell 聚合；六个 profile 元素没有观测而回退几何。场景共享噪声 RNG，不能把 36 个合成日当 36 个独立样本。

本轮完成 A46/A47、B07 与独立进程复跑、D09 失败记录/D10 全量修复与覆盖率复核。没有模型晋级或新增正式成绩。开发和 assessment 的身份保留，最后独立模拟 seeds 仍未开启。

| 编号 | 假设 / 对照 | 实测结果 | 资源 / 决策 |
|---|---|---|---|
| A46 | D12_I405_S 同 autumn fold 的 masked graph cache | 92 fit / 14 cal / 14 assessment，44 nodes；full-LGB profile 和输入重建匹配 | 57.63 秒 CPU；缓存及 start snapshot 保留 |
| A47 | 同强 LGB 的 geometry/attention/density 控制 | cal 选 density α=.75；State .941500 → .941872，gain .000371；日均 CI [−.000654,.004659] | 203.31 秒 GPU，179.63 MiB；未过 .002 gate，14 日 cal standalone 核验通过 |
| B07 | A33 固定 checkpoint 在独立 CTM 的 54 条件 | interpolation .853404，pure attention .848350，density .855957，half-density .864132 | 7.79 秒 GPU、46.59 MiB；mild 退化、CTM-flux 诊断下降，不晋级 |
| B07 repeat | 相同全部条件的独立进程复跑 | 所有 registered/input SHA/候选值/summary 完全一致 | 9.04 秒；3 seeds 而非 54 独立样本 |
| D09 | 冻结点估计 + Hessian 区间/新失配 | 校准倍数 6.402476；汇总时 metadata 覆盖模型身份，assessment aggregate 未保存 | 失败源快照、seal、210 校准数组保留；不接受缺失/NaN 成绩 |
| D10 | 仅修复 D09 元数据隔离/阶段保存 | 同 210 数组和倍数；旧 420 point scores 完全相同；600 assessment method cases、0 failures | 167.91 秒 CPU；模拟 sparse-active −.075310，原三登记条件 CI 跨零，不晋级 |
| D11 | 先核对路径选择是否存在于公开 path set | 十方向全部 OD 对单路径；D10 route-choice 标签实际为既有 demand 扰动 | 2.24 秒 CPU；两类真正新分布 macro CI [−.016780,.051299]，不晋级 |

D11 更正 D09/D10 生成标签的含义：`route_choice_shift` 是历史名称，singleton OD 下 preference 归一化抵消，因此不能称为新的路径选择分布。上表原“三个新分布”指三个登记条件；实际为两类新失配与一次既有 demand 分布复核。原指标未修改，九种实质生成分布对应十个历史标签。以后路径选择候选必须先证明公开资产有相同 OD 的可替代路径，或明示自主生成的替代路径实验范围。

A47 消融保留原 cal 权重：geometry .940372、attention .941929、density .941872；未因 assessment 中 attention 略高而改选。B07 density 的急剧降速 target latent speed RMSE 28.969 → 18.243，但恢复 speed RMSE 22.827 → 25.351；795 / 788 是共享缺失率下的重复 target 计数，不是独立事件数。事故片段收益不能抵消 mild 或恢复失误。

D10 empirical interval 的 raw coverage 在七个旧分布为 .3048–.7801，校准后 .9498–.9996；新区间宽度约真实平均路径流量的 2.54–3.45 倍。校准只改变诊断区间，不改变预测点。Sparse-active 的 .9766 边际覆盖率没有转成准确路径恢复或任何全路径覆盖保证。精确范围、所有 case 与失败原因见原 JSON；独立 dense Hessian inverse、equal-case calibration tests 核验数值。

本轮完成 A40–A44、C31–C34、D06–D08。I5_N 三 seed 与 I405_N 两季的注意力收益均相对 full-label LGB 复核；三种子等权组合保留为开发模型。Queue 两季 cal-selected 波速图 CI 均跨零，不晋级。熵反演在同一生成分布的数值修复后通过模拟门槛，真实 OD 仍未知；原正式最佳不改变。

| 编号 | 固定操作 / 对照 | 测量 | 资源 / 决策 |
|---|---|---|---|
| A40/A42 | I405_N 冬季图缓存、500k/full 同参数控制 | 153/14/14 日；full 3,985,807 labels；State .923899 → .925958 | 459.01 秒准备、497.09 秒 CPU 控制；非正式分项 |
| A41 | I5_N 第三 NN/LGB seed 20261107 | cal 选 attention α=.75；State .942330 → .950416；日均 CI [.006530,.008567] | 504.70 秒 GPU，845.07 MiB；14 日 standalone cal 复跑通过 |
| A43 | I405_N 冬季，同 NN 协议与 full-label 强基线 | State .925958 → .933986；CI [.007056,.009472]；increment proxy .765975 → .726403 | 705.18 秒 GPU，1180.95 MiB；14 日 standalone 复跑通过 |
| A44 | 三个原 cal 成员等权 vs 同三 LGB 等权 | State .942742 → .951168；CI [.007304,.008681]；proxy L1 .756319 → .671424 | 108.25 秒；共享 14 日，不冒充 42 日；一个整日 wrapper/save/reload 核验 |
| A45 | I10_W 监督注意力 vs 同日期 full-LGB | State .951973 → .955810；CI [.003496,.004039]；proxy L1 .719817 → .701798 | 401.02 秒 GPU，359.92 MiB；14 日部署复跑通过；原 cal 选择保持 |
| C31 | 固定 checkpoint 的 class-weight 概率修正 | graph Brier .015419 → .012059；IoU 组合仍选 α=0 | 187.20 秒；概率误差下降不代表 Queue IoU 增益 |
| C32 | 秋季 direct/15min/30min 波速图范围对照 | cal 选 30min α=.25；去重 gain .006606，CI [−.004692,.018938] | 176.90 秒 GPU；不反选 assessment 更好的 15min |
| C33 | 冬季普通因果 temporal/graph，正确加权概率 | cal 选 graph；去重 gain .008485，CI [−.006698,.028666] | 219.89 秒 GPU；不晋级 |
| C34 | 冬季同 C32 的三范围对照 | cal 选 30min α=.25；gain .011703，CI [−.001776,.032738] | 185.75 秒 GPU，111.27 MiB；跨季区间仍跨零 |
| D06 | 七种已知真值分布，熵双空间 Newton vs Ridge/Bayes | 69 次优化失败；有效配置未过最差开发场景门槛，选 Ridge | 769.35 秒；不删除失败 |
| D07 | 浮点舍入预算内接受残差下降步 | 原 69 例全部恢复；新增 1 例失败；dev 选 τ=.05/floor20 | 356.32 秒；该失败不在选定配置，继续数值修复 |
| D08 | 用 Newton decrement 判定浮点精度附近步骤 | 全量 0 failures；Ridge 分数完全相同；既有成功 entropy 最大漂移 2.32e−9 | 341.75 秒；tol 2e−6 不变，全部生成/选择条件保持 |

D08 的独立生成模拟 ODME：Ridge .691063、固定旧 Bayes .728094、开发选定 entropy .747273。family/seed bootstrap 的 entropy gain CI [.043844,.068733]，只适用于这七种生成分布与三个 assessment seed；final seeds 尚未生成。旧 Bayes 在模拟上更好却已正式失败，不能把新模拟收益当正式增益。公开 train 计数只能做稳定性诊断；entropy 约 99% 的变化在零空间。

本轮源代码与模型结果均另存；A45 监督复核完成，不能抵消 A37 零样本失败。平台工作按用户指示暂缓，只有本地 Git 归档。43 research-env tests、CPU 38 passed / 3 optional-Torch module skips；数值修复还由 D06→D08、D07→D08 同场景比较核验。

采样审计 C23 发现组织方公告 742350 已明确 onset 的最后一步属性。首轮代理和 C01–C22 不匹配该时序，全部保留为一般事件压力测试。只有 C24 起的 `--onset-last-step` 对照近似这一公开属性；仍未取得 underlying-state queue truth。

| 编号 | 假设 / 操作 | 数据与划分 | 结果 | 资源 / 决策 |
|---|---|---|---|---|
| F00 | 冻结首轮正式提交 | 本人 Kaggle 回下载，与本地文件 SHA 比对 | 正式 Total **0.82940**，ref 57138874；四项正式分数未知 | 106 个冻结文件，712,127,414 bytes；保留基准 |
| P00 | 公共拓扑是否导致部分方向无法评分 | 公开拓扑、原版 scorer、人工理想两时刻状态 | 十方向全部有可评分连接，否定结构性 0.950 上限猜想 | 本地诊断漏掉外部连接；修复并测试 |
| B01 | CTM、缺失、事故和噪声压力实验 | 独立 54 个场景，3 seed × 6 条件 × R1/R2/R3 | 守恒误差 ≤6.76e-13 vehicles；State observation oracle 1.0，latent oracle 0.952006，时空插值 0.887547，RTS 0.848378 | 7.60 秒 CPU；模拟数据不等于正式生成器 |
| D01 | 分层贝叶斯 OD 对比 Ridge | 两走廊、独立已知真值模拟 | 初步增益，但 zone set 的迭代顺序依赖 Python hash seed | **不作为可复现结论**；保留记录，改为 sorted 后重跑 D02 |
| D02 | 固定组顺序的分层贝叶斯 OD | 十走廊，每方向 3 生成分布 × 3 assessment seed；selection seed 独立 | 模拟 ODME Ridge 0.825047 → Bayes 0.870706；少数 case 退化，最差 −0.003804 | 72.95 秒 CPU；public_reference 从未读取 |
| D03 | 不同 Python hash seed 独立进程复跑 | D12_I5_N / D7_I210_W | 除耗时和运行指纹外，全部数值一致 | 支持重复性，不证明隐藏真值准确 |
| C01 | 随机事件采样、onset/ongoing 分别建模 | D12_I5_N rolling_autumn；带噪观测代理 | old joint 0.426710 → random joint 0.558105 / separate 0.570598 | 同 14 天抽样种子重叠；非独立样本 |
| C02 | 跨方向复核事件模型 | D7_I10_W rolling_autumn | old joint 0.770357 / random joint 0.781109 / separate 0.745426 | **独立事件模型退化**，不晋级 |
| C03 | 跨季节复核事件模型 | D12_I5_N rolling_winter | old joint 0.596339 / random joint 0.657088 / separate 0.646768 | separate 不如共享模型，不晋级 |
| A01 | 缺失感知时间卷积 / 图扩散 / 图+FD 联合损失 | D12_I5_N rolling_autumn，同 92 天 fit / 14 天 selection / 14 天 assessment | selection 选择 graph+LGB α=.5；assessment State 0.936655 → 0.940205；日均配对增益 0.002739，95% block CI [0.001963,0.003462] | GPU 56.43 秒，峰值 344.41 MiB；只有单走廊单 seed，不晋级 |
| A01-FD | FD auxiliary weight .01 | 同 A01 | selection 低于纯 graph；未按 assessment 反选 | 保留负面消融 |
| A02 | 冬季 rolling 对照 | D12_I5_N rolling_winter | State 0.940342 → 0.942887 | 同架构、同参数，α=.5 |
| A03/A04 | 第二个 NN fit seed 20261029 | 秋季 / 冬季，同固定 LGB 比较 | selected State 0.939143 / 0.944371 | 增益继续存在；LGB fit seed 固定 |
| A07/A08 | 第三个 NN fit seed 20261107 | 秋季 / 冬季，同固定 LGB 比较 | selected State 0.939584 / 0.943671 | 多 seed 仍只覆盖单方向 |
| A05 | 零样本跨 family transfer | D12_I5 → D7_I10_W，目标 family 标签不进输入 / 无目标 calibration | 插值 0.928998，纯 graph 0.913238，固定半数组合 0.927536；日均 CI 完全负 | **迁移失败**，不能据此部署共享全走廊模型 |
| A06 | 第三个 rolling 时间段 | D12_I5_N rolling_august | 300k LGB 0.944993 → graph α=.25 0.946550 | 52.92 秒 GPU；增益低于 .002 |
| A09 | public-train accumulation increment 联合损失 | 同 A01；有观测节点固定，仅预测 target | λ=.01；按 State 仍选择普通 graph，未选 auxiliary | 71.17 秒 GPU；不将 density increment proxy 称正式 Physics |
| A10 | 由 cal density 指标选联合候选 | 同 A01；State 至少不低于 baseline 的约束 | State 0.938577；target-pair L1 .824392 → .786185 | 单方向、单 seed、300k 比较组；不是正式 Physics |
| A11 | density-increment 权重 .1 | 同 A01 | selection 选择 α=0，无 State 增益 | 24.66 秒 GPU；强约束失败 |
| A12 | I405_N 逐方向图专家 | D7_I405_N autumn，352 sensors | 300k LGB .917777 → graph FD α=.5 .920775 | 78.50 秒 GPU；仍待强基线预算对照 |
| A13 | I10_W 逐方向图专家 | D7_I10_W autumn | cal 选择 α=0，assessment 保持 .951806 | 86.63 秒 GPU；无独立增益 |
| A14 | 同参数 LGB 标签预算控制 | D12_I5_N autumn，300k → 全部 1,804,908 标签 | State .936655 → **.942340**，超过原 graph 组合 .940205 | 191.10 秒 CPU；修正架构收益解释 |
| A15 | 强 LGB 上重做图组合 | 同 A14，仅 cal 选 graph FD α=.25 | assessment **.943417**，比强 LGB +.001077；日均 block CI [.000717,.001161] | 低于 .002 晋级门槛；无正式 State/Physics 测量 |
| C04 | 发布 T 时刻数据的合法可用率 | validation/private 的 at-T 时间谓词扫描 | 平均可用约 44%–59%；不读未来数值 | Rules 允许 ≤T，原模型只用 <T |
| C05 | 同窗口 at-T 消融 | I5_N autumn | before-T 600/63 .540552 → at-T .612418，cal 选择 at-T | 73.45 秒 CPU；10 个 cal 窗口 |
| C06 | 分离 latest-valid bookkeeping 作用 | 同 C05，增加两个 before-T 对照 | cal 选中 before-T-latest 600/63，assessment **.526387** | 98.76 秒 CPU；选择失稳，不能按 test 改选 .612418 |
| C07 | at-T 跨方向检查 | I10_W autumn | cal 选 at-T 600/63 .802211 vs before-T .765435；at-T 400/23 退化 | 42.22 秒 CPU；仅 6 个 cal 窗口 |
| C08 | at-T 跨季节检查 | I5_N winter | cal 选 at-T 400/23 .697673 vs before-T .670006 | 132.37 秒 CPU；仅 10 个 cal 窗口 |
| D04 | 扩大独立 prior 失配分布 | 十方向、4 种 reverse noisy prior、3 assessment seed | Bayes 在 missing-prior-path 模拟下 .606048 vs Ridge .611359，负增益 | 50.28 秒 CPU；未解释正式全量退化，不以此调隐藏反馈 |
| A16 | 冬季全标签 LGB 控制 | I5_N winter，全部 3,042,465 个标签 | State **.946854**，超过此前最佳冬季图组合 .944371 | 290.08 秒 CPU；强基线结论跨季节成立 |
| C09–C11 | 扩大事件 cal 配额至每条件 30，assessment 至 20 | I5_N 秋/冬、I10_W 秋，实际配额由可用事件限制 | cal 选定候选未稳定改善；唯一 origin 的 at-T 配对 CI 均跨零 | 不晋级；保留 65 / 14 / 59 个唯一 origin 的按日 block 诊断 |
| A17 | 冬季强 LGB 上重做图组合 | 同 A16，cal 选 graph FD α=.25 | State .946854 → .947299；日均增益 CI [−.000010,.000610] | 低于 .002 门槛，CI 跨零 |
| A18 | 较大 I405_N 的全标签 LGB | 2,438,669 fitting labels，同 A12 | assessment State **.921455**，超过原图组合 .920775 | 245.22 秒 CPU |
| A20 | I405_N 强 LGB 上重做图组合 | 同 A18，cal 选 graph FD α=.25 | assessment State **.922655**，增益 .001200 | 低于 .002 门槛 |
| C12–C14 | occupancy 当前值/趋势/上游下游/速度密度一致性 | 同 C09–C11，全部特征 timestamp ≤T | I5_N 秋/冬去重代理增益 .062191 / .048073，CI 正；I10_W **−.050232** | 双方历史只用共同发布 occupancy，不使用 window_history 缺失的 density 字段 |
| C15–C17 | 45 天 cal / 30 天 assessment；事件条件/正负类均衡损失 | I5_N 秋/冬、I10_W 秋；独立新开发划分 | cal 均选 occupancy 600/63；assessment .680572 / .658247 / .574586；I10_W before-T 400/23 为 .622312 | 更长 cal 未消除跨方向退化；事件权重候选未由 cal 选中 |
| A19 | 全标签 direct density expectation 与 q/v 联合投影 | 同 A14；复用原 State 权重，仅拟合密度回归器 | cal State 选 .1，assessment .942174；joint 选 .5，State .940472、density L1 .727239 vs .771083 | 195.49 秒 CPU，峰值 2619 MiB；没有正式 Physics |
| A23/A23b | 发布 occupancy/derived-density 的 State 树特征 | 同 A14，全 1,804,908 labels | A23 在拟合前因只读数组失败，保留；显式 copy 后 A23b State **.942621** vs .942340 | 219.06 秒 CPU；微小增益不晋级 |
| C18/C19/C20 | 扩大合法训练事件预算 | 同 C15/C16/C17 的日期；fit 配额 500，spacing 30 分钟；assessment 仍六小时 | cal 选 occupancy，assessment I5_N 秋 .747888 / 冬 .725320；I10_W .657542 vs before-T .681712，仍退化 | 不将增加训练窗口视为增加独立事件样本 |
| D05 | 坐标可识别性与完整网络负载诊断 | 公开网络/计数/本队两份 OD 预测 | 所有计数列模式均不同，**无重复列**；仍没有一般线性可识别的单独路径，max leverage 平均 .02779；K01 未计数链路负载漂移约 8.2% | 否定重复列猜想；大零空间来自线性组合，不读取隐藏 path truth |
| A24/A25 | 与正式 500k 预算对应的强基线 / 第二个 full fit seed | I5_N autumn，同日期与模型参数 | 500k State .938184；full seed 20261029 .942295；seed 20261013 full .942340 | 145.94 / 200.76 秒 CPU；其余 seed 控制继续进行 |
| A24–A28 汇总 | 三种子 500k vs 全标签配对 | I5_N autumn，同 14 assessment days | full 平均 pooled State 增益 **.003896**；平均跨 seed 的日均 gain .003146，block CI [.002445,.003846] | 三个 fit seed 不冒充 42 个独立日期；满足这一开发方向的 .002 门槛 |
| C21/C22 | 合法公开 condition 与观测推导 condition 对照 | 同 C18/C19，所有窗口/标签匹配 | cal 仍选普通 occupancy；published-condition 候选没有稳定独立收益 | 保留失败，模型部署支持该显式可选标记但不默认开启 |
| B02 | 学习 Δdensity 并固定可见观测的三对角同化 | 同 A14 全标签基线；public-train target-pair 标签、geometry 权重 | State 选 strength=0；joint 选 .1，assessment State .942160 vs .942340、density L1 .743408 vs .771083 | 317.45 秒 CPU；没有正式 Physics，可见锚硬固定 |
| A29/A30 | 六个其他方向的 500k/full 对照 | autumn，同早期 profile / fit / selection / assessment | full pooled State gain：I5_S .000838 / I10_E .001078 / I10_W .000059 / I210_E .000873 / I210_W .000937 / I405_S .001565 | 全部低于 .002；124.28 / 207.56 / 141.96 / 358.97 / 115.91 / 119.43 秒 CPU；训练前源码 ZIP |
| C23 | 公开 onset 公告与采样审计 | 40 个公开 train 窗；本队 val/private 预测与公开窗口 metadata | 39 个带噪 train 首次 queue at T+30，1 个无观测 queue；本队仅 5 个早期正预测 | 组织方允许时序属性；不读 val/private 未来观测，不视为真实标签评分 |
| C24 | 与公告 onset 时序对齐的 at-T / occupancy 对照 | I5_N autumn，45 天 cal / 30 天 assessment | 109 个去重 origin；before-T .871047 → at-T .897422，gain .026375，CI [−.007666,.065189] | cal 选 at-T；366.66 秒 CPU；occupancy 显著退化，不晋级 |
| C25 | 公告对齐的跨方向检查 | I10_W autumn，同 C24 方法 | 26 个去重 origin；before-T .606343 → cal 选 occupancy .628012；CI [−.064953,.082322] | 80.95 秒 CPU；仅 5 个 onset，证据弱，不晋级 |
| C26 | 公告对齐的跨季节检查 | I5_N winter，同 C24 方法 | 103 个去重 origin；before-T .876610 → cal 选 at-T .868240；gain −.008370，CI [−.032518,.009781] | 401.91 秒 CPU；保留退化，不按 assessment 改选 occupancy |
| K02 | 组织方已公开时序的 Queue-only 修正 | 原正式提交，所有其他值逐行保留 | 完整 diff 仅 5 个 Queue 值，private 完全相同；未上传，正式分未知 | 合并 125.42 秒；保留本地，正式比较优先具有 private 改善空间的模型 |
| A31 | 补齐三方向 500k/full 对照 | 同秋季，D12_I405_N/S、D7_I405_N | gain .000741 / 0 / .002521；I405_S 两组均已覆盖全部 fit 标签 | 145.61 / 89.65 / 392.02 秒 CPU |
| A32 | 冬季 500k/full 对照 | I5_N，同 A16 日期/参数 | .941587 → .946854，gain .005268；日均 CI [.004170,.005316] | 398.53 秒 CPU；full 复跑 pooled State 与 A16 一致 |
| A29–A31 汇总 | 按官方五族权重合并十方向 | 秋季，seed 20261013，同 14 天 | .937382 → .938659，global gain .001277；日均 CI [.000823,.001327] | 52.93 秒重建日统计/汇总；低于全局门槛 |
| C27/C28/C29 | onset 最后一步 / ongoing 六步专家与 expected-IoU | matched C24/C25/C26 事件与合法特征 | cal 都选专家 threshold；去重 gain −.007850 / +.034186 / +.013551，CI 全跨零 | 208.83 / 50.54 / 218.37 秒 CPU；不反选未被 cal 选中的胜者 |
| C24–C29 复跑 | 共享对照从公开帧重建并拟合 | 三组原对照与独立新研究 | 树文本、阈值、逐窗 proxy IoU 完全一致 | 不等于真实 Queue 或正式全量复现 |
| B03/B04 | 无新训练的 masked density 插值投影 | I5_N 秋/冬，强 LGB；cal State 容忍 .005 | State 都选 baseline；joint 秋 .937994/.757656、冬 .946067/.729182 | 39.58 / 37.03 秒 CPU；诊断收益小，State 代价明确 |
| B05/B06 | L1 条件中位数密度回归 | 全 fit labels，强 LGB；cal State 容忍 .005 | State 都选 baseline；joint .5 秋 .939019/.732209、冬 .944348/.707246 | 215.40 / 262.97 秒 CPU；秋季不如 A19 L2；无正式 Physics |
| A33 | 几何图 / attention / attention+density 的强基线匹配对照 | I5_N 秋，seed 20261013，全标签 LGB，cal 选择 | State .942340 → .949192，α=.75；日均 CI [.006348,.007157] | 489.27 秒 GPU，845.07 MiB；standalone 14 日独立复跑通过 |
| A34 | 同协议跨季节 | I5_N 冬，seed 20261013 | State .946854 → .954329，纯 attention α=.75；CI [.006259,.007608] | 1139.94 秒并行 GPU，845.07 MiB；不是正式分项 |
| A35 | 同协议第二 NN/LGB fit seed | I5_N 秋，seed 20261029 | State .942295 → .948993，attention+density α=.75；CI [.005566,.006983] | 1140.11 秒并行 GPU，845.07 MiB；与 A33 共享日期 |
| A36 | 同协议第二个走廊族的监督模型 | I405_N 秋，seed 20261013，352 sensors | State .921455 → .930130，纯 attention α=.75；CI [.006482,.009089]，R3 +.010771 | 1212.19 秒 GPU，1180.95 MiB；14 日 standalone 复跑通过 |
| A33–A35 消融 | 所有架构的原 cal checkpoint / blend 均封存 | same assessment；原赢家保留 | attention 比几何图的三组日均 CI 均正；density 比 attention 两组跨零、冬季全负 | 不反选 A35 的 assessment 胜者；所有原赢家指标复现 |
| A37 | 新 attention 模型跨 family 零样本迁移 | I5_N → I10_W；目标标签仅用于评分 | 插值 .928998 / 半数组合 .922352 / 纯 NN .884405；半数组合 CI 全负 | 31.44 秒；失败，不部署共享模型 |
| A38 | 急降速、恢复、拥堵观测、缺失诊断 | 同 A33–A35，按 station/link 五分钟差 | 三组 R3 均改善；少数急降速目标改善，完整 RMSE/计数/日区间记录 | 诊断 score 跨 regime 按 cell pooled，不冒充正式宏平均或真实事故标签 |
| P01 | 全公开 train 的生产准备 | I5_N，45 日 frozen profile 后 228 日 | 4,409,727 labels，全标签 LGB / NN 缓存覆盖完整 | 444.90 秒 CPU；没有 holdout 或正式成绩 |
| P02 | 固定两种生产 NN 假设 | P01 的全部公开 fit 日期，24 epochs / α=.75 | attention 与 attention+density CPU 权重及 standalone 均保存 | 891.96 秒 GPU，844.85 MiB；仅训练记录，没有 holdout |
| K03/K04 | 两种生产假设的完整本地控制候选 | 只改 I5_N State/Physics 输入；其他任务固定 | 各 6,980,503 行，全行 diff 通过，仅 1,131,560 State rows 改变 | 242.97 / 242.22 秒；未上传，正式成绩未知，原最佳不覆盖 |
| C30 | 因果十三步历史预测六步 Queue | matched C27，同事件/标签；NN 使用合法 occupancy/condition | cal 原树 .898170，纯 temporal .456449、纯 graph .775337；组合都选 α=0 | 253.87 秒 GPU，110.75 MiB；拒绝晋级，原树 assessment .897422 保持 |
| A39 | 较大方向的场景诊断 | 同 A36，station/link 五分钟差 | 严重降速诊断 +.115282（275），恢复 +.033326（641） | 带噪观测，不能视为真实事故标签 |
| K01 | 一次 OD-only 正式控制实验 | 同首轮所有 State / Queue 数值，全量 6,980,503 行 diff | **正式 Total 0.76474，下降 0.06466，拒绝晋级**；ref 57140644 | SHA 4976e6bc1fe45207b6ea05dda2031d7885c9b38e1d607efc62c97887ba1e9055 |
| ENV01 | 独立 CUDA 环境 | Python 3.13.1 / RTX 4060 Laptop 8GB | torch 2.8.0+cu128 已安装；实际 CUDA 运算通过 | 当前沙箱临时目录访问失败，改为 E:\\ieee\\.cache\\build_tmp 后构建成功；原 .venv-repro 不修改 |

复现入口：`scripts/register_research_protocol.py`、`audit_physics_topology.py`、`simulate_stress.py`、`od_probability_experiment.py`、`queue_event_experiment.py`。
正式反馈只读本队提交。禁止隐藏标签、组织方内部资产和其他队伍预测。

A33–A36 已完成，四个 standalone 各 14 日重建/预测验证通过。I5_N 两季两 seed、I405_N 单季单 seed 的开发证据均正；A37 跨 family 迁移失败。各架构的匹配 assessment 消融保留 cal 选择，attention 增益稳定，density 独立 State 收益不稳定；不能按已看的 assessment 改选 A35。P01/P02 完成，K03/K04 各 6,980,503 行覆盖与限定任务/方向的全行 diff 通过；正式成绩未知。C30 未超越树模型，拒绝晋级。两个 GPU 训练进程合计曾观察约 3.2 GiB / 8 GiB。原正式 profile 与开发 profile 的数值最大差约 4e-12，仅浮点舍入，不能据 equals=False 推断日期变化。

当前研究环境 35 tests passed（14.32 秒），CPU 30 passed / 3 optional-Torch module skips（8.73 秒）。首轮场景诊断改为按 station/link 差分后，对原模型与原日期独立复跑，所有分组计数和指标仍与历史 I5_N 记录一致；原冻结结果保留。

当前完整模型/配置快照 `model_artifacts_checkpoint_adaptive_v2.json` 保存 **334 项**已完成 artifact SHA。`K03_K04_local_state_controls.json` 保存完整候选 SHA、大小、覆盖及全行 diff；`adaptive_verification_record.json` 记录实际测试与独立复跑的范围。所有文件仅在本地，重要代码与报告通过本地 Git 提交归档。

下一阶段已启动，尚无成绩：A40 为 I405_N 冬季图缓存准备；A41 为 I5_N 第三 NN/LGB fit seed 20261107 的同协议架构对照；A42 为 I405_N 冬季 500k/full 强基线。运行入口、完成标记与续接动作记录于 `research_checkpoint_adaptive_phase.json`。缓存准备拒绝已存在目录，修改 NumPy 数组时显式 copy，开始前保存源码快照。

专家/密度阶段 31 tests passed。expected-IoU 与小规模独立枚举全部标签、全部预测集合一致；独立 Bernoulli 假设不代表交通单元真实独立。density 投影网格严格核对完整 key，并保留缓存预测不变；测试发现 Pandas 可返回只读 NumPy 视图，已显式 copy 并验证与直接预测一致。新实验在开始前保存源代码与依赖锁。

A01 的 CI 是按天配对的诊断，不是非线性 pooled State 或 Kaggle Total 的置信区间。A14 已证实标签预算混杂：全标签 LGB 超过原神经网络组合；A15 的独立增益低于登记门槛。后续必须继续比较强基线和多个 fit seed。

**K01 的迁移失败优先于 D02 模拟增益。** 其他任务值逐行完全相同，按 0.20 权重，ODME 总体分项差约 −0.32330（显示分数舍入误差约 ±0.00005），绝对分项值仍未知。候选可能存在严重 prior distribution 失配，不能仅因模拟更优继续增加其 ensemble 权重。下一轮需扩大生成分布、反向构造 noisy weak prior、检查 nullspace 漂移与真实可用稳定性；不依照这一个正式反馈细调参数。

K01 public-only 诊断：validation 的 Bayes-vs-Ridge L2 差异平均 99.94% 位于观测矩阵零空间；计数误差几乎看不到这些变化。Bayes 总需求约为 Ridge 的 80.74%，相对 prior 的偏移量约为 Ridge 的 1.56 倍。这里只解释模型的行为，不能证明隐藏 path truth 或精确归因正式损失。

现有首轮 January–February test 已经查看；新 rolling assessment 仍是公开 train 开发证据。独立模拟 final seed 7907/8017/9109 尚未生成，需在架构和参数封存后一次性评估。

C09/C10/C11 的去重、按 condition 等权、日期 block bootstrap 增益分别为 .016495 / .024714 / .014972，95% CI 分别为 [−.012655,.060063] / [−.117752,.076207] / [−.017293,.099297]。比较的 before-T 与 at-T 模型都只按各自 selection 选定，assessment 未参与选择。相同 origin 的重复抽样去重，缺失某 condition 的 bootstrap draw 明确排除；I10_W 的事件特别稀少，因此区间证据较弱。

本轮新增模型与配置 116 项 SHA 已记录于 `reports/research/model_artifacts_checkpoint_20261013_a.json`。正在运行的后续模型不在此快照中。CUDA 环境 21 tests passed；测试临时目录使用工作区 `.cache`，系统沙箱 TEMP 不可写。

公告对齐阶段归档：`model_artifacts_checkpoint_20261013_e.json` 保存 235 项已完成模型/配置 SHA；当前 CUDA 环境 25 tests passed，CPU 24 passed / 1 optional-Torch skipped。K02 完整候选与全行 diff 保存在 `artifacts/research/K02_Queue_only_public_timing`，正式成绩仍未知。安全/条件收益界和不上传的决定由 `scripts/record_onset_control_bounds.py` 生成，证据为 `reports/research/K02_public_timing_decision.json`。
# C42–C48：窗口 IoU 损失与公开 TRAIN 事件审计

三种 loss 使用相同九通道历史、30min graph、初始张量、batch 序列及 20×80/batch4 预算；checkpoint 统一为每五 epoch 的最低 cal Brier。seed1=20261013，seed2=20261029；事件 sampler 仍固定 seed1，不把重新训练 seed 误称独立事件。所有 alpha/threshold 选择在 assessment 文件加载前封存。

| 实验 | cal 主选择 | 去重 assessment 观测 proxy IoU | 结论 |
|---|---|---:|---|
| C42 秋 seed2 | tree / α0 | .872422 | 三 loss 均退回 tree；相对旧 seed1 混合 −.006606，CI 跨零 |
| C43 冬 seed2 | Lovász / α.25 | .878723 | 同 seed BCE .881507；差 −.002783，CI 跨零，保留原选择 |
| C47 秋 seed1 | BCE / α.25 | .879028 | soft .842850，Lovász fallback .872422；原 BCE 权重完全复现 |
| C48 冬 seed1 | BCE / α.25 | .879943 | soft .878782，Lovász α.5 .857370；原 BCE 权重完全复现 |
| C44 公开 TRAIN 80 窗口 | 不选模型 | 非成绩 | 1 个 onset 无观测正例；history condition 全部一致 |
| C45 质量投影 | 不选模型 | 非成绩 | 有限速度正例 7100→eligible6457；异常例在质量过滤前仍无正例 |
| C46 等 alpha 次要对照 | 原 cal 设置不改 | 16 个对照 | 冬 seed1 Lovász α.25 +.003753，未由 cal 选中且 seed2 不复现；不反选 |
| 独立部署 | 四组全部 cal windows | 非性能实验 | 原始概率/完整预测各 660 次精确复现；两个原 BCE 参数控制精确 |

四组 GPU wall time 199.11/216.23/203.29/216.44 秒；约 110.33 MiB allocated 峰值。末组与部署复核并行。首次部署 job7692 退出1：pandas 拒绝 numpy.str_，未产生成绩；原失败 zip 和记录保留，str 转换后新目录复跑。部署两份成功记录 73.53/29.37 秒。71 research passed，CPU59 passed/4 Torch skips。损失数学测试枚举 64 个离散真值/误差顶点，并检验 onset/unavailable 零梯度与窗口等权。正式 Total 仍 .82940，无模型晋级、无新完整候选替换、无平台动作。
# C49–C54：两种子集成、选择敏感性与已知真值标签契约

| 实验 | 模型/数据 | 结果与资源 |
|---|---|---|
| C49 v2 秋 | 两 seed mean-BCE/calα.25 | .880013 vs旧.879028，+.000985，CI[−.001310,.003718]；全体cal argmax仍旧BCE |
| C49 v2 冬 | 两 seed mean-BCE/calα.5 | .858806 vs旧.879943，−.021137，CI[−.040511,−.004430]；cal仅+.000160 |
| C49 primary | 21候选/季，1 paired-gap SE 简化 | 两季都tree，−.006606/−.011703；仅in-sample sensitivity，不是独立CI保证 |
| C50 | 重构后的全部旧概率/决策接口 | 660次window/model输出精确，94.75秒GPU；旧tensor控制一致 |
| C51/C52 | 自主CTM，3seeds×6cases×4noise | 72场景，18组latent/quality匹配；503重叠origins/噪声，观测subset495/495/492/490；9.29秒CPU |
| C53 | 10导出角色/全部110cal windows | score/binary各550次精确，134.66秒GPU；重复角色不算独立样本 |
| C54 | 6旧graph NPZ | 公共FD/拓扑/节点顺序全部数组精确重建，补入hash清单，无新模型增益 |

C49首次job20156在cal seal之后、assessment之前因旧清单缺graph NPZ失败；zip/seal/失败JSON保留。v2先核对SHA保留模型中的geometry，再登记图字节，258.92秒CPU完成；C54再独立公开重建六份。C51首次CPU import旧GPU脚本CASES失败，尚未注册/建输出；旧源快照与记录保留，共用场景配置拆入无Torch模块后复跑。共享六case值不变。78 research passed；CPU66 passed/4 Torch skips。没有新增训练、fit seed、正式成绩、晋级或新完整候选；原/K06不变。预测只接收≤T历史，所有latent/未来label/qc只在独立模拟selector或scorer中使用。
# C55–C61：五阶段、新方向与采样条件平衡

四个balanced-v2研究统一预注册：I5_S/I10_W，fit_end10-15/11-30；之后checkpoint15/decision30/selection15/assessment30日。三optimizer seeds共享各组fit事件；Checkpoint最低Brier每5epoch，20×80/batch4/AdamW.001。两tree为600/63 plain与合法occupancy+provided-condition；图采用30min public kernels，三单seed/mean均固定α.25。decision确定强参考及challenger；后段仅gain≥.01且95CI下界正时替换，不重新调参。

| 研究 | checkpoint onset | decision/最终选择 | assessment proxy IoU | graph_mean次要gain/CI |
|---|---:|---|---:|---|
| C55 I5_S秋 balanced-v2 | 10 | rich tree | .781151 | +.007361 [−.002108,.019077] |
| C56 I5_S冬 balanced-v2 | 9 | plain tree | .797255 | −.007431 [−.045926,.033042] |
| C57 I10_W秋 balanced-v2 | 3 | rich tree | .557659 | +.047203 [−.012979,.100827] |
| C58 I10_W冬 balanced-v2 | 3 | rich tree | .814942 | +.000306 [−.003037,.003516] |

四个decision winner都是incumbent，selection gate零，不晋级；次要graph均CI跨零。C55原版秋 .840835在不同事件人口上；原版冬有15 onset候选却选0、worker退出1，未读decision/assessment，后两子任务未启动。所有失败源、两fit-only tree及status保留。C59同checkpoint/seed条件平衡选9 onset/13 ongoing；默认代码8组旧cache的metadata与labels完全复现。新campaign在独立注册/目录重跑，未修改旧报告。

C60同样balanced assessment冻结模型对照为原.781706 vs新.781151，−.000555/CI[−.025070,.018913]，未见策略收益。25个完成研究的训练模型（15NN/10tree），失败另2tree；四v2 wall316.33/302.96/141.75/154.03秒，allocated81.85/83.08MiB。独立decision输入/score/binary 608个role/window精确，五phase/date/seal一致；重复角色/seed采样不是独立事件。83研究/71CPU通过，CPU4 optional-Torch skips。Formal仍.82940，K06冻结，无平台操作。
