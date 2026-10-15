# 当前实现与评分接口审计

C87新增纯NumPy/Pandas dense causal输入及可选Torch全六步warm loss。raw只取历史12槽，T仅masked；未来teacher/quality只进监督。C88独立124fit日/33480原点/742事件teacher/372poison核验，补足此前treefeature证明不等于旧神经x证明的范围。

C89初次portable核验漏released_origin_rows导入，保留源码ZIP/partial；原traceback不可得未伪造。C89b仅修import后新目录核验12网络初态/批次/曝光、48权重与Adam状态、792checkpoint+1896eval forward、1128网格/gates/seals，最大差0；316新eval raw窗口、336便携/168乱序/12future拒绝。旧强树feature证明明确复用，没有重新优化12网络。

C90的7联合/4成员统计仅事后描述，同日历同步抽样；成员用冻结均值阈值，无新选择。速度forecast不是State，低Brier不保证Queue，当前没有flow输出或物理联合优化。新增4测试，全151research/129CPU+8Torch skips通过，旧6877artifact/5完整CSV/锁不变，累计7247。[方法和限制](docs/DENSE_DYNAMICS_PRETRAINING_RESEARCH.md)。

C83的local_search只用原假设场景，8步单翻转与独立模拟bank的条件数值筛查；实际future teacher/mask只进decision evaluator。四组held-simulation bank由fit窗口生成并seal后才读decision，源JSON旧结果不称盲测。0提议/0shortlist触发预登记停止，worker未读新selection/assessment数组；不声称真实非零提议验证了数值接受gate。

C84独立候选集合矩阵复核10小law/320全局集合、52onset/5284邻居、MC区间/停止规则与导出；8onset Booster+4参照probability，16新history/48角色+24乱序+12future拒绝通过。新3测试及当前全147research/126CPU通过、7Torch skips。旧raw/概率/场景bank证明明确复用，无新完整submission或正式成绩。

C85仅fit raw/masked日，原点采样只看过去12步quality，future进人口统计；重复target与unique target分计，末步onset准则明确限于观测代理。C86用独立quality谓词/groupby/移位数组和覆盖时间集合重算680日/stride行、12240整数及汇总。checkpoint再逐raw原点集合证明selected与dense5 onset完全相同（72/122/37/58），不作latent覆盖保证。旧6835artifact重hash不变、累计6877，5完整CSV及锁保持固定。[证据](docs/BOUNDED_SEARCH_AND_DENSE_FORECAST.md)。

C79场景库只由C71 fit onset的eligible观测标签/六步mask构造，missing标签不进rate分母；记录289窗口引用、2seed×2048模板与同base-jitter独立/二值关联变换。没有使用拟合内分类器残差、实际评估future mask或重新拟合概率校准。ScenarioOnsetQueueModel只接收合法history/target，ongoing不变，prefix优化不声称全局最优；前五步零标签evidence与末步mask分开。

C80失败是核验器float32 mean取代生产整数ratio，最大rate差2.62e−8；C80b修正独立整数计算并消除局部变量遮蔽，bit-exact标准与模型/选择不变，失败源码zip保留。4场景库逐位一致，3076条ensemble/member曲线用指示矩阵直接交并集复核差0。8decision onset Booster+4参照probability调用，24概率文件沿用C75；109新便携历史支持981角色+504乱序+36future拒绝。旧2196raw及C75概率审核显式复用，本次新全量raw重建0。

C81逐truth/mask集合枚举重核9个已知非空law最大差9.99e−16，13事后同步日历区间逐次复算差6.25e−17，无多重校正/重新选择。checkpoint另存未来评分mask仅用于误差定位的证明：full关联4个改变单元均不可评分。新增3测试、当前全144research/123CPU通过、7Torch skips；manifest旧6745artifact不变、累计6835，5完整CSV/锁不变。详见[联合场景审计](docs/JOINT_SCENARIO_ONSET_RESEARCH.md)。

C74新增自主Bernoulli suffix卷积IoU内核，k0为空/空=1，k>0完整域期望；link-ID打破平分，最小k容差1e−12，future eligibility不进API。非空arm只禁k0，不声称过滤后仍非空。新概率模型复用冻结._onset因果入口、ongoing和保存时Booster cache清除。单调Platt仅用独立15日checkpoint观测标签，两arm先fit/seal，固定window-equal权重/identity正则与边界；未使用分类器拟合日期上的同窗概率来校准，也未按assessment调参。

C75明确复用旧2196raw审核并重hash输入/未变构造；新增62raw校准窗口/124特征臂重建，8个状态另用SLSQP与KKT验证（参数差2.68e−7，投影梯度7.38e−10）。582条曲线用独立Gauss–Legendre积分复核suffix DP，最大差1.90e−14。全部新网格/门控/评分、100冻结Booster、872便携角色+448乱序+32future拒绝通过；含策略别名/重复origin，不当独立事件数，也未重训树。

C76只使用自主5节点已知联合分布及已知mask法则，不读取比赛数据；联合分布最优诊断禁止部署，按分布期望枚举而非实际未来标签选点。C77另用标量集合运算复核7场景，差6.66e−16；11个事后同步日历区间逐次求和误差6.59e−17，未作多重校正、不参与新选择。4新测试及当前全141research/120CPU通过，7可选Torch skips。新manifest旧6649artifact不变、累计6745，5完整CSV与依赖锁不变。详见[概率集合审计](docs/PROBABILISTIC_ONSET_RESEARCH.md)。

C71的四个滚动case在任何新assessment前统一登记；所有plain/rich参照与onset模型先fit再读decision，原checkpoint15日不读值。decision只选incumbent或threshold，fixed-k为次要控制；selection封存后才读assessment。全部24树实际600，每case两baseline/四onset；没有借用较晚训练的旧强参照。

C72初次因DatetimeIndex.asi8的微秒被误当纳秒而失败，源码zip/注册/错误保留；C72b直接比较Timedelta，原30/360分钟标准不变。独立重建32缓存全部2196选中raw/masked-T窗口、两特征臂4392次、teacher/eligibility/事件历史与未来筛选约束；监督行/权重/模型schema及24模型容量均核对。1504次baseline校准分数、阈值/k网格、门控/CI复算；88次直接Booster、436便携角色+224乱序、16future拒绝通过。不是独立重训或重建完整候选宇宙/RNG，旧神经缓存x不进模型且未独立检查。

C73用同日跨走廊共享日期抽样、两个不重叠日历分别抽样，4个事后对照各4000有效复算，另一逐次求和实现最大差8.07e−17。新增3项科学测试通过，原134research/113CPU及7Torch skips是继承记录，未冒充本轮运行。生产src未改；正式最佳与完整submission保持原版本。详见[滚动核验记录](docs/ROLLING_ONSET_DECODER_RESEARCH.md)。

C68新增`OnsetSetDecoderModel`，复用冻结分类器/ongoing，仅改变onset T+30。24维摘要只读合法≤T历史和masked T，标准化及log1p Ridge仅fit；8 Ridge、4常数控制。future eligibility只进fit计数teacher或evaluator，不参与预测域/阈值/排名；预测接口不接受future mask。计数校正有未验证MCAR假设，不能当latent真值。

C69不调用新摘要helper求参考值：665次摘要独立Pandas重建差0，12状态用增广lstsq而非原normal-equation求解，参数最大差3.997e−15；grid、fit-only标准化、计数取整、gate/CI与ongoing精确。56冻结Booster矩阵调用；112新raw历史窗口支撑896便携角色调用（含别名）、448额外乱序、16future poison拒绝。C66旧2345raw输入审核通过SHA/未变源码显式复用，本次不声称重新构造全部raw。

C70独立核对16个配对次要比较，不用于重新选择。C69完成后只删除一行未执行`if False`死代码，实际执行源码zip保留，数值和断言未变。新增3测试通过，全134research/113CPU通过、7可选Torch模块skip；新manifest重hash6374旧artifact不变、累计6453，5完整CSV及锁文件不变。所有新报告保持严格有限JSON和Git原始字节；无新完整submission/正式分数。[详细证据](docs/ONSET_SET_DECODER_RESEARCH.md)。

C65的OnsetTopKQueueModel只改变onset最后T+30，ongoing直接调用冻结FullDomainQueueModel。ranker score不称概率；同分按link ID稳定选择，完整六步各link集合必须一致，禁止把future-eligible子集作为排序域。类保存CPU model strings、不序列化live Booster cache；输入时序复用已有严格queue_features接口。

C66重建32源缓存的2345窗口raw历史/masked T、tree features及future evaluator teacher/eligibility，核对同fit onset行/分组/unit weights；独立解码/网格/评分/去重day-block gate一致。128Booster矩阵调用包括16fit；便携接口测试每evaluation缓存各condition首末共560次、另140乱序，8未来行poison全部拒绝。不是全部窗口都调用所有便携角色，未独立重训16模型。

C67使用公开assessment标签逐窗选最优k，仅保存诊断、不导出oracle预测/模型、不改变选择，`deployment_allowed=false`。旧source报告含已看结果，新的运行仍只按decision/selection数组选择，不称盲测。131research/110CPU通过，7可选Torch skips；现有State/OD/完整submission未改。

C62新增`latent_queue`的严格13slot因果58特征、训练teacher及完整域决策/scorer。模型仅收v/q/valid/static几何/公开condition/lead，sigma/latent/future quality不进features。未来eligible仅用于loss/scoring，onset早五步0，force在完整允许域选位置；新单测覆盖Gaussian解析值、缺测/置换及score mask分离。

C63逐node独立构造特征、selector及erfc teacher，重新生成268scene/7090窗口和272907训练行/权重，feature最大差1.49e−8、teacher差0。六portable booster在cal/同类/stress共18预测差0，300cal配置及57680角色窗口分数差0。C64独立分组核验8个四seed区间。未独立重训树，不宣称真实latent/organizer契约验证。

事件由自主latent选择，不按未来观测正例筛选，包含28/12个零正例heldout窗口。原注册“upper reference”不是性能上界，latent标签控制实测接近hard；不能代替比赛未知真值。全128research/107CPU通过，7可选Torch skips；测试与验证并行计时。

A138新增研究runner，生产src不变。D12_I405_S使用原始公开fit targets，四个新fit分成两个seed的普通/位置配对，共同初始state、独立存储patch计划、实际逐格曝光完全匹配。强性能参照为既有全标签LGB/A47 cal最高者，普通新GRU亦作为候选；全部alpha/checkpoint/reference/primary整体seal后才读旧后留。

A140首轮核验的alpha-grid触发1e−12断言。独立56纯cal调用证明核验器float64乘法改变float32中间舍入；恢复native float32后20网格评分和选中预测差均0。未改训练/模型/选择/预测/容差，失败源码/注册/诊断保留。A140b重建92fit raw输入/teacher/mask、公开图/距离、独立RNG计划/四份曝光，308predict最大lane-unit差2.27e−13，另56纯cal调用及280保存指标、20所选checkpoint网格评分通过。未重放训练优化器或未选择的epoch。

A139实际154导出接口调用；A141全部14日teacher/diagnostics/154指标独立复算，另22predict差0。11导出含参照、组合和别名，只有四个新拟合模型。历史125research/104CPU、7Torch skips不重复计新测试，脚本compileall通过；二月R3及恢复/长缺失速度退步不隐藏。

A134以`--winter`复用同一位置结构，动态reference=`nn_primary`、固定候选=`geo_seed_mean`/`geo_speed_reference_flow`；没有树流量或额外通道网格。A136独立重建258节点几何、两共同初始化与完整实际格点曝光45,754,069/45,844,015次，选中pt/alpha/gamma/导出一致；224次predict最大lane-unit差4.55e−13，196指标score差2.22e−16。cal重新评分仍拒绝两候选，primary为原NN。

A135全部8角色×14日直接调用112次；A137重建14日raw teacher/lanes/diagnostic masks和112保存指标，另首末16predict差0，score差1.11e−16。纯新NN二月分数较高不改变cal-only选择。生产src未变，前阶段125research/104CPU+7Torch skips为历史内核证据；本轮脚本compileall及上述实测复核完成。跨方向汇总仅按共同14日同步bootstrap的等方向日均诊断，不算正式权重或28独立日期。

A130的相对位置只读首fit日masked文件link_id/milepost列，以正间距中位数归一化、clip20。两attention各新增一个有符号系数初值0，保持全部非距离state tensor的初始化字节相同；非零输出head的函数等价及新系数可获梯度有测试。RelativePositionStateModel固定link顺序、保存距离buffers，复用父类同FD裁剪/总flow单位/窗口逻辑，保存时移除网络缓存。

A132独立从公开metadata计算距离，逐seed共同初始化/全部cell曝光精确匹配A97，选中pt/gamma/alpha与导出一致；252次直接predict最大差2.27e−13，224指标差1.11e−16，固定新速度/原flow组合及强参照cal gate通过。A133全14日teacher/diagnostics、126指标和另18predict通过。A131比较reference明确为strong_channel，旧study无该配置仍默认nn_primary；没有将超过弱NN当超过当前强基线。新三项测试及全套125research/104CPU、7Torch skips通过。

A122/A123仅使用原fit日期raw/cache/已保存实际曝光数组。诊断模块不进入推断features；未来两步/邻居标签只作fit观察统计，不作Queue可见历史。原输入base通道6/7的SSE明确区别NN/官方State。独立shift/window核验245日3185数组、3430统计记录，包含raw eligibility、日界不循环及条件分母；浮点SSE最大绝对差3.73e−9。

新speed_jump_pair_loss只在core内两个连续target均为True且float32 abs(label delta)>=.8时监督速度差，flow输出梯度为0（共享网络参数仍可改变flow）；手算endpoint梯度与无pair退回原目标均通过。A126现已独立重建两seed全部cell/pair计数、初始化/patch序列，核对选中pt与导出权重/alpha、196predict及168指标；最大预测差4.55e−13、score差2.22e−16。A127新14日98指标及14次predict另行通过。原122research/104CPU测试为前阶段记录，此阶段生产src未改；kernel通过不等于模型成功，实际两个日历均退步。

A128只读首个public TRAIN的link_id/milepost列及公开网络CSV；A129用直接CSV ID/union-find独立核验10图。图不连通不证明reader缺陷：AdaptiveSpatialBlock的attn_mask是有限bias，非邻居仍可通信。固定CPU探针3个孤立节点的9个mask值全有限，跨节点梯度L1=.6854989528656006；源码SHA和复现入口保留。不把这个kernel或静态图计数当模型收益/物理连通证据。

A116仅新增两个各含400棵树的流量模型，原速度booster和NN显式复用。原153fit日及4point视图逐块写入磁盘映射；float64 scalar teacher/base/weight、int32 row keys与全部60维float32特征保存。训练不调用NN推断，cal/旧后留NN预测来自A107已核验的冻结输出。A118在拟合并行时独立实现随机坐标、history/guard及所有测量删除，765组完整重建、8次完整60维poison通过；raw流量teacher误差1.19e−7归一化单位、weight差0。

A117只对四个基本角色和固定transfer逐日直接推断，共70次；其余ensemble角色由固定数组组合。A119全部14日teacher/diagnostics和126保存指标独立重算，另18次直接导出调用最大差4.55e−13。不把126指标写成126次新推断。A121优化器只用14日cal数组，源报告JSON含已查看后留摘要，cal增益.00001953不作独立验证；未导出新模型。新阶段生产src未变，原单元测试记录不重复计作新证据。

A120仅hash绑定并复用A118，字段明确本次重建0组、既有验证765组。全部训练文件及raw源重新hash不变，28日×9导出252次predict最大差4.55e−13，238指标最大score差2.22e−16。原速度booster与基线逐字相同、每个新流量booster400树、cal gate和selection seal/read order均核验。source注册/数据manifest/审计报告SHA链完整，模型没有按新forward改变。

ChannelStateEnsemble使用固定model×2channel非负权重，不读取teacher、regime或误差决定专家；检查profile/features一致与键顺序，输出保持总流量单位。递归清除临时网络缓存，保存后恢复当前进程缓存。A112原28日、A113新4role×14日实际部署与固定通道公式差0。

A115b独立raw TRAIN重建28日teacher/lanes、112指标与全部50cal候选/eligibility，核对cal-only winner、封存顺序及序列化权重，最大score差1.11e−16；没有独立重训。A114重建新14日teacher/诊断mask、56保存指标，另8次predict差0。A115最初缺Torch导致嵌套pickle读入失败，注册/源码/错误记录保留，未修改数值断言。全套118 research/102 CPU通过，5可选Torch模块跳过；初次temp setup错误及工作区basetemp修复均记录。

新native reference helper支持A73/A75冬季原控制，检查280role-day部署核验源、模型SHA、三个构造源码与原start archive一致，并重现shared初始化hash。全初始化由同seed/未变构造确定；不把冻结控制写成新训练。旧A97无新增字段时仍解析11角色，已实际smoke验证；新source明确记录reference replay路径/SHA与forward日期。

A103对same-view训练表使用完整公开fit teacher与原profile，newBoundedState只作总流量/速度的公开FD上限裁剪。A110重建全部920组feature/base/teacher/row-key/weight，168纯树role-day及4选中策略直接调用，最大lane-unit差2.27e−13；392保存指标复算。A111全14日teacher/变化mask/缺失RLE独立计算，210指标误差≤1.11e−16，另18次predict差0。新forward verifier兼容纯保存模型与派生hybrid，逐项核对固定混合，不把210指标写成210次新推断。

冬季A1051224view和16真实loss-target poison检查已完成，teacher精度在既有1e−6归一化容差内。A104四模型/selection seal现已完成，assessment/forward在封存后读取，结果见RESULTS。前阶段checkpoint仅冻结当时已完成角色和start archive，原manifest不覆盖；本阶段完成所有核验后另建后继manifest。冻结旧控制显式复用，不算新训练。

A107独立RNG/core/view/实际标签曝光数组4份精确重建，原cal/assessment全部28日×12角色336次直接predict最大差4.55e−13 lane units。A108重建新14日teacher/lanes/indices和全部诊断mask，168保存指标最大差2.22e−16，另24次直接predict差0。两项均不是独立模型重训，所有完整候选及旧manifest固定。

A96只用原fit日期公开raw观测生成4个seed的新Bernoulli link/time mask，rate取原当日R1/R2/R3，不复制隐藏/official stable-hash答案。Point和Queue-shaped共享点坐标；历史可见/未来90min空缺只按公开或metadata模拟origin，后者不读取condition/future coverage/拥堵标签。两区间都不作Task1 loss。拟合期公开Queue origin实际为0，模拟并非官方selector。

所有teacher目标的speed/flow/occupancy/derived density/pct-observed先移除、flags更新，再从头重建StateFeatures。Profile固定于原45日。A99不调用remask helper，独立重建736输入/loss/hidden arrays；16个实际loss-target poison输入exact，独立float64 raw求和与float32训练label差2.38e−7归一化单位，固定容差1e−6。Builder补齐相邻日history/guard窗口后，逐日确认92fit原origin列表均不变；原cache源快照/数组未改。

A97初始化及regime/day/core中心匹配旧GRU，原控制权重显式复用A87/A90，不计本轮新训练。新loss mask会监督模板之外的公开fit标签，实际曝光/新增位置单独记录，不能称等标签人口。全4模型/checkpoint/alpha/双seed均值/primary仅按cal选择并整体seal后才读原assessment；一月14日预先固定，未按新结果重选。116 research/100 CPU tests通过，5可选Torch跳过。

A100独立RNG/core/view和标签曝光重建与实际数组精确一致；对全部原cal/assessment的12role共336次predict重放，输入/teacher/目标顺序核验，最大lane-unit差2.27e−13。A101用逐列RLE和raw前5min速度另算全部诊断mask，14日168保存指标独立重算最大差2.22e−16；另24次直接predict差0。没有将168写成168次新推断，也没有将复用control写成新训练。

A92首次线性反推等alpha失败来自保存neural float32乘法的量化丢失；四原selected-alpha输出和baseline重放exact。A92b直接重推全部728model/case，未放宽旧1e−8预测复现容差，原失败注册/源码保留；其已查看日历明确是次要诊断。

`GapGatedState`只读取features.raw中speed/flow的有限性、五分钟timestamp网格及合法target lookup；不读teacher、regime、score-mask或人工失效metadata。36步门限与新日历/seed/8条件在A93读取新标签前冻结。两状态通道共同切换、空闲专家跳过，保存时递归清除临时网络缓存。只用于离线State，同日已发布未来观测不允许传给Queue。

`outage_shift.py`仅从timestamp/link/milepost和RNG选区域，8条件对测量poison均不变；完整UTC日/连续跨午夜日与同一sensor grid有校验。A94从修改后的frame完整重建特征，cross-midnight分别按日推断、合并SSE/count后评分，避免平均非线性日分数。A95另以eligible-groupby可见性和逐列RLE重建393份gate数组，并重建全部15普通/378segment teacher，核对60普通/1512segment/1344case分数；44次直接保存模型predict差0。114 research/98 CPU测试通过，5可选Torch跳过。

普通15日gate目标为0，全走廊4h合成分数明显下降，该门控不晋级。官方MASK_SPEC的20/30/50%稳定点mask与Queue90min空缺已重新核查，SHA记录于研究文档；不把严重额外断测成绩宣传为正式Task1增益。

A86/A87只改变fit的可见输入，原y/mask不增补。`rebuild_after_outage`不接受teacher；移除所有测量/质量flag后重新构建完整StateFeatures。A89用独立Pandas移除实现重建368份输入与region逐值一致；首末日期8个真实视图先poison移除测量再删除也一致。旧strict12中不能仅清零measurement而保留旧base，新增测试明确检查base实际变化。

训练用独立patch/augmentation RNG流，概率或view改变不影响原regime/day/core。两seed×两臂记录完整plan及实际逐格点曝光，各517171目标都被监督；两个control checkpoint与原GRU所有张量逐值一致。Cal选择完成、整体seal之后才读原assessment，11/27–12/10及新mask seeds在fit前登记。旧文档错误写seed29 GRU alpha.5，本轮依据原JSON更正为.75；原TCN .5、原模型与实测不变。103 research/87 CPU tests通过，5可选Torch跳过。

A90独立重建实际RNG与core监督数组，核对control alpha及旧权重，并对全28日×10roles作280次predict重放，最大lane-unit误差2.27e−13。A91核验未来日历/seed/尺寸与fit前协议相同，重建全部teacher/indices；126普通指标及1512合成分数独立重算最大差1.11e−16，另54次真实predict调用差0。1512不是独立推断次数；嵌套case、共享日历及多重比较限制保留。

A72b的GRU位于原graph/attention编码后，原参数同seed逐项匹配；增加容量21313，不冒充等参数因果证明。固定窗口只接收strict12输入、图与FD；无label/score-mask参数。batch独立性、core先选边界全覆盖、每格点一次输出、zero-head/zero-gain与梯度均有测试。原A72 center-only源快照/partial权重保留，主动停止时未产生selection seal或读assessment。

两fold八个完成模型仅在fit监督；所有checkpoint/alpha/seed均值/primary由cal选择并写整体seal后才读assessment。每个模型保存真实按日逐格点曝光数组，四对架构同seed数组精确一致，两组517171/3042465目标均实际覆盖；原neural标签按link×time均值，不把它和full-LGB的station行数混为完全相同更新预算。Cal/ass输入从masked帧独立重建，raw teacher只用于loss/evaluator/transition诊断。

A74因JSON str路径类型首次导出失败，原源快照保留；新A74b和A75各导出10种冻结策略、28日280次model/day预测重放，合计560次。输入、公开图/FD数组exact，最大lane-unit预测差4.55e−13，绝对容差1e−8；均值单位换算可有roundoff。原primary不因assessment单seed更高而改选。A77c两方向二月各28日固定策略完成，日期与各自profile/fit/cal/assessment不重叠；A84重建56日teacher、560份保存指标，并另作8次固定日期直接推断，最大score差1.11e−16、预测差0。不是560次新推断或未查看过的final测试。

A81额外失效geometry仅依赖公开timestamp/link/milepost及RNG；每次从modified masked frame重新构建全部features，不复用含移除测量的base。speed/flow/occupancy/derived density/pct-observed均移除，flags按原bool/int dtype一致写入。首次dtype错误在首个stress case前退出，旧注册/源码保留，新A81b目录复跑。168嵌套case共享42个中心，672分数不当独立样本；A82重建全部mask/teacher、分数差0，另16次stress直接推断及全部control。六小时退步保留，没有按其结果改选模型。101 research/85 CPU测试通过，5可选Torch模块跳过；所有正式完整提交保持冻结。

D21/D24新增两阶段OD模型：log-demand中心与KL路径修正共享同一公开计数，明确为顺序点估计，未声称joint posterior。alpha0精确调用固定entropy参考并跳过group；prior-center floor不是flow下界，独立primal finite-difference optimizer及低计数反例通过。可选floor参数默认2；D21运行结束后才加入，旧源快照与所有原流量预测保留。

D24网格在D21 assessment结果查看前按development失败条件独立登记；worker只读D21 development_progress/seal和development NPZ，不读其assessment。1350原预测明确标记reuse，记录original_method_seconds，不能当新训练/测速。新两个floor20配置仍使用同一270开发问题，后续933xxx数据只在新seal之后生成。两个gate均失败，primary固定entropy，次要hypothesis或六seed汇总不覆盖选择。

D22/D25从公开train算子与自主生成器重建每份输入、核对NPZ/预测SHA，用独立评分公式重算；另外直接调用无缓存完整model API，覆盖两个方向的旧/新法律和floor20开发候选。D23/D26使用公开A的标准SVD rowspace，不从nullity宣称具体路径识别；逐预测检查平方L2正交和A*null≈0。组件加权分解不是L1几何分解，六seed汇总是次要诊断。两轮1440case记录中270开发重复，实际1170唯一问题，不扩大独立样本数。最终93 research/81CPU测试通过，4可选Torch跳过；未访问reference/隐藏标签，无平台操作或正式新成绩。

新增 `log_demand_od.py` 仅接受公开incidence/counts/weakprior和origin/destination元数据，输出非负流量与局部数值诊断；不读reference或组织方隐藏path truth。四配置development选择封存后才首次生成三个research seed。D17修复后仍保留D14原选择，recomputed_dev_choice只是回归字段；其assessment明确属于已查看seed的数值回放，不是final。270个唯一评估问题与deployment alias分开，失败不删除。D19验证全覆盖、7条失败恢复及1080参考预测/分数逐值不变。

Polish解析Hessian包括count Jacobian项、指数模型二阶项和Gaussian prior；使用Cholesky阻尼/步长回溯，在浮点预算内允许gradient减半，最终projected-gradient仍2e−4。两起点至少一份证书有效才返回，证书只说明局部stationarity。独立trust-region使用3-point数值Jacobian；选中6/6通过，全grid16/18，两不一致病例及NPZ保留。固定prior乘性模型不能表示真零、没有path-level residual或校准posterior，不声称global/identifiable。五项kernel测试及全部88 research/76 CPU测试通过，4个Torch模块在CPU环境跳过。

K07构建入口检查formal baseline/component SHA、完整study与原selection seal、选中参数独立oracle，再对公开split的OD模板逐path/zone核对。State/Queue使用原组件，真实Physics输入因此保持不变。完整主key合并之后另做全行OD-only diff；候选不上传、不替代已测正式最佳。所有模拟/公共count诊断的formal_ODME和formal_Total字段均为空。

C38完整保存tree与三个graph的全模板概率，future labels/score mask独立存为evaluator数据。C39推断函数full_queue_decisions不接受score mask；只有FullForecastScorer校准utility使用label/mask。两套时间规则先在cal封存，之后才读assessment；不重训、没有新fit seed。fast scorer与逐窗scorer含无eligible evidence=0的手算一致。before模式nonempty只能在公开允许的onset T+30候选中选择，避免旧清零顺序删掉强制点；实际selected force=false，所以两季结果相同。

FullDomainQueueModel只接受history、完整template、public net/topology、origin与condition；UTC标准化、拒绝>T测量及缺失template。导出CPU weights，保存前清除GPU cache；实际220次cal-window由Parquet≤T谓词和公开T行重建，binary predictions精确复现。validation/private推断各15480行，完整master-key diff证明K06只改I5_N415个Queue值，State/OD/其他字段逐行不变。67 research /59 CPU tests、3可选Torch跳过；新测试含time-before-force反例、scorer一致性、完整预测域及因果/完整模板拒绝。

C40oracle使用模拟future mask仅作非法决策域的诊断，没有进入任何导出model/候选；其未来真值来自自主模拟，不是组织方latent标签。81条policy/case共享27个事件case，不能当405000独立事件。不同采样总体绝对分数不当算法gain，实际组织方额外选择规则仍未知。

A66/A68四个strict12/vq组的初始化、采样与预算完全相同。新增raw TRAIN y_full及public Mode A pair masks仅进入loss/evaluator；cal/assessment推断从masked帧重建，导出的AdaptiveStateModel只存已有baseline、CPU weights与public topology，不含full labels。两组State-only/legacy参数分别与A52/A34逐张量一致；不算新fit seed。public coverage metric用完整有效对分母与R1/R2/R3等权，单独字段避免与legacy proxy混淆。手算单元测试核验12/42及6/18分母、regime mean非pooled比率。完整分母等价batch强度重缩放，不能宣称新增物理信息。

C35发现sample_events在构造ProxyData时按未来available筛行；pointwise threshold本身不依赖集合，expected prefix与force-nonempty却会利用这个未知未来域。C35/C37固定旧策略重放分别48/32组旧指标精确一致；新的预测函数不接受score mask，完整预测完成后才调用score_full_forecast。未来全不可评分window按官方规则为0，不返回empty-empty满分。反例p=[.9,.4]、future mask=[False,True]说明旧域可将合法full预测的IoU0变成1。没有隐藏label读取；这是公共train代理评测接口的泄漏，不改变已正式提交的全template预测文件。

旧Queue校准runner含future-eligible集合规则，仅保留历史复现；新晋级必须使用完整forecast域并重新校准。公开latent nonempty不保证score-eligible nonempty；old force之后onset清零次序也可能清掉刚强制的早期cell，下一轮单独检验。当前审计保留原cal choices，没有按更正后assessment反选。63 research /55 CPU tests、3 optional-Torch skips；无正式Physics/Queue新成绩。

A58–A65新增纯NumPy/SciPy二次残差同化，冻结State先验，不读raw标签当measurement。严格finite观测优先，extra观测需已知pct；未知质量和缺失值权重0。正prior precision令全grid Hessian SPD；手工dense normal equations及实际24-slot SuperLU reference核验。3836次实验channel CG都满足真实相对线性残差<=约1e−8；这不构成物理动力学正确性证据。

CF按照固定public panel/date/grid/seed分组，仅临时遮四个measurement字段，全部visible cells各遮一次；原frame不改，原target prior不改。测试直接读取假prior的输入帧验证自身四字段不可见。仍可用State公开前后观测，禁止将结果作Queue未来输入。三个paired案例28日prior/strict/quality/edges逐值相同。旧A58/A59 pickle缺少anchor属性时default0的独立cal复跑通过；保存嵌套prior前清除GPU网络cache。

A58第一版因pandas datetime64[us]整数单位误判5分钟间隔，在cal前失败；原源snapshot/注册保留，之后显式as_unit('ns')，测试还拒绝真正10分钟间隔。A65另加full raw TRAIN标签与公共Mode A pair validity，全部parent arrays及scored labels逐值保留；raw标签仅在loss/评估使用，不是推断输入。I210_W的120日没有非target eligible input blanks，因此该fold无新buffer锚点信息；旧loss多惩罚177808个fit pairs且分母不同。新loss kernel手算检验12/42全分母、coverage后的1/3、gradient[-.25,1/3]及非目标零梯度；未训练。测试编辑边界错误的snapshot/失败保留，修复后59 research /51 CPU tests与3可选Torch模块跳过。无正式boundary_flux访问、无正式Physics分数。

A51–A57 新质量输入严格来自 supplied masked frame，六通道为 any-quality v/q、两 availability、pct_observed 与其 availability；首12通道、标签和 target lookup 逐值核验。Torch 为质量基础函数的可选依赖。新增 input 权重零初始化，旧全部共享参数不变；A52 strict12 的最终张量与 A49 精确复现。旧 AdaptiveStateModel pickle 缺少新属性时默认 strict12/vq，14日 cal 原值重建；没有重写旧模型。

密度参数化使用公开观测基准速度把 density residual 转为 flow-equivalent units，保证初始化的非负流量及密度与 speed head 解耦；独立手算/有限梯度测试通过。输出 clipping 与 LGB 混合会改变这项解析关系，不能把它当完整离散守恒解。新模型14日cal输入、预测和保存/重载验证通过；53 research /46 CPU tests、3 optional Torch skips。

重新读公开 scorer 后确认：非目标 eligible RAW observations 进入正式 Task3 state_for_regime，即使对应 masked 输入因 Task2 horizon/buffer 已空；只替换 Task1 targets。旧 mask|strictvisible 与 target-touching 分母不等价。A55 v2 仅在公开 train 对齐这些 evaluator anchors、observable topology、Mode A start-ramp validity 和 all valid adjacent pairs，不访问 boundary_flux，不输出正式 S_LWR。v1 未统一 LGB 上界，源 snapshot 为 postrun/preedit，均明确保留；v2 有 start snapshot。A55/A57 GPU 标志默认 false 的元数据错误在独立记录更正，原 JSON 与 SHA 不改写。后续 code 修复实际 network-device 标志及 pandas copy；A54 joint selector 文字补充 optional strict12 reference，原 source/seal 保留，本次 primary 更高使数值门槛不受该文字修正影响。

本轮 A49 的14日cal独立输入重建及部署预测通过；B08独立进程72组完全相同，B09扩展参数的72组原默认对照也精确相同。适配器默认保留既有temporal-first语义，新 weighted 选项按StateFeatures的权重构造，手工 age/space 案例及缺失prior回退测试通过。CTM历史profile只使用独立 dev eligible observations，未向网络输入latent density/flux，未作occupancy合成。

Zero-aware MAP 与独立 SLSQP/903-path augmented TRF-LSMR 对照通过；Huber核验原目标KKT而非代理二次目标。D12旧540评分逐项相同，1,350个assessment method/case完整。其保守规则在30份实际输入不触发，明确不视为实用增益。D13选择仅从development-only cache重算，三个新research seed与final seed及偏移不重叠，两种当前代码分支的预测SHA复跑相同；不提供未经验证的离散hurdle概率。

完整K05只改70,708个OD path_flow；所有State/Queue和其他字段逐行相同，6,980,503行全覆盖。20个真实输入split-panel实际都是entropy，稀疏专家分支没有触发；正式分数未知，未上传。新raw JSON继续保留字节级SHA，旧已验证最佳文件未改。

D11 读取十方向公开 path_set.csv 并逐 OD 分组，全部 max group size=1。D10 历史 route_choice_shift 标签的归一化 preference 完全抵消，是 lognormal .25 OD demand 扰动而非路径选择证据；原 JSON/生成器未改，仅由独立审计明确更正。两种真正新增失配的配对区间另算并仍跨零；报告不再把十个标签当十种独立生成分布。

本轮参与 SHA 引用链的 JSON 采用局部 `.gitattributes -text` 保护，避免 Git 自动 CRLF/LF 转换改变报告校验值。旧 D08 和 replication-v4 manifest 仅把现有工作区原始字节保存进 Git，其 JSON 内容与旧 HEAD 相同；原模型与数字未改。新旧报告的 staged blob / 工作区字节对照另存 `ctm_od_report_bytes_v1.json`。该修复保障回放时的来源校验，不表示模型改进。

本轮新增 adapter 以明确参数接口排除 CTM latent truth/目标标签，质量不合格值扰动不能进入输入；occupancy 缺失不从 density 合成。B07 两个独立进程的全部 54 case 数值相同。三缺失率共享真值，bootstrap 仅以三个 seed 为单位。两种自主 flux contract 和 target-only anchoring 保留，不冒充官方边界。

OD curvature 使用 measurement-space Woodbury，独立 path-space dense Hessian inverse 对照通过；经验区间只校准统一尺度，未改点预测。D09 曲率 metadata 的 method 覆盖模型身份导致汇总失败，源快照及 seal 保留；D10 已将诊断嵌套、固定身份字段放在最终记录层、增加阶段保存和完整唯一键检查。210 calibration arrays 精确复现，旧 420 point scores 漂移为零，600 assessment method cases 全覆盖、0 optimizer failures。区间宽、稀疏预测差的负面结果公开保留；不宣称后验可靠性。

本轮新增核验：六个 adaptive State 的独立 14 日 calibration 重建/预测均通过；A44 各成员在所有 cal/assessment 日复现，组合 wrapper/save/reload 只额外核验一个完整 assessment 日，不能把它写成组合的完整独立训练。Queue 原树在 C31–C34 的相同事件上逐窗复现；所有 NN 输入均来自 T 及此前一小时，波速图只用公开几何。

Entropy 双空间 Newton 与独立 path-space 优化器及大数消减 scalar oracle 通过对照。D06 的 69 个失败来自目标值浮点分辨率附近线搜索停滞；D07 出现一个新失败，D08 改用与误差预算匹配的 Newton decrement 后全部闭合。tol 2e−6 未放宽；生成器/种子/全局选择 gate 不变，Ridge 原分数完全相同，既有成功 entropy 分数漂移最多 2.32e−9。数值回归不等于独立泛化或正式 OD 成绩；约 99% 解变化在计数零空间。

当前研究环境 **50 passed（10.25 秒）**，CPU **45 passed / 3 optional-Torch module skips（5.78 秒）**。A49 的独立 14 日 calibration 重建/预测也通过。Causal Queue 的 class-weight 修正是显式 flag，默认保留 C30 原行为。旧源代码和所有失败实验均在各自 start snapshot/报告中保留；后续源文件修复不改旧模型或提交。

基准源码固定在 `15036c8`；正式提交 ref 57138874 回下载 SHA 与本地文件一致。官方代码为 `c88cddf533bbf0afa4ff1fc6c031d760a08e7a31`，2026-10-13 再次 ls-remote 核查仍为此版本。完整公开 Rules、Evaluation、Foundational Rules 与 Timeline 已读取并保存于 `data/published_competition_pages.json`。

| 路径 / 接口 | 审计结论 | 证据 / 限制 |
|---|---|---|
| `data.py` / Release | raw observations 仅允许 train；target 模板、ID、quality 和 ramp missing/structural-zero 分开 | 原全数据 fingerprint 与 contract tests；不读取隐藏 split raw |
| `submission.py` | SQLite 全局唯一键、ID、完整覆盖、有限非负、Queue binary、跨 chunk 重复检查 | 两次完整 6,980,503 行校验；K01 再做逐行单任务 diff |
| `metrics.py` / official evaluator | 本地 State / proxy IoU / S_link 有定义；真实其他三分项不可本地计算 | Evaluation 的完整任务缺行记零比 vendor 局部 missing-zero 更严格；上传前完整校验使这一区别不会进入候选 |
| `state.py` | mask-only 特征；离线插值与 RTS 可以使用目标之后的公开观测 | 合法 offline；未把 unmasked targets 放回输入 |
| `experiments.py` | 首轮单时间划分、calibration 选择 | 首轮 test 均已查看；新 rolling 不冒充未知最终 test |
| `queue.py` / `queue_calibration.py` / `causal_queue.py` | 旧模型保持严格 [T−60,T)；新模型显式允许已发布的 T，拒绝任何 >T 输入；nonempty prior 来自公开合约 | T 行用 Parquet 谓词读取；未来值扰动不改变输入的测试通过；旧代理采样分布有偏差 |
| `physics.py` | 外部连接以前被误当作 unavailable，导致五方向诊断覆盖为零 | 已修复并新增 external-vs-internal 回归测试；仍是 incomplete topology fallback，不能称正式 Physics |
| `odme.py` | measurement-space 非负 Ridge，KKT 与官方数值 parity 已核查 | counts 可识别维数远少于路径维数；S_link 不代表 path truth |
| `bayesian_od.py` | 显式分层 Gaussian prior，nonnegative MAP、未截断 posterior SD | D02 自建模拟增益没有迁移；K01 正式总分 0.76474；拒绝晋级 |
| `digital_twin.py` | 独立 Godunov/CTM、CFL 子步、显式 ramp supply、车辆数守恒 | B01 54 场景最大守恒误差 6.76e-13；两种模拟 flux contract 并列 |
| `graph_state.py` / graph experiment | missing-aware temporal conv + directed graph diffusion；occupancy 仅来自 masked 输入；joint speed/flow residual | 日前 profile、相同 fit/selection/assessment 日期、selection seal 后 assessment；GPU 344 MiB；多 seed/跨走廊继续研究 |
| `queue_events.py` | 均匀打乱候选、条件配额、六小时间隔、quality 和 persistence 筛选；C24 起可选 onset 首次观测 queue at T+30 | 组织方公告 742350 公开时序属性；其余 selector 未公开，仍为带噪代理。C01–C22 不匹配时序，仅压力测试；不同 seed 可重复 origin |
| `runtime.py` / evidence | CPU 实验 source fingerprints、进程峰值、训练前源码 ZIP | A29/A30 新实验在训练前保存源代码、依赖锁与逐文件 SHA；过去实验仅有 Git/运行指纹与模型校验 |
| 下载 / API / 发布 | 请求官方 public participant APIs 与本队 submissions；秘密和签名 URL 不写日志 | 用户最新指示先做模型；本阶段只做本地 Git 提交，不处理论坛/Notebook、GitHub 推送或新正式上传 |

上一阶段独立 CUDA 环境 **35 tests passed**（14.32 秒），CPU 环境 **30 passed / 3 optional-Torch module skips**（8.73 秒），源码 compileall 已通过。测试核验事件权重、density projection、target 隔离、锚点同化、daily pooling、onset 时序；新增独立枚举 expected-IoU 最优集合、density 缓存 key 对齐/不变、因果 Queue 十三步历史拒绝未来值/quality 过滤/无观测 fallback/onset-loss domain 检查。测试临时目录位于工作区 .cache。A01 与 A09 的共同模型 tensor/SHA 完全一致；C24–C29 三组共享模型独立重建/拟合后树文本、阈值、逐窗 IoU 一致。均不等于正式七百万行提交的全量独立复现。

A14 审计发现重要比较混杂：原 graph 与 LGB 的日期相同，但 LGB 只用 300k 标签。全标签 LGB 1,804,908 行得到更高 State，A15 重做强基线对照后 graph 独立收益仅 0.001077。所有旧实验保留，解释同步修正。

开放瓶颈：缺乏正式分项；独立模拟分布与赛事分布失配；Queue latent labels 未公开；物理 flux 投影使单纯 CTM surrogate 不可靠；OD 大零空间；Graph 单走廊模型的跨 family 泛化尚不确定。

A33–A36：可变节点的 FD 几何条件与拓扑偏置图注意力通过形状/有限梯度测试。四个 standalone 各在完整 14 天 calibration 独立构造输入并预测，数组逐通道一致，State/密度诊断匹配封存结果。配对架构消融复现原赢家 assessment，原选择保持。A38/A39 场景差分按 station AND link 分组，且严格相隔五分钟；原历史 scenario 脚本也已修正，当前 I5_N 原 50 日复跑的全部指标/计数仍相同，新文件另存。训练前源码快照是主记录；运行结束的 on-disk fingerprint 可能包含期间新增的部署代码。当前正式成绩仍为 .82940，未上传新候选。

K03/K04 各 6,980,503 行严格自然 key 覆盖通过，独立 streaming diff 同时验证 task、数值字段、方向与 master-key 身份。只有 I5_N 的 1,131,560 State rows 改变；原 Queue/OD 文件 SHA 相同，其他 State 方向字段逐行相同。不是全量独立训练复現或正式成绩验证。A38 初版 runtime 的 `gpu_used=false` 是调用 evidence() 时漏传参数，不能用它判断设备或 GPU 峰值；数值与部署验证不受影响。脚本现在记录实际 CUDA availability，A39/生产候选记录为 true；A38 原记录与训练源码快照保留。

`public_onset_timing()` 只约束标记为 onset 的 T+5…T+25，保留 ongoing 与 T+30。模型默认不开启，新公告对齐实验显式设置标记，旧冻结模型行为不变；针对条件/lead 的测试通过。K02 使用本队冻结预测与公开 metadata，未读任何隐藏标签或 val/private 未来数值；完整 6,980,503 行独立 diff 仅 5 个 Queue 值。计分 eligibility 未公开，因此更紧的 .001369 Total 界必须注明条件，安全界为 .0075。
# 窗口损失及独立复核 C42–C48

`queue_overlap_loss.py` 的 availability 只用于 TRAIN 监督损失，推断仍沿用不接收 future mask/label 的完整域模型。onset 前五步排除分类 overlap，未来 speed 标签仅用于训练辅助回归。独立实现 Lovász 的排序/前缀 Jaccard 增量，64 个二值顶点与枚举集合损失一致；soft-IoU 空 foreground 使用显式 mean-p 扩展，不能声称该分支是 exact IoU。

四组初始化与 batch digest 内部相同；全部选择封存后读取 assessment。相同 optimizer seed 的两个 BCE 参数控制与 C32/C34 全张量完全相同。独立从 `<T` RAW TRAIN 及仅 `T` masked 公共帧重建全部 cal histories，raw probability/导出全模板 binary 各660次精确。C46 固定 alpha 只引用已封存设置，后验诊断不反写 primary。首次 verifier numpy.str_ 错误保留失败源快照并在新目录修复复跑。

C44/C45 明确限定 TRAIN，将 future reads 标为 evaluator-only；future_coverage、future qc 和 raw future speeds 从未进入预测。80 窗口仅为公开观测审计，不能将 thresholded speed 或低质量 finite speed 冒充官方 queue_true。新增 Lovász/soft loss 的 sigmoid(logit-log4) 仅作可校准 score；加 overlap 后不保证 proper posterior。
# 两种子概率接口及模拟标签契约 C49–C54

`FullDomainQueueModel`抽取共同完整域概率路径，binary始终在canonical域决策后恢复target顺序；旧PKL接口660次原始cal输出精确复现。新ensemble只平均冻结网络的corrected score，状态保留CPU tensor，保存清除live GPU cache；全部550个导出角色/window的score及binary独立重建精确。未来 mask/label 仍不在forecast API参数中。

`select_with_cal_uncertainty`使用相同cal-window/key/condition、全部45日含空日期、4000个7日块重采样；gap SE不再除sqrt(resamples)。这是已拟合cal上的敏感性，不能宣称held-out coverage。五项测试检验确定增益、单日敏感性、key错配、完整域逆序恢复及缓存序列化。两项CTM基线测试验证只有13个past/current slots、当前缺测仍合法外推及全缺测fallback。

独立CTM forecaster只接收past noisy speed/history qc和提供condition，latent仅决定自主模拟事件，future qc/label在预测后评分。C51 shared CASES移到CPU-only模块，六值未改。C49初次清单缺项和C51初次可选Torch import失败均保留。旧六geometry NPZ此前未纳入manifest，当前bytes先登记并由C54公共网络逐数组重建；不追溯声称先前字节完全覆盖。全部旧模型/完整submission/依赖锁SHA保持不变。
# 五个互斥机制阶段与可选条件平衡 C55–C61

`queue_phase_split.py`严格要求全部TRAIN日历、fit之后15/30/15/30日、不重叠；event origin必须属于注册phase。worker请求decision/selection/assessment必须先存在checkpoint/decision/selection seal，逐读记录原seal SHA；部署复核核对全部原seal及date。NN参数只按checkpoint期Brier选择，threshold/参考仅decision期，后段gate不调参，assessment在gate封存后读取。

`queue_events.select_event_candidates`新可选按condition分池/最小已填quota轮换，仍保持全窗口global spacing；default仍原RNG permutation，旧metadata不新增字段。两测试检验密集class不挤掉rare class、spacing及default兼容，八组旧cache metadata/labels精确重构。C59冬15/944候选旧选0/22，balanced选9/13；这不是组织方selector或latent truth。

FullDomain树features按保存baseline的occupancy/provided-condition flags构建，default旧模型不变。新测试核查该输入契约；5份完整研究全部decision输入、score/binary 608role/window独立raw历史重建精确，查验两个tree原特征以及9channel causal arrays。C60两个冻结策略在相同新assessment完整域比对，新selected原指标精确重放。原失败冬只有fit数据及checkpoint标签，无decision/assessment标签读取；不虚构缺失condition的分数。研究/CPU83/71最终测试与失败记录均保留。
