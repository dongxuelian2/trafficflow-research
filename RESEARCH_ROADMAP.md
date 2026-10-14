# 0.950+ 研究路线

目标为可复现的正式 Kaggle **0.95000+**；当前正式 **0.82940**。不将阶段性较低分数作为终点。

C49 的两种子 mean-BCE 只有秋+.000985/CI跨零，冬 cal 只多.000160却有−.021137 assessment代价；简化到 tree 的 bootstrap-gap primary 也退化。暂不部署这些新政策，不扩张平均权重/同一阈值网格。下一步优先训练与校准数据真正分开的选择机制：固定 checkpoint 或仅在前段 cal 选 checkpoint/threshold，再在后段保留原强参考而非默认退回更弱 tree；该后段必须未参与这次权重选择，不能把已有模型的 cal 切片包装成独立验证。先增加一个新方向及其合法时间划分，保留 source 参考与小幅收益拒绝门槛。

C51/C52 确认已知 latent 队列、带噪 threshold labels 和未来可评分域可以给同样预测不同 IoU；零噪声也会丢失正例。优先结合公开 TRAIN 历史与独立模拟做噪声/未来可用率建模，不能使用实际 future mask 推断。503 origins 重叠、六 case 共用三个 seed，不把72个counterfactual当独立样本。C54 补齐旧 geometry 的公开重建和校验；当前hash不追溯冒充以前完整覆盖。原 .82940/K06继续冻结，无平台动作，final seeds不生成。

C42–C48 的 12 个配对模型否定了“直接加 λ=.25 窗口 IoU surrogate 即可稳定改善”的当前设计：两季两 optimizer seed 的 primary 均无可靠新收益，部分有明确代价。暂不扩张这一固定 loss/旧 Brier-checkpoint 网格。C46 的原 seed Lovász α.25 次要正例不能按 assessment 晋级；另一个 seed 没有复现。下一步可封存跨种子平均概率和校准选择的不确定性规则，先与同数据的多 seed BCE 组合比较，再做跨方向；不将多模型或单个正 CI 作为成功标准。

C44/C45 在实际公开 TRAIN 窗口发现 1/40 onset 没有有限速度阈值正例，质量过滤不是原因。故观测正例 sampler 的缺口已有公开实例，仍不能识别潜在 queue_true。优先用独立 CTM 的已知 latent/带噪观测作事件采样、标签阈值和未来可用率联合诊断；显式分离观察标签、latent 标签和 scorer mask。允许历史估计未知未来可用率，禁止把实际 future mask 放入预测。最终模拟 seeds 仍不生成；原 .82940/K06 冻结，平台按用户指示暂缓。

C38–C41已完成正确全域calibration、时间约束before/after的独立对照、可部署模型及完整LOCAL K06。两季选择均force=false，时序修正gain为0；不要继续扫同一threshold网格。模型与tree的proxy gain仍CI跨零，保持一个fit seed的研究候选，不推广生产、不上传K06。

下一轮优先检验已发布TRAIN窗口及rolling历史对当前观测正例采样的偏差，并增加跨方向/新fit seed。C40的已知真值模拟表明条件化scored-positive会改变nonempty评价，但不能推断实际组织方附加采样规则。可比较事件发生/传播范围和联合概率模型，明确未知未来质量仅可用历史估计，真实future eligibility不得进入推断。所有新概率方法必须先在已知真值模拟与同一完整域tree对照，再进行合法观测proxy验证。

K06只作完整Queue-only控制，State/OD/Physics输入保持原最佳；后续正式测量继续等待用户恢复平台任务。最终模拟seed不启用。0.95000+正式目标不改变，不以当前可部署接口或零增益对照作为完成标准。

A66–A71再次证明density loss不是稳定State增益：秋cal获选方案有.002184 State代价，30日复核CI仍负；冬只有未获primary的.000074–.000077变化。暂停扩张这一固定λ网格，保留State-only和仅作proxy challenger的模型。完整分母约把有效λ减半，若未来继续这条路线，必须比较等平均强度及观测噪声/潜在动力学模型，而不是把λ效应包装成物理结构收益。

**下一步优先修正Queue模型选择的完整forecast域。** C35/C37已证实filtered未来eligible域会改变global-prefix/nonempty决策；只把mask放在scorer。C27–29原专家权重可复用，但需从完整域重新校准候选阈值/集合规则；C32波速checkpoint的Brier训练仍为逐点scorer，旧nonempty绝对IoU需要更正。独立对照“先按公开onset时序限制候选，再做nonempty”与旧先force再清零次序，不能静默改旧成绩。之后再比较事件级概率/空间相关集合模型及观察标签噪声，继续多个fit seeds、跨方向与rolling验证。

公开latent horizon有queue并不保证future eligible评分域仍有positive，nonempty不是安全增益定理；旧代理采样还要求观测future positive，存在选择偏差。需要合法的已知真值模拟压力测试及已发布train windows的观测诊断，始终与真实Queue分数区分。OD继续count/prior失配路线；最后模拟seeds与正式最佳保持冻结。平台任务按用户指示继续暂缓。

正式冻结Queue中D7_I10_W及D7_I405_N为force=true；它们的实际推断使用完整模板，正式分数可信，但原本地校准应加入新的完整域对照。先校准和评估完整模板上的集合规则，再建立未知未来可用率的概率模型；不能把真实future scoring mask当决策输入。

A58–A64提供了明确停止扩张同化网格的证据：三组最多只有.000124/.000112的State gain，I210_W无收益，额外五次先验推断的代价较大。保留可复现代码和cal fallback，暂不全网生产或继续扫更多近邻超参数。CF策略两冬季日均CI为正仍低于原晋级门槛；不能用统计显著替代实际增益大小。

下一轮优先用已完成A65标签缓存做**相同strict12/vq初始化、patch、budget下仅改变density loss coverage/分母**的实验。审计已更正最初假设：I210_W本fold没有额外eligible buffer锚点信息，旧loss有约26.4%相邻对不属公开Mode A coverage。新full标签不能进入模型输入；真实边界流仍未知。检查State代价及公开coverage proxy，避免把旧target-touching分母的下降称正式Physics增益。

随后优先增加高权重Queue的观测/latent标签偏差检验、事件级概率校准及跨走廊rolling对照；不根据刚查看的State日期重调参数后称final。OD仍检验可见计数不一致与稀疏先验失配，保留K01正式失败。最后模拟seeds封存；平台发布/推送/上传继续按用户指示暂缓。

最新 A51–A57 推翻了“直接增加低质量通道即可带来独立 State 增益”的假设：14日退化，追加30日仅小幅变化且 CI 跨零，不扩展全网部署。密度坐标的 State 也未晋级，但更接近公开 coverage 的增量误差代理改善；下一步优先检验噪声校准、隐含边界状态与 State/密度联合目标的可迁移性，在另一季和方向保留纯 v/q 控制。不可根据刚看到的30日结果重调参数后再称独立复核。

质量路线应从可见质量与上下游一致性估计观测误差，再在开发/cal 中验证显式置信度门控或同化，而不是持续盲目加通道。旧12通道与零新增权重嵌套初始化已提供可复现控制。A55 的真实 organizer flux 不可见，公开 train 真值仅用于 evaluator anchors；新 proxy coverage 更接近规则仍不能代替正式 Physics。真正增益必须先有跨季 State 验证，后续正式测量待用户恢复平台任务时再处理。

继续保留 Queue 因果/公告采样路线及 OD 噪声、稀疏先验失配路线。此次没有开启最终模拟 seeds、没有新正式提交，也没有更换已核验最佳完整文件。先完成本地模型与误差证据，不处理论坛或其他发布渠道。

本轮 A49 在 D7_I210_W 强基线下有 .006476 pooled gain、日均区间为正，但只有秋季/单 fit seed；继续补相反方向与跨季。A50 显示 graph 的质量过滤会形成长输入不可用段，R3 p99 60min，而 target p99 25min。下一项结构实验应将合法低质量数值与可靠度显式提供给网络，与现有12-channel/full-LGB在相同划分比较，不能用不可用时段数直接推断官方 mask generator。

D12 的保守规则虽然模拟为正，30 份公开 weak prior 完全不触发。D13 改为 entropy fallback，在三个新 research seed 上九类均值都有收益；完整 K05 已生成并核验，只改 OD，当前20份待测输入实际全部 entropy。保留为可评估的本地控制，原正式最佳不变；下一步扩大 count-noise/prior 分布变化、分析非空负例与路径质量，不能因新模拟 CI 正就宣称正式收益。Final seeds 仍封存，平台继续暂缓。

B09 证明部分 CTM 结论受 temporal-first 弱适配器影响；用现有 StateFeatures 权重后基线更强。事故120min缺失仍极差，history-only对固定时钟过拟合；下一步优先已知 latent/noisy 区分、边界/匝道潜变量及与 State 对照的联合密度增量目标，使用完整 target–observed 锚定诊断，正式 Physics 仍未知。不要继续堆积与现有输入不匹配的零样本测试。

最新证据改变下一步顺序：A47 第四走廊族 D12_I405_S 的 pooled gain 仅 .000371、日均 CI 跨零，不能把前三族的收益推广为全网保证。继续补 D7_I210 和相反方向；在同一 cal 上封存可退回 LGB 的选择规则，按原门槛检查实际退化，而非强制每方向使用 NN。未由 cal 选中的 attention assessment 略高也不反选。

D11 前提审查优先于路径选择架构：公开 path set 的每个 OD 对只有一条 path，D10 的 route-choice preference 会抵消，实际是既有独立 demand 分布。今后统一选择要覆盖九种实质分布（十个历史标签）；若构造替代路径，只能明确标为自主网络实验，不能当现有 Task 4 可识别的 route-choice 参数。先审查特征、变量和生成机制的实际自由度，再增加模型。

B07 固定混合在平均 observation State 上改善，但 mild 与真实 CTM-flux 诊断下降，density 的急降速收益与恢复误差方向相反。优先用分别可控的观测噪声、缺失历史 profile 和已知 latent 守恒目标定位原因，再设计软状态同化；不要把增量代理损失继续加大当作物理突破。现有完整 K03/K04 仍保留，未上传。

D10 冻结 entropy 在 OD demand 同类复核 / outlier 新分布有收益，却在 sparse-active 比 Ridge 差 .075310，两种真实新增分布 macro CI 跨零。下一轮应从独立 development 真值检验能保留零路径的稀疏或 hurdle 先验、合适的计数噪声模型，并对九种实质分布统一选择后再评估；不依据已经查看的 sparse assessment 单独调 floor。Hessian 区间需扩张 6.4 倍仍约真实平均 flow 的三倍宽，不作为可靠 posterior。最后模拟 seeds 继续封存，真实 OD 和 Physics 仍未知。

A41 第三 seed、A43 I405_N 冬季均完成，相对强 LGB 的 State gain 为 .008086 / .008028，日均区间为正；六组匹配 attention-vs-geometry 区间均正。A44 三种子组合 .951168 vs 同三 LGB .942742，但只有一个季节，不扩展生产组合。A45 I10_W 监督 gain .003837、区间为正；A37 零样本失败仍保留。P01/P02 与完整 K03/K04 均已保存，正式最佳不变。C31 校准修正未单独提高 IoU；C32/C34 波速图的 cal-selected 两季区间仍跨零。D08 熵反演通过广义先验模拟 gate，但约 99% 改变量在计数零空间，真实路径效果未知。按用户最新指示，先做模型与本地验证，不处理平台发布。

近期优先级：把同协议 State 对照继续推广到剩余走廊族及方向，判断全网收益与最差方向损失；组合跨季复核优先于生产扩张。Queue 增加 fit seeds/跨方向的同事件波速范围对照，并研究观测噪声与 latent 标签差异；不得把已看 assessment 的 15min 反选为赢家。OD 的 τ=.05/floor20 已在开发端封存，下一步用新的失配分布与可校验的不确定性实验检验；原正式 Ridge 保留。B 路线需在独立 CTM 上验证真实守恒和突变鲁棒性，不能仅依赖 public target-pair increment。独立模拟 final seeds 尚未生成，架构封存后一次性使用。

1. **A：强 State 比较基线与独立残差收益。** A14 的全标签 LGB 已超过原图组合。先在冬季和较大方向核对标签预算，再用强基线评估图网络的独立增益，区分速度收益与流量退化。A15 增益 0.001077 低于晋级门槛；不能依照旧的 300k 比较组继续宣称架构优势。GPU 限制 8GB，使用小批次。
2. **B：交通数字孪生。** 独立 CTM 已通过守恒与密度界检查。先评估模型在未知事故位置、失效比例、需求扰动下的鲁棒性；辨识 public train 动力学时只估计一般参数。官方 flux 是观测投影，模拟 CTM flux 与 observation-projected flux 要并列报告。修正外部连接后的本地残差依旧不等于正式 Physics。
3. **C：公开采样对齐与事件决策。** 组织方公告 742350 允许使用 onset 在 T+30 才发生的属性。首轮和 C01–C22 随机代理仅作为一般事件压力测试保留。C24–C26 对齐这一时序后，at-T 收益仍跨季节不稳定，全部 CI 跨零。下一步优先 onset T+30 的位置/范围专用模型与 ongoing 六步模型，使用相同事件、公开 train 标签、按日不确定性比较；不要按 assessment 选模型。未来数值仍不可作为输入，其他 selector 与 latent truth 仍未知。
4. **D：概率 OD。** D02 在独立模拟中有增益，但 K01 正式 OD-only 控制结果下降 0.06466，拒绝此候选。研究 broad prior distribution 失配、矩阵 rank/nullspace 和 interval coverage；增加反向 noisy-prior 生成与整体/局部 bias。S_link 接近 1 不构成路径恢复成功，不从这一次正式反馈细调 Bayes 参数。
5. **组合。** 每次正式上传前登记假设、基准 SHA、改变的任务与离线证据；每天最多两个新研究提交，低于官方五次限制。负增益候选保留，最佳版本不可覆盖。private split 最终成绩依然未知。
6. **封存与奖项。** 最终独立模拟 seed 在选择封存后启用；全量独立复跑、锁定依赖、模型 checksum、代码和 PDF 是奖项准备的一部分。奖项注册邮件及团队资料尚未核验；不自动发送邮件。

A01/A02 相对 300k LGB 在两个季节、三个 NN seed 改善，但 A14/A15 表明收益主要由标签预算解释。A05 跨 family 迁移失败，A13 有标签的逐方向图模型也无增益。A10 的 public-train density-increment 约束仅改善代理诊断，A11 强权重失败。下一轮先完成强基线控制与合法 Queue 输入实验，避免继续增加没有独立收益的结构。

A16–A20 进一步确认强 LGB 对照下的图增益小于 .002。C12–C14 的 occupancy 特征在 I5_N 两个季节存在正向事件 block 证据，但 I10_W 明显退化；下一步优先 45 天 calibration / 30 天 assessment、事件均衡损失和分方向专家的稳定选择，而不是统一增加该特征并上传。A19 以全标签重检 direct-density 回归；第一轮 joint-LightGBM 失败仍保留。

更长 Queue 评估 C15–C17 削弱了 occupancy 的独立增益，I10_W 仍退化；C18/C19 改用更多合法训练事件，但保留原 assessment spacing。A23b 的 State occupancy 增益仅 .000281，暂不追加复杂度。A24 与正式模型同类的 500k 预算为 .938184，full 为 .942340；A25 第二个 full seed 为 .942295。优先完成预算控制的多 seed / 多方向证据，再考虑一份保留原 Queue 与 OD 的 State–Physics 正式控制候选。

D05 未发现重复计数列或完全不被计数的路径。可识别性不足源于低秩线性组合，不能用列重复来解释；所有单独路径的线性 leverage 都远小于 1。K01 在未计数网络路段上的负载变化约 8.2%，仍无其真实负载可供评分。

当前优先级：500k/full 三种子增益已在 I5_N autumn 达到登记门槛，继续核查其他方向和冬季，再准备保留原 Queue/OD 的 State–Physics 控制候选。Queue 的训练事件预算比复杂事件损失更有收益，但 I10_W 的 at-T 模型持续退化，不能按一个方向全局替换。B02 学习密度增量并做硬锚同化，首先用强 State 基线与 target-pair 诊断筛选。官方 Physics 仍需受控正式测量。

源代码公告未发布，Notebook 为未发布草稿。按用户最新“先做模型，不要管平台发布”指示，当前只做本地研究和 Git 归档，暂缓论坛、Notebook、GitHub 推送与正式上传。

最新优先级：A31/A32 补齐十方向与冬季预算对照，全网秋季 gain .001277，预算本身不足以完成目标。C27–C29 专家两组正增益但全部区间跨零，I5_N 秋季退化；expected-IoU 独立性假设无稳定证据。B03–B06 仍是 State/诊断取舍。下一步投入自适应上下游传播、缺失可信度权重的神经模型，与全标签 LGB 比较，保持日期隔离和选择封存；不依赖低预算比较组，不挑未由 calibration 选中的胜者。

C23/K02 证明简单时序清零只能改变原提交两个 validation 窗口的 5 个值，private 完全不变。保留已核验完整候选，不把这条公开规则包装成高分突破。正式比较优先选择可能改善 private 的模型；下一轮研究重点是 T+30 queue 的稀疏位置恢复和 ongoing 波传播。A29/A30 已完成六方向预算控制，增益全在 .002 以下；不能将 I5_N 的单方向强增益推广到全网。

B01 揭示 flux contract 对诊断影响极大：同一个观测 oracle 在实际 CTM flux 下模拟 Physics 0.340882，在我们按观测真值投影的 flux 下 0.995759。纯 CTM 守恒不能直接视为比赛评分信号；物理损失需结合状态误差和受控正式评测。

矩阵数值 rank 比测量行数更小：例如 D12_I5_N 92 / 4,278 paths，nullity 4,186。rank/n 是行空间维数比例，不是“某些具体路径已经可识别”的证明。区间是未做非负截断的高斯近似，不能冒充可靠 posterior coverage。

规则：当前公开 Kaggle 截止 2026-11-07 06:55 UTC，每日 5 次、团队最多 4 人。奖项最终评价还包含报告。源规则记录在 `data/published_rules_response.json`；需要持续核对 Evaluation 页面，不能用旧榜成绩比较当前重建数据。
