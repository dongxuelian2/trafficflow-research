# V2 文献研究：证据等级、可复现性与比赛适配

检索与阅读日期：2026-10-16。本文与 [数学推导](V2_MATHEMATICAL_DERIVATION.md)、[架构](V2_ARCHITECTURE_PROPOSAL.md)、[决策](V2_RESEARCH_DECISION.md) 配套。没有执行第三方论文代码，也没有把论文指标转换成比赛预期分数。

## 检索和证据边界

检索覆盖 CTM traffic state estimation、sparse traffic data assimilation、moving horizon estimation、physics-informed reconstruction、graph estimation、incident particle filter、probabilistic queue、OD uncertainty/identifiability、differentiable traffic simulation。使用 arXiv 全文、出版社页面、作者机构记录与作者 GitHub；检索命中 Transportation Research B/C、IEEE TITS/CDC、AMRX、AAAI、ACM TOG、Annals of Applied Statistics 等。没有为凑齐期刊名而纳入不相关 Transportation Science 论文。

Google Scholar 直接检索入口与部分 IEEE/DOI 全文入口在本次工具环境不可访问；不绕过访问限制。下列“全文重点阅读”指已读取方法、相关实验/表格和限制，不表示逐行复现全部证明。“摘要/元数据”绝不作为已验证实验依据。中文适配判断和资源建议是本项目分析，不是作者结论。没有找到代码仅表示本轮未确认，不代表代码不存在。

## L1 — CTM 的道路离散化基础

**Carlos F. Daganzo (1994). The cell transmission model: A dynamic representation of highway traffic consistent with the hydrodynamic theory.** Transportation Research Part B 28(4), 269–287. [DOI: 10.1016/0191-2615(94)90002-7](https://doi.org/10.1016/0191-2615(94)90002-7)；[作者机构记录](https://its.berkeley.edu/node/4777)。

- 阅读等级：机构摘要/元数据；本次未把原文数值实验当作已核验结果。
- 核心：用离散单元的发送/接收能力表达宏观守恒。本文采用的通量形式为 `f=min(D,R)`，并以有限体积更新车辆数。
- 条件：路段几何、基本图、初边值与数值步长要明确；离散守恒不等于逆问题可识别。
- 实验规模/结果：不引用未读取的原文实验数字。公开代码：未确认原论文配套实现。
- 本赛适配：提供物理转移的基础，但必须另建噪声观测映射、未知边界和五分钟聚合。相对仓库图平滑，新增的是方向性动态传播。
- 工程判断：CPU 实现难度中等；主要风险是短 link 的 CFL 成本、边界错设与检测器流量误作接口流。8GB GPU 不是正确性的限制条件。

## L2 — 网络 CTM 与节点约束

**Carlos F. Daganzo (1995). The cell transmission model, part II: Network traffic.** Transportation Research Part B 29(2), 79–93. [DOI: 10.1016/0191-2615(94)00022-R](https://doi.org/10.1016/0191-2615(94)00022-R)；[1994 技术报告 The Cell Transmission Model: Network Traffic](https://escholarship.org/uc/item/9pz309w7)；[机构条目](https://its.berkeley.edu/node/2251)。

- 阅读等级：机构摘要、报告记录；1994 报告与 1995 期刊版分开记载。未声称逐页阅读扫描版。
- 核心：将 CTM 扩展到网络，节点守恒与转向/路径信息决定分流。与单一路段不同，节点解不是任意相邻平均。
- 条件：所用需求/转向规律必须给定或另行估计；论文网络传播结果不等于未知 OD 的唯一反演。
- 实验规模/数字与代码：本次未核验，故不填补。
- 本赛适配：拓扑缺失连接应成为边界 nuisance；不能把所有“无传感器 link”直接删去，也不能将里程相邻视为已知连接。需要保留匝道和瓶颈的聚合规则。
- 工程判断：小节点 LP 可在 CPU 完成；转向比例与容量/需求同时自由会严重不可识别。优先借用节点约束思想，不复制一个假定完整网络的求解器。

## L3 — 速度观测与 EnKF 的关键理论限制

**Daniel B. Work, Sébastien Blandin, Olli-Pekka Tossavainen, Benedetto Piccoli, Alexandre M. Bayen (2010). A Traffic Model for Velocity Data Assimilation.** Applied Mathematics Research eXpress 2010(1), 1–35. [DOI: 10.1093/amrx/abq002](https://doi.org/10.1093/amrx/abq002)；[出版全文](https://academic.oup.com/amrx/article/2010/1/1/272197)。

- 阅读等级：全文重点阅读，尤其速度变换、Godunov、边界与 EnKF、Mobile Century 部分。
- 方法/条件：连续速度守恒形式与密度 LWR 的相应等价只在特定线性速度—密度关系成立，不能推广到任意 FD；一般情形用离散 Godunov 后变换。局部 EnKF 用样本交叉协方差进行观测更新。
- 实验：I-880 Mobile Century，100 辆装手机车辆、8 小时，约 2%–5% 流量覆盖；论文比较移动观测估计与 PeMS。这里不把图示一致性改写为未报告的 RMSE 提升。
- 代码：本次未确认可直接复现的论文版本。
- 本赛适配：支持“密度输运 + 速度观测映射”，反对直接给速度套一个任意守恒 PDE。低覆盖率结果不等于本赛 R3 随机遮挡下的保证。
- 工程判断：局部 32–64 ensemble 可先 CPU 验证；协方差退化、边界误差及事故多峰是主要失败点。更适合作为 MHE 对照与不确定性初始化。

## L4 — MHE、已知输入假设与实时性证据

**Suyash C. Vishnoi, Sebastian A. Nugroho, Ahmad F. Taha, Christian G. Claudel. Traffic State Estimation for Connected Vehicles using the Second-Order Aw-Rascle-Zhang Traffic Model.** [arXiv:2209.02848](https://arxiv.org/abs/2209.02848)，2022 首稿；所读 PDF 标注 2024 IEEE TITS in-press。不补写未核验的出版 DOI。

- 阅读等级：[PDF](https://arxiv.org/pdf/2209.02848) 方法、MHE 附录及实验表重点阅读。
- 方法：二阶 ARZ 离散化、观测/动态线性化，arrival、观测与动态误差组成带盒约束 QP。其推导含已知输入假设；不能直接替代本赛隐藏入口估计。
- 实验：VISSIM，9 主线段加 1 入匝道、2 出匝道。表 II 每步时间 EKF .002s、UKF .006s、EnKF .016s、MHE .075s；环境为 i7-8750H、16GB、MATLAB R2020a。它们不是 RTX 4060 性能预测。论文亦出现 MHE 上游拥堵起始偏早等误差。
- 代码：本次未确认作者可运行仓库。
- 本赛适配：可借用局部 QP、约束与 arrival cost，但须补 unknown-input 模块，并比较窗口扩大后的偏差，不能声称“窗口越长越准确”。
- 工程判断：稀疏 CPU QP 中等难度；二阶状态、未知输入和多事件一起释放会扩大病态方向。首版采用更小 CTM，再决定是否需要 ARZ 松弛项。

## L5 — CTM 故障容错 MHE：仅作检索线索

**Stelios Timotheou, Christos G. Panayiotou, Marios M. Polycarpou (2015). Moving horizon fault-tolerant traffic state estimation for the Cell Transmission Model.** IEEE CDC, 3451–3456. [DOI: 10.1109/CDC.2015.7402753](https://doi.org/10.1109/CDC.2015.7402753)。

- 阅读等级：标题、书目信息与相关论文引用核验；IEEE 全文未成功取得。
- 相关问题：CTM 下故障容错滑窗状态估计，与传感器误差和动态约束相符。
- 核心公式、故障假设、实验规模/指标、公开代码：**未全文核验**，不据标题推断已验证的方法。
- 本赛适配/工程判断：可作为后续获取全文的优先对象；不能据此决定具体鲁棒损失或声称可在 8GB 复现。本文 MHE 方程来自自有推导及 L4，不冒充 L5 原公式。

## L6 — 物理信息学习：稀疏传感器下参数失真的直接例子

**Rongye Shi, Zhaobin Mo, Kuang Huang, Xuan Di, Qiang Du.** 期刊标题 **A Physics-Informed Deep Learning Paradigm for Traffic State and Fundamental Diagram Estimation**，IEEE TITS，2021 online / 2022 volume 23(8), 11688–11698. [DOI: 10.1109/TITS.2021.3106259](https://doi.org/10.1109/TITS.2021.3106259)。所读预印本标题为 **A Physics-Informed Deep Learning Paradigm for Traffic State Estimation and Fundamental Diagram Discovery**：[arXiv:2106.03142v2 全文](https://arxiv.org/html/2106.03142v2)。

- 阅读等级：全文方法、环路实验表 I 与 NGSIM 设置重点阅读；版本标题差异保留。
- 方法：观测误差 + PDE 配点残差 + FD 结构/凹性约束，同时估计状态与 FD。
- 规模/结果：环路网格 960×240、100,000 配点，文中 GPU Titan RTX 24GB。表 I 两传感器设置的相对误差 .6021，三传感器 .03327；两传感器估计扩散参数约 3.12573，而生成值 .005。NGSIM US-101 为约 680m、预处理 21×1770 网格。
- 代码：本轮未确认与所读版本匹配的作者实现。
- 本赛适配：最有价值的是“低观测下物理参数也可严重失真”的证据。其规则网格、连续场配点与比赛缺失拓扑、噪声评分目标不同。
- 工程判断：可做小型 FD/观测映射辨识；不优先复刻全域 PINN。配点内存、权重敏感性、错误闭合和低可识别性都可能抵消正则收益。

## L7 — 图上的 Dirichlet 重构与已有模型的重叠

**Qishen Zhou, Yifan Zhang, Michail A. Makridis, Anastasios Kouvelas, Yibing Wang, Simon Hu. Network-wide Freeway Traffic Estimation Using Sparse Sensor Data: A Dirichlet Graph Auto-Encoder Approach.** [所读 arXiv:2503.15845v1（2025）](https://arxiv.org/html/2503.15845v1)；[版本/出版记录](https://arxiv.org/abs/2503.15845)，IEEE TITS 26(12), 22161–22177 (2025)，[DOI: 10.1109/TITS.2025.3610911](https://doi.org/10.1109/TITS.2025.3610911)。

- 阅读等级：v1 方法与结果表；不把后续修订当作已阅读全文。
- 方法：潜空间 Dirichlet 传播、自由/拥堵方向编码、动态遮挡；已知节点固定，未知节点更新可写 `x_u'=A_uo x_o+A_uu x_u`。
- 实验：METR-LA 207、PEMS-BAY 325、PEMSD7(M) 228 节点，五分钟数据，25% 虚拟无传感器节点；表 II METR-LA RMSE 8.07 mph，对照 KITS 9.02。文中 RTX4090；7:3 分割且 validation 与训练集合相同，需注意选择协议。
- [作者代码](https://github.com/ZJU-TSELab/DGAE)：已确认链接，未安装/运行。
- 本赛适配：全节点缺失与随机 cell 遮挡不同，论文只重构速度；仓库已经有全局 attention、remask 和图残差平滑，不能把换名字当新增信息。
- 工程判断：小图潜空间传播可落在 8GB，实际峰值未测；主要风险是图断连、传播平滑急剧降速及协议迁移。作为消融对照，不选为两条主路线。

## L8 — 离散事故与连续交通的多模型粒子法

**Ren Wang, Daniel B. Work, Richard Sowers (2016). Multiple model particle filter for traffic estimation and incident detection.** IEEE TITS 17(12), 3461–3470. [DOI: 10.1109/TITS.2016.2560769](https://doi.org/10.1109/TITS.2016.2560769)；[作者机构记录](https://experts.illinois.edu/en/publications/multiple-model-particle-filter-for-traffic-estimation-and-inciden/)。

- 阅读等级：机构摘要与元数据；全文未获得。
- 已确认问题/方向：联合连续交通估计与事故离散模式，多模型粒子滤波；摘要描述 CORSIM 实验及与粒子法/事故检测基线的比较。
- 公式、实验规模、数值优势与代码：本次未全文验证，不引用具体性能或精确递推。
- 本赛适配判断：研究因果事件模式有价值，但 incident detection 看到事故后的观测，不等于 onset forecasting 能提前预测独立事故。本文提出的半马尔可夫持续时间与 Rao–Blackwell 化是候选设计，不声称是该文已证实的比赛方案。
- 工程判断：完整高维粒子状态易退化；只采样少量位置/容量/持续模式，再条件估计连续状态，可控性更好。资源与收益均待本地小实验。

## L9 — 可微交通模拟的梯度与资源证据

**Sanghyun Son, Yi-Ling Qiao, Jason Sewall, Ming C. Lin (2022). Differentiable Hybrid Traffic Simulation.** ACM Transactions on Graphics 41(6). [DOI: 10.1145/3550454.3555492](https://doi.org/10.1145/3550454.3555492)；[arXiv:2210.08046v1 全文](https://arxiv.org/html/2210.08046v1)。

- 阅读等级：宏微观模型、可微桥接、梯度与实验表重点阅读。它采用宏观 ARZ / 微观 IDM，不是 CTM 论文。
- 方法：分段数值更新的解析导数、宏微观转换、跨时间反向传播，用于参数/初态优化和控制。
- 实验结果：表 1 的 50 步、5000 cells 宏观反传，autograd 71.08s、解析 3.79s，约 18.77 倍；仅代表论文环境，不能当成本机测速。
- [作者代码](https://github.com/SonSang/diff-hybrid-traffic-sim)：已核验存在，MIT，未运行。
- 本赛适配：可借用局部解析梯度/检查点思路。可微性只解决优化工具，不解决未知边界、噪声观测和初态的可识别性。
- 工程判断：直接移植混合微观系统成本高且收益不明；本项目不需要车辆级状态。先写小型离散守恒核及有限差分检查，避免全轨迹 autograd 激活占满 8GB。

## L10 — OD 分布可识别与一次实现可恢复是两回事

**Martin L. Hazelton (2015). Network tomography for integer-valued traffic.** Annals of Applied Statistics. [DOI: 10.1214/15-AOAS805](https://doi.org/10.1214/15-AOAS805)；[arXiv:1501.02514 全文](https://arxiv.org/html/1501.02514)。

- 阅读等级：模型、可识别性 Proposition 1、采样方法与大网络实例重点阅读。
- 方法：`y=Ax`，观测似然对非负整数可行路径向量求和；用 MCMC 进行路径/参数推断。独立非负整数流、充分支持、不同非零列等条件下可识别分布参数；采样构造另有全幺模等适用条件。
- 实验：Leicester 案例，大实例 127 路径、27 计数 link；文中两轮 10,000 pilot 加 20,000 迭代少于 10 分钟。这里关注可行域混合，不解释成隐藏路径精度保证。
- 代码：本轮未确认版本匹配的完整复现包。
- 本赛适配：纠正“秩低就所有统计参数永远不可识别”的过强说法；但本赛缺少同分布重复计数，且目标是一次实现，不能据该定理推翻 D05 零空间风险。
- 工程判断：数千路径的整数 MCMC 难以成为第一候选；须先验证矩阵结构和重复观测条件。GPU 不能弥补条件缺失。

## L11 — 随机 OD 计数推断的经典前提

**Martin L. Hazelton (2000). Estimation of origin–destination matrices from link flows on uncongested networks.** Transportation Research Part B 34(7), 549–566. [DOI: 10.1016/S0191-2615(99)00037-5](https://doi.org/10.1016/S0191-2615(99)00037-5)；[出版页面](https://www.sciencedirect.com/science/article/pii/S0191261599000375)。

- 阅读等级：出版社摘要/可见预览；全文实验未核验。
- 已确认方法方向：无拥堵网络上的随机 OD 与 link 计数，考虑 Poisson/路径随机性及近似似然。
- 理论细节、实验规模/数字、代码：不据摘要补写。
- 本赛适配判断：须先区分均值 OD、某期真实 OD 和先验偏差；公开计数分割并非足够的同分布重复样本。弱先验与隐藏真值的生成分布未知是关键限制。
- 工程判断：小型似然近似主要消耗 CPU；决定风险的是分布错设，而非优化器速度。与最大熵、经验贝叶斯一起作为先验敏感性研究，暂不替代冻结 Ridge。

## L12 — 神经非线性状态空间与结构化变分后验

**Rahul G. Krishnan, Uri Shalit, David Sontag (2017; arXiv 2016). Structured Inference Networks for Nonlinear State Space Models.** AAAI 2017. [arXiv:1609.09869v2](https://arxiv.org/abs/1609.09869v2)。

- 阅读等级：摘要/书目，不声称已阅读全文实验。
- 已确认方向：非线性生成状态转移与观测模型，RNN 参数化结构化变分近似。本文使用的通用 ELBO 见数学文档式 (18)，不是从摘要推断的某个专有公式。
- 实验规模/数值、代码：本轮未核验；不拿其 held-out likelihood 声称交通 IoU 提升。
- 本赛适配判断：可用共享潜状态、不同因果范围的推断网络；但大解码器可能吸收物理状态、导致不可识别和 posterior collapse。
- 工程判断：小网络可行；先验证显式状态模型确有观测增益，再考虑摊销推断以降低 MHE 成本。首轮不训练大规模生成模型。

## L13 — Neural ODE：工具参考，不作为首选交通闭合

**Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud (2018). Neural Ordinary Differential Equations.** [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)。页面显示后续修订；本轮仅核验标题、作者和摘要，不将新版图表当作已读证据。

- 核心方向：以 `dx/dt=f_theta(x,t)` 定义连续深度模型，并通过 ODE 求解/伴随进行学习。
- 阅读等级、实验规模：摘要级；不引用具体实验指标或资源保证。未在本轮安装任何 ODE 库。
- 本赛适配判断：可描述低维边界需求的平滑演化，但交通激波、离散容量事件、节点 min 和传感器聚合需额外处理。连续 ODE 形式本身不包含守恒或可识别性保证。
- 工程判断：自适应函数评估数、刚性和反向重构误差可能抵消所谓内存优势；应与固定步离散 CTM 比较同等误差容限，不能仅按模型名称选择。

## 跨论文比较与研究缺口

| 路线 | 文献真正支持的内容 | 与本赛仍未接通的环节 | 本轮决策 |
|---|---|---|---|
| CTM + MHE/EnKF | 守恒离散、边界处理、有限窗约束估计 | 未知边界、短路段、观测算子、隐藏 LWR flux | 首选可否决原型 |
| PINN/可微模拟 | 可对物理参数/状态求导，部分设置可重构 | 条件数、错误闭合、观测噪声与比赛评分 | 采用离散导数与受限残差，不全域 PINN |
| 图重构 | 已知节点传播、缺失节点重构 | 相对现有全局 attention 的新增信息 | 对照，不独立主推 |
| 混合事件模型 | 事故模式与连续状态的联合表示 | 提前预测与事后检测差别、稀少事件、选窗偏差 | 第二候选，先非神经事件基线 |
| OD 贝叶斯/网络层析 | 明确生成假设下的不确定性、部分分布识别 | 一次计数、零空间、先验分布失配 | 诊断优先，不以模拟分晋级 |

没有找到且验证过一个“已在本赛同等遮挡、同等隐藏边界、同等 noisy/latent 双目标下大幅超过当前模型”的公开方法。现有证据支持结构化研究的必要性，不支持 .90 或 .95 的承诺。最有价值的可复用资产是明确的守恒算子、边界不确定性、条件后验和评价协议，而非某篇论文的排名。
