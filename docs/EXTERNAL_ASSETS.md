# 外部资产和许可证声明

官方 TrafficFlowBench 工具包：MIT，完整 LICENSE 保留于 vendor/official/LICENSE，固定提交和每个文件 Git blob SHA 见 UPSTREAM.json。
本项目 src/trafficflow 为独立实现，许可证为根目录 MIT。
第三方参赛方案仅作公开架构阅读，没有导入代码、训练数据、预训练模型或预测。
正式比赛数据使用 Kaggle 当前公开包，共 9,688 个文件，创建于 2026-09-11。下载归档 SHA-256 和文件大小核验见 `reports/official/download_verification.json`；完整逐文件指纹保存在本机 audit artifact。没有使用旧包或组织方文件。
人工 fixture 由本项目生成，仅用于测试；不作为外部 measured traffic data 或比赛成绩。

数值依赖：NumPy（BSD）、pandas（BSD）、SciPy（BSD）、PyArrow（Apache-2.0）、scikit-learn（BSD）、LightGBM（MIT）、CatBoost（Apache-2.0）。
新研究神经网络依赖官方 PyTorch 2.8.0+cu128（BSD 风格许可），没有使用外部预训练权重。CUDA/cuDNN 为官方轮子随附运行时，按其随附许可证使用；GPU 为本机 RTX 4060 Laptop 8GB。新增 Python 依赖版本见 requirements.research.lock.txt。
官方下载使用 requests（Apache-2.0）/Kaggle API。完整版本见 requirements.lock.txt；分发时应保留依赖自带 LICENSE。
开发过程中使用 Codex AI 辅助编程，运行时不依赖其 API，预测只运行本地代码。
没有付费交通数据、预测 API、第三方 submission、手工隐藏标签或组织方私有资源。

CTM 与 OD 模拟器为本项目独立实现，使用公开网络和 FD/weak-prior 参数作为输入，生成自己的随机场景及真值。它们不是官方生成器，也没有试图确定官方隐藏随机种子。
