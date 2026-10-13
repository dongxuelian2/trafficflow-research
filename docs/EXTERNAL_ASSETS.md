# 外部资产和许可证声明

官方 TrafficFlowBench 工具包：MIT，完整 LICENSE 保留于 vendor/official/LICENSE，固定提交和每个文件 Git blob SHA 见 UPSTREAM.json。
本项目 src/trafficflow 为独立实现，许可证为根目录 MIT。
第三方参赛方案仅作公开架构阅读，没有导入代码、训练数据、预训练模型或预测。
正式比赛数据使用 Kaggle 当前公开包，共 9,688 个文件，创建于 2026-09-11。下载归档 SHA-256 和文件大小核验见 `reports/official/download_verification.json`；完整逐文件指纹保存在本机 audit artifact。没有使用旧包或组织方文件。
人工 fixture 由本项目生成，仅用于测试；不作为外部 measured traffic data 或比赛成绩。

数值依赖：NumPy（BSD）、pandas（BSD）、SciPy（BSD）、PyArrow（Apache-2.0）、scikit-learn（BSD）、LightGBM（MIT）、CatBoost（Apache-2.0）。
官方下载使用 requests（Apache-2.0）/Kaggle API。完整版本见 requirements.lock.txt；分发时应保留依赖自带 LICENSE。
开发过程中使用 Codex AI 辅助编程，运行时不依赖其 API，预测只运行本地代码。
没有付费交通数据、预测 API、第三方 submission、手工隐藏标签或组织方私有资源。
