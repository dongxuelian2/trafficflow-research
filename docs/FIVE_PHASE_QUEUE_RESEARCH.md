# 新方向 Queue 五阶段研究与条件采样诊断

正式已测 .82940，.95000+仍未达到。研究只用公开TRAIN观测，无新平台动作。

## 时间与模型机制

| fit_end | checkpoint（权重选择） | decision（阈值/参考） | selection（固定challenger gate） | assessment |
|---|---|---|---|---|
| 2030-10-15 |10-16…10-30 |10-31…11-29 |11-30…12-14 |12-15…2031-01-13 |
| 2030-11-30 |12-01…12-15 |12-16…2031-01-14 |01-15…01-29 |01-30…02-28 |

fit统一2030-07-16开始。每机制内日期互斥，全部checkpoint权重先封存才读取decision，决策先封存才读取selection，gate先封存才读取assessment。三个optimizer seeds20261013/20261029/20261107分别训练30min CausalGraphQueue，20×80/batch4/AdamW.001/wd1e-4/clip1；checkpoint仅按前15日Brier每五epoch比较。未将旧cal-selected权重切片包装成独立选择数据。

参照为同fit的plain600/63树与合法occupancy/provided-condition树；decision期先选择更强参考。三个图与三seed平均均固定α.25，不扩张平均权重网格；完整预测域先决策后scorer mask。后15日只在已定challenger比incumbent gain≥.01且paired95CI下界正时替换，没有重调参数。全部窗口有未来监督标签，但图/树features只来自≤T历史，T只读公共masked partition。

## 原方案失败与修订

原C55秋训练完整、CP只有1 onset，最后plain tree .840835；原C56冬在CP选0 onset而退出1，尚未读decision/selection/assessment，原C57/C58未启动。保留原zip/注册/模型/status/失败JSON。C59重新检查同样已读CP：15 onset/944 ongoing候选；总pool permutation被6h spacing截断后选0/22。不是候选不存在。

新可选condition-stratified proxy按各condition候选独立随机排列，轮换较少已填quota，保持global spacing。相同CP/seed选9/13，新的I5_S秋CP选10/11。该方法缓解我们自己的condition starvation，不是组织方selector；仍筛观察正例，latent/observation偏差没有解决。默认行为保持原RNG和metadata，两个旧causal cache的8个cal/assessment sampler及eligible labels精确复现。

四个balanced-v2新研究在相同worker/sampler SHA下预注册并另建目录，公开日期仍是开发数据；四fold/方向的日历跨研究重叠，不能称四个独立盲测。

## 结果与限制

| case | incumbent/selected | assessment proxy | fixed graph mean / paired gain CI |
|---|---|---:|---|
| I5_S秋 | rich tree | .781151 | .788512 / +.007361 [−.002108,.019077] |
| I5_S冬 | plain tree | .797255 | .789825 / −.007431 [−.045926,.033042] |
| I10_W秋 | rich tree | .557659 | .604862 / +.047203 [−.012979,.100827] |
| I10_W冬 | rich tree | .814942 | .815249 / +.000306 [−.003037,.003516] |

所有decision winner都是incumbent。后段gate为零比较，数据读序有效，但没有非零替换案例，不能据此证明该gate能降低过拟合。图均未选中；不根据assessment把I10_W秋的+.047203改为primary。其CP仅3 onset，I10_W冬也3；样本与事件标签信息不足仍是限制。

原I5_S秋 .840835 与新 .781151 的人群不同。C60将两份冻结selected放到同样新balanced assessment，原.781706 vs新.781151，差−.000555、CI[−.025070,.018913]。不能宣称采样改动提高或大幅降低模型性能；此控制改变fit/checkpoint/decision/selection的策略整体，不是单个梯度效应。

## 复现与完整性

新campaign命令：`.venv-research\Scripts\python.exe scripts/run_queue_five_phase_campaign.py --balanced-v2`。已注册目录不能覆盖；复现须新campaign名/输出并保留原配置。`summarize_queue_five_phase_campaign.py`验证四worker及sampler源SHA相同；`verify_queue_five_phase.py`验证互斥日历、全部seal读序、独立构建plain/rich features/9channel输入及四个导出角色的score/binary。

15个完整NN/10个tree训练模型保留，失败另2个fit-only tree。四v2 wall316.33/302.96/141.75/154.03秒、GPU allocated81.85/83.08MiB；与部署/测试部分并行，不将sum当整体elapsed。5份完成研究共608个decision role/window调用精确，其中原156/修订452，角色及事件重复不算独立统计样本。83 research passed；CPU71 passed/4 optional Torch skips。

Formal Queue/Physics/OD/Total的新增收益均未知；原正式完整文件、K06以及其他候选保持冻结，final simulator seeds未生成，无模型晋级、上传或推送。下一步需要真正的事件信息或latent/observation模型突破，并转回State动态与概率OD探索；不继续把采样和基础设施本身当高分成果。
