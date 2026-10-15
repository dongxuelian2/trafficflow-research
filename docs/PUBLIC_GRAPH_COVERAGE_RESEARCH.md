# 公开测量图覆盖与注意力范围：A128/A129

正式最佳仍0.82940。这里没有训练候选或新增预测成绩；仅检查公开静态图及现有实现的信息范围。

A128只读网络CSV、首个public TRAIN分区的link_id/milepost列，按当前neighbors()计算方向图。A129不用该helper，按CSV直接ID边和union-find独立核对全部10个图；当前每个topology行都是测量link，直接过滤与遍历一致。两者耗时.92/.36秒。

| 方向 | 测量节点 | 双方向均无可达测量邻居 | 无向分量 |
|---|---:|---:|---:|
| D12_I405_N | 89 | 12 | 19 |
| D12_I405_S | 44 | 27 | 34 |
| D12_I5_N | 258 | 0 | 1 |
| D12_I5_S | 100 | 46 | 70 |
| D7_I10_E | 146 | 52 | 61 |
| D7_I10_W | 107 | 61 | 79 |
| D7_I210_E | 297 | 5 | 7 |
| D7_I210_W | 81 | 53 | 66 |
| D7_I405_N | 352 | 18 | 20 |
| D7_I405_S | 109 | 80 | 94 |

外部link引用不在公开测量表中，不能擅自补为隐藏道路状态；图不连通也不是reader错误的证据。里程邻近或order_index相邻不自动证明物理道路连接。

## 现有注意力已全局通信

最初看到I210_W的53个孤立点时，提出它可能限制图信息传播。源码进一步检查明确：GraphBlock的定向聚合依赖上述图，但AdaptiveSpatialBlock的attn_mask是**有限relation bias**，不是用−inf排除非邻居；注意力已经全连接。因此不能把孤立数量当成“模型无法读取其他节点”的证据。

小型CPU探针将3节点的两方向邻接全置0，9个注意力mask元素全部有限，节点0输出对另一孤立节点1的输入仍有非零梯度，L1=.6854989528656006。固定seed20261015、hidden48/heads2、输入shape(1,48,2,3)、目标output[0,:,0,0].sum()；这只是实现范围证明，不是预测增益。源码及原结果SHA记录于A128_attention_scope_note，复现命令：

```powershell
.venv-repro\Scripts\python.exe scripts/audit_public_graph_coverage.py --output reports/research/A128_public_graph_coverage_audit.json
.venv-repro\Scripts\python.exe scripts/verify_public_graph_audit.py --study reports/research/A128_public_graph_coverage_audit.json --output reports/research/A129_public_graph_independent_verification.json
.venv-research\Scripts\python.exe scripts/probe_global_attention.py --note reports/research/A128_attention_scope_note.json
```

旧目录禁止覆盖，重跑换新名。原attention没有显式相对里程bias；输入base/profile可能间接包含空间信息，不能说完全缺位置。该假设后由A130–A133完成I210小对照，取得局部State收益，见[相对里程研究](RELATIVE_POSITION_STATE_RESEARCH.md)。跨方向结果仍未知，不冒充真实拓扑修复或官方LWR约束。
