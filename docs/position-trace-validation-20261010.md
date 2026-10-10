# 检索位置日志与典型测试（2026-10-10）

本次仅增加可观测性，未修改召回、排序、审核提示词或证据链策略。

## 位置记录

位置均从 1 开始，未出现表示该阶段未包含该 ID。记录存在不等于语义支持成立。

- `retrieval_queries`：每次底层调用的独立 `call_id`、Top-K 与位置，并行分支不会覆盖。
- `positions.candidate`：候选池顺序，不能当成 reranker 分数排名。
- `positions.rerank`：重排完成后的位置；`reranker_applied=false` 时只是未重排顺序。
- `positions.post_policy`：时间融合、时间排序及配额策略之后的位置。
- `positions.selected` / `returned`：底层截断选中、经过后续处理实际返回的位置。
- 每轮 `positions.review_input` / `model_supports` / `validated_supports`：送审、模型选择、程序校验后的来源位置。
- 请求级 `positions.fused` / `pre_bundle`：融合排序、适用性处理后打包前排序。
- `positions.final_results` / `final_source_positions`：最终响应的位置；证据包同时记录结果位置与包内来源位置，在状态选择和披露处理后记录。

详细 trace 仍由调用者接收；本机根 `.env` 已配置 `RAG_POSITION_LOG_PATH=data/logs/search-positions.jsonl`。可将该项置空关闭持久化；`.env.example` 默认为空。日志每个文件上限 10 MiB，保留三个轮转文件。持久化事件仅含来源 ID、位置、阶段、调用编号及状态，不写问题、原文、模型引用或凭据。测试完整 report 仍包含原文，不能与精简持久化日志混淆。

没有新增模型调用。日志仍有序列化和磁盘开销：此次两条请求日志共约 543 KB，不适合无限期保留。未部署服务器。

## 真实测试

真实模型/embedding/reranker，隔离数据库，FastAPI TestClient `/search`。沿用证据链试验配置，Top-K=10；原问题底层取12条，子查询取20条，审核窗口12条。

执行：

```powershell
python scripts/test_evidence_chain_live.py --cases dataset_51 dataset_75 --only-on
python -m pytest -q -p no:cacheprovider
```

两次测试目录：`data/evidence-chain/20261010T094208Z`、`data/evidence-chain/20261010T094417Z`。两次目标来源的阶段位置一致。后一次同时确认两条请求成功写入轮转日志。完整轨迹在 `report.json`，重点位置表在 `position_summary.json`。

| 目标原文 | 调用 | 候选位置 | 重排位置 | 底层 Top-K | 底层返回位置 | 审核可见 | 最终返回 |
|---|---:|---:|---:|---:|---:|---|---|
| 宠物 D2:24 | 原问题 | 108 | 26 | 12 | 无 | 否 | 否 |
| 宠物 D2:24 | 子查询 | 134 | 24 | 20 | 无 | 否 | 否 |
| 冲浪 D29:34 | 原问题 | 282 | 20 | 12 | 无 | 否 | 否 |
| 冲浪 D29:34 | 子查询1 | 336 | 19 | 20 | 19 | 否 | 否 |
| 冲浪 D29:34 | 子查询2 | 283 | 19 | 20 | 19 | 否 | 否 |
| 冲浪 D29:34 | 子查询3 | 317 | 18 | 20 | 18 | 否 | 否 |

这些记录中的 post_policy 位置与 rerank 相同，因此没有证据表明本次是时间/配额策略将其压出。

## 定位结论

1. 宠物题：D2:24 的“一年前买的”进入候选池，也完成重排，但排在单路12/20条返回上限之外，尚未到多跳审核层。
2. 冲浪题：D29:34 的“下个月”在子查询中已返回多跳层，但始终落在12条审核输入之外。不能再把它概括成底层没有召回。
3. 两条目标原文都未送审，不能将其缺失归为模型看过后错误拒绝；模型在其他可见片段上的判断错误仍是另一类问题。
4. 下轮修复应分别验证上下文成组重排/有界邻接补查，以及审核窗口的新证据保留机制。本轮没有调整这些算法，所以未宣称召回质量提升。

最终自动测试：386 passed（27.18秒）。新增测试覆盖真实排序位置、并行分支隔离、最终证据包成员位置，以及日志不写入问题和原文。
