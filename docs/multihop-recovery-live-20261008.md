# 多跳恢复真实场景测试

测试脚本使用根 `.env` 的真实凭据和模型，但覆盖为独立配置：

- `RAG_MULTIHOP_MODE=llm`
- `RAG_MULTIHOP_RECOVERY=on`
- `RAG_MULTIHOP_BINDINGS=on`
- `RAG_MULTIHOP_NEEDS=on`
- `RAG_MULTIHOP_VALIDATION=strict`
- BGE ONNX reranker，目标原文模式
- 独立数据库，不读取或修改已有记忆库

测试 4 题：两道依赖链、一道独立拆分、一道直接问题。共 4 次 HTTP `/add` 和 4 次 `/search`，
全部返回 200；8/8 条标注原文被返回，未触发 fallback。

| 问题类型 | 策略 | search_calls | 恢复变体数 | 命中证据 |
|---|---|---:|---:|---:|
| spouse → employer | chain | 3 | 1 | 2/2 |
| spouse → employer → CEO | chain | 3 | 1 | 3/3 |
| bicycle + tent | split | 6 | 3 | 2/2 |
| direct residence | direct | 1 | 0 | 1/1 |

两道 chain 题实际 trace 例如：

```text
原问题
→ Who is Leona married to?
→ Who is Leona married to? Context question: Where does the spouse of Leona work?
```

变体与原始首跳并行，证据进入同一候选池；最终分别停止于 `sufficient`。
split 题因为恢复变体消耗了查询预算，停止于 `query_budget`，但两条价格证据仍已命中。
这说明恢复宽度必须和 query budget 一起调，不能无条件增加查询数量。

平均搜索耗时约 7.78 秒；该值包含真实 gpt-4o-mini 路由/审核调用、BGE 重排和 embedding，
不代表生产吞吐。结果及逐题 trace：`data/multihop-recovery-live/20261008T130957Z/results.json`。

另外修复了 SearchService 传递空 trace 时使用 `trace or {}` 丢失调用方 trace 的问题，
现在真实请求可记录 strategy、history、recovery_queries、recovery_sources 和 stop。

局限：本次是人工构造英文小样本，且 chain 证据同时可能被原始问题召回；没有证明恢复变体
在真实长历史数据上提高 Recall。下一步应在固定多跳集上对比 recovery off/on，控制相同总查询预算，
分别统计首跳命中、全链证据命中、错误路径和延迟。
