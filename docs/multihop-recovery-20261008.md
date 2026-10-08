# 多跳首跳失败恢复

分支：`codex/multihop-recovery`。

## 实现

新增 `RAG_MULTIHOP_RECOVERY=off/on`。本机 `.env` 已开启，示例配置默认关闭。
开启后，标准 `long` 多跳模式的每条 chain 查询会保留原始子查询，并在预算内增加：

- 首跳：`子查询 + Context question: 原始问题`
- 后续跳：`子查询 + Evidence: 已检索原文摘要`
- 后续跳：`原始问题 + Evidence: 已检索原文摘要`

这些查询使用 `ThreadPoolExecutor` 并行调用同一个底层检索器。查询文本只拼接程序已持有的
原始问题或检索原文，不让恢复逻辑自行生成实体。每个查询仍携带原来的 source_id/bridge，
不会因为拼接上下文而绕过严格来源校验。

原始问题始终单独检索并进入候选池；所有变体结果去重后统一进入现有证据审核和最终重排。
因此第一跳所有变体失败时仍返回原问题基线，不会因为恢复逻辑把结果变为空。
`RAG_MULTIHOP_RECOVERY_WIDTH` 控制每个规划查询最多保留的变体数（2–4，默认 3）。
恢复查询计数和使用的上下文来源记录在 trace 的 `recovery_queries`、`recovery_sources`。

当前 focused 需求模式沿用自己的按需求执行器，尚未复用这组普通模式变体；配置默认 prompt_style
为 long。多跳开启时仍优先于 dual 分区路由，每个恢复查询直接调用 AtomicRetriever。

## 小范围测试

新增两个恢复测试，覆盖首跳原查询失败但上下文变体命中、所有变体失败仍保留 baseline。
全套自动测试：**272 项通过**。

测试构造了：

```text
问题：Where does the spouse of Alice work?
规划首跳：Who is Alice married to?
原始首跳：返回噪声
带原问题上下文的变体：返回 Alice is married to Bob.
下一跳：Where does Bob work?
返回：Bob works at Atlas.
```

验证结果：恢复变体被并行执行，`recovery_queries>=1`、`recovery_sources>=1`，
spouse 与 employer 两条证据均进入结果。第二个测试验证所有恢复查询失败时仍返回原始 baseline。

## 边界

这不是自由查询扩展：上下文变体不等于证据验证，不会自动把相关词当作已解析实体。
如果首跳没有任何可用证据，恢复只能扩大召回机会，不能凭空完成下一跳；下一跳仍需由审核器
从真实 source_id 中绑定实体。变体增加搜索次数和 reranker 成本，受原有 query/round/LLM/time
预算约束。开启前应在固定多跳集上比较首跳命中、全链证据命中、错误路径率和延迟。
