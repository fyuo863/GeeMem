# 检索职责与专类问题

## 已实现的统一底层

`AtomicRetriever.query(AtomicQuery(...))` 是统一底层入口。四类 SourceRetriever
只选择类型并转发请求，不再维护第二份向量/BM25 排序代码。
实际算法仍由 VanillaMemory._search_direct 实现，专类查询也使用配置的 reranker。

```python
from memory.atomic_retriever import AtomicQuery
result = backend.atomic_retriever.query(AtomicQuery(
    user_id='user-001', query='小林喜欢哪支足球队？',
    memory_types=('profile',), session_id=None,
    fallback=True, include_evidence=True, top_k=10,
))
```

- memory_types 为空表示全库，多个类型取并集；同一片段不重复返回。
- fallback=False 只允许指定类型进入候选与结果，包括结果窗口扩展。
- fallback=True 允许全库参与，指定类型获得一次轻量候选排名加权，重排仍按相关性。
  不是结果不足时才查全库，也不保证某个类型占固定名额。
- session_id 是确切会话过滤；user_id 始终隔离。
- include_evidence 附加来源、说话者、消息时间、匹配类型及索引上下文。
  没有类型索引的补充结果可能没有附加上下文。

SourceRetriever 构造参数现在是 `(atomic_retriever, memory_type)`；
EventRetriever 接受 atomic_retriever。推荐使用 backend.<type>_retriever。
不再支持为专类检索器单独传入数据库和 embedder，避免两套配置漂移。

外部 /search 请求响应不变，自动问题路由尚未接入，仍默认全库原子检索。
本次不增加模型调用、自动拆题或答案生成。

## 专类问题的上层方案（待实现）

复用通用选择器，让模型只选择 profile/relationship/rule/event/general，允许多选。
第一版保留原问题，无法确定类型时用 general；低置信度允许全库补充。
用户 ID 由请求绑定，不能由模型改写，不要求模型凭空生成实体 ID 或时间。

| 问题 | 渠道 | 底层查询 | 上层额外职责 |
|---|---|---|---|
| 小林最喜欢哪支球队？ | profile | 原问题 | 确认目标人物 |
| 小林的老师是谁？ | relationship | 原问题 | 区分关系方向及同名人物 |
| 小林的老师在哪里工作？ | relationship + profile | 查老师，再查该人的职业 | 用前一步原文绑定中间人 |
| 写报告有哪些格式要求？ | rule | 查要求与例外 | 判断适用主体、条件和例外 |
| 小林参加过哪些比赛？ | event | 查比赛相关原文 | 汇总去重；Top-K 不保证列表完整 |
| 小林最近一次比赛在哪？ | event | 查赛事及时间 | 比较事件发生时间，不能用消息时间替代 |
| 小林现在住哪里？ | profile + event | 查住所与搬家 | 结合更正和时间判断有效值 |

类型索引不保证昵称消歧、关系链推理或时间有效性。先建立只选渠道的基线，
测试渠道召回率、Hit@10、Recall@10 与延迟；再逐步增加带证据的上层判定。
