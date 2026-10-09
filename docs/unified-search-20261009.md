# 统一检索链路测试（2026-10-09）

SearchService 移除独立入口路由判断。规划器判断 direct/split/chain，每次查询经
PartitionSearch 后调用 AtomicRetriever，审核缺口后继续检索。原问题、子查询和
恢复查询共用此路径；每个分支的分区 trace 独立保存。

自动化回归：276 passed。覆盖独立分支并发、逐查询分区、请求范围保持、
分区错误回退全局；已有多跳来源校验、预算和回退测试继续通过。

真实 API 测试使用 FastAPI TestClient（非公网部署测试），gpt-4o-mini、
text-embedding-v4、BGE ONNX，隔离数据库。/add 开启分类与类型索引写入，
/search 开启多跳及 dual 分区。数据为 4 道人工构造题，非真实数据集抽样。

| 场景 | HTTP | 目标证据命中@10 |
|---|---|---|
| 配偶的公司 | 200 | 2/2 |
| 配偶公司的 CEO | 200 | 3/3 |
| 自行车与帐篷费用 | 200 | 2/2 |
| Mira 居住地 | 200 | 1/1 |

Hit@10、宏平均 Recall@10、Micro Recall@10、全证据命中@10 均为 100%。
平均 /search 耗时 10.38 秒，无 fallback。split 停于 no_grounded_new_query，
说明返回金标证据齐全不等于审核器认定充分。恢复查询没有触发。

结果：data/unified-search-live/20261009T023236Z/results.json。
旧恢复测试平均 8.11 秒，但本轮开启分类写入和分区，且非重复控制实验，
不能把耗时差或全命中解释为稳定性能提升。

限制：每次检索增加分区模型判断；分区最多三次尝试，不计入 MultiHop 的
LLM 预算。时间限制不会硬中断正在进行的网络请求。程序引用校验不保证
模型语义判断正确，小样本也不能证明复杂多跳泛化能力。
