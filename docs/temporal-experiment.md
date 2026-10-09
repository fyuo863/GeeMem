# 查询绝对时间解析实验（方案 B）

共同基线：f7095cc。分支：codex/temporal-absolute-query。方案 A 位于同基线的 codex/temporal-soft-score；本分支不包含 A 的加分模块。

配置：RAG_TEMPORAL_MODE=on、RAG_TEMPORAL_EXPERIMENT=absolute_query；参数来自根目录 .env。默认 off。

/search 新增可选 reference_time（Unix 毫秒）及 reference_timezone（IANA 时区，默认 UTC）。这两个字段向分区/子查询继承；不提供时不猜测相对日期，不使用系统当前日期。原 /search 返回内容不变。

与 A 共用同一程序解析器和时间口径。支持明确年月日、昨天/今天/明天、last weekday、上周几、数字/英文 one 到 twelve 的天/周前、过去 N 个月、上周/上月/去年。缺少年份、多个约束、比较/否定表达等暂不强行解释。last weekday 指严格前一个该星期几；中文上周几指上一日历周。区间按日历计算，月底截到有效日期。

解析成功后保留原问题，追加简短 Time constraint，包含绝对日期/区间和星期。向量查询、BM25、重排均使用该同一文本；不会改写或重嵌入已保存的原文，不修改调用者的请求对象。解析失败退回原始查询；本分支没有时间匹配加分。

测试对照：两组均是已开启时间增强的共同基线，只有实验组启用 absolute_query。使用相同 60 道完整会话、text-embedding-v4/MiniLM/400候选、Top10、关闭多跳与分区、无新增 LLM 标注。数据集 question_date 转为 UTC 参考时间，双方都传入。改写查询有独立真实 embedding，预热后记录检索耗时；不能据此报告含 API 延迟的端到端速度。

注意：本次60题只有4题可以被当前保守规则确定时间约束。结果需同时报告全部题、时间题和触发题，不能把未触发的回退当成规则泛化成功。官方请求若不提供参考时间，相对时间改写不会自动生效。源时间索引与查询应使用一致日历口径（本轮均 UTC）。

实测结果：时间20题 Recall@5 68.67%→78.67%、Hit@5 85%→95%；Recall@10保持84.83%。全60题2题Top5改善、58题持平，无指标退化。全量单测298项通过。比较报告见 temporal-strategy-comparison-20261009.md，原始数据见 data/temporal-sequence-live/20261009-absolute-query60。
