# 文本全量成绩偏低：归因诊断

## 结论

首要原因是输入与重排配置发生变化：说话者身份被测试适配丢弃、上下文重排关闭、原目标评分与邻句支持未启用。不是仅凭本轮就能归因于gpt-4o-mini理解能力差或text-embedding-v4质量差。此前约90%的历史ABC成绩与本轮54.55%并非同条件复测。

上一轮报告已说明未传speaker，但漏报了`RAG_RERANK_CONTEXT=0`及目标融合关闭的关键配置差异；因此不应把分数下降直接解释成新旧算法能力下降。本诊断补充这一点，不改写原测试结果。

## 受控实测：40题固定候选池

从全部10组对话各按固定哈希选4道有金标纯文本题，不按是否答错选题。使用原5,882条原文及其原v4向量、相同400候选预算、同一个GPU MiniLM；不调用LLM，不重新嵌入，不将金标输入评分器。数据库使用独立副本。

说话者条件仅恢复公开原始对话自带的speaker，不生成或猜测姓名；上下文取同会话前后各一条。各变体使用完全相同候选ID集合，金标只用于结果评分。它是原因诊断，不是新的全量上线成绩。

|重排输入/配置|Recall@10|Recall@100|
|---|---:|---:|
|本轮原配置：无说话者、无上下文|55.42%|84.58%|
|仅增加前后上下文|77.08%|93.75%|
|仅恢复说话者|80.83%|93.75%|
|说话者 + 前后上下文|85.00%|96.67%|
|再启用目标评分 + 邻句支持|87.50%|96.67%|

原候选池的宏平均证据覆盖为96.67%。相当多证据已进入重排池，却在孤立句评分时被压到Top10甚至Top100以外。各提升不能相加：身份和上下文信息存在重叠贡献。最后一行不是完整复刻历史ABC，仅为当前已有目标评分、邻句支持的诊断组合。

## 具体原因与证据

### 1. 测试适配删除说话者身份

`validate_text_release.py`将speaker A/B映射成user/assistant，content只保留原text，没有传递speaker，也没有把姓名放进文本。实查全部5,882行rag_sources，speaker非空数为0。

LoCoMo的两位参与者都是人物，并不天然等同于真实用户和AI助手。将人物改成通用role后，“我”的身份信息消失；关系/画像分类也可能受到影响，但本轮没有单独量化构建器归属误差。

例如“What is Jon offering to the dancers at his dance studio?”（Jon给舞蹈工作室的舞者提供什么？），证据D13:7原文为“Besides the dance classes and workshops, I'm offering one-on-one mentoring and training to help dancers reach their full potential.”。原配置排名26，只恢复原说话者Jon后排名1。

官方标准字段不支持自定义speaker不意味着应直接丢弃已知身份；有合法身份信息时，可在符合协议的输入适配或内部索引中保留。但若官方本来只提供role，不能假定生产时能额外获得公开数据集speaker。本实验恢复姓名的收益以身份确实可得为前提。

### 2. 重排配置偏离原高分路线

历史`data/abc-text-benchmarks/20261001T_textonly/manifest.json`记录context=1、target=on、selection=context_support、second_pass=on。本轮manifest为context=0、selection=direct，未设置target，代码默认off。本轮也更换了embedding、上层检索路径和请求top_k，不能把两个总分直接相减做单因素归因。

例如“What inspires Joanna to create drawings of her characters?”（什么促使Joanna绘制人物角色？），证据D25:8是“Thanks, Nate! They're visuals of the characters to help bring them alive in my head so I can write better.”，前一句才明确问到drawings和inspired。孤立句排名82；只补相邻上下文后排名1。

例如“What workshop did Caroline attend recently?”（Caroline最近参加了什么工作坊？），D4:13明确讲LGBTQ+ counseling workshop，但原文只有“I”而没有Caroline。原排名68；上下文条件排名3，姓名条件排名1。

### 3. 上层多跳再次使用缺少元数据的证据

`MultiHop.packet()`给审查器传id/content/created_at，未传speaker和邻句；默认只展示12条证据。最终fusion再次对`pool[i]['content']`调用reranker，不使用底层的带身份/上下文评分文本。因此即使底层后来恢复信息，上层仍可能再次丢失这些信息，需要一起处理。

全量对照本身已说明：无LLM上层的直接检索Recall@10为53.22%，完整流程为54.55%。低分在底层就已存在，不能主要归咎于13次LLM回退。

77道数据集类别1题中，62题被规划为direct、8题chain、6题split、1题规划未成功。但数据集“多跳”标签不必然要求拆分检索，不能把62题全部认定为误判。原全量仅保存了压缩trace，缺少完整规划/审查输入输出，不能从现有日志准确归因每一道上层失败。

### 4. 时间查询格式覆盖不足

861题中112题含英文月份；`resolve_query_time`在无reference_time时对全部原问题均返回None。查询入口PATTERN支持ISO和中文完整日期，但不支持英文月份日期，所以不仅是缺少提问时间。

实际原题“What community service did Maria mention that she was involved in on 31 July, 2023?”解析失败；将同一日期改为2023-07-31后无需reference_time即可解析成功。这里的日期格式缺口使目标窗口路径不触发；不能说整个时间模块完全没运行，因为TemporalRanker仍可能排序。

消息timestamp已写入，并非丢失了所有时间。缺少提问时间影响的是相对日期锚定，应与英文绝对日期不受支持分开修复。

## 建议的修复顺序

1. 固定符合赛事实际输入的适配口径；有据可查的身份和会话顺序不能丢弃，不让模型凭空补姓名。
2. 恢复并验证底层上下文重排、目标评分和邻句支持；保留原文返回，将增强文本限定为内部索引/评分材料。
3. 让多跳审查和最终重排复用同一证据表示，避免底层刚补的身份和上下文再次丢失；仍保持模型输入预算。
4. 补齐常见英文绝对日期解析，并分别测试有/无提问时间、日期筛选和事件时间语义。
5. 在相同模型、输入、top_k和候选预算下重新对照。40题归因结果不作为自动部署依据；本轮没有修改生产逻辑、配置、提交或部署。

原始诊断：`data/final-text-validation/diagnosis-20261009/cases.json`及`summary.json`；脚本：`scripts/diagnose_text_release.py`。固定样本哈希盐为`input-diagnostic-v1`。
