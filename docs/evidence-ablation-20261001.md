# 分支实现与八组合对照实验（2026-10-01）

## 目的与比较约定

目标是提高固定十条原文预算的 Hit@10。主指标 Hit@10；并列时依次比较 Micro Recall@10、宏 Recall@10、Search P50。所有方案同时报告全证据命中、Hit/Recall@5 及延迟，不通过增大返回条数提高成绩。

共同基础提交 `47d9fab` 从 `v1` 的 `b80c405` 提取原有打分函数，未改变排序规则，基础接口/检索相关 32 项测试通过。

| 标记 | 独立分支 | 改动 |
|---|---|---|
| A | `codex/evidence-metadata` | 调用者可选提供 speaker 和 session_timestamp，保存来源索引，供重排使用 |
| B | `codex/evidence-target-rerank` | 目标句独立打分的有界修正，及针对提问/纯寒暄的软降权 |
| C | `codex/evidence-second-pass` | 按需执行一次原查询与有来源线索驱动的全局补检 |
| 比较 | `codex/evidence-factorial` | 合并三条实验分支，分别开关，运行 base/A/B/C/AB/AC/BC/ABC |

三项均有独立实现提交和单项单元测试。第四项“不采用强制分散/会话配额”落实为所有组合均无这些规则，不额外建立无代码分支。

## A：可靠元数据与原文账本

`/add` 保持原有字段；新增可选 `messages[].speaker`、请求级 `session_timestamp`（Unix 毫秒）。speaker 必须由调用者可靠提供，未提供时未知。role 保留为 user/assistant，不当作真实人名。

新 `rag_sources` 表按 chunk 关联 `memory_id`，存放 role、speaker、session_timestamp、程序生成的 source_id、session 内按写入顺序递增的 source_index、字符区间 `[char_start,char_end)`。字符位置按 Python 字符索引计算，不是 UTF-8 字节位置。source_id 根据 user/request/message_index 生成；chunk ID 保持原来的生成方式。单条原文分多个 chunk 时 source_id/source_index 相同，各有准确字符位置。

旧表不删除、不改写，初始化仅回填可由旧记录确定的 ID 和位置；旧数据未知的 speaker、role、时间、字符区间保留 NULL。未提供新字段的旧请求保持原有幂等摘要，新元数据变化会触发冲突。迁移为新增来源表，应备份数据库再升级；回退旧代码可忽略该表，旧代码写入的数据再次升级时仅能回填可确定信息。

`RAG_METADATA_MODE=on` 只改变给重排器的检索视图：`speaker said, on date: 原文`，附带明确 yesterday/tomorrow/last month/next month 的日期解释。相对时间只作为索引辅助，不视作抽取的新事实。消息 timestamp 优先于 session_timestamp。向量索引仍编码原文，返回 content 仍是原始 chunk，不含前缀、上下文或时间推导。

LoCoMo 数据源给出姓名与会话时间，评测适配器显式传入；无时区信息统一按 UTC 保留日历日期，这只是本地基准的约定。线上若没有这些元数据，不能预期获得相同提升；系统不会读 QA 的 gold 来补人物或日期。

## B：目标消息归因

用同一冻结 CrossEncoder 分别打分上下文视图 `context` 与目标句视图 `target`，修正为：

`score = context + 0.25 * clip(target - context, -2, 2)`。

有界修正让省略主语/指代的短回答仍保留上下文的收益。随后照常进行单跳邻句支持；最终对不超过 35 个词、以问号结束的候选软减 0.6 分（query 明确问“问了什么”时不罚），对完整匹配的短寒暄模板软减 0.6 分。包含事实的 “Thanks, I moved to Paris.” 不按寒暄过滤。没有硬过滤、session 配额或手工数据集人名规则。

人物/日期区分主要依赖 A 提供的可靠视图和 CrossEncoder，B 不会凭人名出现位置断言事实归属。规则覆盖有限，不能宣称已经解决所有间接表达。

## C：一次按需全局补检

若第一阶段最高 logit 小于 4、第一与第十候选分差小于 2，或问句有时间/列表线索，则允许补检。这是冻结的启发式触发条件，logit 不是概率。

对 top3 中分值不低于 4 的原文，提取至少在两个来源中出现的规则关键词，最多取三个稀有线索词。它们是“多来源出现”的线索，不代表已完成语义真伪验证。

以原查询及其扩展查询分别对同 session 前后两条消息组成的片段运行 BM25/RRF，保留全局原查询通道，最多加入 40 个之前不在候选池中的消息。新消息用原问题重新重排，与旧候选共同排序，最终仍返回十条原文，不拼接整段绕过 K 的限制。不要求候选来自不同 session。

## 实验协议和验证

- 固定 BGE small 与 ms-marco MiniLM-L6 本地快照；GPU RTX 4070；基线候选 400、context=1、单跳邻句支持 penalty=2、result window=0。
- 10 个公共 LoCoMo-Refined 会话，5,882 条消息，859 道有效纯文本证据题；多模态题和无 gold 题不计入本轮。
- 同一次完整写入保存公共数据库，三开关做 2×2×2 因子对照；特性关闭时不使用对应行为。
- 精确 query/document 模型打分可复用，只用于准确率；分值以文本 hash 持久化。实现、配置、源数据有指纹，配置/代码变化不能续写同一目录。
- 延迟另用 30 个固定问题（每个会话源顺序前三题），不复用打分/查询缓存，轮换八个方案的运行顺序。计时包含 SQLite、查询 embedding、候选检索、所有重排和选择，不含网络传输；均为模型加载后的暖态。
- 记录十条原文的逐题 ID；检查内容一致、用户隔离、不重复、不超过 K。另做 HTTP Add/Search 八组合测试及真实模型 API 对照。
- 数据已多次被观察，本轮选择的是此公共测试集上的最优方案，不是未知官方评测成绩。旧开发/验证划分只作分组报告，不再声称盲测。
- 描述性不确定性通过按会话聚类的 10,000 次 bootstrap；只有十个会话，区间不能视为充分的泛化证明。

正式结果使用 `data/factorial-benchmarks/20261001-v2/`。之前 `20261001/` 是未完成预跑；代码复查后修复寒暄大小写识别，重新独立跑正式目录，不混合两次结果。

## 复现

在 `codex/evidence-factorial` 上运行：

```powershell
python scripts/benchmark_factorial.py --out data/factorial-benchmarks/reproduce-abc
python scripts/benchmark_factorial.py --out data/factorial-benchmarks/reproduce-abc --latency
python scripts/summarize_factorial.py --out data/factorial-benchmarks/reproduce-abc
```

可在相同实现/配置下续跑中断的准确率任务。计时必须使用 `--latency`，不要将缓存准确率任务的耗时作为检索延迟。


## 全量结果与合并决策

A=元数据；B=目标句重排；C=按需补检。以下百分比按同一批 859 题计算。

| 方案 | Hit@10 | Recall@10 | Micro Recall@10 | 全证据@10 | P50 / P95 (ms) |
|---|---:|---:|---:|---:|---:|
| base | 82.42% | 78.822% | 74.02% | 75.44% | 193.4 / 214.2 |
| A | 92.08% | 89.076% | 84.54% | 85.56% | 233.8 / 250.7 |
| B | 82.42% | 78.880% | 74.12% | 75.55% | 315.9 / 338.8 |
| C | 82.54% | 78.968% | 74.21% | 75.55% | 305.3 / 348.2 |
| AB | 92.67% | 89.998% | 85.75% | 86.85% | 361.7 / 387.1 |
| AC | 92.20% | 89.164% | 84.64% | 85.68% | 335.2 / 421.7 |
| BC | 82.65% | 79.142% | 74.39% | 75.79% | 435.4 / 490.9 |
| ABC | 92.78% | 90.056% | 85.75% | 86.96% | 482.7 / 583.8 |

**按预先声明的 Hit@10 优先标准，选择 ABC 合并回 v1。** ABC 命中 797/859，比基线 708/859 净增 89 题（+10.36 个百分点），98 题新增命中、9 题丢失；十条原文预算不变。

- A 是主要贡献：单独带来 +83 道净命中。A→AB 再净增 5 道，AB→ABC 再净增 1 道。
- B 单独 8 胜/8 负，Hit 无净提升；C 单独 2 胜/1 负。因此组合收益不可简单相加。
- ABC 相比 AB 只新增命中 conv-42 (57)：询问 Joanna 何时计划去 Nate 家看乌龟，找回 D28:32。Micro Recall 相同，不能把这 1 道变化说成全面提升。
- 准确率代价明确：ABC P50 482.7 ms，相比基线 193.4 ms 约 2.50 倍；仅 A 为 233.8 ms，已达到 92.08% Hit。需要较低延迟时可以关闭 B/C，但这不是本次“最高 Hit”所选的默认实验配置。
- ABC 宏 Recall@10 为 90.056%，确实超过 90%；AB 的 89.998% 只是四舍五入后显示 90.00%，仍略低于 90%。ABC 的 Micro Recall@10 仍为 85.75%（921/1074），尚未实现按总证据计的 90% 覆盖。
- 原开发组 Hit 从 87.55% 到 92.53%；原验证组从 80.42% 到 92.88%。这些数据已被观察，本轮不得称为独立盲测。
- 按会话聚类 bootstrap 得到 ABC 相对基线 Hit 差值的描述性 95% 区间为 +7.36 至 +13.45 个百分点；它没有校正多方案选择，也不替代新外部测试。

Top5：Hit 从 73.57% 到 87.78%，Recall 从 69.13% 到 84.08%，Micro Recall 从 63.78% 到 78.58%，全证据命中从 64.96% 到 80.33%。

多证据题仍较困难：154 题的宏 Recall@10 从 58.50% 到 71.81%，全证据命中从 39.61% 到 54.55%。未来应继续优化跨轮/多条证据，不宜因为总 Hit 超过 90% 就停止诊断。

此前分析的 Caroline 婚恋状态、John 三月参加的活动、Jolene 发言活动、Evan 盆景含义、James 订票国家等案例已找回标注证据；冲浪约定时间、反复丢钥匙、同义替代证据案例仍未命中 gold，保留为后续失败案例。

## 启用方式

以下值写入项目根 `.env`，不读取系统环境变量；模型快照需已下载。API 使用包含 speaker/session_timestamp 的新写入才有相应可靠信息，旧数据不会自动凭空补全。

```dotenv
RAG_METADATA_MODE=on
RAG_TARGET_MODE=on
RAG_SECOND_PASS=on
RAG_RERANK_MODE=local
RAG_RERANK_CONTEXT=1
RAG_RERANK_CANDIDATES=400
RAG_RERANK_SELECTION=context_support
RAG_RERANK_NEIGHBOR_PENALTY=2
RAG_RESULT_WINDOW=0
RAG_TAG_MODE=off
RAG_DEVICE=cuda
RAG_RERANK_DEVICE=cuda
```

CPU 环境可把两个 DEVICE 改为 cpu，但不得沿用本报告的 GPU 延迟数字。保存完配置后，新建服务实例读取新配置；已经运行的实例不自动热更新。

已有正式输出目录绑定当时配置指纹。修改 `.env` 后如触发指纹不一致，应使用新的 `--out` 目录重跑，不修改旧报告的指纹来强行续跑。

## 最终验证证据

- 全套 102 项测试通过；覆盖幂等、兼容迁移、重复内容的字符偏移、来源序号、元数据冲突、日期推导、跨用户/会话边界和八组合 API 合约。
- 基线 859 道 top10 与此前冻结版本逐题完全一致，确认公共基础重构没有悄悄改变基线。
- 240 次无分值缓存的实际模型搜索与八组合准确率结果完全一致；另 16 次实际模型 HTTP `/search` 校验也全部一致。
- 切回 A/B/C 各独立分支，用冻结的真实模型分值分别重放全部 859 题；三条分支各 859/859 top10 与组合分支对应单项完全一致，缺失打分键会直接失败而不会回退。提交分别为 `0e8f565`、`b8cdf17`、`1dab910`。
- 完整 `/add` 写入 5,882 条消息，来源表中的 speaker、session_timestamp、字符区间全部存在；同 user/session 的 source_index 没有指向多个源消息。一次 GPU 初始写入总计约 10.27 秒，包含逐批数据库写入；该数字不是八组 Add 吞吐对照。
- 汇总、置信区间和分组指标见 `docs/evidence-ablation-results-20261001.json`；原始逐题输出、模型分值、输入/实现指纹保存在忽略目录 `data/factorial-benchmarks/20261001-v2/`。

本地根 `.env` 采用 ABC + CUDA 配置；`.env.example` 提供相同算法、CPU 设备的可移植默认模板。需要运行新服务实例才会读取配置变更。实验分支保留用于后续对照。

根 `.env` 选中配置已通过真实模型 HTTP `/add`→`/search` 冒烟：重复写入成功、元数据变化返回 409、原文含空格保持一致、未知用户无结果、来源身份与索引正确。实验数据库与 `.env` 均被 Git 忽略，未写入提交。
