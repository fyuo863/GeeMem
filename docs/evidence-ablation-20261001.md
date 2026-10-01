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
python scripts/benchmark_factorial.py --out data/factorial-benchmarks/20261001-v2
python scripts/benchmark_factorial.py --out data/factorial-benchmarks/20261001-v2 --latency
python scripts/summarize_factorial.py --out data/factorial-benchmarks/20261001-v2
```

可在相同实现/配置下续跑中断的准确率任务。计时必须使用 `--latency`，不要将缓存准确率任务的耗时作为检索延迟。
