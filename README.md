# GeeMem：Agent 长期记忆与原文证据检索

GeeMem 通过同步 `POST /add` 保存对话，通过 `POST /search` 返回可追溯的原文证据，供上层 Agent 生成回答。当前使用 Agent Memory Leaderboard 文本赛道适配器，入口为 `memory.aml_api:app`。

检索器负责召回、选择和组织证据，不生成最终答案，不裁决因果链是否成立或哪个冲突事实有效，也不将模型生成的摘要当成原文返回。

## 当前 smoke 版本

| 项目 | 当前值 |
| --- | --- |
| 版本标签 | [`smoke-raw-audit-20261010`](https://github.com/fyuo863/GeeMem/tree/smoke-raw-audit-20261010) |
| 实现提交 | `04e6403`（部署版本以本表标签指向的提交为准） |
| 运行行为 | 原文证据包开启；候选翻页与分类缺口补查增强关闭 |
| LLM | `gpt-4o-mini`：分类、检索规划与来源选择 |
| Embedding | 百炼 `text-embedding-v4` API，1024维 |
| Reranker | 百炼 `qwen3.7-text-rerank` API |
| 存储 | SQLite，按 `user_id` 隔离 |
| 配置 | 只读取根目录 `.env`，不读取系统环境变量，不执行变量插值 |

本 README 从部署提交更新，文档提交不表示服务器运行代码改变。标签固定代码，运行行为还依赖 `.env`；复现时使用下文配置，不要直接沿用 `.env.example` 中的历史实验默认值。密钥不提交 Git。

## 接口地址与鉴权

| 接口 | 当前地址 |
| --- | --- |
| Add | `http://210.16.160.209:18092/add` |
| Search | `http://210.16.160.209:18092/search` |
| Health | `http://210.16.160.209:18092/health` |

Add/Search 使用 `Authorization: Bearer <AML_API_KEY>`。新 smoke 版本沿用已有服务 Key。多个榜单版本如果绑定相同地址，都会访问该地址当前运行的服务；Git 标签不会自动创建独立服务。

`user_id` 是数据分区键，上游调用方负责将身份正确绑定到对应分区。

## /add：分类、保存原文与建立索引

1. 验证鉴权、参数和请求幂等性。
2. 分类器选择记忆类型及依据消息；程序绑定来源ID，并为选中消息补充前两条上下文。
3. 向量写入器按消息分块，默认每块320 lexical tokens、重叠40，保存原文、角色、时间、来源位置、可用说话者信息和embedding。
4. 类型写入模块保存人物画像、人物关系、事件、规则等选中原文及索引，不自由生成事实。无须进入类型构建的内容仍保存为向量原文。
5. 完成写入及所需索引后发布可检索版本。失败请求可沿用相同内容和相同 `request_id` 恢复未完成工作。

```http
POST /add
Authorization: Bearer <AML_API_KEY>
Content-Type: application/json
```

```json
{
  "request_id": "write-001",
  "user_id": "user-001",
  "session_id": "session-001",
  "messages": [
    {"role": "user", "content": "Veda has a brother named Niko.", "timestamp": 1704067200000},
    {"role": "user", "content": "Niko takes lessons from teacher Selene.", "timestamp": 1704153600000},
    {"role": "user", "content": "Selene teaches the clarinet.", "timestamp": 1704240000000}
  ]
}
```

成功返回：

```json
{"success": true, "request_id": "write-001", "user_id": "user-001", "session_id": "session-001"}
```

同一用户下，相同 `request_id` 与相同内容重试不会重复写入；相同ID搭配不同内容返回409。单次支持1–200条消息，角色为 `user` 或 `assistant`。

消息 `timestamp` 可省略，提供时为Unix毫秒。扩展字段包括消息的 `speaker` 和请求的 `session_timestamp`。角色不等于人物姓名，系统不会从 `user/assistant` 猜测真实身份，也不能恢复输入中缺失的姓名或时间。

## /search：统一检索与原文证据包

1. 固定本次请求的已发布数据版本，避免多跳过程中混入后来完成的写入。
2. 规划器选择直接检索、独立拆分或依赖式多跳，并选择查找内容和分区。
3. 底层检索器统一执行向量与BM25混合召回，`dual` 分区兼顾目标类型与全局候选，再由Qwen API结合可用上下文重排。
4. 来源选择器选择相关原文，必要时提出后续查询；程序保留各轮已选中间来源。新旧记录与事件阶段均以原文交付，不裁决事实有效性。
5. 程序绑定来源、去重、排序和打包，再截取 `top_k` 返回。规则适用性及基础脱敏按服务器配置执行。

多跳编排共享查询、调用和时间预算：当前最多6次检索、5次规划/审核LLM调用，每轮最多查看12条证据，每条原文与上下文合计1200字符。独立分区分类等模块的调用不计入该多跳计数，5不是整个请求所有模型调用的绝对上限。

```http
POST /search
Authorization: Bearer <AML_API_KEY>
Content-Type: application/json
```

```json
{"user_id": "user-001", "query": "What instrument does Veda's brother's teacher teach?", "top_k": 10}
```

返回普通原文或原文包。以下仅展示结构，ID与分数为示例：

```json
{
  "data": [
    {
      "id": "bundle_example",
      "content": "[Related original sources; relationships, current validity and event stages are not adjudicated. source_time is message time, not event time.]\n[source_id=\"source-1\"; source_time=\"2024-01-01T00:00:00Z\"]\nVeda has a brother named Niko.\n\n[source_id=\"source-2\"; source_time=\"2024-01-02T00:00:00Z\"]\nNiko takes lessons from teacher Selene.\n\n[source_id=\"source-3\"; source_time=\"2024-01-03T00:00:00Z\"]\nSelene teaches the clarinet.",
      "score": 1.02
    }
  ]
}
```

- `top_k` 必填，范围1–100；可选 `options` 为选择题选项数组，不作为证据。
- 无结果返回 `{"data": []}`；`score` 是排序信号，不是可信概率。
- 每段包含来源ID、消息时间和完整原文；数据库保存包ID到来源ID的映射。
- 每包最多4段、6000字符，当前模式最多组织前8条已选来源入包；超长单条保留为普通原文。每个包占一个TopK位置。
- 包不提供整体 `created_at`，因为各段时间不同；普通结果的来源时间未知时也省略该字段。
- 消息时间不等于事件时间，排列顺序不证明因果或事实有效性。证据不完整时也可返回已有原文，不补写缺失事实。

接口接受扩展参数 `reference_time`（Unix毫秒）与 `reference_timezone`。当前时间增强关闭，不能将接受参数解释为启用了时间窗口检索。

## 安装与上线配置

Python 3.11+，建议使用独立虚拟环境：

```sh
python -m pip install -e ".[rag,test]"
```

首次运行可复制 `.env.example` 为 `.env`，已有文件不要覆盖。以下是当前版本主要配置；供应商URL需匹配实际开通地域或工作空间，密钥自行填写：

```dotenv
MEMORY_BACKEND=vanilla
AML_AUTH_MODE=bearer
AML_API_KEY=<服务鉴权Key>
AML_ADD_CONCURRENCY=0
AML_SEARCH_CONCURRENCY=0
RAG_MEMORY_DB=data/aml/text-embedding-v4.sqlite3

LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=<OpenAI Key>
LLM_PROXY=

RAG_EMBEDDING_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
RAG_EMBEDDING_API_MODEL=text-embedding-v4
RAG_EMBEDDING_API_KEY=<百炼Embedding Key>
RAG_EMBEDDING_DIMENSIONS=1024
RAG_EMBEDDING_BATCH_SIZE=10
RAG_RERANK_MODE=local
RAG_RERANK_API_PROTOCOL=dashscope
RAG_RERANK_API_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
RAG_RERANK_API_MODEL=qwen3.7-text-rerank
RAG_RERANK_API_KEY=<百炼Reranker Key>
RAG_RERANK_API_TIMEOUT=60
RAG_RERANK_API_RETRIES=1
RAG_RERANK_SELECTION=direct
RAG_RERANK_CONTEXT=1
RAG_RETRIEVAL_MODE=hybrid
RAG_RESULT_WINDOW=0
RAG_METADATA_MODE=on
RAG_TARGET_MODE=off
RAG_FUSION_QA=off
RAG_TAG_MODE=off

RAG_WRITE_GATE_MODE=on
RAG_BUILD_MODE=on
RAG_PARTITION_MODE=dual
RAG_MULTIHOP_MODE=llm
RAG_MULTIHOP_PROMPT_STYLE=long
RAG_MULTIHOP_RECOVERY=on
RAG_MULTIHOP_ROUNDS=3
RAG_MULTIHOP_QUERIES=6
RAG_MULTIHOP_LLM_CALLS=5
RAG_MULTIHOP_EVIDENCE=12
RAG_MULTIHOP_EVIDENCE_CHARS=1200
RAG_MULTIHOP_BINDINGS=off
RAG_MULTIHOP_NEEDS=off

RAG_EVIDENCE_BUNDLE_MODE=raw
RAG_VERSION_MODE=on
RAG_DISCLOSURE_MODE=mask
RAG_RULE_APPLICABILITY_MODE=on
RAG_SUPPLEMENTAL_MODE=on
RAG_POSITION_LOG_PATH=data/logs/production-search-positions.jsonl

RAG_RAW_COVERAGE_MODE=off
RAG_EVIDENCE_GROUP_MODE=off
RAG_EVIDENCE_CHAIN_MODE=off
RAG_FACT_REPLACEMENT_MODE=off
RAG_STATE_EVIDENCE_MODE=off
RAG_SEMANTIC_PRIVACY_MODE=off
RAG_TEMPORAL_MODE=off
RAG_TIME_ANNOTATION_MODE=off
RAG_TEMPORAL_EXPERIMENT=off
```

`RAG_RERANK_MODE=local` 是已有适配器的开关名称；非空API URL使其实际调用HTTP Qwen重排，而不是本地MiniLM。更换embedding需要匹配的新库并重建向量。显式代理填写于 `.env`，不能依赖系统环境变量。

并发配置为0只取消应用层准入限制，不代表数据库或提供商没有容量限制。启动命令：

```sh
python -m uvicorn memory.aml_api:app --host 0.0.0.0 --port 18092 --workers 1 --no-access-log
```

线上目录为 `/GeeAI/AIAgent/CSIG`，用户级服务为 `csig-aml-v1.service`，生产库为该目录下的 `shared/data/aml/text-embedding-v4.sqlite3`。修改 `.env` 后重启生效。

位置日志只记录来源ID、排序位置和状态，不记录问题、原文和密钥。多个实例须使用不同日志文件或关闭日志，避免滚动文件竞争。

## 验证结果与边界

部署前在服务器凭据和生产库副本上验证，再切换正式服务。2026-10-10正式接口smoke检查：health、add、重复写入、search均为200；其他用户查询为空；未鉴权请求为401。关系题交付3/3段原文，新旧信息题交付2/2段原文，均生成证据包。切换时原有1702条来源全部保留，配置和数据库快照已保存。

代码自动化回归403项通过。19题本机对照中，上一版raw交付全部目标来源16/19题、目标来源总数26/32；覆盖增强仍为16/19题、27/32条，但增加调用和延迟，因此当前smoke关闭增强。样本包含6道人工题和13道同一真实会话的题，不代表跨会话泛化或官方成绩。历史861题全量结果属于早期配置，不应直接作为本标签的全量成绩。

当前限制：

- 选择器只能看到有限候选，存在漏选；已选来源也可能因包容量或TopK未交付。
- 包内可能混入其他人物、事件的材料，不保证完整或无噪声。
- 新旧事实和事件阶段打包不等于事实替代、完整状态管理或因果推理。
- 基础脱敏不是通用语义隐私或按用途授权系统。
- 规则适用性模块仍会筛选候选，尚未实现完整规则版本与废止关系管理。
- `/health` 仅证明服务存活，模型可用性需真实Add/Search验证。

## 测试与历史报告

```sh
python -m pytest -q -p no:cacheprovider
```

真实测试需要根 `.env` 中的有效凭据，应使用隔离数据库。下列报告描述各自实验当时的状态；其中“未部署”等历史描述不覆盖本README的当前状态：

- [原文证据包与8题测试](docs/raw-evidence-bundles-20261010.md)
- [候选翻页与缺口补查实验，当前关闭](docs/raw-coverage-validation-20261010.md)
- [位置日志与问题定位](docs/position-trace-validation-20261010.md)
- [历史Qwen全量验证](docs/qwen-full-validation-20261009.md)
- [历史接口审查](docs/aml-contract-audit-20261002.md)

代码保留实验模块供后续研究；“代码存在”不表示“线上开启”。

本版本已启用[详细检索审计日志](docs/search-audit.md)，记录问题、每轮审查证据及最终证据包。生产配置 `RAG_SEARCH_AUDIT_PATH=data/logs/search-audit.jsonl`；日志包含评测原文，须与数据库一并按评测数据保留要求清理。
