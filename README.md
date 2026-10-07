# GeeMem：Agent 长期记忆与原文证据检索


GeeMem 是一个通用 Agent 长期记忆模块，通过同步 `POST /add` 保存对话，通过 `POST /search` 返回相关原文证据。当前提供 Agent Memory Leaderboard 文本赛道适配接口，入口为 `memory.aml_api:app`。

本分支 `codex/llm-multihop` 从 `main` 的 `b90ea62` 创建，增加可选的 **gpt-4o-mini 检索编排**：直接检索、独立子问题并行检索、依赖式逐步检索，以及带来源引用的证据缺口检查。默认 `RAG_MULTIHOP_MODE=off`；启用时只有 Search 调用生成式 LLM，Add 和赛事响应结构保持原样。实验尚未部署。配置、预算、失败回退及测试见 [多跳实验说明](docs/llm-multihop.md)。下文参赛版本说明对应关闭该实验开关的基线。

当前选择 **ABC＋方案 1＋方案 3（FM）**：混合召回、上下文与目标评分融合、局部问答关联，以及人物/日期软召回。多人子查询方案 2 保留实现但关闭。Add/Search 不调用生成式 LLM，不生成最终答案。

## 当前版本

- 当前参赛运行版本：`text-embedding-v4-fm-20261002`，代码提交 `c7e5da09b2e8ab178d7bf38d300c96499e2f7287`，已合并 `main` 并在服务器部署。
- 历史 tag `competition-final-20261002` 对应旧 BGE 版本，未移动；请勿用它复现当前学术榜模型配置。
- Embedding：百炼官方 `text-embedding-v4` API，1024 维，每批最多 10 条。
- Reranker：`cross-encoder/ms-marco-MiniLM-L-6-v2`。
- 存储：SQLite，按 `user_id` 隔离，保存原文、来源位置、元数据和向量。
- 配置只读取项目根目录 `.env`，不读取系统环境变量，不执行变量插值；`.env` 不提交 Git。
- 部署目录 `/GeeAI/AIAgent/CSIG`，服务 `csig-aml-v1.service`；生产库 `shared/data/aml/text-embedding-v4.sqlite3`。原 BGE 库保留但未迁移，当前服务使用新库。

历史图方法使用 LLM，与当前 RAG 路径不同。保留的 `LLM_MODEL=gpt-4o-mini` 不表示当前 Add/Search 调用了该模型，也不替代对实际 embedding/reranker 的模型披露。

## /add 流程

1. 验证请求、鉴权和用户分区，检查 `request_id` 幂等性。
2. 按原消息分块，默认每块 320 lexical tokens，重叠 40；保留原文及来源位置。
3. 保存角色、时间，以及调用者提供的可选 `speaker`、`session_timestamp`。未知信息不凭空推断。
4. 调用 text-embedding-v4 编码，保存到 SQLite；同步写入成功后即可搜索。

人物身份与 API 角色不同：`user/assistant` 不能自动替代具体姓名。建议调用者提供真实 `speaker`，以帮助人物相关检索。时间戳单位为 Unix 毫秒。

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
  "session_timestamp": 1704067200000,
  "messages": [
    {"role": "user", "speaker": "Xiaolin", "content": "I live in Hangzhou and train for a marathon three times a week.", "timestamp": 1704067200000},
    {"role": "assistant", "content": "Which marathon are you preparing for?", "timestamp": 1704067260000},
    {"role": "user", "speaker": "Xiaolin", "content": "The Hangzhou Marathon.", "timestamp": 1704067320000}
  ]
}
```

成功返回：

```json
{"success": true, "request_id": "write-001", "user_id": "user-001", "session_id": "session-001"}
```

同一用户下相同 `request_id`、相同内容重试不会重复写入；使用相同 ID 提交不同内容返回 409。单次请求支持 1–200 条消息；当前评测入口的角色仅支持 `user` 和 `assistant`。

## /search 流程

1. **用户隔离与混合召回**：在指定用户内进行 text-embedding-v4 稠密检索和 BM25 检索，通过加权 RRF 合并候选。
2. **人物/日期软召回（方案 3）**：用规则识别人名与日期线索，追加最多 60 个元数据候选，补救未进入原候选池的证据；不将人物或日期作为硬过滤条件。
3. **MiniLM 重排与 ABC**：基础重排候选上限为 400，分别评估上下文与目标文本，结合可靠人物/日期元数据，并在需要时进行二次关键词补检。
4. **融合与局部问答关联（方案 1）**：有限幅度地调整上下文与目标评分，保护依赖上下文的短回答；利用同会话、相邻消息位置及不同说话者/角色关联提问与回答。
5. **返回原文**：排序后返回 `top_k` 个结果，不将上下文或模型生成内容伪装成证据原文。长消息可能返回其原文分块。

上述人物、日期和问答规则均不调用生成式 LLM。方案 2 的多人子查询当前关闭，因为本次完整测试中降低了 Hit@10。

```http
POST /search
Authorization: Bearer <AML_API_KEY>
Content-Type: application/json
```

```json
{"user_id": "user-001", "query": "Which marathon is Xiaolin preparing for?", "top_k": 10}
```

返回结构如下，ID 和分数仅为示例：

```json
{
  "data": [
    {
      "id": "example-chunk-id",
      "content": "The Hangzhou Marathon.",
      "score": 5.2,
      "created_at": "2024-01-01T00:02:00Z"
    }
  ]
}
```

`top_k` 必填，范围 1–100；**官网正式外部评测固定传 100**，上面的 10 仅为本地示例。选择题还会传顶层 `options` 字符串数组，例如 `"options": ["A. Hangzhou", "B. Shanghai"]`，开放题省略。无结果返回空数组。`score` 是排序分数，不是概率。`created_at` 来自已知源时间，未知时省略。`user_id` 是数据分区键，调用方仍须负责正确绑定用户身份。

官网标准输入不保证提供扩展字段 `speaker`、`session_timestamp`；缺省时接口仍正常运行，但人物/日期通道可用信息会减少。此前带元数据的公开测试结果不能直接代表这种输入条件下的成绩。

2026-10-02 已重新核对官网：[接口兼容性审查及容量、数据保留边界](docs/aml-contract-audit-20261002.md)。

## 安装与配置

Python 3.11+，建议使用独立虚拟环境。在项目根目录执行：

```sh
python -m pip install -e ".[rag,test]"
```

首次运行将 `.env.example` 复制为 `.env`；已有文件不要覆盖。填入独立的 `AML_API_KEY`，保留模板中的 ABC 参数及以下配置：

```dotenv
MEMORY_BACKEND=vanilla
AML_AUTH_MODE=bearer
AML_API_KEY=<独立服务密钥>
AML_BASE_URL=http://127.0.0.1:18092
AML_ADD_CONCURRENCY=1
AML_SEARCH_CONCURRENCY=4
RAG_MEMORY_DB=data/aml/text-embedding-v4.sqlite3
RAG_EMBEDDING_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
RAG_EMBEDDING_API_MODEL=text-embedding-v4
RAG_EMBEDDING_API_KEY=<百炼API密钥>
RAG_EMBEDDING_DIMENSIONS=1024
RAG_EMBEDDING_BATCH_SIZE=10
RAG_EMBEDDING_API_PROXY=
RAG_QUERY_PREFIX=
RAG_QUERY_INSTRUCTION=
RAG_RERANK_MODE=local
RAG_RERANK_PATH=data/models/ms-marco-MiniLM-L-6-v2
RAG_RERANK_DEVICE=cpu
RAG_METADATA_MODE=on
RAG_TARGET_MODE=on
RAG_SECOND_PASS=on
RAG_RERANK_SELECTION=context_support
RAG_RERANK_CONTEXT=1
RAG_RESULT_WINDOW=0
RAG_FUSION_QA=on
RAG_MULTI_QUERY=off
RAG_SOFT_RECALL=on
```

MiniLM GPU 部署时将 `RAG_RERANK_DEVICE` 改为实际设备，例如 `cuda:0`，并安装匹配的 PyTorch/CUDA。Embedding 使用百炼 API，不加载本地 BGE；保持 `RAG_RERANK_API_URL` 为空或不配置。API Key 必须对应账户开通地域；上面的官方兼容地址已实测可用，业务空间专属地址以控制台为准。

首次下载本地重排模型：

```sh
python scripts/download_reranker.py
```

下载器使用固定版本并记录 manifest，代理通过 `.env` 的 `RAG_DOWNLOAD_PROXY` 显式指定。更换 embedding 必须使用新库并从原文重新写入，不可复用旧 BGE 向量。[迁移说明](docs/text-embedding-v4-migration.md)。

启动服务：

```sh
python -m uvicorn memory.aml_api:app --host 0.0.0.0 --port 18092 --workers 1 --no-access-log
```

修改配置后重启进程生效。`GET /health` 为健康检查，`/docs` 为交互式接口文档。远端现有服务为用户级 `csig-aml-v1.service`；部署前核对其工作目录、实际代码版本、模型路径和持久化数据库路径。旧 `deploy/update-geeai.sh` 面向历史 test_base，不应直接用于本分支。

## 测试结果

2026-10-02，使用真实 HTTP `/add` 重新写入 10 组公开对话、5,882 条消息，再对全部 861 道纯文本问题执行 `/search`（top_k=100）；859 道有效证据题计分，共 1,074 条证据出现次数。

| 配置 | Hit@10 | Recall@10 | Micro Recall@10 | 全证据命中@10 |
|---|---:|---:|---:|---:|
| 历史 BGE＋ABC＋1＋3 | 93.25% | 90.55% | 86.13% | 87.43% |
| **当前 v4＋ABC＋1＋3** | **93.25%** | **90.51%** | **86.03%** | **87.43%** |

Hit@10 命中题目集合完全一致；部分证据总命中少 1 条。当前 Hit@100 为 98.60%、Recall@100 为 97.57%、Micro Recall@100 为 95.90%、全证据命中@100 为 96.51%。

- Hit：至少命中一条证据的题目比例。
- Recall：逐题证据召回率的平均值。
- Micro Recall：所有题目命中证据数除以总证据数。
- 全证据命中：全部标注证据均被召回的题目比例。

完整 861 题 HTTP 检索平均 852ms，P95 1097ms；重新写入累计 540.52 秒。耗时包括 embedding API 网络和本地重排，不含服务启动，不代表并发吞吐。旧 BGE 692ms 是 60 题引擎内抽样计时，不能直接计算增幅。

118 项自动测试通过；独立库完整 HTTP 评测与上线后的合成 Add/Search、鉴权、幂等、立即检索、用户隔离检查通过。公开集包含 caller-supplied 人物/会话日期元数据，已用于开发；不代表官方隐藏输入条件或 Answer 成绩。当前版本尚未触发官方 Smoke/Full。

[完整模型对比](docs/text-embedding-v4-comparison-20261002.md) · [机器可读结果](docs/text-embedding-v4-summary-20261002.json) · [历史八组合报告](docs/abc-extensions-comparison-20261002.md)

自动测试与独立合成接口检查：

```sh
python -m pytest -q -p no:cacheprovider
python scripts/smoke_aml.py --live
```

后者使用 `.env` 的 `AML_BASE_URL` 和服务密钥，写入独立合成用户数据，不是官方评测。当前模型对比脚本为 `scripts/benchmark_embedding_v4.py`；历史八组合脚本仅用于其对应的 BGE 实验配置。

## 分支与回退

`main` 已包含当前 v4 实现；`codex/text-embedding-v4` 保留迁移开发历史。三种独立方案均从检查点 `ddaf4b4` 创建：

| 分支 | 用途 |
|---|---|
| `codex/abc-fusion-qa` | 方案 1：上下文/目标融合与局部问答关联 |
| `codex/abc-multi-query` | 方案 2：多人子查询实验 |
| `codex/abc-soft-recall` | 方案 3：人物/日期软召回 |
| `codex/abc-factorial` | 历史 BGE 三种实现的组合及完整测试 |

要回到原 ABC，保留 ABC 参数，将三个扩展开关全部设为 `off` 并重启。代码缺省值仍为 off，当前模板显式选择 1＋3，便于区分历史配置与新配置。

## 多跳合并版本

`codex/llm-multihop` 已合并进 `main`，合并提交以带注释的 `multihop` tag 标记。选用原长提示词，gpt-4o-mini 规划直接检索、独立拆分或依赖式多跳，再通过来源绑定和证据需求检查追加查询，最终仍返回原文。

当前 `.env.example` 与本机根 `.env` 选择：

```dotenv
RAG_MULTIHOP_MODE=llm
RAG_MULTIHOP_PROMPT_STYLE=long
RAG_MULTIHOP_BINDINGS=on
RAG_MULTIHOP_NEEDS=on
RAG_MULTIHOP_LLM_CALLS=8
RAG_MULTIHOP_ROUNDS=3
RAG_MULTIHOP_QUERIES=6
RAG_MULTIHOP_SECONDS=90
RAG_MULTIHOP_LLM_TIMEOUT=20
RAG_RESULT_WINDOW=0
```

配置仅从项目根 `.env` 读取。代码在未指定模式时仍默认关闭多跳；新部署请采用以上显式配置，并提供原有 LLM 凭据、embedding 和 reranker 配置。设置 `RAG_MULTIHOP_MODE=off` 可回到直接检索。本次合并没有同步或重启远端服务。

上轮 30 题测试中，选中方案 Hit@10 为 96.67%、Recall@10 为 87.44%、全证据命中@10 为 80.00%，平均耗时 10.61 秒；包含 3 次连接回退。这是开发集检索结果，不是官方答案评分。详见 [实现说明](docs/multihop-evidence-repair.md) 与 [最新对照报告](docs/compact-multihop-results-20261004.md)。

## 提示词简化实验（2026-10-04）

`RAG_MULTIHOP_PROMPT_STYLE=long|short|focused` 可比较原提示词、仅缩短提示词、以及逐需求读取证据并独立生成下一查询的流程。缺省为 `long`；`RAG_MULTIHOP_MODE=off` 仍完全关闭 LLM 多跳。`focused` 自带需求依赖与字面来源校验，最终 `/search` 仍只返回原文。

实验配置为 8 次 LLM、6 次检索、90 秒总预算；本次合并已在本机选用其中的 long 配置，远端部署尚未同步。短提示词不保证理解更准确；非法依赖仍会被拒绝并回退。对照结果、输入输出错误案例和复现记录见 [提示词简化实验报告](docs/compact-multihop-results-20261004.md)，逐项数值见 [JSON 汇总](docs/compact-multihop-results-20261004.json)。

## 边界与来源

当前模型及完整测试主要针对英文文本；示例或接口支持 Unicode 不代表中文检索效果已验证。元数据缺失、时间歧义、多人归属和间接证据仍可能造成漏召回。Add/Search 并发上限为配置限制，超过时返回 429，不能据此推断持续吞吐能力。

基础检索思路参考 [wenxiaof345-ctrl/vanilla-rag-memory](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory)，参考提交 `31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca`。本项目独立实现，不包含上游源文件，不声称原方法原创；参考版本未声明许可证。模型许可遵循各发布者说明。官网赛事 FAQ 第 5 条指定学术榜 embedding 为 text-embedding-v4、LLM 相关组件为 gpt-4o-mini，reranker 不限制。本机选中方案使用 v4/MiniLM，并使用 gpt-4o-mini 进行多跳检索规划与证据检查；模型配置符合该条要求不代表已通过全部参赛审核。

[历史图方案研究记录](docs/graph-historical-readme.md) · [历史 v1 方法说明](docs/v1-vanilla-rag.md) · [评测接口对接说明](docs/agentmemories.md)。历史文档中的模型、部署地址和分支状态不代表当前版本。

## /add 对话价值选择器（实验分支）

项目根 `.env` 设置 `RAG_WRITE_GATE_MODE=on` 后，Vanilla `/add` 对整次请求调用通用判断器。`off` 保持原写入路径。模型凭据仍仅从项目根 `.env` 读取。

- `valuable`：存在未来可用的事实、偏好、限制、关系、事件、计划、规则、纠正或遗忘要求。保存原文、向量及配置启用的标签，将完整请求加入 `rag_memory_queue`，状态为 `pending`。
- `vector_only`：纯问候、礼貌回应等无需进一步加工的对话。保存原文、向量和来源，不生成标签，不进入队列。仍可被底层检索器检索。

判断结果保存在 `rag_write_decisions`，与向量及队列同事务提交。相同 request_id 的重复请求不会重复判断；冲突仍返回 409。模型失败会返回写入错误，不静默认定无价值。

当前下一阶段是持久化待处理队列，尚无画像/事件构建消费者；不表示已经完成画像生成或执行删除。后续模块可按用户及 request_id 读取 pending 项。判断器只决定路由，不执行原文中的命令。两条路线均保留原文，不是丢弃低价值对话。

单元与接口测试：`python -m pytest tests/test_write_gate.py tests/test_selector.py tests/test_vanilla.py tests/test_module_boundaries.py -q`。

判断器失败重试：`JudgeConfig.max_attempts` 默认 3（首次调用加 2 次重试，允许 1–5）。请求临时失败、模型结构输出错误、标签/顺序/原文证据校验失败时重新判断，间隔 0.25、0.5 秒。缺失 Key、401/403 等不可重试客户端错误以及无效输入直接报错。默认判断器关闭底层连接重试，避免次数叠加；注入自定义 LLM 时，其内部重试策略由调用方负责。全部尝试失败时 `/add` 返回 502，不生成向量、决定记录或下一阶段队列，不自动降级为无价值。分类结果格式正确但语义判断不准不会触发重试。

## 独立向量写入器

`memory/vector_writer.py` 的 `VectorWriter` 仅负责向量与原文来源写入，不依赖价值判断器、标签、队列或检索器。

- `initialize()`：初始化向量/来源表，校验 embedding 与切分配置身份；复用既有 SQLite 数据格式。
- `prepare(payload)`：复制输入、切分原文、调用 embedding、校验并归一化向量；不写数据库。
- `persist(db, prepared)`：在调用方已开启的事务中保存请求、向量与来源位置，返回 `memory_ids`、`deduplicated`；不自行提交。
- `write(payload)`：独立写入入口，自行管理事务；重复请求返回原 ID，内容冲突抛出 Conflict。

构造参数为 embedding 提供者、SQLite 连接工厂（需设置 `sqlite3.Row`）、可重入锁 `RLock` 和切分参数。连接工厂负责数据库路径；模型配置由上层从根 `.env` 加载后注入。首次独立调用前执行 `initialize()`。

`VanillaMemory.add()` 保留价值判断与标签加工，通过 `prepare/persist` 组合写入，让向量、判断记录、标签和待处理队列共用事务。来源位置与已有片段 ID 保持原生成规则。此轮只分离模块，embedding 传输重试与模型 tokenizer 上限检查尚未新增。

## 多标签记忆类型分类（当前 /add 入口）

`RAG_WRITE_GATE_MODE=on` 现在启用 `MemoryTypeSelector`，取代前述二分类入口；旧 `MemoryValueSelector` 和单标签 `ConfigurableJudge` 保留供显式调用。

一次 LLM 调用评估所有类型：`profile`（画像属性）、`relationship`（人物关系）、`event`（事件/计划）、`rule`（可复用规则/经验）、`other_memory`（其他有价值信息）、`vector_only`（仅向量存储）。前五类可同时命中；`vector_only` 必须单独出现。纯人物关系不额外计为画像。分类治理请求只负责路由，不会执行删除。

通用 `MultiLabelJudge` 通过 `MultiLabelConfig` 配置类别、说明、判定标准及互斥类别，可用于其他业务。返回每类的 `selected`、独立适用度 `score`、`message_indices` 和理由；分数不要求总和为 1。程序按配置排列输出，检查类别齐全且无重复、互斥、索引类型与范围；不把合法的类别输出顺序变化当作失败。失败重试沿用原 3 次尝试策略。

消息索引由程序从 0 编号；模型只选择索引，不生成原文 ID。`MemoryTypeSelector` 将有效索引绑定为稳定 `source_ids`，与向量来源一致。涉及简短确认时应同时引用上下文和回答；索引合法不等于语义支持已被验证。

`rag_write_decisions.result` 保存带 `memory-types-v1` 版本的完整分类结果。`rag_memory_routes` 按用户、请求、类型保存待处理路由、消息索引和来源 ID。`rag_memory_queue` 仍保留一份完整对话，供后续构建器读取上下文。以上内容和原文向量在同一事务写入；只有 `vector_only` 时不创建队列或类型路由。旧数据库新增路由表，旧记录不会自动重新分类。

外部 `/add` 成功响应格式不变。后续画像、事件等消费者仍待实现。本模块只完成分类与队列准备。

### 路由自动补充上下文（memory-types-v2）

分类器选中消息后，程序绑定直接依据的来源 ID，并为每条依据补充本次请求内前两条消息。`message_indices/source_ids` 保留直接依据，`context_indices/context_source_ids` 保存补充上下文；两组互斥，分别去重。不会递归向前扩展，也不跨请求读取历史。模型原始选中索引仍在 classification 中保留。

`builder_messages` 为两组消息的合并结果，按原文顺序排列，包含原文、角色、时间、speaker、message_index、source_id 和 source_kind（evidence/context）。后续构建器从对应类型路由读取此字段即可取得直接依据与解释上下文；context 不等于已确认事实，不可忽略否定、角色与不确定性。

路由新字段与向量、判断结果和完整对话队列同事务保存。旧数据库自动添加字段；旧路由默认空上下文字段，不回填历史记录。旧记录如需交给新构建器，应使用原完整对话及原选中索引重新执行 bind_route。此功能不增加 LLM 调用。

当前分类配置版本为 `memory-types-v3`：已移除 governance。更正按具体内容分类，无法归入具体类型的撤回/遗忘请求归入 other_memory，仅保存请求。历史治理路由保留，新请求不再生成 governance 路由。

## 独立关系记忆模块

`memory/relationship.py` 不依赖旧 `Graph`、`Store` 或旧 `edges` 表。`RelationshipBuilder.extract()` 接收分类器提供的直接依据与上下文，调用 LLM 只抽取明确关系；模型返回名称、类型、关系和程序消息索引，不生成数据库 ID。程序规范化常见关系（如朋友、同事、导师、兄弟姐妹），绑定用户作用域实体 ID，并将对称关系按当前用户到对方的方向存储。

`write()` 使用独立的 `relationship_entities`、`relationship_assertions`、`relationship_evidence` 表。重复的用户/主体/关系/客体会合并并追加证据，实体按用户隔离；证据保存 source_id、消息索引、原文和 evidence/context 标记。关系断言保留语义方向、置信度和 active 状态。

`RelationshipRetriever.find()` 支持按用户、主体、关系、客体和状态查询，并返回带证据的断言；`expand()` 在限定跳数内按实体 ID 扩展无向邻域，但每条结果仍保留 subject/object 的语义方向。共同事件不会自动生成关系，问句也不生成关系。

本模块不负责画像属性、事件抽取、实体消歧的最终决策或多跳答案生成；这些由上层构建器和规划器负责。真实 gpt-4o-mini 小测覆盖朋友、导师、亲属、同事、共同事件和问句，关系集合准确率为 7/7；测试记录位于 `data/relationship-builder-20261007/results-v2.json`。


## 独立人物画像模块

## 独立规则记忆模块（实验）

`memory/rule.py` 提供 `RuleBuilder` 和 `RuleRetriever`。`extract(builder_messages)`
使用根 `.env` 的 LLM 配置，抽取可复用指令、顺序流程和明确经验，保留条件、动作、例外、
适用范围、确定性及置信度。`write(user_id, extraction, builder_messages)` 由程序生成 ID，
绑定消息索引与 source_id，写入独立 `rule_records/rule_evidence` 表。
上下文不能独立生成规则；模块仅存储规则，不执行指令。

完全相同的结构重复写入复用 ID 并追加证据。显式 `supersedes=旧规则ID` 可创建新版本，
旧版本保留为 superseded；不按语义相似度自动覆盖，不自动裁决互相冲突的规则。
第三方规则适用范围依赖抽取质量，未实现权限推断。当前不自动消费 `/add` pending 路由，
调用方需将 rule 路由的 builder_messages 交给构建器。

`find(user_id, scope=..., status=...)` 查询结构和原文；
`find_applicable(user_id, query, scope=..., limit=...)` 在用户的全部有效规则中执行 BM25
候选检索，返回分数、完整条件/例外和原文。命中仅代表候选，不等于条件已经满足；
本版没有规则向量索引、条件推理或自动执行。未知范围不应自动应用全局规则。

测试：`python -m pytest tests/test_rule.py -q`。
读写基准及真实抽取小测：`python -m scripts.benchmark_rule --live`，
结果保存到 `data/rule-tests/results.json`。合成关键词 Hit@1 不代表真实语义召回率。

`memory/profile.py` 的 `ProfileBuilder` 与 `ProfileRetriever` 不复用旧图结构，但和关系模块共享 `relationship_entities` 表及 `entity_key`/实体 ID 规则。因此同一用户的“我的朋友小王”和“小王是医生”会引用同一个实体。

画像抽取保存主体、属性、值、确定性（`confirmed/uncertain/planned/denied`）、置信度和原文来源；`ProfileRetriever` 支持主体、属性和值查询，返回带 source_id 的证据。重复事实追加证据并提高置信度，不把不确定计划写成当前确定值。画像事实使用独立的 `profile_facts`、`profile_evidence` 表。
