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
