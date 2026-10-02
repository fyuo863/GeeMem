# GeeMem：Agent 长期记忆与原文证据检索

GeeMem 是一个通用 Agent 长期记忆模块，通过同步 `POST /add` 保存对话，通过 `POST /search` 返回相关原文证据。当前提供 Agent Memory Leaderboard 文本赛道适配接口，入口为 `memory.aml_api:app`。

当前选择 **ABC＋方案 1＋方案 3（FM）**：混合召回、上下文与目标评分融合、局部问答关联，以及人物/日期软召回。多人子查询方案 2 保留实现但关闭。Add/Search 不调用生成式 LLM，不生成最终答案。

## 当前版本

- 实现分支：`codex/abc-factorial`。
- 完整实验与报告提交：`58443cb`；当前 README 和配置在该提交基础上更新。
- Embedding：`BAAI/bge-small-en-v1.5`。
- Reranker：`cross-encoder/ms-marco-MiniLM-L-6-v2`。
- 存储：SQLite，按 `user_id` 隔离，保存原文、来源位置、元数据和向量。
- 配置只读取项目根目录 `.env`，不读取系统环境变量，不执行变量插值；`.env` 不提交 Git。
- 本次已在本机 `.env` 和 `.env.example` 应用 FM。远端 `/GeeAI/AIAgent/CSIG` 的生产部署未在本次更新中切换；更新本地配置不会自动更新远端服务。

历史图方法使用 LLM，与当前 RAG 路径不同。保留的 `LLM_MODEL=gpt-4o-mini` 不表示当前 Add/Search 调用了该模型，也不替代对实际 embedding/reranker 的模型披露。

## /add 流程

1. 验证请求、鉴权和用户分区，检查 `request_id` 幂等性。
2. 按原消息分块，默认每块 320 lexical tokens，重叠 40；保留原文及来源位置。
3. 保存角色、时间，以及调用者提供的可选 `speaker`、`session_timestamp`。未知信息不凭空推断。
4. 使用 BGE 编码，保存到 SQLite；同步写入成功后即可搜索。

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

1. **用户隔离与混合召回**：在指定用户内进行 BGE 稠密检索和 BM25 检索，通过加权 RRF 合并候选。
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

`top_k` 必填，范围 1–100；无结果返回空数组。`score` 是排序分数，不是概率。`created_at` 来自已知源时间，未知时可为空。`user_id` 是数据分区键，调用方仍须负责正确绑定用户身份。

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
RAG_MEMORY_DB=data/aml/vanilla.sqlite3
RAG_MODEL_PATH=data/models/bge-small-en-v1.5
RAG_RERANK_MODE=local
RAG_RERANK_PATH=data/models/ms-marco-MiniLM-L-6-v2
RAG_DEVICE=cpu
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

GPU 部署时将两个 device 改为实际设备，例如 `cuda:0`，并安装匹配的 PyTorch/CUDA。复现本次结果需使用上述 BGE/MiniLM，保持 `RAG_EMBEDDING_API_URL` 和 `RAG_RERANK_API_URL` 为空或不配置，避免启用历史 Qwen HTTP 适配器。

首次下载模型：

```sh
python scripts/download_rag_model.py
python scripts/download_reranker.py
```

下载器使用固定版本并记录 manifest；代理通过 `.env` 的 `RAG_DOWNLOAD_PROXY` 显式指定。已有完整模型可直接设置模型路径。更换 embedding 模型需要新库并重新写入，不能混用旧向量。仅开启方案 1＋3 不改变向量模型或存储结构，无需为此重新写入记忆。

启动服务：

```sh
python -m uvicorn memory.aml_api:app --host 0.0.0.0 --port 18092 --workers 1 --no-access-log
```

修改配置后重启进程生效。`GET /health` 为健康检查，`/docs` 为交互式接口文档。远端现有服务为用户级 `csig-aml-v1.service`；部署前核对其工作目录、实际代码版本、模型路径和持久化数据库路径。旧 `deploy/update-geeai.sh` 面向历史 test_base，不应直接用于本分支。

## 测试结果

2026-10-02，在相同远端服务器、BGE/MiniLM 和独立数据库快照上比较八种固定组合。10 组完整对话、5,882 条消息；861 道纯文本题全部执行，859 道有有效证据的题参与指标计算，共 1,074 条证据出现次数。

| 配置 | Hit@10 | Recall@10 | Micro Recall@10 | 全证据命中@10 | 平均检索耗时 |
|---|---:|---:|---:|---:|---:|
| 原 ABC | 92.67% | 89.88% | 85.57% | 86.73% | 641 ms |
| **当前 ABC＋1＋3** | **93.25%** | **90.55%** | **86.13%** | **87.43%** | **692 ms** |

- Hit@10：至少召回一条标注证据的题目比例。
- Recall@10：逐题证据召回率的平均值。
- Micro Recall@10：所有题目命中证据数除以证据总数。
- 全证据命中@10：全部标注证据均进入 Top10 的题目比例。

当前方案新增命中 6 题、退化 1 题，Hit@10 净增 0.58 个百分点。延迟是固定 60 题、无缓存且交替执行的引擎计时，不含 HTTP、启动或写入，不代表并发吞吐量。公开数据已用于开发，不是官方隐藏评测成绩。

验证包括 111 项自动测试、480/480 次无缓存排名一致，以及最佳方案真实 HTTP `/search` 的 60/60 次 Top10 ID 一致。HTTP 验证复用了已写入的完整记忆快照，本轮未重复整库 `/add`。

[八组合完整报告与退化案例](docs/abc-extensions-comparison-20261002.md) · [机器可读汇总](docs/abc-extensions-summary-20261002.json)

自动测试与独立合成接口检查：

```sh
python -m pytest -q -p no:cacheprovider
python scripts/smoke_aml.py --live
```

后者使用 `.env` 的 `AML_BASE_URL` 和服务密钥，写入独立合成用户数据，不是官方评测。全组合实验脚本为 `scripts/benchmark_abc_extensions.py`，报告脚本为 `scripts/report_abc_extensions.py`。

## 分支与回退

三种独立方案均从检查点 `ddaf4b4` 创建：

| 分支 | 用途 |
|---|---|
| `codex/abc-fusion-qa` | 方案 1：上下文/目标融合与局部问答关联 |
| `codex/abc-multi-query` | 方案 2：多人子查询实验 |
| `codex/abc-soft-recall` | 方案 3：人物/日期软召回 |
| `codex/abc-factorial` | 三种实现的组合、完整测试与当前推荐配置 |

要回到原 ABC，保留 ABC 参数，将三个扩展开关全部设为 `off` 并重启。代码缺省值仍为 off，当前模板显式选择 1＋3，便于区分历史配置与新配置。

## 边界与来源

当前模型及完整测试主要针对英文文本；示例或接口支持 Unicode 不代表中文检索效果已验证。元数据缺失、时间歧义、多人归属和间接证据仍可能造成漏召回。Add/Search 并发上限为配置限制，超过时返回 429，不能据此推断持续吞吐能力。

基础检索思路参考 [wenxiaof345-ctrl/vanilla-rag-memory](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory)，参考提交 `31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca`。本项目独立实现，不包含上游源文件，不声称原方法原创；参考版本未声明许可证。模型许可遵循各发布者说明。参赛时如实披露实际使用的 BGE/MiniLM，榜单资格以主办方规则与确认为准。

[历史图方案研究记录](docs/graph-historical-readme.md) · [历史 v1 方法说明](docs/v1-vanilla-rag.md) · [评测接口对接说明](docs/agentmemories.md)。历史文档中的模型、部署地址和分支状态不代表当前版本。
