# CSIG Memory

通用 Agent 长期记忆模块第一版：`POST /add` 将整轮对话交给 LLM 提取无向线索图，`POST /search` 将查询解析为关键词，沿节点/关系路径查找并返回消息原文。SQLite 持久保存图与证据，不依赖向量数据库。

## 启动

Python 3.11+，在 PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
# 首次配置：若 .env 不存在，从模板创建，然后填写密钥与模型
if (!(Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
.\.venv\Scripts\python -m uvicorn memory.api:app --host 127.0.0.1 --port 8000
```

提供商须兼容 OpenAI Chat Completions 的 `response_format=json_schema` 严格结构化输出。程序自动读取项目根目录的 `.env`，不依赖启动时的工作目录；配置仅来自该文件，忽略系统环境变量，禁用 `${VAR}` 插值；HTTP 客户端也不读取环境代理或证书配置。未填写的可选项使用代码默认值，密钥无默认值。支持 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL` 、`LLM_PROXY`（可选，例如 `http://127.0.0.1:7897`）和 `MEMORY_DB`，修改后重启服务生效。`.env` 已被 Git 忽略。接口文档：<http://127.0.0.1:8000/docs>；存活探针：`GET /health`（不检查 LLM 连通性）。

## 写入

`POST /add`，Content-Type 为 `application/json`：

```json
{
  "request_id": "write-001",
  "user_id": "user-001",
  "session_id": "session-001",
  "messages": [
    {"role":"user","content":"我叫小林，现在住在杭州，最近正在准备马拉松。","timestamp":1704067200000},
    {"role":"assistant","content":"小林，你计划参加哪一场马拉松？","timestamp":1704067260000},
    {"role":"user","content":"我计划参加今年的杭州马拉松，每周训练三次。","timestamp":1704067320000}
  ]
}
```

返回 `request_id`、`message_ids`、本次抽取的 `nodes`/`edges` 数及 `deduplicated`。时间戳为 Unix 毫秒；原文包含空格均按输入保留。支持 user/assistant/system/tool 角色。

LLM 一次读取完整对话，识别实体、指代、事实和无向关联。例如「用户A向用户B提问」形成 `用户A —[询问]— 用户B`，每条节点和关系都保存来源消息下标。抽取完成并通过引用校验后，消息、图和幂等记录在一个事务中落库。LLM 不可用或输出无效时返回 502，不写半成品，也不静默切换成规则提取。

同一用户下，相同 request_id 和相同内容重试，返回第一次的消息 ID；不同内容复用该 ID 返回 409。不同用户可独立使用相同 request_id。并发重复写入只提交一次（可能发生重复 LLM 调用）。

## 检索

`POST /search`：

```json
{
  "user_id":"user-001",
  "query":"小林参加什么马拉松，每周训练几次？",
  "limit":10,
  "max_hops":2
}
```

可选 `session_id` 限定会话；省略时搜索该用户所有会话。`limit` 为 1–100，`max_hops` 为 0–4。

返回格式：

```json
{
  "data":[
    {
      "id":"49383853-0c51-4c86-bf4a-1ee1215b269d",
      "content":"我计划参加今年的杭州马拉松，每周训练三次。",
      "score":0.8,
      "created_at":"2024-01-01T00:02:00Z"
    }
  ]
}
```

ID 和分数为示意。created_at 来自消息时间而非入库时间。LLM 提取实体/关系关键词，匹配节点名称、键、别名及关系标签，再做有跳数限制的广度优先路径检索。搜索可从连边任一端遍历，提问者等角色信息保留在原文中。只使用指定用户/会话的证据和边。返回节点或路径边关联的**完整原消息**（也可包含 assistant 原消息），不由 LLM 生成答案，不只做原文关键词搜索。

分数为各关键词可到达证据的距离权重均值：直接节点证据为 1，距离 d 的节点证据为 1/(d+1)，遍历边证据按抵达下一跳的权重计算；同一关键词对一条消息取最高权重。按分数降序、消息时间降序、ID 排序；不是概率或向量相似度。无命中返回 `{"data":[]}`。

## 测试与结构

```powershell
python -m pytest -q
```

- `memory/api.py`：HTTP 接口、应用工厂和错误转换。
- `memory/llm.py`：真实 LLM 抽取和关键词解析适配器。
- `memory/models.py`：请求、响应及图结构校验。
- `memory/store.py`：SQLite 事务、证据映射、幂等与图路径搜索。
- `tests/test_memory.py`：API、方向、多跳、隔离、持久化、并发幂等及 LLM 错误测试。

自动测试使用可控 LLM 替身和 HTTP mock，不要求密钥；不代表真实模型的抽取质量已经验证。运行实际示例需填写根目录 `.env`。

## 第一版边界

当前是单机原型，检索加载该用户的图，适合中小规模数据；没有 embedding、重排、自动过期、事实版本消解或删除接口。跨轮实体合并依赖 LLM 稳定键与别名，尚无专门实体消歧模型。图证据下标通过结构校验，但抽取内容是否忠实仍取决于模型。多跳会带回相关上下文，尚未支持复杂逻辑查询。

`user_id` 是调用者传入的分区键，**不是身份认证**。默认仅监听本机；作为远程服务部署时，应由受信任网关鉴权并将身份绑定到 user_id。LLM 会接收整轮对话，请选用符合数据使用要求的服务商。

## Git 与 CodeGraph

本项目使用 Git；CodeGraph 是供开发时查询代码结构的独立索引，与业务记忆图无关。已使用 `@colbymchenry/codegraph@1.5.0 init -i` 初始化，`.codegraph/` 和运行数据库不提交。

系统 PATH 没有 codegraph 时可运行：

```powershell
npx.cmd --yes --cache .cache/npm @colbymchenry/codegraph@1.5.0 sync
```


## 公开数据与单场景测试

下载公开文本评测数据（数据和结果保存到被 Git 忽略的 `data/`）：

```powershell
python scripts/download_datasets.py
python scripts/test_single_scene.py
```

下载器固定上游版本并记录 URL、许可证、文件大小与 SHA-256；重复下载会核对哈希后跳过已有文件。范围为 LoCoMo-Refined、LongMemEval Oracle/S-cleaned、PersonaMem-v2 文本 benchmark 与 32K 历史、BEAM 100K/500K/1M、CL-bench；不含训练集、图片或更大变体，不是 AML 私有正式评测集。

单场景脚本使用 LoCoMo conv-41/session-20 的 18 条原消息，通过 FastAPI TestClient 调用实际 /add 和 /search 处理器，使用 `.env` 配置的真实 LLM；不是 mock，也不测试公网 HTTP 部署。每次运行创建独立数据库和时间戳目录，保存图 JSON、可点击的图 HTML、接口输入输出、检索证据和幂等检查。金标只在构图完成后用于本地评测，不进入记忆。失败记录同样保留。

首次测试说明、数据下载清单见 `data/README.md`。单场景通过不代表抽取语义全部正确，已发现的问题记录在各次测试报告中。


## 无向图与写入成本

当前图为无向证据关联网络。`source` / `target` 保留为兼容字段名，只表示两个无序端点；`directed=false` 明确标识图类型。A—B 与 B—A 在关系标签相同时合并，证据下标和原文摘录取并集；不同关系类型仍可形成多条边。关系不再用端点顺序表达施受、因果或时间方向，具体事实以返回原文为准。

写入采用一次LLM抽取，再进行节点引用、下标和原文摘录精确匹配校验，不再调用第二个LLM做方向审核。该调整减少写入调用量，但不能保证关系主体识别、提问覆盖或关联语义正确。仅转换边方向本身不一定降低耗时；性能对比必须同时说明已取消第二次审核。搜索原本就双向遍历，因此BFS规则不变。

旧数据库在首次打开时事务性迁移：规范端点顺序，合并同标签反向边，保留全部 edge_evidence 和 edge_quotes，之后通过 schema_meta 标记跳过重复迁移。消息和幂等写入记录保留；历史返回的抽取边数仍是原写入时的统计。旧历史图文件不会被覆盖。
