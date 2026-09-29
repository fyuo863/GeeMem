# CSIG Memory

通用 Agent 长期记忆模块第一版：`POST /add` 将整轮对话交给 LLM 提取有向线索图，`POST /search` 将查询解析为关键词，沿节点/关系路径查找并返回消息原文。SQLite 持久保存图与证据，不依赖向量数据库。

## 启动

Python 3.11+，在 PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_API_KEY="你的密钥"
$env:LLM_MODEL="gpt-4.1-mini"
$env:MEMORY_DB="data/memory.sqlite3"
.\.venv\Scripts\python -m uvicorn memory.api:app --host 127.0.0.1 --port 8000
```

提供商须兼容 OpenAI Chat Completions 的 `response_format=json_object`。`.env.example` 是配置示例，不会自动读取 `.env`。接口文档：<http://127.0.0.1:8000/docs>；存活探针：`GET /health`（不检查 LLM 连通性）。

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

LLM 一次读取完整对话，识别实体、指代、事实和有向关系。例如「用户A向用户B提问」形成 `用户A →[询问]→ 用户B`，每条节点和关系都保存来源消息下标。抽取完成并通过引用校验后，消息、图和幂等记录在一个事务中落库。LLM 不可用或输出无效时返回 502，不写半成品，也不静默切换成规则提取。

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

ID 和分数为示意。created_at 来自消息时间而非入库时间。LLM 提取实体/关系关键词，匹配节点名称、键、别名及关系标签，再做有跳数限制的广度优先路径检索。为支持「谁询问了B」等反向问题，搜索可沿入边和出边遍历，存储的关系方向保持不变。只使用指定用户/会话的证据和边。返回节点或路径边关联的**完整原消息**（也可包含 assistant 原消息），不由 LLM 生成答案，不只做原文关键词搜索。

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

自动测试使用可控 LLM 替身和 HTTP mock，不要求密钥；不代表真实模型的抽取质量已经验证。运行实际示例需配置上述环境变量。

## 第一版边界

当前是单机原型，检索加载该用户的图，适合中小规模数据；没有 embedding、重排、自动过期、事实版本消解或删除接口。跨轮实体合并依赖 LLM 稳定键与别名，尚无专门实体消歧模型。图证据下标通过结构校验，但抽取内容是否忠实仍取决于模型。多跳会带回相关上下文，尚未做严格关系方向约束或复杂逻辑查询。

`user_id` 是调用者传入的分区键，**不是身份认证**。默认仅监听本机；作为远程服务部署时，应由受信任网关鉴权并将身份绑定到 user_id。LLM 会接收整轮对话，请选用符合数据使用要求的服务商。

## Git 与 CodeGraph

本项目使用 Git；CodeGraph 是供开发时查询代码结构的独立索引，与业务记忆图无关。已使用 `@colbymchenry/codegraph@1.5.0 init -i` 初始化，`.codegraph/` 和运行数据库不提交。

系统 PATH 没有 codegraph 时可运行：

```powershell
npx.cmd --yes --cache .cache/npm @colbymchenry/codegraph@1.5.0 sync
```
