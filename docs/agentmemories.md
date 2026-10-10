# Agent Memory Leaderboard 文本赛道对接

核对日期：2026-09-30。来源：[参赛规则](https://agentmemories.ai/rules)、[接口指南](https://agentmemories.ai/api-guide)、[文档](https://agentmemories.ai/docs)。页面可能更新，正式申报前应重新核对。公开页面快照保存在本机 data/agentmemories（未提交）。

本分支 test_base 从 baseline/infra-v1（c38a690）创建，使用基线的构图和检索方法，仅增加独立的评测契约层。memory.api:app 是原研究接口；**部署参评使用 memory.aml_api:app**。暂仅申报 Textual，不宣称支持图片、多模态或代码赛道完整能力。

## 契约适配

| 项目 | 本实现 |
|---|---|
| POST /add | 同步持久化成功后 HTTP 200，返回 success=true，原样回显 request_id/user_id/session_id |
| 可选 timestamp | 缺失/null 内部用 0 表示未知，告知模型不可解释成 1970；Search 对时间 0 省略 created_at。实际源时间 0 也不返回日期。重试映射保持一致 |
| POST /search | user_id/query/top_k，options 可选；只按 user_id 检索全部来源会话 |
| 选择题选项 | 作为关键词检索上下文传入，绝不生成最终答案或写入记忆；无需金标 |
| Search 返回 | data 数组，返回原始消息及稳定 ID；按相关性排序且数量不超过 top_k；空结果 data=[] |
| 认证 | Bearer、Token 或 X-Api-Key，根目录 .env 配置；none 仅用于显式允许的 smoke |
| GET /health | 无需鉴权，200 |
| 重试 | 相同 user_id/request_id 和同一规范化载荷幂等；冲突 409；写入失败不部分持久化 |
| 限流 | 单进程默认 Add 1、Search 4；超出返回 429 和 Retry-After: 5 |
| 数据错误 | 422 明确拒绝，避免在错误响应中回显请求正文 |
| 模型 | 评测入口强制检查 gpt-4o-mini；所有业务配置只读根目录 .env，不读系统环境变量 |

本实现公开容量限制：最多 200 条消息/请求，每条 32,000 字符，文本 Search 的 query 不设本地字符上限、不截断，最多 100 个选项、每项 4,000 字符，三个 ID 各 256 字符，top_k=1..100。ID 前后空白被拒绝而不是静默修改；文字内容原样保存。超限 422，不静默截断。官方普通文本一般按 20 条消息或 2,000 词分块，但无全局字符上限；这些本地限制需要在申报材料披露，不能保证覆盖全部 Full 输入。图片数组不支持，返回 422。

## 配置与启动

将 .env.example 所列 AML_* 配置加入根目录 .env，保留已有 LLM_API_KEY 等设置。不得把密钥提交到 Git。

```dotenv
LLM_MODEL=gpt-4o-mini
AML_AUTH_MODE=bearer
AML_API_KEY=<本系统专用长随机密钥>
AML_ALLOW_UNAUTHENTICATED=false
AML_MEMORY_DB=data/aml/memory.sqlite3
AML_ADD_CONCURRENCY=1
AML_SEARCH_CONCURRENCY=4
```

AML_API_KEY 是你给平台调用记忆系统用的 **Memory System Key**，不是 OpenAI key，也不是平台签发的 Eval Key。

```powershell
python -m pip install -e ".[test]"
python -m uvicorn memory.aml_api:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

公网部署应在前面配置 HTTPS 反向代理；Add/Search 上游超时至少 1800 秒，按实际负载配置请求体限制。平台拒绝回环/私网地址；127.0.0.1 仅供本机调试。SQLite 版先使用单 worker，上述并发限制按进程计算，多 worker 不能当作全局限流。跨相同 user 的并发写入支持 SQLite 事务，但不同分块之间没有自动拼接前两条上下文；属于基线方法边界。

可选容器：

```powershell
docker build -f Dockerfile.aml -t csig-aml:local .
docker run --name csig-aml -p 127.0.0.1:8000:8000 --mount "type=bind,source=$((Get-Location).Path)/.env,target=/app/.env,readonly" --mount type=volume,source=csig-aml-data,target=/app/data csig-aml:local
```

容器不打包密钥或数据，.env 由只读文件挂载。容器里的 LLM_PROXY 若指向 127.0.0.1 则是容器本身；使用 Windows Docker Desktop 时需按实际代理地址配置 host.docker.internal。镜像示例尚需在目标 Docker 环境构建验证，不能据此宣称公网部署已就绪。

## 本地兼容性测试

```powershell
python -m pytest -q
python scripts/smoke_aml.py
```

默认 smoke 使用真实 .env LLM、FastAPI TestClient 和自动清理的临时数据库，只写自建合成内容，不使用平台私有数据或金标；不打开公网端口。验证同步写入、重试、可选时间戳、跨 session 检索、选项、Top K、返回原文与 user_id 隔离。输出只有摘要，保存在 data/aml-smoke/，不保存模型候选或请求正文。

对已部署的自己接口运行 HTTP smoke：

```dotenv
AML_BASE_URL=https://your-memory-host.example
AML_HTTP_PROXY=
```

```powershell
python scripts/smoke_aml.py --live
```

live 模式会向该地址写入独立 local-smoke: 前缀的合成记忆，使用 .env 中的鉴权设置。它仍不是 AML 官方 smoke，不会创建榜单任务。

## 正式接入与阻塞项

官网明确要求所有参赛方自托管公网 Add/Search API；仅提供代码或 Docker 不会由平台代部署。

1. 在实际主机配置 .env、独立数据库、TLS、备份和容量；用 HTTP smoke 验证外部可达和鉴权。
2. 固定参评 commit 与版本，填写系统信息、联系人、公开仓库、原创/复用说明、端点、鉴权与容量；可参考 aml-submission.example.json。提交评测申请由平台审核。
3. 平台审核后签发 Eval Key，并绑定系统版本；先运行平台官方 smoke，再考虑 Full。
4. Full 的 Answer/Eval 由平台执行，我们只提供 Add/Search 原文证据。不得根据私有结果实时修改参评版本。

当前需要用户提供/确认的材料：公网主机与域名、部署方式、公开仓库和许可/来源声明、联系人、系统名称与版本、Memory System Key 的安全配置及审核后的 Eval Key。尚未向主办方提交申请、绑定版本或启动官方评测。

规则当前显示：材料截止 2026-10-31 23:59（UTC+8）；评测停止 2026-11-04 23:59；正式 Full 常需 0.5—2 天。Smoke 每小时最多一次、本届每赛道最多 30 次；Full 每赛道最多两次，第二次须距第一次完成至少 30 天。以官网实时规则为准。

## 数据与运行边界

参评数据只用于当前评测，不记录请求正文，不运行研究脚本保存候选图，不用于训练、产品分析或传播。正式数据使用独立 AML_MEMORY_DB；在任务完成后 30 天内删除该任务数据及派生副本/备份。单次专用部署可在停服后删除专用数据库及 SQLite -wal/-shm 和相应备份/容器卷；不要误删研究数据库。若多人共享实例，应先实现按任务清理流程再接收正式数据。

本次适配不证明基线构图语义正确，不保证 Full 规模容量或榜单成绩。正式资格需要平台 smoke、完整 Full、固定版本及官方复核。
