# GeeMem 申请表填写说明

部署实现版本：9fc8797fa221bcb517cb7a5fe67d540a4e94421b；README 后续文档提交不改变此部署版本。

| 字段 | 内容 |
|---|---|
| 工作邮箱 | fyuo863@gmail.com（按用户截图，提交前确认可收信） |
| 姓名 | Leon |
| 组织/团队 | GeeAgent |
| 邀请码 | 没有邀请码则留空 |
| 系统名称 | GeeMem |
| 版本名称 | v1-abc-qwen-9fc8797 |
| Add API | http://210.16.160.209:18092/add |
| Search API | http://210.16.160.209:18092/search |
| 认证方式 | Authorization: Bearer |
| 记忆系统 Key | 远端 /home/agent/csig-aml-v1/shared/.env 的 AML_API_KEY 值；只填值，不带 Bearer 前缀，不用 OpenAI API Key |
| Add 状态地址 | 留空（同步 Add） |
| 健康检查地址 | http://210.16.160.209:18092/health |
| 公开 GitHub 仓库 | https://github.com/fyuo863/GeeMem/tree/v1 |

## 提交说明与运行指南（可粘贴）

GeeMem 是通过同步 Add/Search API 提供原文证据检索的研究系统。部署代码固定为 v1 分支的 9fc8797fa221bcb517cb7a5fe67d540a4e94421b，入口 memory.aml_api:app，由参赛方自行维护。Add 按 user_id/session_id 保存分块原文、向量及程序生成的溯源位置；Search 使用 dense+BM25/RRF、上下文/目标句重排、可靠人物/日期元数据和按需二次补检，返回 top_k 原文，不生成答案。可选人物/日期元数据缺失时不自行编造。

当前模型为服务器已部署的 Qwen3-Embedding-0.6B 和 Qwen3-Reranker-0.6B，通过内网 vLLM API 调用。当前 Add/Search 不调用生成式 LLM；LLM_MODEL=gpt-4o-mini 只用于未启用的历史 graph 后端。申请表对非工业榜有 gpt-4o-mini 限制，而英文 API 指南表示不指定 embedding 模型，请审核方确认该无生成式 LLM、Qwen embedding/reranker 方案是否允许进入学术/开源方法榜；本说明不宣称已经获得豁免。

接口使用独立 Bearer Key；同步 Add 成功即持久化并可检索，无异步状态端点。SQLite 持久化、单进程运行，配置 Add 并发上限 1、Search 上限 4，超限返回 429、Retry-After: 5；模型请求超时 120 秒，内部锁和共享模型可能限制实际吞吐。现有公网接口是 HTTP，无 TLS。已经通过自建合成 HTTP 冒烟，验证幂等、跨会话检索、top_k、原文返回与用户隔离；尚未完成官方 Smoke/Full 或 Qwen 全量准确率与并发测试。公开报告中的 BGE/MiniLM 结果不作为此 Qwen 版本成绩。

公开代码与运行指南：https://github.com/fyuo863/GeeMem/tree/v1 。基础检索方法参考 wenxiaof345-ctrl/vanilla-rag-memory（原仓库作者账户 wenxiaof345-ctrl，参考提交 31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca），独立实现，没有引入其源文件。新增可靠元数据、目标句归因、二次补检、原文溯源、AML 接口和 Qwen API 适配。方法说明及消融报告位于 README.md、docs/v1-vanilla-rag.md、docs/evidence-ablation-20261001.md，无另行发表的技术论文。
