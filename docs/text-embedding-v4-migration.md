# text-embedding-v4 迁移准备

赛事页 https://agentmemories.ai/competition/#faq 第 5 条明确规定学术榜 embedding 为 text-embedding-v4，LLM 相关组件为 gpt-4o-mini，reranker 不限制。当前 BGE 冻结版本不满足此 embedding 要求；不可将它直接作为学术榜合规版本。

迁移分支：codex/text-embedding-v4。保留 ABC＋1＋3 和 MiniLM，只更换 embedding。调用阿里云百炼官方服务需要开通服务并配置该地域的 API Key；不能把本地 Qwen3-Embedding 当作 text-embedding-v4。密钥只读根目录 .env，不从系统环境变量或 OpenAI Key 回退。

根目录 .env 配置示例（北京传统兼容地址；以账户控制台实际地域/工作空间地址为准）：

```dotenv
RAG_EMBEDDING_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
RAG_EMBEDDING_API_MODEL=text-embedding-v4
RAG_EMBEDDING_API_KEY=<百炼密钥>
RAG_EMBEDDING_DIMENSIONS=1024
RAG_EMBEDDING_BATCH_SIZE=10
RAG_EMBEDDING_API_PROXY=
RAG_QUERY_PREFIX=
RAG_QUERY_INSTRUCTION=
RAG_MEMORY_DB=data/aml/text-embedding-v4.sqlite3
RAG_FUSION_QA=on
RAG_MULTI_QUERY=off
RAG_SOFT_RECALL=on
```

保持 MiniLM 本地路径与 RAG_RERANK_MODE=local，不要配置旧的 Qwen reranker API。现有 BGE 数据库保留，新库必须从原始消息经 /add 重新编码，不能只改模型名后使用旧向量。真实评测应先独立 smoke，再对公开测试集重建并评估，同题比较 Hit/Recall/Micro Recall/全证据命中及延迟。旧 BGE 的成绩不能沿用。

HTTP 适配器已支持显式 Bearer Key、每批最多 10 条、float 编码、维度配置，以及返回 index 唯一性/完整性/维度校验。不同向量维度进入数据库模型身份，防止混用。官方 text-embedding-v4 每条输入最多 8192 tokens；当前文档分块保持 320 lexical tokens，不能将二者视为相同 tokenizer 的计数。

当前尚缺可用 API Key/地域确认，真实 text-embedding-v4 连通性、重新写入及准确率测试未执行；线上仍为已部署的 BGE 版本。未修改旧最终 tag、报名表或生产数据库。
