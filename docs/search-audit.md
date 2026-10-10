# 检索审计日志

新增独立详细日志，不改变检索、评分和返回接口，不增加 LLM 调用。

项目根目录 `.env` 配置：

```dotenv
RAG_SEARCH_AUDIT_PATH=data/logs/search-audit.jsonl
RAG_SEARCH_AUDIT_MAX_BYTES=20971520
RAG_SEARCH_AUDIT_BACKUPS=5
```

路径留空关闭。修改后重新创建后端或重启服务。每文件约 20 MiB，保留 5 个轮转文件；单条记录不截断，超大请求可能超过阈值。按容量轮转，不按日期删除。当前单进程服务可直接使用；多个进程须分配不同文件。

## 内容

每个后端 search 请求一条 JSONL，成功和异常均记录。请求开始生成 `trace_id`，与位置日志相同；它不是官方题号，不加入响应 JSON。

- `request`：用户 ID、问题、选项、top_k、参考时间、时区。
- `trace.initial_plan`：策略和初始查询。
- `trace.retrieval_queries`：实际子查询、阶段排名、重排标志与异常类型/HTTP 状态。
- `trace.audit_review_inputs`：标准多跳流程每轮实际提供给模型的截取后证据、历史和解析后的模型输出。供应商解析失败时可能只有输入，没有输出。
- `trace.rounds`：审查判断、缺口、下一跳及来源选择。
- `trace.bundle_sources/final_source_positions`：包成员与最终位置。
- `response`：披露策略处理后后端最终返回的原文、ID 和分数。
- `trace.fallback/error_phase/error_type/cause_type/http_status`：回退及已捕获的原因。
- `error`：未恢复的异常类型和可获得的 HTTP 状态，不保存异常字符串、供应商 URL、请求头或配置。
- 开始/结束时间、耗时、读取版本和调用计数（若流程提供）。

日志包含问题和原文，放在 Git 忽略的 data/ 下，仅供诊断；部署时限制目录访问，不公开托管。原位置日志仍不包含原文。审计写盘失败仅输出固定告警 search_audit_write_failed，不改变结果。

范围是进入 VanillaMemory.search 的请求；鉴权/格式校验失败及进程强制终止不保证有记录。未保存供应商原始 HTTP 输出及逐次重试。旧 smoke 问题无法补录。模型判断充分不等于客观证据完整。

## 验证与启用

57 项回归测试通过：新审计、位置日志、多跳、原文包及 AML 接口。覆盖本地真实写入检索、模拟两跳规划、并发关联、502、回退、轮转、默认关闭、写盘失败。未进行新的真实 LLM 准确率评测。

smoke-raw-audit-20261010 发布配置在本机和服务器启用。正式评测日志及备份属于评测数据派生副本，应与数据库按规定一起清理；容量轮转不替代到期清理。
