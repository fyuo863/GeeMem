# text-embedding-v4 与 BGE＋FM 完整对比（2026-10-02）

两者均为 ABC＋方案1＋方案3、MiniLM 重排；仅 embedding 及其请求前缀切换为对应模型设置。新模型为百炼官方 text-embedding-v4、1024维，地址为 https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings 。该地址已用用户配置的 Key 实测成功，无需猜测杭州专属域名。

10 组公开对话、5,882 条消息重新经真实 HTTP /add 写入独立新库；全部 861 道纯文本问题经 /search 请求 top_k=100，859 道有效证据题计分，1,074 条证据出现次数。没有调用生成式 LLM 或官方评测。

| 指标 | BGE＋FM | text-embedding-v4＋FM | 变化（百分点） |
|---|---:|---:|---:|
| Hit@5 | 87.78% | 87.78% | +0.00 |
| Recall@5 | 84.28% | 84.34% | +0.06 |
| Micro Recall@5 | 78.68% | 78.77% | +0.09 |
| 全证据命中@5 | 80.68% | 80.79% | +0.12 |
| Hit@10 | 93.25% | 93.25% | +0.00 |
| Recall@10 | 90.55% | 90.51% | -0.04 |
| Micro Recall@10 | 86.13% | 86.03% | -0.09 |
| 全证据命中@10 | 87.43% | 87.43% | +0.00 |

## Top100 与时延

| 指标 | text-embedding-v4＋FM |
|---|---:|
| Hit@100 | 98.60% |
| Recall@100 | 97.57% |
| Micro Recall@100 | 95.90% |
| 全证据命中@100 | 96.51% |

重新写入 HTTP 请求累计 540.52 秒；完整 861 题 HTTP Search 平均 0.852 秒，P50 0.800 秒，P95 1.097 秒。包含 embedding API 网络与本地 MiniLM 推理；不包含服务启动。旧 BGE 692ms 是60题引擎内抽样计时，不能直接计算延迟增幅。

## 结果与复现边界

Hit@10 命中集合完全一致：均为 801/859，无新增命中题、无退化命中题。证据总命中从 925 变为 924；不同题目的部分证据变化如下：

- conv-48 / q74：新增 无，丢失 ['D29:26']。

- 数据集 SHA256 与此前冻结 BGE 实验一致；除 embedding 相关项、数据库路径和等效的 FM 开关外，检索参数一致。
- BGE 使用此前冻结 FM 排名，未重新测量其 HTTP 延迟；Top100 没有对应 BGE 参考，因此不报告 Top100 增幅。
- 测试包含公开集的 caller-supplied speaker/session_timestamp，保持与历史对照一致；不保证官方请求包含这两项。缺少扩展元数据的合成 smoke 已通过，但未据此宣称同等准确率。
- 用户提供的 Key 只保存在本地 .env 及权限为0600的远端隔离测试 .env；结果中不包含密钥。
- 原文、ID、排名顺序、返回数量和用户分区检查全部通过。118项自动测试通过；真实模型合成 smoke 通过。
- 临时服务已停止，正式服务健康检查通过，生产仍使用原 BGE。此迁移分支尚未推送、部署为正式版本或修改报名表。
- 新模型满足官网 FAQ 的学术榜 embedding 指定名称，但不等于完整报名资格或官方 Smoke/Full 已通过。

原始记录：data/embedding-v4-20261002/{manifest.json,ingestion.json,cases.jsonl,report.json}。远端实验目录：/GeeAI/AIAgent/CSIG/shared/benchmarks/embedding-v4-20261002。
