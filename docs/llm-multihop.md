# LLM 多跳实验

来源绑定与结构化证据需求的可选增强，见 [实现与消融测试说明](multihop-evidence-repair.md)。两个增强开关默认关闭，便于与原多跳版本比较。

分支 `codex/llm-multihop`，从 `main` 的 `b90ea62` 创建。没有迁入 `codex/evidence-memory` 的事实、规则多跳或证据窗口模块。

## Search 流程

1. gpt-4o-mini 将原问题分类为 direct、split 或 chain。direct 不改写；split 返回最多三个可独立执行的问题；chain 只提出当前可执行的一步。
2. 原问题始终执行一次基线检索，作为候选及失败回退结果。首批子查询使用同一 user_id 和调用者 options；模型无权指定检索用户。
3. 独立查询以最多三个工作线程调度，结果按查询顺序合并；底层共享模型的推理锁仍保留，GPU 推理不会不受控并发。
4. 每轮将已选中间证据及各路前列结果提供给模型检查。模型返回充分性判断、所需证据引用、缺失信息和下一轮查询，不返回最终答案。
5. 程序校验查询桥接词在所引用问题/原文中真实出现，而且该词出现在新查询中；拒绝未知 ID、无出处桥接词及重复查询。只有真实的候选原文 ID 能用于证据引用，引用文字必须存在于实际展示给模型的原文片段。
6. 在证据充分、没有可执行新查询或达到预算时停止。候选按块 ID 去重；结合原问题 MiniLM 评分名次与各检索路的 RRF。模型引用且程序验证来源的证据优先，保护中间链条不被单独按原问题相关性排出。
7. 返回 `data:[{id,content,score,created_at?}]`，content 始终取自存储的原文块。模型规划、解释和引用均不写入记忆，也不加入公开 API 响应。

引用校验验证出处，不证明语义正确。模型仍可能错误判断关系、范围是否完整、证据是否足够。直接给中间证据优先级可能把错误引用推到前面，必须通过回归集检验；不是已证明优于基线的排序方案。

若模型给出的桥接片段为 `spouse of Leona`，而子查询为 `Who is Leona married to?`，程序允许把桥接片段缩短为其中唯一且在子查询中完整出现的英文人名 `Leona`，并记录 shortened_bridges。原桥接片段仍必须先在所引用原文中完整出现；不从别处猜人物，也不修复不在来源中的桥接词。

## 配置

只读取项目根目录 `.env`，不读取系统环境变量；LLM HTTP 客户端设置 `trust_env=False`。默认关闭，不需要额外 LLM 调用。

```dotenv
RAG_MULTIHOP_MODE=llm
LLM_MODEL=gpt-4o-mini
RAG_RESULT_WINDOW=0
RAG_MULTIHOP_ROUNDS=3
RAG_MULTIHOP_QUERIES=6
RAG_MULTIHOP_CANDIDATES=20
RAG_MULTIHOP_EVIDENCE=12
RAG_MULTIHOP_EVIDENCE_CHARS=1200
RAG_MULTIHOP_LLM_TIMEOUT=20
RAG_MULTIHOP_SECONDS=90
```

复用 `.env` 现有 LLM_BASE_URL、LLM_API_KEY 和 LLM_PROXY，以及现有 embedding/reranker。这里只启用检索编排，不更换模型，也不要求重建向量库。测试使用隔离数据库。

- queries 是每次 Search 的总检索调用上限，包含一次原问题检索，默认总计最多 6 次。
- rounds 是证据检查轮数，默认 3；路由一次加检查最多三次，最多 4 次 LLM 调用，不进行隐藏重试。
- candidates 是每条子查询返回候选数，默认 20。
- evidence 是每次检查展示的最大原文块数，默认 12；每块最多展示前 1,200 字符。之前引用的中间证据优先，其余由各路轮流补齐，避免一路占满提示词。
- seconds 是开始新步骤前检查的总时间预算，**不是强制中断所有底层检索的硬时限**；单次 LLM 网络阶段另受 timeout 限制。线程池会等待已经启动的任务结束。

截断可能遮住证据；小 top_k 可能放不下完整链条。内部 trace 的 supports_fit 会指出经校验引用是否能全部容纳，不能因 LLM 判断 sufficient 就声称返回结果已足够。

## 回退、排序与诊断

LLM 连接失败、结构化输出不合法、可捕获的子检索网络错误或融合评分错误时，返回原问题的基线结果。基线检索本身不可用仍会按原服务错误路径处理，不伪造成功。

内部 `store.search(payload,trace={})` 可记录策略、模型调用次数、检索次数、每轮具体子查询、已展示原文 ID、引用、缺口、拒绝数量、停止原因及 fallback。trace 是单请求局部对象，不跨用户共享；真实 API 不暴露它。测试脚本将其保存在本地 data 下，里面可能含原文引用，不应作为普通生产日志公开。

启用编排后的 score 是融合排名信号：原问题重排名次的 RRF 加各路名次 RRF，经过引用校验的证据再获得 1.0 的优先层偏移。它不是模型置信概率，不与关闭编排时的原始 MiniLM 分数比较。全程按最终 score 降序返回。块 ID 去重不会自动合并同一消息的不同重叠块。

## 测试

实际小范围结果见 [2026-10-03 测试报告](llm-multihop-test-20261003.md)：前五证据集中度改善，但真实下一跳存在来源绑定失败，尚不适合替换生产版本。

```powershell
python -m pytest -q -p no:cacheprovider
python scripts/test_multihop_live.py --out data/multihop/new-synthetic-run
python scripts/test_multihop_live.py --out data/multihop/new-public-run --public --public-cache data/evidence-memory/longmemeval-pilot-20261002-cuda/baseline.sqlite3
python scripts/test_multihop_live.py --out data/multihop/new-chain-check --case-id synthetic-3 --top-k 3 --subquery-candidates 1
```

输出目录必须不存在，防止覆盖结果。第二条运行四个自建场景（直接、拆分、两级关系、三级关系），第三条运行之前相同的八题公开 LongMemEval-S 完整历史回归样本。缓存只复制到新库，并由 VanillaMemory 的 embedding/chunk identity 检查兼容性；不会修改原数据库。脚本只在测试进程中切换 off/llm，不改 `.env`。

适量分层测试（六类各五题，排除上述八题与拒答题，保留每题完整历史）：

本次实际结果见 [30 题分层对比报告](llm-multihop-stratified30-20261003.md)：Recall@10 为 78.44% → 80.11%，Hit@10 持平，平均耗时约 5.61 倍。

```powershell
python scripts/test_multihop_full.py --out data/multihop/new-stratified30 --per-category 5 --embedding-workers 6
# 中断后使用相同目录、选择参数及源码/配置续跑：
python scripts/test_multihop_full.py --out data/multihop/new-stratified30 --per-category 5 --embedding-workers 6 --resume
```

该脚本仅缓存文档向量以加速 Add 准备，Search 的查询编码仍实际调用 API。每条结果落盘，可从中断处继续；编码并发只影响准备速度，不改变两版检索实现。省略 `--per-category` 才会运行全部 500 题。拒答题和没有消息级 gold 的题不纳入证据召回分母，分别计数；检索接口不生成答案，不能由此评价拒答正确率。

脚本通过真实 FastAPI Add/Search 处理器测试响应、原文一致性、用户范围、唯一 ID 和降序分数。它报告 Hit、题均 Recall、Micro Recall、全证据命中@3/5/10 及耗时，另列 fallback_count。若全部回退，这些数值只验证基线路径与回退开销，**不能当作多跳性能**。人工构造样例用于功能验证，公开八题已用于此前分析，只能作为开发回归，均不是官网成绩。不会生成最终答案。
