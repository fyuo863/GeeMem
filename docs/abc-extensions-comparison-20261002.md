# ABC 三项改进的八组合完整测试

F：上下文/目标融合与局部问答关联；Q：多人子查询与排序合并；M：人物/日期软召回。表中 F、Q、M 均指在原始 ABC 上追加改进。

共同起点 ddaf4b4；独立分支 codex/abc-fusion-qa（6822ce8）、codex/abc-multi-query（fe872fa）、codex/abc-soft-recall（de43a6e）；组合分支 codex/abc-factorial。

同一远端服务器 GPU 1、相同 BGE/MiniLM、完整记忆库的独立快照。861 道纯文本问题全部执行；859 道有证据问题进入准确率统计，证据总数 1,074。

| 方案 | Hit@10 | Recall@10 | Micro Recall@10 | 全证据@10 | 新增/退化题 | 平均检索秒 | P95 秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FM | 93.25% | 90.55% | 86.13% | 87.43% | +6/−1 | 0.692 | 0.927 |
| F | 93.13% | 90.43% | 86.03% | 87.31% | +5/−1 | 0.645 | 0.865 |
| FQM | 93.02% | 90.30% | 85.75% | 87.19% | +6/−3 | 0.710 | 0.973 |
| FQ | 92.90% | 90.18% | 85.66% | 87.08% | +5/−3 | 0.655 | 0.873 |
| M | 92.78% | 90.00% | 85.66% | 86.85% | +1/−0 | 0.690 | 0.924 |
| ABC | 92.67% | 89.88% | 85.57% | 86.73% | +0/−0 | 0.641 | 0.865 |
| QM | 92.55% | 89.75% | 85.29% | 86.61% | +1/−2 | 0.709 | 0.957 |
| Q | 92.43% | 89.63% | 85.20% | 86.50% | +0/−2 | 0.655 | 0.878 |

按事先固定的选择规则：先最大化 Hit@10，平分时依次比较 Micro Recall、逐题 Recall、平均延迟。本次最优为 **FM**。Hit@10 相对 ABC 增加 0.58 个百分点；Micro Recall 增加 0.56 个百分点。

冻结 ABC 的 Top10 ID 精确一致：861/861；无缓存复核精确一致：480/480。

准确率运行共享完全相同 query/document 的模型分数，仅减少重复推理，未使用答案或金标改变检索。6888 次生产 search 路径执行完成。延迟来自预先固定的均匀抽样 60 题，每种配置独立无缓存推理且交替顺序；不含 HTTP、启动或写入，不是全量延迟或并发吞吐量。

按 10 个会话聚类重采样，FM 相对 ABC 的 Hit 差值 95% 描述性区间为 +0.12 至 +0.95 个百分点。数据已用于开发与诊断，此区间不是独立泛化保证，也未校正多方案选优偏差。

所有实验参数在本次全量测试前固定，没有根据这次逐题结果回调。最优仅指本次八个实现/组合，不代表三类方法的全局最优。线上比赛服务未被切换。

## Top5 指标

| 方案 | Hit@5 | Recall@5 | Micro Recall@5 | 全证据@5 |
|---|---:|---:|---:|---:|
| FM | 87.78% | 84.28% | 78.68% | 80.68% |
| F | 87.66% | 84.17% | 78.58% | 80.56% |
| FQM | 87.89% | 84.45% | 78.77% | 80.91% |
| FQ | 87.78% | 84.33% | 78.68% | 80.79% |
| M | 87.89% | 84.16% | 78.58% | 80.33% |
| ABC | 87.78% | 84.04% | 78.49% | 80.21% |
| QM | 88.13% | 84.38% | 78.77% | 80.56% |
| Q | 88.01% | 84.26% | 78.68% | 80.44% |

## 最优方案的新增命中与退化

### 新增命中

- conv-26 / q104：Why did Melanie choose to use colors and patterns in her pottery project?
- conv-41 / q86：How does John plan to honor the memories of his beloved pet?
- conv-42 / q78：What did Nate think of the coconut milk ice cream he made?
- conv-43 / q136：What activity did Tim do after reading the stories about the Himalayan trek?
- conv-48 / q143：Why did Jolene get the new plant on 30 August, 2023?
- conv-49 / q94：What activity does Evan do to keep himself busy while healing his knee?

### 退化

- conv-30 / q42：What did Jon say about Gina's progress with her store?

## 推荐配置与接口验证

在 `codex/abc-factorial` 分支的项目根目录 `.env` 中配置：

```dotenv
RAG_FUSION_QA=on
RAG_MULTI_QUERY=off
RAG_SOFT_RECALL=on
```

三个开关默认均为 off，以保留原 ABC 行为；组合分支包含全部实现，按上述配置启用本次最优 FM。独立分支保留各自方案，便于继续开发。

F 通过有限幅度的上下文/目标融合和相邻问答关联改善短回答的排名；M 增加人物与日期的候选入口。Q 当前只在 21/861 题触发，没有新增 Hit@10，反而使 2 道原成功题退化，因此本次不启用。F 单独使用的平均耗时接近 ABC，可作为优先低延迟的备选。

FM 在独立快照、临时服务上通过真实 HTTP `/search` 复核：60/60 道题的 Top10 ID 与完整实验完全一致，未认证请求返回 401，健康检查通过。此验证复用已写入的记忆快照，没有重新执行整库 `/add`，因为这三项改动均位于检索路径。临时服务随后停止；比赛生产部署保留原 ABC。

最终本地测试：111 passed。详细原始结果保存在 `data/abc-extensions-20261002/`（不提交数据集与逐题原文）；机器可读汇总见同目录文档 `abc-extensions-summary-20261002.json`。

