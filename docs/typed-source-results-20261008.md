# 四类原文索引测试

实现前检查点：d45239f。默认 /add 的 profile、relationship、rule、event
均只保存分类选中原文的引用，构建阶段模型调用为零。

## 自动测试

全套 258 项通过。新增覆盖四类共享原文与向量、多标签、前两条上下文、
用户/会话隔离、禁止构建阶段调用模型，以及漏分类的全库补充召回。
失败恢复测试验证相同请求继续索引，不重复创建原文。

## 真实小测

使用根 .env 的分类模型和 embedding；独立数据库，关闭额外标签加工及重排。
通过 HTTP /add 写入，再调用各类型内部检索器（fallback=False）检索。
这不是外部 /search 全量质量评测，也不是与旧版的同题性能对照。

| 类型 | 原文 | 写入秒 | 检索秒 | 结果 |
|---|---|---:|---:|---|
| profile | My favorite football team is Arsenal. | 4.890 | 0.126 | 分类正确，原文返回 |
| relationship | Veda is my sister and Arun is her teacher. | 4.487 | 0.162 | 分类正确，原文返回 |
| rule | When writing reports for me, put the conclusion first and always use bullet points. | 5.068 | 0.153 | 分类正确，原文返回 |
| event | I planned to visit Shanghai on Friday, but cancelled the trip. | 4.041 | 0.118 | 分类正确，原文返回 |

4/4 HTTP 200，目标类型索引各一条，4/4 原文一致。
详情：data/typed-source-live-20261008/results.json。
每题只有一条原文，这仅验证端到端连通、分类和保真，不证明复杂库召回准确率提升。

不再自动产出画像属性、三元组、规则条件和事件日期；不自动进行实体消歧、
昵称合并或冲突裁决。分类仍可能误判，补充召回不保证所有漏选都进入 top-k。
旧结构化模块保留作历史实验，默认写入链路不再调用；历史完成路由未自动回填。
