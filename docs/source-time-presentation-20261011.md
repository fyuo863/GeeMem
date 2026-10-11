# 来源日期锚点呈现修复

## 已确认原因与边界

官方 smoke 日志按完成顺序第1题：`When Jon has lost his job as a banker?`

首条原文：`Jon: Hey Gina! Good to see you too. Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business.`

来源 ID 为 `fef7edff46729e57e639038aee8c8262597e5e7f3204f78505d87b3ccf769a46`，`created_at=2023-01-20T16:04:00Z`。该题时间预算耗尽，未完成审查。普通来源正文没有来源时间，证据包正文才有来源时间。官方写回 `Yesterday`、0分，不能据此断言上层一定忽略字段或此次修改必然改变官方分数。

## 实现

在 VanillaMemory 最终返回处统一调用 `present_sources`，在检索、重排、选择、状态选择和证据打包之后处理；包括直接检索及规划失败回退。

- 有效、含时区的单条来源：在 content 前附 `source_id`、`source_time`，说明是消息时间而非事件时间，原文正文完整保留。
- 多来源证据包已有逐条时间标签，使用程序持有的 bundle ID 映射识别，不重复包装、不赋予整包一个日期。
- 缺失、非法、无时区、timestamp=0 的时间不猜测，不使用提问日期或当前时间替代。
- 不改变数据库原文、来源 ID、分数、顺序、top_k 或 created_at；不更改向量输入、重排输入和审查引用；不调用模型、不计算失业日期。
- `source_presentation` 审计记录带日期标签的来源 ID、未知时间来源 ID 和证据包 ID。

Jon 的返回 content 现在同时含 `source_time="2023-01-20T16:04:00Z"` 和原句 `Lost my job as a banker yesterday`。此时间是传入消息的时间，不自行替换成公开数据集日期。引用、转述或显式指定日期的段落可能使用其他时间锚点，标签中明确提示这一边界。

## 验证

相关检索/API/隐私/时间/证据包回归共146项通过；补充审计断言后，来源呈现与审计18项复测通过（属于前述覆盖的子集，不累加）。覆盖规划超时、原子检索、用户隔离、未知日期、时区偏移、零时间戳、长原文、证据包不重复包装，以及原文不入库改写。

回放48条官方日志的最终响应：989个单条来源附加日期标签，56个证据包保留原格式，335个无有效时间来源保持原样。每条响应的顺序、数量、ID、分数、created_at 与原文正文均逐项验证不变。数字按响应出现次数统计，不是去重来源数。

回放产物：`data/selection-fix/date-presentation-replay.json`（Git忽略）。只测试返回呈现，不重新调用LLM、embedding或reranker，不是官方重新评分。

成本是每条已知时间的普通结果增加一段元数据文本；对本地召回调用次数和原文排序无影响，但上层模型输入会略增长。

修改仅本机，尚未提交或部署。
