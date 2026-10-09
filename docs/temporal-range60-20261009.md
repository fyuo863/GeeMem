# 时间增强扩大测试：60 道 LongMemEval 真实题

## 结论

Top10 有小幅收益，但 Top5 出现退化，当前不能称为全面提升，也不建议据此默认开启。

- 20 道时间题 Recall@10：82.17% → 84.83%，提高 2.67 个百分点；金标命中从 32/43 → 34/43。
- 改善仍来自上一轮两题。新增 10 道时间题 Recall@10 为 90% → 90%，没有新收益。
- 40 道其他题型对照的 Top5/Top10 指标和返回排名均未变化。
- 时间题 Recall@5：71% → 68.67%；Hit@5：90% → 85%。必须同时报告这个损失。
- 所有 60 道题 @10 为 2 题提升、58 题持平、0 题退化；@5 为 2 题提升、57 题持平、1 题退化。

### 明确退化：正确证据从第 5 名降至第 8 名

题目 `gpt4_fa19884d`：**我上周五开始听的是哪位音乐人？**

金标原文包含：

> I recently discovered a bluegrass band that features a banjo player and started enjoying their music today.

程序正确把 `today` 标为 `2023-03-31`，但该证据的重排分数从 -7.719253 降到 -8.014204，排名 5 → 8。Top10 仍命中，Top5 从命中变为未命中。

该题不触发 earliest/latest 的最终日期顺序微调。可确认：附加时间文本后的神经重排评分下降；不能假定添加正确日期就一定提高相关性。当前也没有把查询的 last Friday 按提问参考时间归一化。后者是实现缺口，但尚未通过消融证明它是本次降分的唯一原因。

### 新增时间题中完全未命中的例子

题目 `6e984302`：**我四周前提过为比赛做了一笔投入，买了什么？**

金标原文说购买了雕塑工具（modeling tool set、wire cutter、sculpting mat），并提到 today。两组 Top10 均未命中。问题使用“为比赛投入”的间接描述，单靠消息上的日期补充不足以建立对应关系。当前尚无查询参考时间归一化、明确时间区间召回和跨片段事件关联。

### 后续优先方向（本轮未修改检索实现）

1. 在固定候选池上分别比较原文评分和时间增强评分，确定降分来源，评估保留原始相关性的融合方法。
2. 为查询提供明确参考时间，将 last Friday、four weeks ago 等转换成可比较的时间约束；不能使用机器当前日期代替历史提问时间。
3. 对新增时间题和本次退化题验证改动，继续同时观察 @5 与 @10，避免只优化单一截断位置。

## 协议

- 20 道 temporal-reasoning，加其他五类各 8 道，均为非拒答题；在已有缓存中按固定哈希选取，未根据表现筛题。
- 保留各题完整会话，运行前逐一检查缓存块 ID 和原文完全一致；共 39974 块、107 条金标消息。
- text-embedding-v4 + 英文 MiniLM，重排池 400，Top10，原文窗口/多跳/分区规划关闭。两组使用同一查询向量，交替执行先后顺序。
- 使用真实本地 /search HTTP 处理链路；向量缓存与时间侧索引回填，不是重新全量 /add，也不是赛事答案正确率。
- 保留上一轮已有模型标注，本轮不新增 LLM 调用；新增会话使用规则时间索引。本实验不能代表完整 LLM 标注能力。
- 检索核心代码保持上一轮版本，仅扩展测试脚本；线上配置未更改。

## 分组 Top10 指标

| 分组 | 题数 | Hit@10 关闭→开启 | Recall@10 关闭→开启 | Micro Recall@10 关闭→开启 | 全证据命中@10 关闭→开启 | 改善/持平/退化 |
|---|---:|---:|---:|---:|---:|---:|
| 全部 | 60 | 96.67% → 96.67% | 87.22% → 88.11% | 80.37% → 82.24% | 76.67% → 78.33% | 2/58/0 |
| 时间题（全部） | 20 | 95.00% → 95.00% | 82.17% → 84.83% | 74.42% → 79.07% | 70.00% → 75.00% | 2/18/0 |
| 时间题（上一轮10题） | 10 | 100.00% → 100.00% | 74.33% → 79.67% | 62.96% → 70.37% | 50.00% → 60.00% | 2/8/0 |
| 时间题（新增） | 10 | 90.00% → 90.00% | 90.00% → 90.00% | 93.75% → 93.75% | 90.00% → 90.00% | 0/10/0 |
| 其他题型对照 | 40 | 97.50% → 97.50% | 89.75% → 89.75% | 84.38% → 84.38% | 80.00% → 80.00% | 0/40/0 |
| knowledge-update | 8 | 100.00% → 100.00% | 87.50% → 87.50% | 87.50% → 87.50% | 75.00% → 75.00% | 0/8/0 |
| multi-session | 8 | 100.00% → 100.00% | 77.92% → 77.92% | 71.43% → 71.43% | 50.00% → 50.00% | 0/8/0 |
| single-session-assistant | 8 | 100.00% → 100.00% | 100.00% → 100.00% | 100.00% → 100.00% | 100.00% → 100.00% | 0/8/0 |
| single-session-preference | 8 | 87.50% → 87.50% | 83.33% → 83.33% | 81.82% → 81.82% | 75.00% → 75.00% | 0/8/0 |
| single-session-user | 8 | 100.00% → 100.00% | 100.00% → 100.00% | 100.00% → 100.00% | 100.00% → 100.00% | 0/8/0 |

## Top5 指标

| 分组 | Hit@5 | Recall@5 | Micro Recall@5 | 全证据命中@5 |
|---|---:|---:|---:|---:|
| 全部 | 90.00% → 88.33% | 78.94% → 78.17% | 69.16% → 70.09% | 70.00% → 70.00% |
| 时间题（全部） | 90.00% → 85.00% | 71.00% → 68.67% | 60.47% → 62.79% | 60.00% → 60.00% |
| 时间题（新增） | 90.00% → 80.00% | 90.00% → 80.00% | 93.75% → 87.50% | 90.00% → 80.00% |
| 其他题型对照 | 90.00% → 90.00% | 82.92% → 82.92% | 75.00% → 75.00% | 75.00% → 75.00% |

## 排名稳定性与不确定性

- 时间题（全部）：触发时间规则 20/20，排名变化 9 题；Recall@10 差值的配对 bootstrap 95% 区间 [0.00, 7.00] 个百分点。
- 时间题（新增）：触发时间规则 10/10，排名变化 6 题；Recall@10 差值的配对 bootstrap 95% 区间 [0.00, 0.00] 个百分点。
- 其他题型对照：触发时间规则 8/40，排名变化 0 题；Recall@10 差值的配对 bootstrap 95% 区间 [0.00, 0.00] 个百分点。

区间为本样本的探索性统计，不能修复缓存样本选择偏差；新增时间题只有 10 道。其他数据集题型也可能包含时间约束，触发时间规则不自动等于误触发。

上一轮 10 题开启组排名复现：10/10。

## 逐题金标召回（@10）

| ID | 类型 | 原题 | 命中数关闭→开启 / 金标数 |
|---|---|---|---:|
| gpt4_f420262d | temporal-reasoning | What was the airline that I flied with on Valentine's day? | 1 → 1 / 3 |
| 71017276 | temporal-reasoning | How many weeks ago did I meet up with my aunt and receive the crystal chandelier? | 1 → 1 / 1 |
| gpt4_2f584639 | temporal-reasoning | Which gift did I buy first, the necklace for my sister or the photo album for my mom? | 2 → 2 / 2 |
| a3838d2b | temporal-reasoning | How many charity events did I participate in before the 'Run for the Cure' event? | 3 → 3 / 6 |
| gpt4_85da3956 | temporal-reasoning | How many weeks ago did I attend the 'Summer Nights' festival at Universal Studios Hollywood? | 1 → 1 / 1 |
| cc6d1ec1 | temporal-reasoning | How long had I been bird watching when I attended the bird watching workshop? | 2 → 2 / 2 |
| 71017277 | temporal-reasoning | I received a piece of jewelry last Saturday from whom? | 1 → 1 / 1 |
| gpt4_7f6b06db | temporal-reasoning | What is the order of the three trips I took in the past three months, from earliest to latest? | 1 → 1 / 3 |
| gpt4_45189cb4 | temporal-reasoning | What is the order of the sports events I watched in January? | 2 → 3 / 3 |
| b46e15ed | temporal-reasoning | How many months have passed since I participated in two charity events in a row, on consecutive days? | 3 → 4 / 5 |
| gpt4_1a1dc16d | temporal-reasoning | Which event happened first, the meeting with Rachel or the pride parade? | 2 → 2 / 2 |
| c9f37c46 | temporal-reasoning | How long had I been watching stand-up comedy specials regularly when I attended the open mic night at the local comedy club? | 2 → 2 / 2 |
| 370a8ff4 | temporal-reasoning | How many weeks had passed since I recovered from the flu when I went on my 10th jog outdoors? | 2 → 2 / 2 |
| gpt4_1e4a8aeb | temporal-reasoning | How many days passed between the day I attended the gardening workshop and the day I planted the tomato saplings? | 2 → 2 / 2 |
| gpt4_fa19884d | temporal-reasoning | What is the artist that I started to listen to last Friday? | 1 → 1 / 1 |
| gpt4_4edbafa2 | temporal-reasoning | What was the date on which I attended the first BBQ event in June? | 2 → 2 / 2 |
| 4dfccbf8 | temporal-reasoning | What did I do with Rachel on the Wednesday two months ago? | 1 → 1 / 1 |
| gpt4_7bc6cf22 | temporal-reasoning | How many days ago did I read the March 15th issue of The New Yorker? | 1 → 1 / 1 |
| c8090214 | temporal-reasoning | How many days before I bought the iPhone 13 Pro did I attend the Holiday Market? | 2 → 2 / 2 |
| 6e984302 | temporal-reasoning | I mentioned an investment for a competition four weeks ago? What did I buy? | 0 → 0 / 1 |
| c960da58 | single-session-user | How many playlists do I have on Spotify? | 1 → 1 / 1 |
| 86f00804 | single-session-user | What book am I currently reading? | 1 → 1 / 1 |
| 118b2229 | single-session-user | How long is my daily commute to work? | 1 → 1 / 1 |
| dccbc061 | single-session-user | What was my previous stance on spirituality? | 1 → 1 / 1 |
| 4fd1909e | single-session-user | Where did I attend the Imagine Dragons concert? | 1 → 1 / 1 |
| faba32e5 | single-session-user | How long did Alex marinate the BBQ ribs in special sauce? | 1 → 1 / 1 |
| 8550ddae | single-session-user | What type of cocktail recipe did I try last weekend? | 1 → 1 / 1 |
| 3b6f954b | single-session-user | Where did I attend for my study abroad program? | 1 → 1 / 1 |
| e8a79c70 | single-session-assistant | I was going through our previous conversation about making a classic French omelette, and I wanted to confirm - how many eggs did you say we need for the recipe? | 1 → 1 / 1 |
| 5809eb10 | single-session-assistant | I'm looking back at our previous conversation about the Bajimaya v Reward Homes Pty Ltd case. Can you remind me what year the construction of the house began? | 1 → 1 / 1 |
| 6ae235be | single-session-assistant | I remember you told me about the refining processes at CITGO's three refineries earlier. Can you remind me what kind of processes are used at the Lake Charles Refinery? | 1 → 1 / 1 |
| b759caee | single-session-assistant | I was looking back at our previous conversation about buying unique engagement rings directly from designers. Can you remind me of the Instagram handle of the UK-based designer who works with unusual gemstones? | 1 → 1 / 1 |
| e3fc4d6e | single-session-assistant | I wanted to follow up on our previous conversation about the fusion breakthrough at Lawrence Livermore National Laboratory. Can you remind me who is the President's Chief Advisor for Science and Technology mentioned in the article? | 1 → 1 / 1 |
| ceb54acb | single-session-assistant | In our previous chat, you suggested 'sexual compulsions' and a few other options for alternative terms for certain behaviors. Can you remind me what the other four options were? | 1 → 1 / 1 |
| 4c36ccef | single-session-assistant | Can you remind me of the name of the romantic Italian restaurant in Rome you recommended for dinner? | 1 → 1 / 1 |
| 7e00a6cb | single-session-assistant | I'm planning my trip to Amsterdam again and I was wondering, what was the name of that hostel near the Red Light District that you recommended last time? | 1 → 1 / 1 |
| 1a1907b4 | single-session-preference | I've been thinking about making a cocktail for an upcoming get-together, but I'm not sure which one to choose. Any suggestions? | 1 → 1 / 1 |
| 32260d93 | single-session-preference | Can you recommend a show or movie for me to watch tonight? | 1 → 1 / 1 |
| 6b7dfb22 | single-session-preference | I've been feeling a bit stuck with my paintings lately. Do you have any ideas on how I can find new inspiration? | 1 → 1 / 1 |
| 95228167 | single-session-preference | I'm getting excited about my visit to the music store this weekend. Any tips on what to look for in a new guitar? | 0 → 0 / 1 |
| b6025781 | single-session-preference | I'm planning my meal prep next week, any suggestions for new recipes? | 1 → 1 / 1 |
| 505af2f5 | single-session-preference | I was thinking of trying a new coffee creamer recipe. Any recommendations? | 1 → 1 / 1 |
| 35a27287 | single-session-preference | Can you recommend some interesting cultural events happening around me this weekend? | 2 → 2 / 3 |
| 07b6f563 | single-session-preference | Can you suggest some useful accessories for my phone? | 2 → 2 / 2 |
| gpt4_f2262a51 | multi-session | How many different doctors did I visit? | 2 → 2 / 5 |
| a346bb18 | multi-session | How many minutes did I exceed my target time by in the marathon? | 2 → 2 / 2 |
| gpt4_2f8be40d | multi-session | How many weddings have I attended in this year? | 2 → 2 / 3 |
| ba358f49 | multi-session | How many years will I be when my friend Rachel gets married? | 1 → 1 / 2 |
| 1c549ce4 | multi-session | What is the total cost of the car cover and detailing spray I purchased? | 2 → 2 / 2 |
| 1a8a66a6 | multi-session | How many magazine subscriptions do I currently have? | 2 → 2 / 3 |
| e25c3b8d | multi-session | How much did I save on the designer handbag at TK Maxx? | 2 → 2 / 2 |
| 92a0aa75 | multi-session | How long have I been working in my current role? | 2 → 2 / 2 |
| 0977f2af | knowledge-update | What new kitchen gadget did I invest in before getting the Air Fryer? | 1 → 1 / 2 |
| 08e075c7 | knowledge-update | How long have I been using my Fitbit Charge 3? | 2 → 2 / 2 |
| 6a27ffc2 | knowledge-update | How many videos of Corey Schafer's Python programming series have I completed so far? | 2 → 2 / 2 |
| dad224aa | knowledge-update | What time do I wake up on Saturday mornings? | 1 → 1 / 2 |
| 26bdc477 | knowledge-update | How many trips have I taken my Canon EOS 80D camera on? | 2 → 2 / 2 |
| ce6d2d27 | knowledge-update | What day of the week do I take a cocktail-making class? | 2 → 2 / 2 |
| 42ec0761 | knowledge-update | Do I have a spare screwdriver for opening up my laptop? | 2 → 2 / 2 |
| db467c8c | knowledge-update | How long have my parents been staying with me in the US? | 2 → 2 / 2 |

原始运行目录：`data\temporal-sequence-live\20261009-range60`。`cases.jsonl` 保留每题两组原文结果，`comparison.json` 保存新增/丢失金标消息 ID，`manifest.json` 保存配置及代码/数据集哈希。
