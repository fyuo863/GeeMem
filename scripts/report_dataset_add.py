"""Human audit notes after inspecting every saved source and model output."""
import json
import statistics
from pathlib import Path
from collections import Counter,defaultdict

ROOT=Path('data/add-dataset-audit-20261008')
# Coverage refers to structured target clues, not downstream answer accuracy.
NOTES=[
('John 在篮球技术以外有哪些职业目标？','完整','慈善、代言和个人品牌均进入事件；但把多个目标重复写入画像，还把童年到被球队选中的经历压成画像属性。'),
('John 练过哪些武术？','部分','跆拳道进入计划；kickboxing 未生成记录。分类器未把 kickboxing 选入 event；画像虽看见该消息却返回空，说明路由与模块边界共同造成漏写。'),
('Caroline 研究了什么？','完整','研究领养机构正确进入 ongoing 事件；但 Melanie 的露营计划、赞美语混进画像，next month 未解析且缺参考时间。'),
('Jon 与 Gina 有哪些共同点？','部分','一条 event 路由 ValueError/502，失业和创业事件链不完整；画像包含问候语、失业事件及创业计划。后续 Gina 广告事件已写入但出现字符串 null。错误日志只有类型，不能断定 ValueError 的具体校验项。'),
('John 与 Maria 都做过什么志愿服务？','部分','John 在收容所分发物资已保存；Maria 只写捐车，未明确保存长期志愿服务。2022-12-22 说 yesterday 应为 12-21，却写成 12-22。'),
('Dave 有哪些梦想？','完整','开店、经典汽车、从零造车都保留；但 maybe 被标 confirmed，人物画像推断 mechanic，Cal 别名引用的是 Calvin 的回答而非称呼原文，且存在字符串 null。'),
('Caroline 何时参加 LGBTQ 支持小组？','部分','事件和 end=2023-05-07 正确，但 start 是字符串 null，日期字段不对称；不能视为完整可排序日期。'),
('Jon 何时失去银行工作？','错误','2023-01-20 的 yesterday 应为 01-19，模型写成 01-16，置信度仍为 1；画像还保留 banker 而未明确旧职业，并写入问候语。'),
('Audrey 哪年收养前三只狗？','完整','宠物名及持有三年准确入画像，原证据和时间保留；未推断收养日期合理，但三年是随参考时间变化的属性，尚无结构化时间锚点。'),
('James 在 2022-03-16 做了什么休闲活动？','缺失','分类已选中保龄球原文，event 却只输出游戏进展与邀请，保龄球完全漏抽；perhaps 的邀请还被加强为 confirmed。'),
('Jolene 在 2023 年 1 月初做什么项目？','完整','电气工程项目完成正确保存；last week 未归一化，字符串 null 污染。题目时间与所选原文的 last week 不完全一致，不能用标准答案强行填日期。'),
('Calvin 何时第一次去东京？','完整','东京音乐活动正确保留；原文只有 just went，不足以推出完整答案区间，保留未知时间合理。缺失值用字符串 null 不合理。'),
('Nate 除 Joanna 外可能还有朋友吗？','完整','与团队玩 CS:GO 的事实保留，没有把队友推断成确定朋友，符合写入边界；仍有字符串 null。'),
('Caroline 可能继续学习哪些领域？','完整','教育计划、咨询和心理健康兴趣均保存；画像重复存教育计划，并把明确兴趣与自我感受标 uncertain，事件则把兴趣增强为确定计划。'),
('James 在 2022 年 4 月有女朋友吗？','部分','保留陌生人外貌激发角色设计及一见钟情，但把 two weeks ago 挂到角色创建而非散步，未建立独立散步事件；unknown 实体及短暂情绪污染画像。不应直接写“没有女朋友”。'),
('Nate 如何昵称 Joanna？','缺失','只写染发事件，没有 Jo 别名记录；片段中只有 Jo，缺乏显式 Joanna 映射，不应强行合并，但应保存待解析称呼线索。'),
('John 描述的彩色数字牌游戏是什么？','原文保留','路由 other_memory/stored_only，游戏玩法原文仍在向量库；没有结构化记录，也没有无依据写入 UNO。此题需要外部知识推断，不能按漏写标准答案判错。'),
('John 算爱国的人吗？','完整','参军服务兴趣及家人朋友支持线索保存；但旅行建筑回忆被误写为 Maria residence=London，聊天已发生却标 ongoing，family/friends 是泛称实体。'),
('Jon 和 Gina 如何减压？','部分','只写 Jon 喜欢舞蹈和开舞蹈室计划；Gina 明确说用舞蹈减压，却未被 profile 选中，属于分类器选消息遗漏。'),
('Jon 为什么决定开舞蹈室？','部分','event 路由失败，画像只存失业和创业，未完整体现热爱舞蹈与分享动机；不能因原文仍在就称事件构建成功。'),
('Jolene 喜欢哪些书？','缺失','Sapiens 的明确偏好与 Avalanche 阅读经历全部进 other_memory，没有画像或事件。标准答案将读过与喜欢并列，不应机械把 Avalanche 升格最爱，但 Sapiens 偏好确实漏写。'),
('Jolene 和伴侣在一起多久？','完整','三年、未婚与 partner 关系保存；但确定信息标 uncertain，Deborah 仅提问却被推断 friend，戒指描述混入人物属性，未命名伴侣使用通用 unknown。'),
('Gina 最喜欢哪种舞蹈？','缺失','原文确认 contemporary，但整段进 other_memory，没有画像。'),
('Jon 最喜欢哪种舞蹈？','缺失','profile 只选前文一般舞蹈热爱，没有选中 contemporary is my top pick；存了泛化兴趣却漏掉更具体偏好。'),
]

def main():
    cases=[json.loads((ROOT/f'{i:02}.json').read_text(encoding='utf8')) for i in range(1,25)]
    requests=[x for c in cases for x in c['requests']]
    routes=[x for c in cases for x in c['routes']]
    record_count=Counter(); evidence_count=0; bad_links=0
    for c in cases:
        sources={}
        from memory.provenance import source_id
        for req in c['requests']:
            p=req['payload']
            for i,m in enumerate(p['messages']):sources[source_id(p['user_id'],p['request_id'],i)]=m['content']
        for table,rows in c['records'].items():record_count[table]+=len(rows)
        for rows in c['evidence'].values():
            for e in rows:
                evidence_count+=1
                bad_links+=sources.get(e['source_id'])!=e['content']
    metrics=dict(cases=24,requests=len(requests),http_success=sum(r['status']==200 for r in requests),
        cases_without_http_error=sum(all(r['status']==200 for r in c['requests']) for c in cases),
        routes=dict(Counter(r['status'] for r in routes)),records=dict(record_count),
        evidence_links=evidence_count,invalid_source_links=bad_links,
        mean_request_seconds=statistics.mean(r['seconds'] for r in requests),
        p95_request_seconds=sorted(r['seconds'] for r in requests)[int(.95*len(requests))],
        mean_case_seconds=statistics.mean(c['elapsed_s'] for c in cases),
        manual_target_coverage=dict(Counter(n[1] for n in NOTES)))
    (ROOT/'summary.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 24 题 /add 写入逐题审计','',
        '代码基线 c339259。LoCoMo-Refined public，category 1/2/3/4 各 6 题。筛除多模态题目，证据及前两条消息只传文本、speaker、role 和会话时间。会话未声明时区，统一按 UTC 适配，不声称真实时区。',
        '题目和答案不传模型；按题独立用户、按会话调用 /add。选择证据片段是有监督的小测，不代表全会话或在线分布。标签/重排关闭，真实 .env LLM 与 embedding。没有更改业务代码，也没有重试掩盖失败。',
        '分类没有官方记忆类型标注，不能报告严格分类准确率。人工覆盖评级针对结构化题目线索；完整覆盖不等于所有写入字段正确，更不是回答准确率。原文保留单独记账。',
        '', '## 实测指标','', '```json',json.dumps(metrics,ensure_ascii=False,indent=2),'```','',
        '来源链接通过仅证明 ID 和原文一致，不证明引用语义正确。', '', '## 逐题检查','']
    for c,(zh,coverage,note) in zip(cases,NOTES):
        q=c['question']; lines += [f"### {c['number']:02}. {q['qa_id']} · category {q['category']}",'',
            f"原题：{q['question']}",f'译文：{zh}',f"数据集参考答案：{q['answer']}",
            f'结构化关键线索覆盖：**{coverage}**',f'人工检查：{note}','', '输入原文：','']
        for req in c['requests']:
            for m in req['source']:lines.append(f"- {m['dia_id']} / {m['speaker']}：{m['text']}")
        lines += ['', '分类与构建输出：','']
        for r in c['routes']:
            lines += [f"- {r['request_id']} / {r['memory_type']} / 选中索引 {r['message_indices']} / {r['status']} / {r['error'] or '无运行错误'}"]
            if r['extraction']:lines += ['```json',json.dumps(json.loads(r['extraction']),ensure_ascii=False,indent=2),'```']
        lines += ['',f"完整数据库记录与证据：[原始结果]({c['number']:02}.json)",'']
    (ROOT/'audit.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(metrics,ensure_ascii=False))

if __name__=='__main__':main()
