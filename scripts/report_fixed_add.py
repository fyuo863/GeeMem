"""Paired audit with manually reviewed outcomes; no LLM grading."""
import json
from pathlib import Path
from . import report_dataset_add as base

NOTES=[
('完整','慈善、代言和品牌目标仍完整；画像仍重复写入目标，不能视为边界已修复。'),
('完整','kickboxing 与 taekwondo 均已进入事件，前者的漏抽修复；还正确保留 Maria 的瑜伽与志愿活动。'),
('完整','领养研究正确，next month 已程序解析；但画像出现 aliases=:[], 异常值，并混入活动类属性。'),
('完整','原事件路由失败已消失，双方失业、创业和舞蹈兴趣保留；日期由原文程序解析；画像仍混入失业与创业计划。'),
('部分','捐车日期纠正为 2022-12-21；John 志愿活动保留，Maria 长期志愿服务仍未独立结构化，目标覆盖未改善。'),
('完整','三项梦想保留，maybe 的造车计划改 uncertain，未直接推断 mechanic；但 Cal 别名仍引用 Calvin 回答而非称呼原文。'),
('完整','支持小组 start/end 均为 2023-05-07，无字符串 null。'),
('完整','失业日期纠正为 2023-01-19；画像仍以 uncertain banker 保存旧职业，画像时态问题未解决。'),
('完整','宠物名和三年保留，与基线一致；时变时长仍缺专用时间锚点。'),
('完整','保龄球与两次 strike 补回，日期为 2022-03-16；perhaps 的邀请改 uncertain；承诺更新仍被写 ongoing。'),
('完整','工程项目保留，last week 程序归一为 2023-01-16 至 01-22。数据集题面 January beginning 与该原文有差异，不用题目改写来源日期。'),
('完整','东京音乐事件保留；just went 不足以算日期，维持未知。'),
('完整','团队游戏线索保留，未把队友自动写成朋友；新增 had a blast 事件与游戏事件重复。'),
('完整','咨询/心理健康兴趣明确 confirmed；画像仍混入教育计划和经历。'),
('部分','保留角色设计和陌生人启发；日期未错挂，但散步时间未单独保留。出现明显退化：didn’t approach 被写成 planned approach。'),
('缺失','仍只有染发事件，Jo 未存为可解析别名线索。无显式映射不能强行合并，此版未实现独立称呼库。'),
('原文保留','游戏玩法仍 other_memory，未凭外部知识写入 UNO，合理保留原文。'),
('完整','爱国相关兴趣线索保留，Maria residence=London 误写消失；新增家人被标 friend_of，proud_of opportunity 也混入关系；已讨论仍错误 planned。'),
('部分','仍只有 Jon 舞蹈兴趣，Gina 减压漏写。完整输入没有解决模型忽略另一说话者。'),
('部分','失业和创业事件成功写入，但缺分享舞蹈热情的完整动机。运行改善不等于关键线索完整。'),
('完整','复查后进入 profile/event；Sapiens 偏好与 Avalanche 阅读事件恢复，two weeks ago 正确解析。画像仍重复存 recently read。'),
('完整','未婚和三年改 confirmed；未知伴侣按来源隔离。Deborah 的问题仍被写成 relationship status inquiry，戒指对象属性仍混入画像。'),
('完整','复查后 Gina 和 Jon 的 contemporary 偏好均保存，原分类缺失修复。'),
('完整','Jon 的 contemporary 偏好补回；同时被事件误记为 completed leisure activity，属于新增边界问题。'),
]

def main():
    old_notes=base.NOTES
    base.ROOT=Path('data/add-dataset-fixed-20261008')
    base.NOTES=[(old[0],new[0],new[1]) for old,new in zip(old_notes,NOTES)]
    base.main()
    # Correct the baseline report header for the fixed variant.
    audit=base.ROOT/'audit.md'
    text=audit.read_text(encoding='utf8').replace('代码基线 c339259。','修复实验（基线审计快照 94a2699）。').replace('没有更改业务代码，也没有重试掩盖失败。','本轮业务修复包含覆盖审计与有界结构修复；逐条保留尝试记录，不重复 HTTP 请求掩盖失败。')
    audit.write_text(text,encoding='utf8')
    before=json.loads(Path('data/add-dataset-audit-20261008/summary.json').read_text(encoding='utf8'))
    after=json.loads((base.ROOT/'summary.json').read_text(encoding='utf8'))
    cases=[json.loads((base.ROOT/f'{i:02}.json').read_text(encoding='utf8')) for i in range(1,25)]
    for i,c in enumerate(cases,1):
        old=json.loads(Path(f'data/add-dataset-audit-20261008/{i:02}.json').read_text(encoding='utf8'))
        assert c['question']==old['question']
        assert [r['payload'] for r in c['requests']]==[r['payload'] for r in old['requests']]
    after['literal_null_event_fields']=sum(v=='null' for c in cases for e in c['records']['event_records'] for v in e.values())
    after['builder_attempts']=sum(len(json.loads(r.get('audit') or '[]')) for c in cases for r in c['routes'])
    before_cases=[json.loads(Path(f'data/add-dataset-audit-20261008/{i:02}.json').read_text(encoding='utf8')) for i in range(1,25)]
    before['literal_null_event_fields']=sum(v=='null' for c in before_cases for e in c['records']['event_records'] for v in e.values())
    comparison={'before':before,'after':after,'paired_inputs_equal':True}
    (base.ROOT/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# /add 修复前后 24 题对比','',
        '相同 24 题、29 次 HTTP 请求、86 条输入消息。模型/embedding 取根 .env，标签重排关闭。按题隔离用户。单次配对实验无置信区间，不代表泛化性能。',
        '人工完整覆盖只考察关键线索，不代表所有额外写入都正确。已有 24 题被用于开发，需独立留出集后才能判断泛化。','',
        '| 指标 | 修复前 | 修复后 |','|---|---:|---:|']
    for label,key in [('HTTP 成功数','http_success'),('全部写入成功题数','cases_without_http_error'),('平均请求耗时（秒）','mean_request_seconds'),('请求 P95（秒）','p95_request_seconds'),('事件字符串 null 字段数','literal_null_event_fields'),('来源链接错误数','invalid_source_links')]:
        a,b=before[key],after[key]
        lines.append(f'| {label} | {a:.2f} | {b:.2f} |')
    lines+=['',f"关键线索覆盖：{before['manual_target_coverage']} → {after['manual_target_coverage']}",'',
        '## 逐题复核','', '| 题号 | 修复前覆盖 | 修复后覆盖 | 人工检查 |','|---|---|---|---|']
    for i,(old,new) in enumerate(zip(old_notes,NOTES),1):lines.append(f'| {i} | {old[1]} | {new[0]} | {new[1]} |')
    lines+=['','结论：时间解析和运行可靠性改善，部分关键线索补回；画像污染、说话者遗漏和否定理解仍有问题。第 15、18、24 题出现新增错误，不建议据此直接认定生产准确性达标。',
            '', '完整输入、候选、审计、数据库记录见 data/add-dataset-fixed-20261008/01.json 至 24.json；详细原文报告见该目录 audit.md。']
    Path('docs/add-write-fix-comparison-20261008.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(comparison,ensure_ascii=False))

if __name__=='__main__': main()
