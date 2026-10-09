"""Summarize fixed 60-question window ablation and four-question pool replay."""
import json
from pathlib import Path
from test_temporal_live import metrics


def read(folder):
    return [json.loads(line) for line in (folder/'cases.jsonl').read_text(encoding='utf8').splitlines()]


def main():
    root = Path(__file__).resolve().parents[1]
    runs = root/'data/temporal-sequence-live'
    cases = read(runs/'20261009-window60')
    diag = read(runs/'20261009-window-diagnostic4')
    previous = {c['id']: c for c in read(runs/'20261009-absolute-query60') if c['variant']=='on'}
    baseline = {c['id']: c for c in cases if c['variant']=='off'}
    assert len(baseline) == 60 and set(baseline) == set(previous)
    assert all(c['ranked_messages']==previous[i]['ranked_messages'] for i,c in baseline.items())
    assert all(c['ranked_messages']==baseline[c['id']]['ranked_messages'] for c in diag)
    groups = {name: {arm: {str(k):metrics([c for c in cases if c['variant']==arm and predicate(c)],k)
                          for k in (5,10)} for arm in ('off','on')}
              for name,predicate in [('全部60题',lambda c:True),
                                     ('时间20题',lambda c:c['category']=='temporal-reasoning'),
                                     ('对照40题',lambda c:c['category']!='temporal-reasoning')]}
    details = []
    for c in diag:
        if c['variant']!='on': continue
        trace=c['trace']; window=trace['temporal_window']
        other=next(x for x in diag if x['id']==c['id'] and x['variant']=='off')
        details.append(dict(id=c['id'],question=c['question'],matched=window['matched_count'],
                            promoted=len(window['promoted_indices']),gold_total=len(set(c['gold'])),
                            gold_in_pool=len(set(c['gold'])&set(c['candidate_messages'])),
                            candidate_sets_equal=set(trace['candidate_ids'])==set(other['trace']['candidate_ids'])))
    report=json.loads((runs/'20261009-window60/report.json').read_text(encoding='utf8'))
    result=dict(groups=groups,pool_diagnostics=details,paired_report=report,
                prior_B_rankings_reproduced=True)
    target=root/'docs/temporal-window-results-20261009'
    target.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 时间窗口候选召回实验','',
           '基线提交：1605c6b（方案 B 检查点）。实验在 B 的绝对日期查询增强之上增加窗口候选预留。',
           '', '## 实现','',
           '- 配置：RAG_TEMPORAL_MODE=on，RAG_TEMPORAL_EXPERIMENT=window；默认仍 off，未改根目录 .env。',
           '- 用原文时间片段的归一化区间与问题区间求交集；显式置信度低于0.8、非法日期不匹配。规则片段没有置信度字段，作为候选提示使用，不视为已验证事件。',
           '- 固定400候选时先保留全局前120条，再补最多280条窗口内候选；重复去除、空位由全局补齐；统一重排。不额外调用LLM或embedding。',
           '- 本版不自动扩窗，不把消息时间当事件时间；目标无法解析时维持 B 行为。',
           '- 这是重排前候选池约束：仍计算用户范围内的向量/BM25分数，复用同一排序。不是数据库 WHERE 时间裁剪，不宣称减少全库计算量。',
           '- 如启用分区 dual，后续既有分区预留仍生效，最终比例可能变化；本次分区和多跳关闭。',
           '', '## 协议','',
           '相同LongMemEval 20道时间题＋40道控制题；39974块、107条金标消息。真实本地HTTP /search、text-embedding-v4＋MiniLM、400候选、Top10、无结果邻居扩展。缓存完整历史向量及既有时间侧索引，没有重新全量 /add；本轮新增LLM标注调用为0。提问时间采用数据集question_date（UTC）。',
           '本轮对照逐题复现上次 B 的返回排名。查询向量预热，耗时不含远端embedding。参数未按测试结果调优。',
           '', '## 指标','']
    for name,group in groups.items():
        lines += ['### '+name,'','| 指标 | B | B＋窗口 |','|---|---:|---:|']
        for k in ('5','10'):
            for key,label in [('hit','Hit'),('recall','Recall'),('micro_recall','Micro Recall'),('all_evidence','全证据命中')]:
                lines.append(f"| {label}@{k} | {group['off'][k][key]:.2%} | {group['on'][k][key]:.2%} |")
        lines.append('')
    lines += ['## 候选池诊断','', '| 题目ID | 窗口匹配片段 | 新进入池的片段 | 池内金标/全部金标 | 候选集与B相同 |', '|---|---:|---:|---:|---|']
    for d in details:
        lines.append(f"| {d['id']} | {d['matched']} | {d['promoted']} | {d['gold_in_pool']}/{d['gold_total']} | {d['candidate_sets_equal']} |")
    lines += ['', '60题Top5/Top10召回指标均无改善或退化，所有最终排名一致。实际触发只有4题；候选预算400已包含全部窗口匹配片段，全部金标消息也都已进入候选池。旅行题仍只返回1/3证据，比赛购物题仍未命中，当前主要瓶颈是重排而非候选进入池。',
              '',f"平均检索耗时：B {report['off']['mean_seconds']*1000:.1f} ms，窗口 {report['on']['mean_seconds']*1000:.1f} ms。单次配对测量，不能据此宣称提速。", 
              '', '## 验证与结论','',
              '全套302项测试通过，随后新增HTTP窗口专项用例并运行5项窗口测试通过（合计303个不同测试）。专项验证池外证据进入重排、无时间全局保留、缺参考时间回退、用户隔离、候选预算和无事件日期时不能用消息时间替代。',
              '当前实验保留为可选配置，不推荐仅凭本轮结果替换 B。下一步应检查金标在重排阶段的日期关联与语义评分；若继续评估窗口召回，使用新的未调参数据和预先确定的小候选预算，不把候选不足等同于证据充分。',
              '', '产物：data/temporal-sequence-live/20261009-window60；诊断复跑：20261009-window-diagnostic4。两次运行各自manifest中的脚本哈希对应运行时版本；之后脚本仅补充复现元数据。','']
    target.with_suffix('.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(result['pool_diagnostics'],ensure_ascii=False,indent=2))


if __name__=='__main__': main()
