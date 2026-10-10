"""Source-aware full-run report; preserves return-slot and expanded-source metrics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import numpy as np
from validate_text_release import PROJECT_ROOT, load_lines, metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    cases=[c for p in sorted(args.run.glob('*/cases.jsonl')) for c in load_lines(p)]
    previous=[c for p in sorted((PROJECT_ROOT/'data/final-text-validation/20261009-qwen-full').glob('*/cases.jsonl')) for c in load_lines(p)]
    key=lambda c:(c['sample_id'],c['question_index'])
    assert len(cases)==861 and len(set(map(key,cases)))==861, 'Full run incomplete'
    assert set(map(key,cases))==set(map(key,previous)), 'Question set changed'
    arms={'prior_full':(previous,'current'),'direct':(cases,'direct'),'current':(cases,'current')}
    stats={a:{str(k):metrics(cs,n,k) for k in (5,10,100)} for a,(cs,n) in arms.items()}
    expanded=[]
    for c in cases:
        r=dict(c['current']);r.pop('ranked_groups',None)
        expanded.append(dict(c,current=r))
    stats['current_expanded_sources']={str(k):metrics(expanded,'current',k) for k in (5,10,100)}
    diagnostic={}
    for a,(cs,n) in arms.items():
        diagnostic[a]=dict(http_errors=sum(c[n]['status']!=200 for c in cs),
            fallbacks=sum(c[n].get('fallback',False) for c in cs),
            routing_errors=sum(c[n].get('routing_errors',0) for c in cs),
            p50_seconds=float(np.median([c[n]['seconds'] for c in cs])),
            p95_seconds=float(np.percentile([c[n]['seconds'] for c in cs],95)),
            mean_seconds=float(np.mean([c[n]['seconds'] for c in cs])),
            llm_calls=sum(c[n].get('llm_calls',0) for c in cs),
            mean_chars_top10=float(np.mean([sum(len(h['content']) for h in c[n].get('hits',[])[:10]) for c in cs])) if a!='prior_full' else None)
    previous_by_id={key(c):c for c in previous}
    changes=[]
    def recalled(c,n):
        return set(c['gold']) & {s for g in c[n].get('ranked_groups',[[s] for s in c[n]['ranked']])[:10] for s in g}
    for c in cases:
        if not c['gold']:continue
        p=previous_by_id[key(c)];old=recalled(p,'current');new=recalled(c,'current')
        if old!=new:
            changes.append(dict(sample_id=c['sample_id'],question_index=c['question_index'],question=c['question'],
                category=c['category'],gold=c['gold'],before=sorted(old),after=sorted(new),
                lost=sorted(old-new),gained=sorted(new-old),trace=c['current'].get('trace',{})))
    annotations=Counter();fact_relations=[];writes=[]
    for p in args.run.glob('*/memory.sqlite3'):
        if p.parent.name not in {c['sample_id'] for c in cases}: continue
        with sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            annotations['memories']+=db.execute('select count(*) from rag_memories').fetchone()[0]
            annotations['publications']+=db.execute('select count(*) from rag_publications').fetchone()[0]
            for r in db.execute('select * from rag_fact_relations'):
                fact_relations.append(dict(r,sample_id=p.parent.name))
        writes.extend(load_lines(p.parent/'add.jsonl'))
    traces=[c['current'].get('trace',{}) for c in cases]
    report=dict(questions=len(cases),scored=sum(bool(c['gold']) for c in cases),metrics=stats,diagnostics=diagnostic,
        categories={str(cat):{a:metrics([c for c in cs if c['category']==cat],n,10)
                    for a,(cs,n) in arms.items()} for cat in sorted({c['category'] for c in cases if c['gold']})},
        changed_questions=changes,improved=sum(len(c['after'])>len(c['before']) for c in changes),
        regressed=sum(len(c['after'])<len(c['before']) for c in changes),
        controls=dict(rule_status=dict(Counter(t.get('rule_applicability',{}).get('status','not_triggered') for t in traces)),
            bundle_status=dict(Counter(t.get('evidence_bundle',{}).get('status','not_triggered') for t in traces)),
            supplemental_calls=sum(t.get('supplemental_calls',0) for t in traces),
            supplemental_new_candidates=sum(t.get('supplemental_new_candidates',0) for t in traces)),
        add=dict(annotations,attempts=len(writes),failed_attempts=sum(r['status']!=200 for r in writes),
                 p95_seconds=float(np.percentile([r['seconds'] for r in writes],95))),fact_relations=fact_relations)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 当前版本全量纯文本检索测试','',
        '公开 LoCoMo-Refined：861 题，859 题有证据标注。不是官方私有评分，也没有测试上层答案生成。',
        '本轮新建隔离库，经真实 Add/Search 接口调用 text-embedding-v4、Qwen reranker 和 gpt-4o-mini。',
        '每题请求 top_k=100，再从同一返回列表截取 @5/@10；直接请求 top_k=10 的初始候选池可能不同。',
        'prior_full 是此前保存的完整流程结果，direct 是本轮同语料同模型的底层检索对照。',
        'current 按返回项截取 TopK 后展开可追溯来源；expanded 按展开后的原文条目截取 TopK，避免只看证据拼接带来的槽位收益。',
        '结果按来源 ID 计分；若原文内容已遮蔽，这不保证金标中的敏感答案仍可见。','',
        '|指标|此前完整流程|本轮底层检索|当前返回项|当前展开原文|','|---|---:|---:|---:|---:|']
    for k in ('5','10','100'):
        for m in ('recall','hit','micro_recall','all_evidence'):
            lines.append('|'+m+'@'+k+'|'+'|'.join(f'{stats[a][k][m]:.2%}' for a in stats)+'|')
    lines+=['','|运行指标|此前完整流程|本轮底层检索|当前完整流程|','|---|---:|---:|---:|']
    for m in ('http_errors','fallbacks','routing_errors','p50_seconds','p95_seconds','llm_calls'):
        lines.append('|'+m+'|'+'|'.join(f'{diagnostic[a][m]:.2f}' for a in arms)+'|')
    lines+=['',f"相对此前完整流程：改善 {report['improved']} 题，退化 {report['regressed']} 题。逐题证据变化、控制触发情况和事实关系见同名 JSON。",'',
        '本轮开启版本固定、基础脱敏、显式事实替代、规则适用性、阶段选择与证据拼接；时间增强和语义隐私仍关闭，不能用本轮结果声称这两项已验证。',
        '规则、授权等专项验证需单独结合人工构造测试；公开检索集不能覆盖全部治理与安全能力。',
        '不同调用存在模型/API 波动，历史对比不是同时重跑；不将全部变化归因于某一个组件。','']
    args.out.write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({k:report[k] for k in ['questions','metrics','diagnostics','improved','regressed','controls','add']},indent=2))

if __name__=='__main__':main()
