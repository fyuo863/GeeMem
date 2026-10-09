"""Compare frozen soft-score / absolute-query runs without fitting parameters."""
import argparse
import json
from pathlib import Path
from test_temporal_live import metrics


def load(path):
    cases=[json.loads(l) for l in (path/'cases.jsonl').read_text(encoding='utf8').splitlines()]
    result={}
    for c in cases:
        arm=result.setdefault(c['id'],{})
        if c['variant'] in arm: raise ValueError('Duplicate case arm')
        arm[c['variant']]=c
    assert len(result)==60 and all(set(arms)=={'off','on'} for arms in result.values())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a',type=Path,required=True)
    parser.add_argument('--b',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    a,b=load(args.a),load(args.b)
    assert set(a)==set(b)
    assert all(a[i]['off']['question']==b[i]['off']['question'] and a[i]['off']['gold']==b[i]['off']['gold'] and
               a[i]['off']['reference_time']==b[i]['off']['reference_time'] for i in a)
    baseline_equal=all(a[i]['off']['ranked_messages']==b[i]['off']['ranked_messages'] for i in a)
    arms={'基线':{i:p['off'] for i,p in a.items()},'A 时间加分':{i:p['on'] for i,p in a.items()},'B 绝对时间查询':{i:p['on'] for i,p in b.items()}}
    groups={'全部60题':list(a),'时间20题':[i for i in a if a[i]['off']['category']=='temporal-reasoning'],
            '其他40题':[i for i in a if a[i]['off']['category']!='temporal-reasoning'],
            '实际解析触发题':[i for i in a if a[i]['on']['resolved_time']]}
    summary={}
    for group,ids in groups.items():
        summary[group]={}
        for arm,items in arms.items():
            summary[group][arm]={str(k):metrics([items[i] for i in ids],k) for k in (5,10)}
    details=[]
    for i in a:
        d=dict(id=i,question=a[i]['off']['question'],category=a[i]['off']['category'],time=a[i]['on']['resolved_time'])
        for name,items in arms.items():
            c=items[i];gold=set(c['gold'])
            d[name]=dict(ranks=[r+1 for r,x in enumerate(c['ranked_messages']) if x in gold],
                         recall5=metrics([c],5)['recall'],recall10=metrics([c],10)['recall'])
        details.append(d)
    result=dict(baseline_rankings_identical=baseline_equal,groups=summary,cases=details)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    lines=['# 两种时间增强方案对照','',
           '共同基线：`f7095cc`（已经开启原时间增强）。A 与 B 均直接从该提交建立独立分支，接口及保守时间解析器共用，检索策略不同。',
           '',
           '- A：原查询不变，重排后对明确匹配的时间片段加分，限前50候选、奖励为稳健分数跨度10%、最多移动2位。',
           '- B：原问题后追加解析出的绝对日期/区间与星期，同一文本用于向量召回、BM25和重排，没有 A 的加分。',
           '- 60题相同，包括20时间题与40其他题；相同39974块原文、107条金标消息、text-embedding-v4/MiniLM/400候选/Top10。关闭多跳、分区及新LLM标注。',
           '- 两组都收到数据集 question_date（UTC）；仅实验组使用它。不会用当前系统时间。旧时间侧索引原样复用。',
           '- 规则在60题中仅解析出4道明确约束，其余回退。参数测试前固定，本次没有按结果调参。',
           f'- A/B对照组的60题返回排名完全一致：{baseline_equal}。',
           '', '## 指标','']
    for group,arms_metrics in summary.items():
        lines+=['### '+group,'','| 指标 | 基线 | A 时间加分 | B 绝对时间查询 |','|---|---:|---:|---:|']
        for k in ('5','10'):
            for key,label in [('hit','Hit'),('recall','Recall'),('micro_recall','Micro Recall'),('all_evidence','全证据命中')]:
                values=[f'{arms_metrics[arm][k][key]:.2%}' for arm in arms]
                lines.append(f'| {label}@{k} | '+' | '.join(values)+' |')
        lines+=['']
    lines+=['## 实际触发题的金标排名','','| ID | 原题 | 基线 | A | B |','|---|---|---:|---:|---:|']
    for d in details:
        if d['time']:
            values=[str(d[name]['ranks']) for name in arms]
            lines.append(f"| {d['id']} | {d['question']} | "+' | '.join(values)+' |')
    lines+=['','## 改善/退化题数（相对基线）','']
    for arm in list(arms)[1:]:
        for k in (5,10):
            ds=[d[arm]['recall'+str(k)]-d['基线']['recall'+str(k)] for d in details]
            lines.append(f'- {arm} Recall@{k}：改善 {sum(x>0 for x in ds)}、持平 {sum(x==0 for x in ds)}、退化 {sum(x<0 for x in ds)}。')
    lines+=['','## 局限','',
            '这是带可靠提问时间的、缓存完整会话的本地真实 /search 检索实验，不是官方答案正确率，也不是全量重新 /add。实际触发仅4题，只能比较本轮实现，不能证明策略普遍优劣。A最多两位调整会限制修复较大的排名偏差；本轮未再搜索权重/位移参数。B依赖调用方提供参考时间；没有该字段时相对时间增强不会自动生效。源/查询日历口径需一致（本轮UTC）。',
            '',
            '各运行 manifest.json 保存代码/数据集哈希，cases.jsonl 保存原文结果。检索耗时不含预热后的查询 embedding API 时间，不代表实际端到端延迟。',
            '',f'A产物：`{args.a}`；B产物：`{args.b}`。','']
    args.out.write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(summary['时间20题'],ensure_ascii=False,indent=2))


if __name__=='__main__': main()
