"""Report full public textual evaluation; never choose policies using gold."""
import argparse
import json
from pathlib import Path
import sys
import sqlite3
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate_text_release import load_lines,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    cases=[c for f in sorted(a.run.glob('*/cases.jsonl')) for c in load_lines(f)]
    if not cases: raise SystemExit('No completed cases')
    assert len({(c['sample_id'],c['question_index']) for c in cases})==len(cases)
    scope={'all':cases,'development':[c for c in cases if c['sample_id'] in ['conv-26','conv-30','conv-41']],
           'remaining':[c for c in cases if c['sample_id'] not in ['conv-26','conv-30','conv-41']]}
    summary={name:{str(k):{arm:metrics(group,arm,k) for arm in ['direct','current']}
                   for k in [5,10,100]} for name,group in scope.items() if group}
    changes={}
    for k in [5,10,100]:
        rows=[]
        for c in cases:
            if not c['gold']:continue
            g=set(c['gold']);b=len(g&set(c['direct']['ranked'][:k]));n=len(g&set(c['current']['ranked'][:k]))
            if b!=n:rows.append(dict(sample_id=c['sample_id'],question_index=c['question_index'],question=c['question'],
                before=b,after=n,gold_count=len(g),strategy=c['current']['strategy'],fallback=c['current']['fallback'],
                lost=sorted((g&set(c['direct']['ranked'][:k]))-set(c['current']['ranked'][:k])),
                gained=sorted((g&set(c['current']['ranked'][:k]))-set(c['direct']['ranked'][:k]))))
        changes[str(k)]=dict(wins=sum(x['after']>x['before'] for x in rows),losses=sum(x['after']<x['before'] for x in rows),cases=rows)
    categories={str(category):{
        'questions':len(group),
        'metrics':{str(k):{arm:metrics(group,arm,k) for arm in ['direct','current']} for k in [5,10,100]}
    } for category in sorted({c['category'] for c in cases},key=str)
       if (group:=[c for c in cases if c['category']==category]) and any(c['gold'] for c in group)}
    conversations={sid:{str(k):{arm:metrics(group,arm,k) for arm in ['direct','current']} for k in [10,100]}
                   for sid in sorted({c['sample_id'] for c in cases})
                   if (group:=[c for c in cases if c['sample_id']==sid])}
    diag={arm:dict(http_errors=sum(c[arm]['status']!=200 for c in cases),
        fallback=sum(c[arm]['fallback'] for c in cases),routing_errors=sum(c[arm]['routing_errors'] for c in cases),
        p50=float(np.median([c[arm]['seconds'] for c in cases])),
        p95=float(np.percentile([c[arm]['seconds'] for c in cases],95))) for arm in ['direct','current']}
    adds=[r for f in a.run.glob('*/add.jsonl') for r in load_lines(f)]
    successful={r['request_id']:r for r in adds if r['status']==200}
    route_counts=Counter();decision_counts=Counter();queue_counts=Counter()
    for path in a.run.glob('*/memory.sqlite3'):
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
            for kind,status,count in db.execute('SELECT memory_type,status,count(*) FROM rag_memory_routes GROUP BY memory_type,status'):
                route_counts[kind+':'+status]+=count
            for decision,count in db.execute('SELECT decision,count(*) FROM rag_write_decisions GROUP BY decision'):
                decision_counts[decision]+=count
            for status,count in db.execute('SELECT status,count(*) FROM rag_memory_queue GROUP BY status'):
                queue_counts[status]+=count
    screen_path=a.run/'reranker-screen-summary.json'
    screen=json.loads(screen_path.read_text(encoding='utf8')) if screen_path.exists() else None
    report=dict(questions=len(cases),expected=861,complete=len(cases)==861,groups=summary,diagnostics=diag,changes=changes,
                categories=categories,conversations=conversations,reranker_screen=screen,
                add=dict(successful_batches=len(successful),messages=sum(r['messages'] for r in successful.values()),
                         failed_attempts=sum(r['status']!=200 for r in adds),routes=dict(route_counts),
                         decisions=dict(decision_counts),queue=dict(queue_counts)),
                protocol='Fresh Add + Search via in-process HTTP; full public LoCoMo-Refined textual subset; no reference_time or speaker; top_k=100. Not official Answer accuracy.')
    if not a.out:
        print(json.dumps({k:v for k,v in report.items() if k not in ['groups','changes','categories','conversations','reranker_screen']},indent=2));return
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 完整文本协议验证','',f"完成 {len(cases)}/861 题，完整：{report['complete']}。",'',report['protocol'],'',
           '生成模型：gpt-4o-mini；向量：text-embedding-v4 1024维；重排：MiniLM。当前组启用写入分类/类型存储、多跳、分区及时间模式；直接组复用相同原文和向量，关闭上层检索。',
           '不向接口传递金标、额外姓名或提问时间。在不传reference_time的条件下，当前解析器未从861个原问题中提取出目标时间区间；这不意味着题目没有时间信息，也不能用本轮成绩证明时间窗口收益。',
           '每题只请求一次top_k=100，@5和@10由该返回列表截取，并非分别用top_k=5/10请求。对照是同版本底层直接检索，不是旧线上版本复测。',
           '公开对话中的两位人物映射为user/assistant角色，原speaker字段未另行注入正文。因此本轮是公开数据向标准API字段的一种投影，不能确认等同于官方适配，也不能直接与保留人物姓名的旧测试成绩对比。',
           '前3组对话用于24题reranker预筛，剩余7组没有用于本次模型选择；整个公开数据集历史上已用过，不是未见测试集。','']
    for name,values in summary.items():
        lines+=['## '+name,'','|指标|直接检索|当前完整流程|','|---|---:|---:|']
        for k in ['5','10','100']:
            for key in ['hit','recall','micro_recall','all_evidence']:
                lines.append(f"|{key}@{k}|{values[k]['direct'][key]:.2%}|{values[k]['current'][key]:.2%}|")
        lines.append('')
    lines+=['## 按题型与增退化统计','','题型编号沿用公开数据集，不等同于官网能力维度。','',
            '|类别|题数|直接 Recall@10|当前 Recall@10|直接 Recall@100|当前 Recall@100|',
            '|---|---:|---:|---:|---:|---:|']
    for category,item in categories.items():
        m=item['metrics']
        lines.append(f"|{category}|{item['questions']}|{m['10']['direct']['recall']:.2%}|{m['10']['current']['recall']:.2%}|{m['100']['direct']['recall']:.2%}|{m['100']['current']['recall']:.2%}|")
    lines+=['','|对话|直接 Recall@10|当前 Recall@10|直接 Recall@100|当前 Recall@100|',
            '|---|---:|---:|---:|---:|']
    for sid,m in conversations.items():
        lines.append(f"|{sid}|{m['10']['direct']['recall']:.2%}|{m['10']['current']['recall']:.2%}|{m['100']['direct']['recall']:.2%}|{m['100']['current']['recall']:.2%}|")
    lines+=['','|K|召回证据增加的题数|减少的题数|','|---|---:|---:|']
    for k,item in changes.items():lines.append(f"|{k}|{item['wins']}|{item['losses']}|")
    lines+=['','### Top100退化题（完整列表）','']
    for item in changes['100']['cases']:
        if item['after']>=item['before']:continue
        lines.append(f"- {item['sample_id']} / 原QA索引{item['question_index']}：{item['question']}；命中证据 {item['before']} → {item['after']} / {item['gold_count']}；丢失 {', '.join(item['lost'])}；策略 {item['strategy']}。")
    if screen:
        lines+=['','## 重排器预筛','','固定24道开发题、相同v4候选和文档；未进行参数微调。','',
                '|模型与运行环境|Recall@10|Hit@10|Recall@100|平均纯重排耗时|',
                '|---|---:|---:|---:|---:|']
        for name,item in screen.items():
            label='MiniLM / 本机GPU' if name=='minilm' else 'BGE base ONNX / 本机CPU'
            lines.append(f"|{label}|{item['recall10']:.2%}|{item['hit10']:.2%}|{item['recall100']:.2%}|{item['seconds']:.3f}秒|")
        lines+=['','本轮保留MiniLM：BGE在小样本中召回较好，但当前可用运行配置的重排耗时明显增加；多跳会重复重排。本结果不代表BGE在GPU上的速度，也不证明MiniLM具有最高准确率。','']
    lines+=['## 稳定性与延迟','','```json',json.dumps(diag,ensure_ascii=False,indent=2),'```','',
            '延迟包含本地HTTP编排、真实embedding、LLM和重排。多会话并发、共享GPU，且两组工作量不同；不能将其当作隔离的模型速度基准。','',
            '正式成绩仍需官方Smoke/Full及统一Answer评分；未调用官方评测渠道或消耗官方配额。','']
    lines+=['## 写入完成情况','','```json',json.dumps(report['add'],ensure_ascii=False,indent=2),'```','',
            'completed表示任务已落库，不代表逐条分类或实体归属已通过人工准确性审核。','']
    a.out.write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(diag,indent=2))


if __name__=='__main__':main()
