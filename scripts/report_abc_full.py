"""Summarize fresh paired full-corpus evaluation without modifying retrieval."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT
from memory.evaluation import evidence_metrics

NAMES=['baseline','ABC']
KS=[5,10,20,50,100]


def metric(cases,name,k):
    valid=[c for c in cases if c['evidence'] and not c['missing_evidence']]
    rows=[];rr=[];ndcg=[];precision=[]
    for c in valid:
        gold=set(c['evidence']);ranked=c[name]['ranked_evidence'][:k]
        found=set(ranked)&gold
        rows.append(dict(evidence=c['evidence'],result=dict(hit_evidence=list(found))))
        rr.append(next((1/(i+1) for i,d in enumerate(ranked) if d in gold),0))
        seen=set();dcg=0
        for i,d in enumerate(ranked):
            if d in gold and d not in seen:dcg+=1/math.log2(i+2)
            seen.add(d)
        ideal=sum(1/math.log2(i+2) for i in range(min(k,len(gold))))
        ndcg.append(dcg/ideal);precision.append(len(found)/k)
    result=evidence_metrics(rows,'result')
    result.update(total_questions=len(cases),excluded_no_gold=sum(not c['evidence'] for c in cases),
                  excluded_missing_gold=sum(bool(c['missing_evidence']) for c in cases),
                  mrr=float(np.mean(rr)) if rr else None,ndcg=float(np.mean(ndcg)) if ndcg else None,
                  precision=float(np.mean(precision)) if precision else None)
    return result


def metrics(cases):
    return {str(k):{name:metric(cases,name,k) for name in NAMES} for k in KS}


def change(c,k=10):
    gold=set(c['evidence'])
    before=len(gold&set(c['baseline']['ranked_evidence'][:k]));after=len(gold&set(c['ABC']['ranked_evidence'][:k]))
    return before,after


def paired(cases,k=10):
    pairs=[change(c,k) for c in cases if c['evidence'] and not c['missing_evidence']]
    return dict(hit_wins=sum(a==0 and b>0 for a,b in pairs),hit_losses=sum(a>0 and b==0 for a,b in pairs),
                evidence_wins=sum(b>a for a,b in pairs),evidence_losses=sum(b<a for a,b in pairs),
                evidence_ties=sum(a==b for a,b in pairs))


def perf(cases,name):
    times=[c[name]['seconds'] for c in cases]
    if not times:
        return dict(queries=0,mean_seconds=None,p50_seconds=None,p95_seconds=None,p99_seconds=None,
            total_query_seconds=0,sequential_queries_per_second=None,mean_rerank_documents=None,
            mean_rerank_calls=None,mean_rerank_seconds=None,max_observed_process_rss_bytes=None,
            max_observed_gpu_allocated_bytes=None)
    return dict(queries=len(times),mean_seconds=float(np.mean(times)),p50_seconds=float(np.median(times)),
        p95_seconds=float(np.percentile(times,95)),p99_seconds=float(np.percentile(times,99)),
        total_query_seconds=sum(times),sequential_queries_per_second=len(times)/sum(times),
        mean_rerank_documents=float(np.mean([c[name]['rerank_documents'] for c in cases])),
        mean_rerank_calls=float(np.mean([c[name]['rerank_calls'] for c in cases])),
        mean_rerank_seconds=float(np.mean([c[name]['rerank_seconds'] for c in cases])),
        max_observed_process_rss_bytes=max(c[name]['rss_bytes'] for c in cases),
        max_observed_gpu_allocated_bytes=max(c[name]['gpu_peak_allocated_bytes'] or 0 for c in cases))


def main():
    p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args();out=Path(args.directory)
    if not out.is_absolute():out=PROJECT_ROOT/out
    completed=json.loads((out/'complete.json').read_text(encoding='utf-8'))
    cases=[json.loads(l) for l in (out/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(cases)==completed['queries']
    assert len({(c['sample_id'],c['question_index']) for c in cases})==len(cases)
    scopes={'text':[c for c in cases if not c['multimodal']],
            'multimodal_text_input':[c for c in cases if c['multimodal']],'all_text_input':cases}
    report=dict(completion=completed,manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8')),
        ingestion=json.loads((out/'ingestion.json').read_text(encoding='utf-8')),
        metrics={scope:metrics(group) for scope,group in scopes.items()},
        paired={scope:{str(k):paired(group,k) for k in KS} for scope,group in scopes.items()},
        performance={scope:{name:perf(group,name) for name in NAMES} for scope,group in scopes.items()},
        text_by_conversation={sid:metrics([c for c in scopes['text'] if c['sample_id']==sid]) for sid in sorted({c['sample_id'] for c in cases})},
        text_by_category={str(cat):metrics([c for c in scopes['text'] if c['category']==cat]) for cat in sorted({c['category'] for c in cases})},
        text_by_evidence_count={label:metrics([c for c in scopes['text'] if (len(c['evidence'])>1)==multiple and c['evidence']]) for label,multiple in [('single',False),('multiple',True)]},
        cases_sha256=hashlib.sha256((out/'cases.jsonl').read_bytes()).hexdigest())
    # Descriptive paired cluster interval, preserving within-conversation dependence.
    ids=sorted(report['text_by_conversation']);rng=np.random.default_rng(20261001);draw=rng.integers(0,len(ids),(10000,len(ids)))
    counts=np.array([report['text_by_conversation'][sid]['10']['baseline']['evaluated_questions'] for sid in ids])
    gains=np.array([report['text_by_conversation'][sid]['10']['ABC']['hit_questions']-report['text_by_conversation'][sid]['10']['baseline']['hit_questions'] for sid in ids])
    report['text_hit10_gain_cluster95']=np.percentile(gains[draw].sum(axis=1)/counts[draw].sum(axis=1),[2.5,97.5]).tolist()
    previous_path=PROJECT_ROOT/'data/factorial-benchmarks/20261001-v2/cases.jsonl'
    previous=[json.loads(l) for l in previous_path.read_text(encoding='utf-8').splitlines()] if previous_path.exists() else []
    old={(c['sample_id'],c['question_index']):c for c in previous}
    parity={name:0 for name in NAMES}
    for c in scopes['text']:
        key=(c['sample_id'],c['question_index'])
        if key not in old:continue
        for name,old_name in [('baseline','base'),('ABC','ABC')]:
            parity[name]+=set(c[name]['hit_evidence'])==set(old[key][old_name]['hit_evidence'])
    report['prior_text_hit_evidence_parity']=dict(parity,compared=len(old))
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    sources=json.loads((out/'source-map.json').read_text(encoding='utf-8'))
    losses=[];wins=[]
    for c in cases:
        if not c['evidence'] or c['missing_evidence']:continue
        before,after=change(c)
        if before==after:continue
        detail=dict(sample_id=c['sample_id'],question_index=c['question_index'],question=c['question'],multimodal=c['multimodal'],
                    evidence=c['evidence'],before=before,after=after,
                    ABC_gold_ranks={d:next((i+1 for i,item in enumerate(c['ABC']['ranked_evidence']) if item==d),None) for d in c['evidence']},
                    gold=[v for v in sources.values() if v['sample_id']==c['sample_id'] and v['evidence_id'] in c['evidence']],
                    baseline_top10=[sources[mid] for mid in c['baseline']['ids'][:10]],ABC_top10=[sources[mid] for mid in c['ABC']['ids'][:10]])
        (wins if after>before else losses).append(detail)
    losses.sort(key=lambda c:(c['multimodal'],c['after']!=0,-(c['before']-c['after']),c['sample_id'],c['question_index']))
    (out/'evidence-regressions.json').write_text(json.dumps(losses,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'evidence-improvements.json').write_text(json.dumps(wins,ensure_ascii=False,indent=2),encoding='utf-8')
    details=['# 所有 Top10 证据覆盖退步案例','',
        '按严格标注 ID 计算，包含从部分命中变成命中更少的情况。排名为空表示 ABC 前100没有出现该证据，不能据此断定前400候选也缺失。以下显示 gold 原文及 ABC 排名前三的原文；双方完整 top10 保存在同名 JSON。','']
    for c in losses:
        details += [f"## {c['sample_id']} / {c['question_index']}（{'多模态标记，仅文本输入' if c['multimodal'] else '纯文本'}）",'',c['question'],'',
                    f"命中证据：{c['before']} → {c['after']} / {len(c['evidence'])}",'','标注证据：','']
        for g in c['gold']:
            details += [f"- {g['evidence_id']}，{g['speaker']}，{g['date']}；ABC 排名 {c['ABC_gold_ranks'][g['evidence_id']]}：{g['content']}"]
        details += ['','ABC 前三条：','']
        for g in c['ABC_top10'][:3]:details += [f"- {g['evidence_id']}，{g['speaker']}：{g['content']}"]
        details += ['']
    (out/'evidence-regressions.md').write_text('\n'.join(details),encoding='utf-8')
    def pct(x):return f'{100*x:.2f}%' if x is not None else '—'
    def table(group):
        lines=['| K | Hit 基线→ABC | Recall 基线→ABC | Micro Recall 基线→ABC | 全证据 基线→ABC |','|---|---|---|---|---|']
        for k in KS:
            a,b=group[str(k)]['baseline'],group[str(k)]['ABC']
            lines.append('| '+str(k)+' | '+' | '.join(pct(a[key])+' → '+pct(b[key]) for key in ['hit_rate','recall','micro_recall','all_evidence_hit_rate'])+' |')
        return '\n'.join(lines)
    lines=['# ABC 完整重测与基线对照','',f"运行目录：`{out.relative_to(PROJECT_ROOT).as_posix()}`；代码 `{report['manifest']['commit'][:7]}`；基线 `{report['manifest']['baseline_commit'][:7]}`。",'',
        f"两套新数据库完整写入 {report['ingestion']['ABC']['messages']:,} 条消息，对本次 {len(cases):,} 道题分别进行真实模型 Top10 推理（{completed['searches']:,} 次）。不同 K 的统计使用同次新推理产生的排序，不读取历史分值缓存。",
        '基线为 ABC 前的 BGE + BM25/RRF + 400 候选上下文 CrossEncoder + 单跳邻句支持。ABC 是当前根 `.env` 配置。旧版写入实现使用基线 Git 提交源码，双方轮换先后顺序。',
        '全部方案仅输入对话文本。多模态标记题单列，不能称为完整视觉测评；没有答案生成或答案正确率评分。无 gold 或缺失 gold 题从证据指标排除，但仍执行检索并计入延迟。',
        '题目数量与排除：'+'；'.join(f"{scope} 总数 {len(group)}、有效 {report['metrics'][scope]['10']['ABC']['evaluated_questions']}、无 gold {report['metrics'][scope]['10']['ABC']['excluded_no_gold']}、缺失 gold {report['metrics'][scope]['10']['ABC']['excluded_missing_gold']}" for scope,group in scopes.items())+'。',
        '', '## 纯文本题证据指标','',table(report['metrics']['text']),
        '', '## 全题目（仅文本输入）','',table(report['metrics']['all_text_input']),
        '', '## 多模态标记题（仅文本输入）','',table(report['metrics']['multimodal_text_input']),
        '', '## 排名与精度（纯文本题）','', '| K | Precision 基线→ABC | MRR 基线→ABC | nDCG 基线→ABC |','|---|---|---|---|']
    for k in KS:
        a,b=report['metrics']['text'][str(k)]['baseline'],report['metrics']['text'][str(k)]['ABC']
        lines.append('| '+str(k)+' | '+' | '.join(pct(a[key])+' → '+pct(b[key]) for key in ['precision','mrr','ndcg'])+' |')
    lines+=['','Precision 按每题命中证据数/K；MRR 为首条证据倒数排名；nDCG 使用二元证据相关性，同一原文重复块只计一次。指标按标注证据 ID 计算，同义替代证据不自动算命中。',
        '', '## 检索成本：全量真实 Top10 查询','', '| 指标 | 基线 | ABC | ABC/基线 |','|---|---:|---:|---:|']
    pa,pb=report['performance']['all_text_input']['baseline'],report['performance']['all_text_input']['ABC']
    for key,label,scale in [('mean_seconds','平均延迟 ms',1000),('p50_seconds','P50 ms',1000),('p95_seconds','P95 ms',1000),('p99_seconds','P99 ms',1000),('sequential_queries_per_second','顺序查询/秒',1),('mean_rerank_documents','平均重排文本对数',1),('mean_rerank_calls','平均重排调用次数',1),('mean_rerank_seconds','平均模型重排耗时 ms',1000)]:
        lines.append(f'| {label} | {pa[key]*scale:.2f} | {pb[key]*scale:.2f} | {pb[key]/pa[key]:.2f}× |')
    lines+=['','不包含 HTTP 网络传输、模型首次加载或并发排队。双方共享已加载的模型并顺序执行；QPS 是顺序吞吐，不能外推为并发服务容量。计时期间不复用模型打分。',
        '', '## 写入与存储','', '| 指标 | 基线 | ABC | ABC/基线 |','|---|---:|---:|---:|']
    ia,ib=report['ingestion']['baseline'],report['ingestion']['ABC']
    for key,label,scale in [('total_seconds','全部写入秒数',1),('messages_per_second','消息/秒',1),('database_bytes','数据库 MiB',1/1048576)]:
        lines.append(f'| {label} | {ia[key]*scale:.3f} | {ib[key]*scale:.3f} | {ib[key]/ia[key]:.2f}× |')
    for key in ['p50','p95','p99']:
        va,vb=ia['batch_seconds'][key],ib['batch_seconds'][key]
        lines.append(f'| 写入批次 {key.upper()} ms | {va*1000:.2f} | {vb*1000:.2f} | {vb/va:.2f}× |')
    lines+=['','每个请求最多 40 条消息，批次大小随会话而变化；双方使用同批原文并轮换执行顺序。存储大小在 WAL checkpoint 后读取。模型占用另记于 manifest；每查询的进程 RSS/GPU 高水位在 report 中，因为共享模型并带有测试记录器，不将它们伪装成独立线上服务内存对照。',
        '', '## 按会话、类别和证据数量（纯文本 Top10）','', '| 分组 | 有效题数 | Hit 基线→ABC | Recall 基线→ABC |','|---|---:|---|---|']
    for key in ['text_by_conversation','text_by_category','text_by_evidence_count']:
        for label,data in report[key].items():
            a,b=data['10']['baseline'],data['10']['ABC']
            lines.append(f"| {key.replace('text_by_','')}:{label} | {a['evaluated_questions']} | {pct(a['hit_rate'])} → {pct(b['hit_rate'])} | {pct(a['recall'])} → {pct(b['recall'])} |")
    lines+=['','类别保留数据集原始编号；没有未经核验地赋予语义标签。','', '## 进步与退步','']
    for scope in scopes:
        p=report['paired'][scope]['10'];m=report['metrics'][scope]['10']['ABC']
        lines.append(f"- {scope}：新增命中 {p['hit_wins']} 题，丢失命中 {p['hit_losses']} 题；证据条数增加 {p['evidence_wins']} 题、减少 {p['evidence_losses']} 题、不变 {p['evidence_ties']} 题。ABC 全命中 {m['all_evidence_hit_questions']} 题、部分命中 {m['hit_questions']-m['all_evidence_hit_questions']} 题、零命中 {m['evaluated_questions']-m['hit_questions']} 题。")
    lines+=['','所有证据减少的题目（包含仍有部分命中的情况）及双方完整 top10/原文保存在 `evidence-regressions.json`；改善题在 `evidence-improvements.json`。以下列出纯文本题从有命中变为零命中的全部案例。',
        '', '| 会话/题号（从0开始） | 问题 | 标注证据 |','|---|---|---|']
    for c in losses:
        if not c['multimodal'] and c['after']==0:
            lines.append(f"| {c['sample_id']}/{c['question_index']} | {c['question'].replace('|','/')} | {', '.join(c['evidence'])} |")
    ci=report['text_hit10_gain_cluster95']
    lines+=['',f'纯文本 Hit@10 提升的按会话聚类 95% 描述性区间：+{ci[0]*100:.2f} 至 +{ci[1]*100:.2f} 个百分点（10,000 次重采样）。只有十个会话，且数据已经用于方案开发，不是独立泛化证明。',
        '',f"与前次结果的命中证据集合一致数：基线 {parity['baseline']}/{len(old)}；ABC {parity['ABC']}/{len(old)}（历史文件缺失时不做此核对）。完整输出保存在本次独立运行目录，未覆盖历史结果。",'',
        '## 复现与产物','', '`python scripts/benchmark_abc_full.py` 新建运行目录，然后执行 `python scripts/report_abc_full.py <运行目录>`。需要根 .env 的 ABC 配置与已下载本地模型。',
        '', 'manifest.json：模型、源数据、代码、配置指纹；ingestion.json：完整写入与存储；cases.jsonl：逐题原始排序和耗时；report.json：所有分组指标；complete.json：完整执行与原文检查。运行失败不会生成 complete.json，也不会把未测题当成成功。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(text10=report['metrics']['text']['10'],performance=report['performance']['all_text_input'],ingestion=report['ingestion'],paired=report['paired']['text']['10']),indent=2))
    print('REPORT',out/'report.md')


if __name__=='__main__':main()
