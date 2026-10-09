"""Summarize paired evidence retrieval without calling any model or endpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
from test_temporal_live import metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--markdown',type=Path,required=True)
    args=parser.parse_args()
    cases=[json.loads(l) for l in (args.run/'cases.jsonl').read_text(encoding='utf8').splitlines()]
    previous={c['id']:c for l in (args.previous/'cases.jsonl').read_text(encoding='utf8').splitlines()
              if (c:=json.loads(l))['variant']=='on'}
    pairs={}
    for c in cases:
        if c['variant'] in pairs.setdefault(c['id'],{}): raise ValueError('Duplicate arm')
        pairs[c['id']][c['variant']]=c
    expected=json.loads((args.run/'selection.json').read_text())
    assert set(pairs)=={s['id'] for s in expected}
    assert all(set(p)=={'off','on'} for p in pairs.values())
    groups={'全部':list(pairs),
            '时间题（全部）':[i for i,p in pairs.items() if p['off']['category']=='temporal-reasoning'],
            '时间题（上一轮10题）':[i for i in pairs if i in previous],
            '时间题（新增）':[i for i,p in pairs.items() if p['off']['category']=='temporal-reasoning' and i not in previous],
            '其他题型对照':[i for i,p in pairs.items() if p['off']['category']!='temporal-reasoning']}
    for category in sorted({p['off']['category'] for p in pairs.values()}):
        if category!='temporal-reasoning':
            groups[category]=[i for i,p in pairs.items() if p['off']['category']==category]
    def recall(c,k): return len(set(c['gold']) & set(c['ranked_messages'][:k]))/len(set(c['gold']))
    summary={}
    for name,ids in groups.items():
        if not ids: continue
        delta=np.array([recall(pairs[i]['on'],10)-recall(pairs[i]['off'],10) for i in ids])
        rng=np.random.default_rng(20261009)
        means=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(axis=1)
        summary[name]=dict(count=len(ids),off={str(k):metrics([pairs[i]['off'] for i in ids],k) for k in (5,10)},
                          on={str(k):metrics([pairs[i]['on'] for i in ids],k) for k in (5,10)},
                          improved=int((delta>0).sum()),same=int((delta==0).sum()),regressed=int((delta<0).sum()),
                          recall10_difference_bootstrap95=np.percentile(means,[2.5,97.5]).tolist(),
                          changed_rankings=sum(pairs[i]['off']['ranked_messages']!=pairs[i]['on']['ranked_messages'] for i in ids),
                          triggered=sum(pairs[i]['on']['temporal_triggered'] for i in ids))
    details=[]
    for i,p in pairs.items():
        a,b=p['off'],p['on'];gold=set(a['gold']);old=set(a['ranked_messages']);new=set(b['ranked_messages'])
        details.append(dict(id=i,category=a['category'],question=a['question'],gold=len(gold),
                            before=len(gold&old),after=len(gold&new),triggered=b['temporal_triggered'],
                            gained=sorted((gold&new)-old),lost=sorted((gold&old)-new),
                            recall5_before=recall(a,5),recall5_after=recall(b,5)))
    repeat={i:previous[i]['ranked_messages']==pairs[i]['on']['ranked_messages'] for i in previous if i in pairs}
    result=dict(groups=summary,cases=details,previous_on_ranking_reproduced=repeat)
    (args.run/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    lines=['# 时间增强扩大测试：60 道 LongMemEval 真实题','',
           '## 协议','',
           '- 20 道 temporal-reasoning，加其他五类各 8 道，均为非拒答题；在已有缓存中按固定哈希选取，未根据表现筛题。',
           '- 保留各题完整会话，运行前逐一检查缓存块 ID 和原文完全一致；共 '+str(sum(s['chunks'] for s in expected))+' 块、'+str(sum(s['gold_count'] for s in expected))+' 条金标消息。',
           '- text-embedding-v4 + 英文 MiniLM，重排池 400，Top10，原文窗口/多跳/分区规划关闭。两组使用同一查询向量，交替执行先后顺序。',
           '- 使用真实本地 /search HTTP 处理链路；向量缓存与时间侧索引回填，不是重新全量 /add，也不是赛事答案正确率。',
           '- 保留上一轮已有模型标注，本轮不新增 LLM 调用；新增会话使用规则时间索引。本实验不能代表完整 LLM 标注能力。',
           '- 检索核心代码保持上一轮版本，仅扩展测试脚本；线上配置未更改。',
           '', '## 分组 Top10 指标','',
           '| 分组 | 题数 | Hit@10 关闭→开启 | Recall@10 关闭→开启 | Micro Recall@10 关闭→开启 | 全证据命中@10 关闭→开启 | 改善/持平/退化 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in summary.items():
        cells=[f"{s['off']['10'][k]:.2%} → {s['on']['10'][k]:.2%}" for k in ('hit','recall','micro_recall','all_evidence')]
        lines.append(f"| {name} | {s['count']} | "+' | '.join(cells)+f" | {s['improved']}/{s['same']}/{s['regressed']} |")
    lines+=['','## Top5 指标','','| 分组 | Hit@5 | Recall@5 | Micro Recall@5 | 全证据命中@5 |','|---|---:|---:|---:|---:|']
    for name in ['全部','时间题（全部）','时间题（新增）','其他题型对照']:
        s=summary[name];cells=[f"{s['off']['5'][k]:.2%} → {s['on']['5'][k]:.2%}" for k in ('hit','recall','micro_recall','all_evidence')]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','## 排名稳定性与不确定性','']
    for name in ['时间题（全部）','时间题（新增）','其他题型对照']:
        s=summary[name];lo,hi=s['recall10_difference_bootstrap95']
        lines.append(f"- {name}：触发时间规则 {s['triggered']}/{s['count']}，排名变化 {s['changed_rankings']} 题；Recall@10 差值的配对 bootstrap 95% 区间 [{lo*100:.2f}, {hi*100:.2f}] 个百分点。")
    lines+=['','区间为本样本的探索性统计，不能修复缓存样本选择偏差；新增时间题只有 10 道。其他数据集题型也可能包含时间约束，触发时间规则不自动等于误触发。',
            '',f"上一轮 10 题开启组排名复现：{sum(repeat.values())}/{len(repeat)}。",'',
            '## 逐题金标召回（@10）','','| ID | 类型 | 原题 | 命中数关闭→开启 / 金标数 |','|---|---|---|---:|']
    for d in details:
        lines.append(f"| {d['id']} | {d['category']} | {d['question'].replace('|','/')} | {d['before']} → {d['after']} / {d['gold']} |")
    lines+=['','原始运行目录：`'+str(args.run)+'`。`cases.jsonl` 保留每题两组原文结果，`comparison.json` 保存新增/丢失金标消息 ID，`manifest.json` 保存配置及代码/数据集哈希。','']
    args.markdown.write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
