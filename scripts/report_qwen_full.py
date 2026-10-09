"""Compare frozen public full-text runs; no retrieval or parameter tuning."""
import json
from pathlib import Path
from collections import Counter
import sqlite3
import numpy as np
from validate_text_release import metrics,load_lines,PROJECT_ROOT

def main():
    run=PROJECT_ROOT/'data/final-text-validation/20261009-qwen-full'
    old=PROJECT_ROOT/'data/final-text-validation/20261009-official861'
    current=[c for f in sorted(run.glob('*/cases.jsonl')) for c in load_lines(f)]
    previous=[c for f in sorted(old.glob('*/cases.jsonl')) for c in load_lines(f)]
    def key(c):return c['sample_id'],c['question_index']
    assert len(current)==861 and len(set(map(key,current)))==861
    assert set(map(key,current))==set(map(key,previous))
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf8'))
    arms={'previous':(previous,'current'),'direct':(current,'direct'),'current':(current,'current')}
    totals={name:{str(k):metrics(cs,arm,k) for k in [5,10,100]} for name,(cs,arm) in arms.items()}
    diagnostics={name:dict(http_errors=sum(c[arm]['status']!=200 for c in cs),
        fallbacks=sum(c[arm]['fallback'] for c in cs),routing_errors=sum(c[arm]['routing_errors'] for c in cs),
        p50=float(np.median([c[arm]['seconds'] for c in cs])),p95=float(np.percentile([c[arm]['seconds'] for c in cs],95)))
        for name,(cs,arm) in arms.items()}
    categories={str(cat):{name:metrics([c for c in cs if c['category']==cat],arm,10)
        for name,(cs,arm) in arms.items()} for cat in sorted({c['category'] for c in current})
        if any(c['gold'] and c['category']==cat for c in current)}
    lookup={key(c):c for c in previous};changes={}
    for name in ['previous','direct']:
        rows=[]
        for c in current:
            if not c['gold']:continue
            before=lookup[key(c)]['current'] if name=='previous' else c['direct']
            g=set(c['gold']);b=len(g&set(before['ranked'][:10]));a=len(g&set(c['current']['ranked'][:10]))
            rows.append(dict(sample_id=c['sample_id'],question_index=c['question_index'],question=c['question'],before=b,after=a,gold=len(g)))
        changes[name]=dict(improved=sum(x['after']>x['before'] for x in rows),
            regressed=sum(x['after']<x['before'] for x in rows),unchanged=sum(x['after']==x['before'] for x in rows),
            differences=[x for x in rows if x['after']!=x['before']])
    adds=[r for f in run.glob('*/add.jsonl') for r in load_lines(f)]
    successful={r['request_id']:r for r in adds if r['status']==200}
    routes=Counter()
    for p in run.glob('*/memory.sqlite3'):
        with sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True) as db:
            for kind,status,n in db.execute('SELECT memory_type,status,count(*) FROM rag_memory_routes GROUP BY memory_type,status'):
                routes[kind+':'+status]+=n
    report=dict(questions=len(current),manifest=manifest,metrics=totals,diagnostics=diagnostics,categories=categories,
        changes=changes,strategies=dict(Counter(c['current']['strategy'] for c in current)),
        add=dict(batches=len(successful),messages=sum(x['messages'] for x in successful.values()),
                 failed_attempts=sum(x['status']!=200 for x in adds),route_status=dict(routes)))
    path=PROJECT_ROOT/'docs/qwen-full-validation-20261009.md'
    path.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 当前配置完整纯文本测试（2026-10-09）','',f"测试提交：`{manifest['commit']}`。861 题全部完成，其中 859 题有标注证据；2 题无金标，只做接口测试，不计入召回分母。",'',
        '公开 LoCoMo-Refined，10 个完整对话样本。真实 v4 embedding、Qwen rerank、gpt-4o-mini，通过 FastAPI TestClient 调用 Add/Search；不是官网私有评测，也不是答案生成正确率。',
        '从根目录 .env 读取配置，仅覆盖隔离数据库及测试鉴权。最多 3 个会话同时测试，按会话排队；实际配置见配套 JSON 清单。',
        '当前 .env 未配置 RAG_TEMPORAL_MODE，因此采用默认 off；本轮不是时间窗口增强实验。旧测试脚本曾强制启用时间窗口，本轮已去除这项覆盖。',
        '每题请求 top_k=100，@5/@10 从同一返回列表截取。保存全部原文并校验 ID、内容一致、无重复和排序。',
        '新输入使用标准 content 中的显式 speaker 前缀保留数据集姓名；传入会话消息时间戳，不注入提问 reference_time。旧测试丢弃姓名且强制开启部分实验，新旧差异不能单独归因于 Qwen。',
        '本轮直接检索与完整流程使用相同新语料、embedding 和 Qwen，直接组关闭规划、分区和时间增强。','',
        '|指标|上轮旧配置完整流程|本轮直接检索|本轮当前完整流程|','|---|---:|---:|---:|']
    for k in ['5','10','100']:
        for metric in ['recall','hit','micro_recall','all_evidence']:
            lines.append('|'+metric+'@'+k+'|'+'|'.join(f"{totals[a][k][metric]:.2%}" for a in arms)+'|')
    lines+=['','Recall 为逐题宏平均；Micro Recall 为标注证据加权；Hit 表示至少命中一条；all_evidence 表示全部证据命中。','',
        '|诊断|上轮旧配置|本轮直接|本轮完整|','|---|---:|---:|---:|']
    for m in ['http_errors','fallbacks','routing_errors','p50','p95']:
        lines.append('|'+m+'|'+'|'.join(f'{diagnostics[a][m]:.3f}' for a in arms)+'|')
    lines+=['','延迟单位为秒，包含真实 API 与并发等待，不能当作隔离模型速度；远端服务未修改。','',
        '|数据集题型|上轮 Recall@10|本轮直接 Recall@10|本轮完整 Recall@10|','|---|---:|---:|---:|']
    for cat,values in categories.items():lines.append('|'+cat+'|'+'|'.join(f"{values[a]['recall']:.2%}" for a in arms)+'|')
    lines+=['','题型编号沿用数据集，不等同于官网能力维度。','', '## 逐题变化','']
    for name,item in changes.items():lines.append(f"- 相对 {name}：改善 {item['improved']}，退化 {item['regressed']}，持平 {item['unchanged']}。完整差异列表见 JSON。")
    lines+=['','## 写入与配置','','```json',json.dumps(report['add'],ensure_ascii=False,indent=2),'```','',
        '首轮 conv-43 的一个 Add 批次连续两次 502。单独复查成功后以原输入恢复测试；conv-44 另有一次 502，重试成功。共 3 次失败尝试全部保留。完成状态只说明写入成功，不代表分类逐条人工审核。',
        '', '## 结论与限制', '',
        '本轮完整流程 Recall@10 为 94.59%，Micro Recall@10 为 91.15%，达到这份公开集上的 90% 证据覆盖目标。相对旧全量 Recall@10 提升 40.04 个百分点，但同时改变了上下文、人物身份投影、reranker 和实验开关，不能将全部提升归因于 Qwen。',
        '同输入与模型下，上层流程相较直接检索只增加 0.37 个百分点 Recall@10、0.65 个百分点 Micro Recall@10；中位延迟从 1.29 秒增至 9.95 秒。收益主要体现在 Top5（Recall +2.00 个百分点），不能说它对所有题都必要。',
        '题型 1 的完整流程 Recall@10 为 79.87%，全证据命中@10 只有 64.94%；题型 3 则相较直接检索退化（78.68% → 74.26%）。整体高分不能代表多跳长链与推断题已经解决。',
        '只验证证据检索，没有统一 Answer 模型答题评分；数据集已在开发中多次使用，不是未见测试集。字段投影无法保证等同于官方私有输入，模型随机性及 API 波动也尚未通过多轮重复估计。',
        '数据与逐题记录：`data/final-text-validation/20261009-qwen-full/`。未调用官方评测、未推送部署。','']
    path.write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(dict(metrics=totals,diagnostics=diagnostics,add=report['add'],changes={k:{a:b for a,b in v.items() if a!='differences'} for k,v in changes.items()}),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
