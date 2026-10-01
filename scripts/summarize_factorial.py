"""Report fixed-budget metrics, paired changes and conversation-cluster uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT
from memory.evaluation import canonical_evidence,evidence_metrics
from memory.vanilla import chunks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='data/factorial-benchmarks/20261001-v2');args=parser.parse_args()
    out=PROJECT_ROOT/args.out
    report=json.loads((out/'report.json').read_text(encoding='utf-8'))
    cases=[json.loads(l) for l in (out/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(cases)==859 and len({(c['sample_id'],c['question_index']) for c in cases})==859
    raw=json.loads((PROJECT_ROOT/'data/locomo-refined/data/raw/locomo_refined.json').read_text(encoding='utf-8'))
    mapping={}
    for sample in raw:
        for session,messages in sample['conversation'].items():
            if not session.startswith('session_') or not isinstance(messages,list):continue
            for start in range(0,len(messages),40):
                for i,m in enumerate(messages[start:start+40]):
                    for j,_ in enumerate(chunks(m['text'])):
                        for prefix in ['factorial:','public-full:']:
                            mid=hashlib.sha256(json.dumps([prefix+sample['sample_id'],session+':'+str(start),i,j]).encode()).hexdigest()
                            mapping[mid]=canonical_evidence([m['dia_id']])[0]
    names=list(report['variants'])
    at5=[]
    for c in cases:
        row=dict(c)
        for name in names:
            found=set(mapping[mid] for mid in c[name]['ids'][:5]) & set(c['evidence'])
            row[name]=dict(hit_evidence=sorted(found))
        at5.append(row)
    by_conversation={s['sample_id']:{name:evidence_metrics([c for c in cases if c['sample_id']==s['sample_id']],name) for name in names} for s in raw}
    rng=np.random.default_rng(20261001)
    resampled=rng.integers(0,len(raw),size=(10000,len(raw)))
    sizes=np.array([by_conversation[s['sample_id']]['base']['evaluated_questions'] for s in raw])
    paired={}
    for name in names:
        changes=np.array([by_conversation[s['sample_id']][name]['hit_questions']-by_conversation[s['sample_id']]['base']['hit_questions'] for s in raw])
        deltas=changes[resampled].sum(axis=1)/sizes[resampled].sum(axis=1)
        paired[name]=dict(hit_difference_95_cluster_percentile=np.percentile(deltas,[2.5,97.5]).tolist(),
            wins=[dict(sample_id=c['sample_id'],question_index=c['question_index'],question=c['question']) for c in cases if c[name]['hit_evidence'] and not c['base']['hit_evidence']],
            losses=[dict(sample_id=c['sample_id'],question_index=c['question_index'],question=c['question']) for c in cases if not c[name]['hit_evidence'] and c['base']['hit_evidence']])
    frozen=[]
    for filename in ['dev-all-cases.json','heldout-propagate_2-cases.json']:
        frozen+=json.loads((PROJECT_ROOT/'data/hit10-experiments'/filename).read_text(encoding='utf-8'))
    old={(c['sample_id'],c['question_index']):c for c in frozen}
    parity=sum([mapping[mid] for mid in c['base']['ids']] == [mapping[mid] for mid in old[(c['sample_id'],c['question_index'])]['propagate_2']['ids']] for c in cases)
    latency=json.loads((out/'latency.json').read_text(encoding='utf-8'))
    indexed={(c['sample_id'],c['question_index']):c for c in cases}
    latency_parity=sum(c['ids']==indexed[(c['sample_id'],c['question_index'])][c['variant']]['ids'] for c in latency['cases'])
    assert latency_parity==len(latency['cases']), 'Real uncached engine differs from cached score evaluation'
    ranked=sorted(names,key=lambda n:(-report['metrics']['all'][n]['hit_questions'],
        -report['metrics']['all'][n]['micro_recall'],-report['metrics']['all'][n]['recall'],latency['metrics'][n]['p50_seconds']))
    summary=dict(report=report,at5={n:evidence_metrics(at5,n) for n in names},by_conversation=by_conversation,
        by_evidence_count={label:{n:evidence_metrics([c for c in cases if (len(c['evidence'])>1)==multiple],n) for n in names}
                           for label,multiple in [('single',False),('multiple',True)]},
        paired=paired,latency=latency['metrics'],frozen_baseline_top10_exact_matches=parity,
        uncached_parity_queries=latency_parity,ranked_variants=ranked,winner=ranked[0],
        source_cases_sha256=hashlib.sha256((out/'cases.jsonl').read_bytes()).hexdigest(),
        note='Winner selected on previously inspected public data. Cluster intervals descriptive, only ten conversations; not an independent generalization test. Fixed budget ten original messages, no generated answers.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    for n in ranked:
        m=report['metrics']['all'][n];l=latency['metrics'][n]
        print(n,'Hit',round(100*m['hit_rate'],2),'Recall',round(100*m['recall'],2),
              'Micro',round(100*m['micro_recall'],2),'All',round(100*m['all_evidence_hit_rate'],2),
              'p50ms',round(1000*l['p50_seconds'],1),'p95ms',round(1000*l['p95_seconds'],1))
    print('winner',ranked[0],'baseline exact',parity,'uncached parity',latency_parity)


if __name__=='__main__':main()
