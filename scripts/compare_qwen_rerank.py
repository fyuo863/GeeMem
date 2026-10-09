"""80 fixed public questions, shared v4 pools, no LLM or gold-fed retrieval."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT,load_settings
from memory.embeddings import HTTPEmbedder
from memory.rerank import LocalReranker,HTTPReranker
from memory.vanilla import VanillaMemory
from memory.evaluation import canonical_evidence


def main():
    run=PROJECT_ROOT/'data/final-text-validation/20261009-official861'
    out=PROJECT_ROOT/'data/final-text-validation/qwen-repairs80-20261009';out.mkdir(exist_ok=True)
    cfg=load_settings();cfg.update(json.loads((run/'manifest.json').read_text())['settings'])
    cfg.update(RAG_MULTIHOP_MODE='off',RAG_PARTITION_MODE='off',RAG_TEMPORAL_MODE='off',
        RAG_TEMPORAL_EXPERIMENT='off',RAG_WRITE_GATE_MODE='off',RAG_BUILD_MODE='off',RAG_FUSION_QA='off')
    cfg.pop('RAG_RERANK_API_URL',None)
    mini=LocalReranker(cfg);embedder=HTTPEmbedder(cfg)
    api_cfg=dict(cfg,RAG_RERANK_API_PROTOCOL='dashscope',RAG_RERANK_API_MODEL='qwen3.7-text-rerank',
        RAG_RERANK_API_URL='https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank',
        RAG_RERANK_API_KEY=cfg['RAG_EMBEDDING_API_KEY'],RAG_RERANK_API_PROXY=cfg.get('RAG_EMBEDDING_API_PROXY',''))
    api=HTTPReranker(api_cfg)
    dataset=json.loads((PROJECT_ROOT/'data/locomo-refined/data/raw/locomo_refined.json').read_text(encoding='utf8'))
    previous=json.loads((PROJECT_ROOT/'data/final-text-validation/diagnosis-20261009/cases.json').read_text(encoding='utf8'))
    excluded={(x['sample_id'],x['question_index']) for x in previous}
    logfile=out/'cases.jsonl'
    results=[json.loads(l) for l in logfile.read_text(encoding='utf8').splitlines()] if logfile.exists() else []
    done={(x['sample_id'],x['question_index']) for x in results}
    for sample in dataset:
        sid=sample['sample_id'];dbpath=out/(sid+'.sqlite3')
        with sqlite3.connect((run/sid/'memory.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as src:
            with sqlite3.connect(dbpath) as dst:src.backup(dst)
        store=VanillaMemory(dict(cfg,RAG_MEMORY_DB=str(dbpath)),embedder,reranker=mini)
        with store.connect() as db:
            rows=[dict(r) for r in db.execute('SELECT m.*,s.role,s.speaker,s.session_timestamp,s.source_id,s.source_index FROM rag_memories m LEFT JOIN rag_sources s ON m.id=s.memory_id ORDER BY m.rowid')]
        indices={r['id']:i for i,r in enumerate(rows)};evidence={}
        for row in rows:
            m=sample['conversation'][row['session_id']][int(row['request_id'].rsplit(':',1)[1])+row['message_index']]
            evidence[row['id']]=canonical_evidence([m['dia_id']])[0]
        selected=sorted([(i,q) for i,q in enumerate(sample['qa']) if not q.get('is_multi_modality') and q.get('evidence') and (sid,i) not in excluded],
            key=lambda x:hashlib.sha256((sid+':'+str(x[0])+':rerank-repairs-v1').encode()).hexdigest())[:8]
        for n,(qi,q) in enumerate(selected):
            if (sid,qi) in done:continue
            store.reranker=mini;store.rerank_context=0;store.target_mode='off';store.rerank_selection='direct'
            trace={};store.search(SimpleNamespace(user_id='release-text:'+sid,query=q['question'],top_k=100,options=None,retrieval_trace=trace))
            ids=trace['candidate_ids'];candidates=[indices[mid] for mid in ids];gold=set(canonical_evidence(q['evidence']))
            case=dict(sample_id=sid,question_index=qi,question=q['question'],category=q['category'],group='screen' if n<4 else 'validation',
                gold=sorted(gold),candidate_ids=ids,variants={})
            for name,ranker,context,target,selection in [('old_minilm',mini,0,'off','direct'),
                ('repaired_minilm',mini,1,'on','context_support'),('qwen37',api,1,'off','direct')]:
                store.reranker=ranker;store.rerank_context=context;store.target_mode=target;store.rerank_selection=selection
                start=time.perf_counter();scores=store.score_candidates(q['question'],rows,candidates);seconds=time.perf_counter()-start
                ranked=[evidence[ids[int(i)]] for i in np.argsort(-scores,kind='stable')]
                case['variants'][name]=dict(ranked=ranked,seconds=seconds,
                    metrics={str(k):dict(recall=len(gold&set(ranked[:k]))/len(gold),hit=int(bool(gold&set(ranked[:k]))),
                        found=len(gold&set(ranked[:k])),all_evidence=int(gold<=set(ranked[:k]))) for k in [5,10,100]})
            results.append(case)
            with logfile.open('a',encoding='utf8') as f:f.write(json.dumps(case,ensure_ascii=False)+'\n')
            print('COMPARE',len(results),sid,qi,flush=True)
    report={}
    for group in ['all','screen','validation']:
        cs=[c for c in results if group=='all' or c['group']==group]
        report[group]=dict(questions=len(cs),models={name:dict(
            metrics={str(k):{**{m:float(np.mean([c['variants'][name]['metrics'][str(k)][m] for c in cs])) for m in ['recall','hit','all_evidence']},
                'micro_recall':sum(c['variants'][name]['metrics'][str(k)]['found'] for c in cs)/sum(len(c['gold']) for c in cs)} for k in [5,10,100]},
            mean_seconds=float(np.mean([c['variants'][name]['seconds'] for c in cs])),
            p95_seconds=float(np.percentile([c['variants'][name]['seconds'] for c in cs],95))) for name in ['old_minilm','repaired_minilm','qwen37']})
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
