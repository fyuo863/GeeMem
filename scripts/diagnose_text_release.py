"""Fixed-pool attribution audit. No LLM, no changed corpus vectors or gold-fed ranking."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT, load_settings
from memory.embeddings import HTTPEmbedder
from memory.evaluation import canonical_evidence
from memory.rerank import LocalReranker
from memory.vanilla import VanillaMemory


def main():
    run=PROJECT_ROOT/'data/final-text-validation/20261009-official861'
    out=PROJECT_ROOT/'data/final-text-validation/diagnosis-20261009'
    out.mkdir(exist_ok=True)
    cfg=load_settings()
    cfg.update(json.loads((run/'manifest.json').read_text())['settings'])
    cfg.update(RAG_MULTIHOP_MODE='off',RAG_PARTITION_MODE='off',RAG_TEMPORAL_MODE='off',
               RAG_TEMPORAL_EXPERIMENT='off',RAG_WRITE_GATE_MODE='off',RAG_BUILD_MODE='off')
    cfg.pop('RAG_RERANK_API_URL',None)
    reranker=LocalReranker(cfg)
    embedder=HTTPEmbedder(cfg)
    samples=json.loads((PROJECT_ROOT/'data/locomo-refined/data/raw/locomo_refined.json').read_text(encoding='utf8'))
    results=[]
    variants={'bare':(False,0,'off','direct'), 'context_only':(False,1,'off','direct'),
              'speaker_only':(True,0,'off','direct'), 'speaker_context':(True,1,'off','direct'),
              'speaker_context_target_support':(True,1,'on','context_support')}
    for sample in samples:
        sid=sample['sample_id'];dbpath=out/(sid+'.sqlite3')
        with sqlite3.connect((run/sid/'memory.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as src:
            with sqlite3.connect(dbpath) as dest:src.backup(dest)
        store=VanillaMemory(dict(cfg,RAG_MEMORY_DB=str(dbpath)),embedder,reranker=reranker)
        with store.connect() as db:
            rows=[dict(r) for r in db.execute('SELECT m.*,s.role,s.speaker,s.session_timestamp,s.source_id,s.source_index FROM rag_memories m LEFT JOIN rag_sources s ON m.id=s.memory_id ORDER BY m.rowid')]
        row_indices={r['id']:i for i,r in enumerate(rows)}
        named=[];evidence={}
        for row in rows:
            offset=int(row['request_id'].rsplit(':',1)[1])+row['message_index']
            original=sample['conversation'][row['session_id']][offset]
            named.append(dict(row,speaker=original['speaker']))
            evidence[row['id']]=canonical_evidence([original['dia_id']])[0]
        pool=sorted([(i,q) for i,q in enumerate(sample['qa']) if not q.get('is_multi_modality') and q.get('evidence')],
            key=lambda x:hashlib.sha256((sid+':'+str(x[0])+':input-diagnostic-v1').encode()).hexdigest())[:4]
        for qi,q in pool:
            trace={};store.rerank_context=0;store.target_mode='off';store.rerank_selection='direct'
            response=store.search(SimpleNamespace(user_id='release-text:'+sid,query=q['question'],top_k=100,options=None,retrieval_trace=trace))
            ids=trace['candidate_ids'];indices=[row_indices[mid] for mid in ids]
            gold=set(canonical_evidence(q['evidence']))
            result=dict(sample_id=sid,question_index=qi,question=q['question'],category=q['category'],gold=sorted(gold),
                        pool_recall=len(gold&{evidence[mid] for mid in ids})/len(gold),variants={})
            for name,(use_speaker,context,target,selection) in variants.items():
                store.rerank_context=context;store.target_mode=target;store.rerank_selection=selection
                scores=store.score_candidates(q['question'],named if use_speaker else rows,indices)
                order=np.argsort(-scores,kind='stable')
                ranked=[evidence[ids[int(i)]] for i in order]
                if name=='bare':assert [evidence[h['id']] for h in response['data']]==ranked[:100]
                result['variants'][name]=dict(ranked=ranked,
                    recall10=len(gold&set(ranked[:10]))/len(gold),recall100=len(gold&set(ranked[:100]))/len(gold))
            results.append(result)
            (out/'cases.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
            print('DIAG',len(results),sid,qi,flush=True)
    summary=dict(questions=len(results),pool_recall=float(np.mean([r['pool_recall'] for r in results])),
                 variants={name:{k:float(np.mean([r['variants'][name][k] for r in results])) for k in ['recall10','recall100']} for name in variants})
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
