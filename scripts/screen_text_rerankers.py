"""24 fixed development questions; identical v4 candidate pools and documents."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT,load_settings
from memory.embeddings import HTTPEmbedder
from memory.rerank import LocalReranker,ONNXReranker
from memory.vanilla import VanillaMemory
from memory.evaluation import canonical_evidence


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    cfg=load_settings()
    mini=LocalReranker(dict(cfg,RAG_RERANK_PATH='data/models/ms-marco-MiniLM-L-6-v2',RAG_RERANK_DEVICE='cuda',RAG_RERANK_BATCH_SIZE='32'))
    bge=ONNXReranker(cfg)
    samples=json.loads((PROJECT_ROOT/'data/locomo-refined/data/raw/locomo_refined.json').read_text(encoding='utf8'))[:3]
    pool=sorted([(s,qi,q) for s in samples for qi,q in enumerate(s['qa']) if not q.get('is_multi_modality') and q.get('evidence')],
                key=lambda x:hashlib.sha256((x[0]['sample_id']+':'+str(x[1])+':rerank-screen-v1').encode()).hexdigest())[:24]
    class Capture:
        def score(self,query,docs):
            self.query=query;self.docs=docs
            before=time.perf_counter();self.scores=mini.score(query,docs);self.seconds=time.perf_counter()-before
            return self.scores
    recorder=Capture();results=[]
    for sample,qi,q in pool:
        sid=sample['sample_id'];db=args.run/sid/'memory.sqlite3'
        local=dict(cfg,RAG_MEMORY_DB=str(db.resolve()),RAG_MULTIHOP_MODE='off',RAG_PARTITION_MODE='off',
            RAG_WRITE_GATE_MODE='off',RAG_BUILD_MODE='off',RAG_TEMPORAL_MODE='off',RAG_RESULT_WINDOW='0',RAG_RERANK_CANDIDATES='400')
        store=VanillaMemory(local,HTTPEmbedder(local),reranker=recorder)
        trace={};request=SimpleNamespace(user_id='release-text:'+sid,query=q['question'],top_k=100,options=None,retrieval_trace=trace)
        store.search(request)
        with store.connect() as conn:
            rows={r['id']:dict(r) for r in conn.execute('SELECT id,session_id,message_index,request_id FROM rag_memories WHERE user_id=?',(request.user_id,))}
        evidence={}
        for mid in trace['candidate_ids']:
            row=rows[mid];offset=int(row['request_id'].rsplit(':',1)[1]);m=sample['conversation'][row['session_id']][offset+row['message_index']]
            evidence[mid]=canonical_evidence([m['dia_id']])[0]
        before=time.perf_counter();bge_scores=bge.score(recorder.query,recorder.docs);bge_seconds=time.perf_counter()-before
        gold=set(canonical_evidence(q['evidence']));entry=dict(sample_id=sid,question_index=qi,gold=sorted(gold),variants={})
        for name,scores,seconds in [('minilm',recorder.scores,recorder.seconds),('bge',bge_scores,bge_seconds)]:
            ids=[trace['candidate_ids'][int(i)] for i in np.argsort(-scores,kind='stable')]
            ranked=[evidence[i] for i in ids]
            entry['variants'][name]=dict(seconds=seconds,ranked=ranked,
                recall10=len(gold&set(ranked[:10]))/len(gold),hit10=int(bool(gold&set(ranked[:10]))),
                recall100=len(gold&set(ranked[:100]))/len(gold))
        results.append(entry)
        (args.run/'reranker-screen.json').write_text(json.dumps(results,indent=2),encoding='utf8')
        print('SCREEN',len(results),sid,qi,flush=True)
    summary={name:{key:float(np.mean([r['variants'][name][key] for r in results]))
                    for key in ['seconds','recall10','hit10','recall100']} for name in ['minilm','bge']}
    (args.run/'reranker-screen-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
