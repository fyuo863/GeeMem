"""Conversation-split retrieval diagnosis; no generated answers or gold-aware ranking."""
import argparse,json,sqlite3,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT,load_settings
from memory.vanilla import VanillaMemory,LocalEmbedder,chunks
from memory.rerank import LocalReranker
from memory.evaluation import canonical_evidence,evidence_metrics


def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','heldout'],default='dev');p.add_argument('--variant',choices=['all','plain','context'],default='all');args=p.parse_args()
 cfg=load_settings();cfg=dict(cfg,RAG_RERANK_DEVICE='cuda',RAG_RERANK_BATCH_SIZE='32',RAG_RERANK_MAX_LENGTH='512')
 model=LocalEmbedder(cfg);rerank=LocalReranker(cfg)
 root=PROJECT_ROOT/'data/full-semantic-benchmarks/20260930T122205Z'
 out=PROJECT_ROOT/'data/rerank-benchmarks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out.mkdir(parents=True)
 with sqlite3.connect(root/'baseline.sqlite3') as src,sqlite3.connect(out/'memory.sqlite3') as dst:src.backup(dst)
 store=VanillaMemory(dict(cfg,RAG_MEMORY_DB=str(out/'memory.sqlite3'),RAG_TAG_MODE='off',RAG_RERANK_MODE='off',RAG_RESULT_WINDOW='0'),model)
 samples=json.loads((PROJECT_ROOT/'data/locomo-refined/data/raw/locomo_refined.json').read_text(encoding='utf-8'))
 samples=samples[:3] if args.split=='dev' else samples[3:]
 original={(c['sample_id'],c['question_index']):c for c in map(json.loads,(root/'cases.jsonl').read_text(encoding='utf-8').splitlines()) if c['top_k']==10}
 cases=[];variants=['plain','context'] if args.variant=='all' else [args.variant]
 print('OUTPUT',out,flush=True)
 with (out/'cases.jsonl').open('w',encoding='utf-8') as stream:
  for sample in samples:
   user='public-full:'+sample['sample_id'];mapping={}
   for session,batch_all in sample['conversation'].items():
    if not session.startswith('session_') or not isinstance(batch_all,list):continue
    for start in range(0,len(batch_all),40):
     for i,m in enumerate(batch_all[start:start+40]):
      for j,_ in enumerate(chunks(m['text'])):
       mid=hashlib.sha256(json.dumps([user,session+':'+str(start),i,j]).encode()).hexdigest();mapping[mid]=canonical_evidence([m['dia_id']])[0]
   with store.connect() as db:rows=db.execute('SELECT * FROM rag_memories WHERE user_id=? ORDER BY rowid',(user,)).fetchall()
   position={r['id']:i for i,r in enumerate(rows)}
   for qi,q in enumerate(sample['qa']):
    gold=canonical_evidence(q.get('evidence'))
    if q.get('is_multi_modality',False) or not gold:continue
    before=time.perf_counter();hits=store.search(SimpleNamespace(user_id=user,query=q['question'],top_k=200,options=None))['data'];retrieval=time.perf_counter()-before
    c=dict(sample_id=sample['sample_id'],question_index=qi,question=q['question'],evidence=gold)
    c['original_window']=original[(sample['sample_id'],qi)]['baseline']
    def record(name,chosen,seconds):
     found=set(mapping[h['id']] for h in chosen)&set(gold)
     c[name]=dict(hit_evidence=sorted(found),recall=len(found)/len(gold),seconds=seconds,ids=[h['id'] for h in chosen])
    record('no_window',hits[:10],retrieval)
    for k in [20,50,100,200]:record('ceiling_'+str(k),hits[:k],retrieval)
    for variant in variants:
     documents=[]
     for h in hits:
      text=h['content'];i=position[h['id']]
      if variant=='context':
       prev=rows[i-1]['content'] if i>0 and rows[i-1]['session_id']==rows[i]['session_id'] else ''
       following=rows[i+1]['content'] if i+1<len(rows) and rows[i+1]['session_id']==rows[i]['session_id'] else ''
       text='Target message: '+text+'\nPrevious context: '+prev+'\nNext context: '+following
      documents.append(text)
     before=time.perf_counter();scores=rerank.score(q['question'],documents);elapsed=time.perf_counter()-before
     assert len(scores)==len(hits) and np.isfinite(scores).all()
     order=np.argsort(-scores,kind='stable').tolist()
     record(variant,[hits[i] for i in order[:10]],retrieval+elapsed)
     c[variant]['candidate_scores']=scores.tolist()
    cases.append(c);stream.write(json.dumps(c,ensure_ascii=False)+'\n');stream.flush()
    if len(cases)%20==0:print('SEARCHED',len(cases),sample['sample_id'],flush=True)
 summary={}
 for name in ['original_window','no_window',*[f'ceiling_{k}' for k in [20,50,100,200]],*variants]:
  summary[name]=evidence_metrics(cases,name)
  summary[name]['p50_seconds']=float(np.median([c[name]['seconds'] for c in cases]))
  if not name.startswith('ceiling') and name!='original_window':summary[name].update(wins=sum(c[name]['recall']>c['original_window']['recall'] for c in cases),losses=sum(c[name]['recall']<c['original_window']['recall'] for c in cases))
 report=dict(split=args.split,conversations=[s['sample_id'] for s in samples],metrics=summary,reranker=rerank.identity,candidate_limit=200,top_k=10,llm_calls=0,
  note='Development first 3 conversations; heldout remaining 7. Previously observed aggregate dataset, not an unseen external test. Original baseline latency from prior run; reranker GPU, retrieval CPU.')
 (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));print('REPORT',out/'report.json',flush=True)


if __name__=='__main__':main()
