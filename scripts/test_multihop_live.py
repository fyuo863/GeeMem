"""Small real Add/Search comparison. Reports planner fallback separately from success."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.aml_api import create_app
from memory.config import PROJECT_ROOT, load_settings
from memory.rerank import LocalReranker, HTTPReranker
from memory.vanilla import VanillaMemory, HTTPEmbedder, LocalEmbedder, chunks


class TracedMemory(VanillaMemory):
    def search(self,payload,**kwargs):
        self.last_trace={}
        return super().search(payload,trace=self.last_trace)


def synthetic():
    definitions=[
        ('direct','Where does Mira live?', ['Mira lives in Tallinn.']),
        ('split','What is the total cost of my bicycle and tent?',
         ['I paid $240 for my bicycle.','My tent cost $180.']),
        ('chain','Where does the spouse of Leona work?',
         ['Leona is married to Tomas.','Tomas works at Cedar Labs.']),
        ('chain','Who is the CEO of the company where the spouse of Nadia works?',
         ['Nadia is married to Elias.','Elias works at Solstice Analytics.','The CEO of Solstice Analytics is Ruth Chen.'])]
    out=[]
    for i,(strategy,question,evidence) in enumerate(definitions):
        texts=['We discussed a garden project and an upcoming trip.']*6+evidence+[
            'Mira recommended a book about Paris.', 'Leona works at Harbor Studio.',
            'Nadia visited Cedar Labs last year.', 'A bicycle repair cost $30.',
            'The rental tent cost $20 per day.', 'The CEO of Harbor Studio is Alan.']
        # Distinct sessions prevent adjacency alone from supplying the whole chain.
        messages=[dict(role='user',content=t,timestamp=1704067200000+j*60000) for j,t in enumerate(texts)]
        out.append(dict(id=f'synthetic-{i}',category=strategy,question=question,
            sessions=[(str(j),[m]) for j,m in enumerate(messages)],gold=[f'{j}:0' for j in range(6,6+len(evidence))]))
    return out


def public_samples():
    source=PROJECT_ROOT/'data/longmemeval/longmemeval_s_cleaned.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    result=[]
    for kind in ('multi-session','temporal-reasoning','knowledge-update','single-session-user'):
        selected=sorted((s for s in data if s['question_type']==kind and not s['question_id'].endswith('_abs')),
            key=lambda s:hashlib.sha256(('evidence-pilot-v1:'+s['question_id']).encode()).hexdigest())[:2]
        for sample in selected:
            sessions=[];gold=[]
            for sid,date,messages in zip(sample['haystack_session_ids'],sample['haystack_dates'],sample['haystack_sessions'],strict=True):
                stamp=int(datetime.strptime(date[:10]+' '+date[-5:],'%Y/%m/%d %H:%M').replace(tzinfo=timezone.utc).timestamp()*1000)
                sessions.append((sid,[dict(role=m['role'],content=m['content'],timestamp=stamp) for m in messages]))
                gold.extend(f'{sid}:{i}' for i,m in enumerate(messages) if m.get('has_answer'))
            result.append(dict(id=sample['question_id'],category=kind,question=sample['question'],sessions=sessions,gold=gold))
    return result


def batches(sample,cfg):
    uid=('multihop:' if sample['id'].startswith('synthetic') else 'lme-pilot:')+sample['id']
    requests=[];mapping={}
    for sid,messages in sample['sessions']:
        expanded=[]
        for i,m in enumerate(messages):
            expanded.extend((dict(m,content=m['content'][j:j+30000]),i) for j in range(0,len(m['content']),30000)
                            if m['content'][j:j+30000].strip())
        for start in range(0,len(expanded),20):
            part=expanded[start:start+20];rid=f'{sample["id"]}:{sid}:{start}'
            requests.append(dict(user_id=uid,request_id=rid,session_id=sid,messages=[m for m,_ in part]))
            for i,(m,source_i) in enumerate(part):
                for j,text in enumerate(chunks(m['content'],int(cfg.get('RAG_CHUNK_TOKENS','320')),int(cfg.get('RAG_CHUNK_OVERLAP','40')))):
                    key=hashlib.sha256(json.dumps([uid,rid,i,j]).encode()).hexdigest()
                    mapping[key]=dict(message=f'{sid}:{source_i}',content=text)
    return requests,mapping


def metrics(cases,k):
    recalls=[len(set(c['gold'])&set(c['ranked_messages'][:k]))/len(set(c['gold'])) for c in cases]
    return dict(hit=float(np.mean([r>0 for r in recalls])),recall=float(np.mean(recalls)),
        micro_recall=sum(len(set(c['gold'])&set(c['ranked_messages'][:k])) for c in cases)/sum(len(set(c['gold'])) for c in cases),
        all_evidence=float(np.mean([r==1 for r in recalls])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--public-cache',type=Path,help='Existing compatible public pilot baseline DB; copied, never modified.')
    parser.add_argument('--public',action='store_true',help='Use eight public full-history regression questions instead of four constructed cases.')
    args=parser.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=load_settings()
    cfg.update(RAG_MEMORY_DB=str(out/'memory.sqlite3'),RAG_RESULT_WINDOW='0',RAG_MULTIHOP_MODE='off',
               AML_AUTH_MODE='bearer',AML_API_KEY='local-test')
    if args.public_cache:
        if not args.public: parser.error('--public-cache requires --public')
        with closing(sqlite3.connect(args.public_cache.resolve().as_uri()+'?mode=ro',uri=True)) as src,closing(sqlite3.connect(cfg['RAG_MEMORY_DB'])) as dst:
            src.backup(dst)
    embedder=HTTPEmbedder(cfg) if cfg.get('RAG_EMBEDDING_API_URL') else LocalEmbedder(cfg)
    reranker=HTTPReranker(cfg) if cfg.get('RAG_RERANK_API_URL') else LocalReranker(cfg)
    stores={name:TracedMemory(dict(cfg,RAG_MULTIHOP_MODE=mode),embedder,reranker=reranker)
            for name,mode in [('baseline','off'),('multihop','llm')]}
    manifest=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [PROJECT_ROOT/'memory/multihop.py',PROJECT_ROOT/'memory/vanilla.py',Path(__file__)]},
        settings={k:v for k,v in cfg.items() if (k.startswith('RAG_') or k=='LLM_MODEL') and not any(s in k for s in ('KEY','URL','PROXY'))},
        protocol='Actual in-process Add/Search handlers; actual embedding, reranker and gpt-4o-mini calls. No gold sent to memory. Not an answer-level or official score.',
        evaluation='public regression' if args.public else 'constructed functional scenarios')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    samples=public_samples() if args.public else synthetic()
    cases=[]
    for sample in samples:
        requests,mapping=batches(sample,cfg)
        for name,store in stores.items():
            with TestClient(create_app(settings=cfg,backend=store)) as client:
                client.headers['Authorization']='Bearer local-test'
                for body in requests:
                    r=client.post('/add',json=body)
                    assert r.status_code==200, r.status_code
                    assert r.json()==dict(success=True,**{k:body[k] for k in ('user_id','session_id','request_id')})
                before=time.perf_counter()
                r=client.post('/search',json=dict(user_id=requests[0]['user_id'],query=sample['question'],top_k=10))
                seconds=time.perf_counter()-before
                assert r.status_code==200,r.status_code
                hits=r.json()['data']
                assert len(hits)<=10 and len({h['id'] for h in hits})==len(hits)
                assert all(h['id'] in mapping and h['content']==mapping[h['id']]['content'] for h in hits)
                assert all(a['score']>=b['score'] for a,b in zip(hits,hits[1:]))
                case=dict(variant=name,id=sample['id'],category=sample['category'],question=sample['question'],gold=sample['gold'],
                    ranked_messages=[mapping[h['id']]['message'] for h in hits],hits=hits,trace=store.last_trace,seconds=seconds)
                cases.append(case)
                with (out/'cases.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(case,ensure_ascii=False)+'\n')
                print(name,sample['id'],round(seconds,2),store.last_trace.get('strategy'),store.last_trace.get('stop'),flush=True)
    result={}
    for name in stores:
        arm=[c for c in cases if c['variant']==name]
        result[name]=dict(metrics={str(k):metrics(arm,k) for k in (3,5,10)},
            mean_seconds=float(np.mean([c['seconds'] for c in arm])),p95_seconds=float(np.percentile([c['seconds'] for c in arm],95)),
            fallback_count=sum(c['trace'].get('fallback',False) for c in arm),
            llm_calls=sum(c['trace'].get('llm_calls',0) for c in arm),
            search_calls=sum(c['trace'].get('search_calls',1) for c in arm))
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('COMPLETE',out,flush=True)


if __name__=='__main__': main()
