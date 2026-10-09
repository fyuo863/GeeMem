"""Real provider /add + /search smoke, isolated public conversation subset."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from memory.config import PROJECT_ROOT,load_settings
from memory.aml_api import create_app
from memory.vanilla import VanillaMemory
from memory.evaluation import canonical_evidence

def main():
    out=PROJECT_ROOT/'data/final-text-validation/qwen-smoke-20261009';out.mkdir(exist_ok=True)
    cfg=load_settings();cfg.update(RAG_MEMORY_DB=str(out/'memory.sqlite3'),AML_API_KEY='isolated-test')
    sample=json.loads((PROJECT_ROOT/'data/locomo-refined/data/raw/locomo_refined.json').read_text(encoding='utf8'))[0]
    questions=[q for q in sample['qa'] if not q.get('is_multi_modality') and q.get('evidence') and all(e.split(':')[0] in ['D1','D2','D3'] for e in q['evidence'])][:6]
    store=VanillaMemory(cfg);mapping={};report=dict(add=[],search=[])
    with TestClient(create_app(settings=cfg,backend=store)) as c:
        c.headers['Authorization']='Bearer isolated-test'
        for n in [1,2,3]:
            messages=[]
            for m in sample['conversation'][f'session_{n}']:
                content=f"[speaker: {m['speaker']}] {m['text']}"
                mapping[content]=canonical_evidence([m['dia_id']])[0]
                messages.append(dict(role='user' if m['speaker']==sample['conversation']['speaker_a'] else 'assistant',content=content))
            start=time.perf_counter()
            r=c.post('/add',json=dict(user_id='smoke',session_id=f'session_{n}',request_id=f'add-{n}',messages=messages))
            report['add'].append(dict(session=n,status=r.status_code,messages=len(messages),seconds=time.perf_counter()-start))
            print('ADD',n,r.status_code,flush=True);assert r.status_code==200
        for q in questions:
            trace={};start=time.perf_counter()
            # Same real HTTP adapter, capture internal trace without changing the search path.
            original=store.search
            def traced(payload,**kwargs):return original(payload,trace=trace)
            store.search=traced
            try:r=c.post('/search',json=dict(user_id='smoke',query=q['question'],top_k=10))
            finally:store.search=original
            assert r.status_code==200
            hits=r.json()['data'];assert all(h['content'] in mapping for h in hits)
            gold=set(canonical_evidence(q['evidence']));ranked=[mapping[h['content']] for h in hits]
            report['search'].append(dict(question=q['question'],gold=sorted(gold),ranked=ranked,recall=len(gold&set(ranked))/len(gold),seconds=time.perf_counter()-start,trace=trace))
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
            print('SEARCH',len(report['search']),report['search'][-1]['recall'],flush=True)
        r=c.post('/search',json=dict(user_id='empty-user',query='What does Caroline like?',top_k=10))
        assert r.status_code==200 and r.json()['data']==[]
        report['isolation_passed']=True
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('COMPLETE',len(questions),flush=True)

if __name__=='__main__':main()
