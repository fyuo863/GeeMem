"""Real classification and embeddings, no event extraction."""
from contextlib import closing
from pathlib import Path
import json
import time
from fastapi.testclient import TestClient
from memory.config import load_settings
from memory.vanilla import VanillaMemory
from memory.aml_api import create_app


def main():
    out=Path('data/event-source-live-20261008');out.mkdir(parents=True,exist_ok=True)
    cfg=dict(load_settings(),RAG_MEMORY_DB=str((out/'memory.sqlite3').resolve()),
        RAG_WRITE_GATE_MODE='on',RAG_BUILD_MODE='on',RAG_MULTIHOP_MODE='off',
        RAG_RERANK_MODE='off',RAG_TAG_MODE='off',RAG_TARGET_MODE='off',RAG_FUSION_QA='off',RAG_RERANK_SELECTION='direct')
    b=VanillaMemory(cfg);results=[]
    cases=[('昨天我去了杭州参加马拉松。','杭州马拉松'),
           ('我原本计划周五去上海，但后来取消了。','上海取消'),
           ('下个月我可能去北京出差，还没有确定。','北京出差')]
    with TestClient(create_app(settings={'AML_AUTH_MODE':'bearer','AML_API_KEY':'test'},backend=b)) as client:
        for i,(text,query) in enumerate(cases):
            payload=dict(user_id=f'event{i}',request_id='r',session_id='s',messages=[dict(role='user',speaker='小林',content=text,timestamp=1704067200000)])
            start=time.perf_counter();response=client.post('/add',json=payload,headers={'Authorization':'Bearer test'});elapsed=time.perf_counter()-start
            start=time.perf_counter();hits=b.event_retriever.search(payload['user_id'],query);read=time.perf_counter()-start
            with closing(b.connect()) as db:
                routes=[dict(r) for r in db.execute('SELECT * FROM rag_memory_routes WHERE user_id=?',(payload['user_id'],))]
                sources=[dict(r) for r in db.execute('SELECT * FROM event_sources WHERE user_id=?',(payload['user_id'],))]
            results.append(dict(input=text,status=response.status_code,write_s=elapsed,search_s=read,
                routes=routes,sources=sources,hits=hits,verbatim=bool(hits and hits[0]['content']==text)))
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps([dict(status=r['status'],write_s=r['write_s'],search_s=r['search_s'],event_sources=len(r['sources']),verbatim=r['verbatim'],routes=[x['memory_type'] for x in r['routes']]) for r in results]))

if __name__=='__main__': main()
