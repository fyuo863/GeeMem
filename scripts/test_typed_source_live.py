"""Real classification and embeddings, no structured extraction."""
from contextlib import closing
from pathlib import Path
import json
import time
from fastapi.testclient import TestClient
from memory.config import load_settings
from memory.vanilla import VanillaMemory
from memory.aml_api import create_app


def main():
    out=Path('data/typed-source-live-20261008');out.mkdir(parents=True,exist_ok=True)
    cfg=dict(load_settings(),RAG_MEMORY_DB=str((out/'memory.sqlite3').resolve()),
        RAG_WRITE_GATE_MODE='on',RAG_BUILD_MODE='on',RAG_MULTIHOP_MODE='off',
        RAG_RERANK_MODE='off',RAG_TAG_MODE='off',RAG_TARGET_MODE='off',RAG_FUSION_QA='off',RAG_RERANK_SELECTION='direct')
    b=VanillaMemory(cfg);results=[]
    cases=[('profile', 'My favorite football team is Arsenal.', 'Arsenal'),
           ('relationship', 'Veda is my sister and Arun is her teacher.', 'Veda teacher'),
           ('rule', 'When writing reports for me, put the conclusion first and always use bullet points.', 'reports conclusion'),
           ('event', 'I planned to visit Shanghai on Friday, but cancelled the trip.', 'Shanghai cancelled')]
    with TestClient(create_app(settings={'AML_AUTH_MODE':'bearer','AML_API_KEY':'test'},backend=b)) as client:
        for i,(kind,text,query) in enumerate(cases):
            payload=dict(user_id=f'event{i}',request_id='r',session_id='s',messages=[dict(role='user',speaker='小林',content=text,timestamp=1704067200000)])
            start=time.perf_counter();response=client.post('/add',json=payload,headers={'Authorization':'Bearer test'});elapsed=time.perf_counter()-start
            start=time.perf_counter();hits=getattr(b, kind+'_retriever').search(payload['user_id'],query,fallback=False);read=time.perf_counter()-start
            with closing(b.connect()) as db:
                routes=[dict(r) for r in db.execute('SELECT * FROM rag_memory_routes WHERE user_id=?',(payload['user_id'],))]
                sources=[dict(r) for r in db.execute(f'SELECT * FROM {kind}_sources WHERE user_id=?',(payload['user_id'],))]
            results.append(dict(kind=kind,input=text,status=response.status_code,write_s=elapsed,search_s=read,
                routes=routes,sources=sources,hits=hits,verbatim=bool(hits and hits[0]['content']==text)))
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps([dict(kind=r['kind'],status=r['status'],write_s=r['write_s'],search_s=r['search_s'],typed_sources=len(r['sources']),verbatim=r['verbatim'],routes=[x['memory_type'] for x in r['routes']]) for r in results]))

if __name__=='__main__': main()
