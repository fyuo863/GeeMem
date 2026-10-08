"""Live HTTP add benchmark using root .env models and an isolated database."""
import json
from contextlib import closing
import tempfile
import time
from pathlib import Path
from fastapi.testclient import TestClient
from memory.config import load_settings
from memory.vanilla import VanillaMemory
from memory.aml_api import create_app


def main():
    cases = [
        ('我现在住在杭州，是一名医生。', ['profile']),
        ('王伟是我的同事。', ['relationship']),
        ('以后给我写报告时，先列结论，再列证据。', ['rule']),
        ('我在2024年5月1日参加了杭州马拉松。', ['event']),
        ('你好，谢谢。', ['vector_only']),
        ('王伟是我的同事，他现在住在北京，是一名工程师。', ['profile','relationship']),
    ]
    with tempfile.TemporaryDirectory() as folder:
        cfg = dict(load_settings(),RAG_MEMORY_DB=str(Path(folder)/'db'),RAG_WRITE_GATE_MODE='on',
                   RAG_BUILD_MODE='on',RAG_MULTIHOP_MODE='off',RAG_RERANK_MODE='off',
                   RAG_TAG_MODE='off',RAG_TARGET_MODE='off',RAG_FUSION_QA='off',RAG_RERANK_SELECTION='direct')
        store = VanillaMemory(cfg)
        results=[]
        with TestClient(create_app(settings={'AML_AUTH_MODE':'bearer','AML_API_KEY':'local-test'},backend=store)) as client:
            for i,(text,expected) in enumerate(cases):
                payload=dict(user_id=f'case{i}',request_id='r',session_id='s',messages=[dict(role='user',content=text,timestamp=1704067200000)])
                start=time.perf_counter()
                response=client.post('/add',json=payload,headers={'Authorization':'Bearer local-test'})
                elapsed=time.perf_counter()-start
                with closing(store.connect()) as db:
                    decision=db.execute('SELECT result FROM rag_write_decisions WHERE user_id=?',(payload['user_id'],)).fetchone()
                    routes=[dict(r) for r in db.execute('SELECT * FROM rag_memory_routes WHERE user_id=?',(payload['user_id'],))]
                labels=[r['memory_type'] for r in routes] or (['vector_only'] if decision else [])
                start=time.perf_counter()
                retry=client.post('/add',json=payload,headers={'Authorization':'Bearer local-test'})
                results.append(dict(input=text,expected=expected,labels=labels,classification_correct=set(labels)==set(expected),
                    http_status=response.status_code,elapsed_s=elapsed,retry_status=retry.status_code,retry_ms=(time.perf_counter()-start)*1000,
                    routes=routes))
        path=Path('data/add-builders/results.json');path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps([dict(labels=r['labels'],correct=r['classification_correct'],status=r['http_status'],seconds=r['elapsed_s'],retry_ms=r['retry_ms']) for r in results]))


if __name__=='__main__': main()
