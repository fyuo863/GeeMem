import numpy as np
from fastapi.testclient import TestClient
from memory.aml_api import AMLAdd, AMLSearch, create_app
from memory.vanilla import VanillaMemory


def test_absolute_query_used_consistently_without_mutating_request(tmp_path):
    class E:
        identity='absolute-query-test'
        seen=[]
        def documents(self,texts): return np.tile([1.,0.],(len(texts),1))
        def queries(self,texts):
            self.seen.extend(texts)
            return self.documents(texts)
    class R:
        seen=[]
        def score(self,q,docs):
            self.seen.append(q)
            return [1.]*len(docs)
    cfg=dict(RAG_MEMORY_DB=str(tmp_path/'memory.db'),RAG_TEMPORAL_MODE='on',RAG_TEMPORAL_EXPERIMENT='absolute_query',
             RAG_RESULT_WINDOW='0',AML_AUTH_MODE='bearer',AML_API_KEY='test')
    e,r=E(),R();memory=VanillaMemory(cfg,e,reranker=r)
    text='I started listening to the band today.'
    memory.add(AMLAdd(user_id='u',request_id='r',session_id='s',messages=[dict(role='user',content=text,timestamp=1680220800000)]))
    body=AMLSearch(user_id='u',query='last Friday?',top_k=5,reference_time=1680652800000)
    assert memory.search(body)['data'][0]['content']==text
    assert body.query=='last Friday?'
    assert e.seen[-1]==r.seen[-1]
    assert '2023-03-31 (Friday)' in e.seen[-1]
    with TestClient(create_app(settings=cfg,backend=memory)) as client:
        client.headers['Authorization']='Bearer test'
        response=client.post('/search',json=dict(user_id='u',query='last Friday?',top_k=5))
        assert response.status_code==200
        assert response.json()['data'][0]['content']==text
        assert e.seen[-1]=='last Friday?'
        assert r.seen[-1]=='last Friday?'
