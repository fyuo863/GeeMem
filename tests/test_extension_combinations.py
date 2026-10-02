import itertools
import numpy as np
from fastapi.testclient import TestClient
from memory.aml_api import AMLAdd, create_app
from memory.vanilla import VanillaMemory


class Embedder:
    identity='extension-test'
    def documents(self,texts):return np.tile([1.,0.],(len(texts),1))
    def queries(self,texts):return self.documents(texts)


class Ranker:
    def score(self,query,documents):
        return np.array([3.+float('Paris' in d) for d in documents])


def test_eight_combinations_preserve_api_contract_and_user_isolation(tmp_path):
    common=dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_RESULT_WINDOW='0',RAG_RERANK_CONTEXT='1',
                RAG_RERANK_MODE='local',RAG_RERANK_SELECTION='context_support',RAG_TARGET_MODE='on',
                RAG_METADATA_MODE='on',RAG_SECOND_PASS='on')
    for f,q,m in itertools.product(('off','on'),repeat=3):
        store=VanillaMemory(dict(common,RAG_FUSION_QA=f,RAG_MULTI_QUERY=q,RAG_SOFT_RECALL=m),Embedder(),reranker=Ranker())
        app=create_app(settings=dict(AML_AUTH_MODE='bearer',AML_API_KEY='unit-test'),backend=store)
        with TestClient(app) as client:
            auth={'Authorization':'Bearer unit-test'}
            for user in ['u','other']:
                payload=dict(user_id=user,session_id='s',request_id='r',session_timestamp=1685750400000,
                    messages=[dict(role='user',speaker='Alice',content='Where did you travel?'),
                              dict(role='assistant',speaker='Bob',content='I visited Paris in June.')])
                assert client.post('/add',json=payload,headers=auth).status_code==200
            assert client.post('/search',json=dict(user_id='u',query='Where did both Alice and Bob travel in June 2023?',top_k=10)).status_code==401
            hits=client.post('/search',json=dict(user_id='u',query='Where did both Alice and Bob travel in June 2023?',top_k=10),headers=auth).json()['data']
            assert len(hits)==2 and len({h['id'] for h in hits})==2
            assert {h['content'] for h in hits}=={x['content'] for x in payload['messages']}
            assert client.post('/search',json=dict(user_id='absent',query='Paris',top_k=10),headers=auth).json()=={'data':[]}


def test_selected_fm_official_fields_top100_persistence_and_scores(tmp_path):
    """Official envelopes omit our optional speaker/session metadata."""
    cfg = dict(RAG_MEMORY_DB=str(tmp_path/'official.sqlite3'), RAG_RESULT_WINDOW='0',
        RAG_RERANK_CONTEXT='1', RAG_RERANK_MODE='local', RAG_RERANK_SELECTION='context_support',
        RAG_TARGET_MODE='on', RAG_METADATA_MODE='on', RAG_SECOND_PASS='on',
        RAG_FUSION_QA='on', RAG_MULTI_QUERY='off', RAG_SOFT_RECALL='on')
    auth = {'Authorization': 'Bearer contract-test'}
    payload = dict(request_id='eval:test:chunk-0', user_id='eval:test:conv-0',
        session_id='eval:test:sample:0', messages=[dict(role='user' if i%2==0 else 'assistant',
            content=f'Original memory {i}: I visited Paris.',
            **({'timestamp': 1704067200000+i*1000} if i%2 else {})) for i in range(105)])
    query = dict(user_id=payload['user_id'], query='Which city did I visit?',
        options=['A. Paris', 'B. Tokyo'], top_k=100)
    def app():
        backend = VanillaMemory(cfg, Embedder(), reranker=Ranker())
        return create_app(settings=dict(AML_AUTH_MODE='bearer', AML_API_KEY='contract-test'), backend=backend)
    with TestClient(app()) as client:
        response = client.post('/add', json=payload, headers=auth)
        assert response.status_code == 200
        assert response.json() == dict(success=True, **{k: payload[k] for k in ('request_id','user_id','session_id')})
        assert client.post('/add', json=payload, headers=auth).json() == response.json()
        response = client.post('/search', json=query, headers=auth)
        assert response.status_code == 200
        hits = response.json()['data']
        assert len(hits) == len({h['id'] for h in hits}) == 100
        assert all(h['content'] in {m['content'] for m in payload['messages']} for h in hits)
        assert all(a['score'] >= b['score'] for a,b in zip(hits,hits[1:]))
        for hit in hits:
            source = next(m for m in payload['messages'] if m['content']==hit['content'])
            assert ('created_at' in hit) == ('timestamp' in source)
        assert client.post('/search',json=dict(query,user_id='isolated'),headers=auth).json()=={'data':[]}
    with TestClient(app()) as client:
        assert client.post('/search',json=query,headers=auth).json()['data'] == hits
