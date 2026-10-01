import itertools
import sqlite3
import pytest
from fastapi.testclient import TestClient
from memory.aml_api import create_app
from memory.vanilla import VanillaMemory
from test_vanilla import Embedder


class Scorer:
    def __init__(self):
        self.documents = []

    def score(self, query, documents):
        self.documents.extend(documents)
        return [2.0 + ('Paris' in d) for d in documents]


@pytest.mark.parametrize('a,b,c', itertools.product([False,True],repeat=3))
def test_add_search_all_combinations_keep_source_and_boundaries(tmp_path,a,b,c):
    model=Scorer()
    backend=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'memory.db'),RAG_RESULT_WINDOW='0',
        RAG_RERANK_CONTEXT='1',RAG_RERANK_SELECTION='context_support',RAG_RERANK_CANDIDATES='1',
        RAG_METADATA_MODE='on' if a else 'off',RAG_TARGET_MODE='on' if b else 'off',
        RAG_SECOND_PASS='on' if c else 'off'),Embedder(),reranker=model)
    app=create_app(backend=backend,settings=dict(AML_AUTH_MODE='none',AML_ALLOW_UNAUTHENTICATED='true'))
    with TestClient(app) as client:
        payload=dict(user_id='u',request_id='r',session_id='s',session_timestamp=1681776000000,
            messages=[dict(role='user',speaker='Alice',content='Where did you move?'),
                      dict(role='assistant',speaker='Bob',content='  Paris, last month.  ')])
        assert client.post('/add',json=payload).status_code==200
        assert client.post('/add',json=payload).status_code==200
        foreign=dict(payload,user_id='foreign',messages=[dict(role='user',speaker='Private',content='Secret Tokyo')])
        assert client.post('/add',json=foreign).status_code==200
        response=client.post('/search',json=dict(user_id='u',query='When did Bob move to Paris?',top_k=2))
        assert response.status_code==200
        hits=response.json()['data']
        assert len(hits)==2 and len({h['id'] for h in hits})==2
        assert {h['content'] for h in hits}=={m['content'] for m in payload['messages']}
        assert all('Secret' not in d and 'Private' not in d for d in model.documents)
        assert any('Bob said' in d for d in model.documents)==a
        changed=dict(payload,session_timestamp=1681862400000)
        assert client.post('/add',json=changed).status_code==409
        assert client.post('/search',json=dict(user_id='missing',query='Paris',top_k=10)).json()=={'data':[]}
